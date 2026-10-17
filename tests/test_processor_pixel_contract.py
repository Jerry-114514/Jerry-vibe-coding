from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from processor_pixel_contract import score,item_with_contract,assert_runtime_grid


class ProcessorContractTests(unittest.TestCase):
    def item(self):
        return {'axis':'grounding','width':640,'height':480,'answer_type':'bbox','answer_key':[64,96,320,384],
                'processor_width':320,'processor_height':120,'processor_grid_thw':[1,8,20],
                'question':'Find object in original-image pixel coordinates. The image is 640 pixels wide and 480 pixels high.'}

    def test_anisotropic_mapping_keeps_raw_and_separate_boxes(self):
        text='[32,24,160,96]';result=score(text,self.item())
        self.assertEqual(result['raw_generation'],text)
        self.assertEqual(result['parsed_processor_pixels'],[32,24,160,96])
        self.assertEqual(result['parsed_answer'],[64,96,320,384])
        self.assertEqual(result['iou'],1)

    def test_no_normalized_fallback_or_gt_based_unit_selection(self):
        result=score('[100,200,500,800]',self.item())
        self.assertIsNone(result['parsed_answer']);self.assertEqual(result['correct_bit'],0)

    def test_prompt_declares_exact_dimensions_and_preserves_source(self):
        item=self.item();row=item_with_contract(item,{'processor_width':320,'processor_height':120})
        self.assertIn('320 pixels wide and 120 pixels high',row['question'])
        self.assertIn('640 pixels wide',item['question'])
        item['question']='unknown';
        with self.assertRaises(AssertionError):item_with_contract(item,{'processor_width':320,'processor_height':120})

    def test_runtime_grid_mismatch_is_rejected(self):
        class Grid:
            def tolist(self):return [1,8,24]
        with self.assertRaises(AssertionError):assert_runtime_grid({'image_grid_thw':[Grid()]},self.item())

    def test_other_axes_unchanged(self):
        item={'axis':'pope','answer_type':'yes_no','answer_key':'yes','question':'object?'}
        self.assertEqual(item_with_contract(item,{}),item)
        self.assertEqual(score('yes',item)['correct_bit'],1)


if __name__=='__main__':unittest.main()
