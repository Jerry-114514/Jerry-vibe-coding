import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from evaluate_9b_native import native_item, score_native


class NativeCoordinateTests(unittest.TestCase):
    def item(self):
        return {'axis':'grounding','width':640,'height':480,'answer_type':'bbox_xyxy_pixels',
                'answer_key':[64,96,320,384],
                'question':'Locate object. Return only box in original-image pixel coordinates. The image is 640 pixels wide and 480 pixels high.'}

    def test_anisotropic_conversion_and_raw_preservation(self):
        raw='[100,200,500,800]';result=score_native(raw,self.item())
        self.assertEqual(result['raw_generation'],raw)
        self.assertEqual(result['parsed_answer'],[64,96,320,384])
        self.assertEqual(result['correct_bit'],1)
        self.assertEqual(result['iou'],1)

    def test_invalid_and_out_of_contract_boxes_rejected(self):
        for raw in ('[0,0,1001,500]','[100,200,50,500]','[0,0,NaN,500]','The box is [100,200,500,800]'):
            self.assertEqual(score_native(raw,self.item())['parse_fail_flag'],1)

    def test_native_prompt_is_explicit_and_no_unknown_rewrite(self):
        source=self.item();row=native_item(source)
        self.assertIn('normalized coordinates from 0 to 1000',row['question'])
        self.assertIn('original-image pixel',source['question'])
        source['question']='unrecognized coordinate contract'
        with self.assertRaises(AssertionError):native_item(source)

    def test_non_grounding_metric_unchanged(self):
        row={'axis':'pope','answer_type':'yes_no','answer_key':'yes','question':'object?'}
        self.assertEqual(native_item(row),row)
        self.assertEqual(score_native('yes',row)['correct_bit'],1)

    def test_registered_child_deadline_actually_stops_process(self):
        from run_9b_native_evaluation import child
        with self.assertRaisesRegex(AssertionError,'wall budget exhausted'):
            child([sys.executable,'-c','import time; time.sleep(5)'],'unused-synthetic-token',timeout=.1)


if __name__=='__main__':unittest.main()
