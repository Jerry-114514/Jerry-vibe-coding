from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_7b_grounding_contracts import resized_score


class GroundingAuditTests(unittest.TestCase):
    def test_resized_pixels_map_by_each_axis_and_keep_contract_fixed(self):
        item={'width':640,'height':480,'answer_key':[64,96,320,384]}
        result=resized_score('[32,24,160,96]',item,320,120)
        self.assertEqual(result['parsed_answer'],item['answer_key'])
        self.assertEqual(result['iou'],1)
        self.assertEqual(result['correct_bit'],1)

    def test_out_of_contract_never_falls_back_to_other_units(self):
        item={'width':640,'height':480,'answer_key':[64,96,320,384]}
        self.assertEqual(resized_score('[100,200,500,800]',item,320,120)['parse_fail_flag'],1)


if __name__=='__main__':unittest.main()
