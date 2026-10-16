import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from coordinate_contracts import score_normalized_grounding


class CoordinateContractTests(unittest.TestCase):
    def setUp(self):
        self.item = {'axis': 'grounding', 'answer_type': 'bbox', 'width': 640,
                     'height': 480, 'answer_key': [160, 120, 480, 360]}

    def test_fixed_transform_matches_pixel_box(self):
        result = score_normalized_grounding('[250,250,750,750]', self.item)
        self.assertEqual(result['parsed_answer'], self.item['answer_key'])
        self.assertEqual(result['iou'], 1)
        self.assertEqual(result['correct_bit'], 1)

    def test_small_values_are_still_normalized_not_guessed_pixels(self):
        result = score_normalized_grounding('[1,1,2,2]', self.item)
        self.assertEqual(result['parsed_answer'], [.64, .48, 1.28, .96])
        self.assertEqual(result['correct_bit'], 0)

    def test_out_of_contract_is_failure_not_clamped(self):
        result = score_normalized_grounding('[0,0,1001,1000]', self.item)
        self.assertEqual(result['parse_fail_flag'], 1)
        self.assertEqual(result['correct_bit'], 0)


if __name__ == '__main__':
    unittest.main()
