import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from scoring import box_iou, center_box, consistency_bit


class PhaseAScoringTests(unittest.TestCase):
    def test_iou_boundary_uses_box_area_without_plus_one(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [0, 0, 5, 10]), 0.5)
        self.assertEqual(int(box_iou([0, 0, 10, 10], [0, 0, 5, 10]) >= 0.5), 1)
        self.assertEqual(box_iou([0, 0, 5, 5], [5, 0, 10, 5]), 0.0)

    def test_iou_invalid_predictions_score_zero(self):
        for prediction in ([1, 1, 0, 0], [0, 0, 0, 0], [0, 0, float("nan"), 1]):
            self.assertEqual(box_iou(prediction, [0, 0, 10, 10]), 0.0)

    def test_center_box_is_centered_with_requested_training_area(self):
        box = center_box(640, 480, 10000)
        self.assertAlmostEqual((box[0] + box[2]) / 2, 320)
        self.assertAlmostEqual((box[1] + box[3]) / 2, 240)
        self.assertAlmostEqual((box[2] - box[0]) * (box[3] - box[1]), 10000)
        self.assertEqual(center_box(100, 50, 100000), [0.0, 0.0, 100.0, 50.0])

    def test_consistency_is_all_correct_not_majority_or_agreement(self):
        self.assertEqual(consistency_bit([1, 1, 1, 1, 1]), 1)
        self.assertEqual(consistency_bit([1, 1, 1, 1, 0]), 0)
        self.assertEqual(consistency_bit([0, 0, 0, 0, 0]), 0)

    def test_missing_paraphrase_cannot_pass(self):
        with self.assertRaises(ValueError):
            consistency_bit([1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
