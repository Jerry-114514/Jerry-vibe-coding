from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from experiment_common import AXES, read_json, ROOT
from paired_gate_v2 import JointImageBootstrap, analyze_family


def records():
    rows = []
    for axis in AXES:
        for index in range(100):
            rows.append({'axis': axis, 'stratum': axis+'/one', 'unit': axis+str(index),
                         'image_group': str(index), 'correct_bit': 0, 'parse_fail_flag': 0})
    return [rows, [{**row, 'correct_bit': int(int(row['image_group']) < 40)} for row in rows],
            [{**row, 'correct_bit': int(int(row['image_group']) < 80)} for row in rows]]


class PairedGateTests(unittest.TestCase):
    def test_inner_weights_preserve_same_images_across_axes_and_models(self):
        engine = JointImageBootstrap(records())
        rng = np.random.default_rng(123)
        outer = engine.draw_counts(rng)
        inner = engine.draw_counts(rng, outer, size=32)
        scores, floors = engine.scores(engine.totals(inner))
        for axis in range(1, 4):
            np.testing.assert_allclose(scores['wrong'][:, :, 0], scores['wrong'][:, :, axis])
        np.testing.assert_array_equal(floors['wrong'], 0)

    def test_identical_axes_cannot_create_strict_reversal_or_positive_excess(self):
        protocol = read_json(ROOT/'configs/experiment.json')
        summary, draws = analyze_family(records(), protocol, outer=12, inner=16, seed=123)
        self.assertEqual(summary['point']['strict_scale'], 0)
        self.assertEqual(summary['point']['strict_noise'], 0)
        self.assertAlmostEqual(summary['point']['F_difference'], 0)
        self.assertNotEqual(summary['statistical_candidate_decision'], 'go')
        self.assertTrue(all(d['strict_difference'] == 0 for d in draws))

    def test_inner_draw_cannot_reintroduce_feature_absent_from_outer(self):
        engine = JointImageBootstrap(records())
        parent = [np.zeros_like(counts) for _, counts in engine.blocks]
        for result, (_, counts) in zip(parent, engine.blocks):
            result[0] = counts.sum()
        inner = engine.draw_counts(np.random.default_rng(123), parent, size=32)
        for matrix in inner:
            self.assertTrue(np.all(matrix[:, 1:] == 0))

    def test_unpaired_source_image_rejected(self):
        data = records()
        data[1][0]['image_group'] = 'different-image'
        with self.assertRaises(ValueError):
            JointImageBootstrap(data)


if __name__ == '__main__':
    unittest.main()
