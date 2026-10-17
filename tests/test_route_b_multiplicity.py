"""Multiplicity changes interval coverage, never effect estimates or raw scores."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from report_route_b_multiplicity import intervals


class MultiplicityTests(unittest.TestCase):
    def test_adjusted_interval_contains_pointwise_interval(self):
        result = intervals(np.linspace(-.5, .5, 10001))
        low, high = result['pointwise_ci95']
        adjusted_low, adjusted_high = result['bonferroni_familywise_ci95']
        self.assertLess(adjusted_low, low)
        self.assertGreater(adjusted_high, high)

    def test_zero_effect_remains_zero(self):
        result = intervals(np.zeros(10000))
        self.assertEqual(result['pointwise_ci95'], [0., 0.])
        self.assertEqual(result['bonferroni_familywise_ci95'], [0., 0.])

    def test_family_of_one_needs_no_correction(self):
        result = intervals(np.random.default_rng(42).normal(size=10000), family_size=1)
        self.assertEqual(result['pointwise_ci95'], result['bonferroni_familywise_ci95'])


if __name__ == '__main__':
    unittest.main()
