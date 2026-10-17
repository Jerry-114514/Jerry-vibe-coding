"""Multiplicity changes interval coverage, never effect estimates or raw scores."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from report_route_b_multiplicity import intervals
import report_route_b_multiplicity as report
from experiment_common import AXES
from route_b_staged import SEEDS


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

    def test_full_report_preserves_paired_seed_and_image_draws(self):
        complete = {}
        jobs = [('N', 42)] + [(a, s) for s in SEEDS for a in ('A1', 'A2', 'U')]
        for arm, seed in jobs:
            records = {}
            for axis in AXES:
                for i in range(4):
                    bit = (i // 2 + SEEDS.index(seed)) % 2 if arm in ('A1', 'A2') else 0
                    records[(axis, 'synthetic', str(i))] = {'image_group': str(i // 2), 'correct_bit': bit}
            marker = {'axis_scores': {axis: .5 if arm in ('A1', 'A2') else 0. for axis in AXES},
                      'ledger_sha256': 'synthetic-test-only'}
            complete[(arm, seed)] = (marker, records)
        workspace = Path(__file__).resolve().parents[1]
        temporary_root = workspace / 'tmp'
        temporary_root.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=temporary_root) as folder:
            path = Path(folder)
            self.assertTrue(path.resolve().is_relative_to(workspace))
            (path / 'reporting-amendment.json').write_text('{}', encoding='utf-8')
            plan = {'random_seed': 42, 'coverage_caveat': 'Synthetic test only'}
            with patch.object(report, 'REPORT', path), patch.object(report, 'REPLICATES', 400), \
                 patch.object(report, 'register', return_value=plan), \
                 patch.object(report, 'read_complete_evaluations', return_value=complete), \
                 patch('verify_route_b_stage_outputs.verify'):
                result = report.analyze()
            self.assertEqual(len(result['scores']), 16)
            self.assertEqual(result['family_size'], 12)
            self.assertEqual(result['matched_seeds'], SEEDS)
            # A1 and A2 have variable seed/image outcomes, but they are identical
            # within each paired seed and image. Their paired difference is zero
            # on every draw; independent arm resampling would break this.
            for axis in AXES:
                contrast = result['paired_contrasts']['A1-A2'][axis]
                self.assertEqual(contrast['difference'], 0.)
                self.assertEqual(contrast['pointwise_ci95'], [0., 0.])
                self.assertEqual(contrast['bonferroni_familywise_ci95'], [0., 0.])
                self.assertEqual(result['paired_contrasts']['A1-N'][axis]['difference'], .5)


if __name__ == '__main__':
    unittest.main()
