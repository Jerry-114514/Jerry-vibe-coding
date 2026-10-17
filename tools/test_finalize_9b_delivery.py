"""Reject incomplete/mismatched final conclusions; synthetic data only."""
from copy import deepcopy
import unittest

from finalize_9b_delivery import validate_statistics, SEEDS, AXES, CONTRASTS


class FinalStatisticsTests(unittest.TestCase):
    def fixture(self):
        models = [('N', 42)] + [(a, s) for s in SEEDS for a in ('A1', 'A2', 'U')]
        scores = {f'{a}-seed{s}': {axis: .5 for axis in AXES} for a, s in models}
        plan = {'family_size': 12, 'bootstrap_replicates': 10000, 'bootstrap_seed': 2026101703}
        data = {'status': 'all16_verified', 'matched_seeds': SEEDS, **plan, 'scores': scores,
                'ledger_sha256': {name: name + '-synthetic' for name in scores},
                'paired_contrasts': {label: {axis: {'difference': 0., 'pointwise_ci95': [0., 0.],
                                      'bonferroni_familywise_ci95': [0., 0.]} for axis in AXES} for label in CONTRASTS}}
        evaluations = {(a, s): ({'axis_scores': scores[f'{a}-seed{s}'],
                                 'ledger_sha256': data['ledger_sha256'][f'{a}-seed{s}']}, {}) for a, s in models}
        return data, plan, evaluations

    def test_zero_touching_intervals_do_not_claim_a_direction(self):
        rows = validate_statistics(*self.fixture())
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row['direction'] == 'includes_zero' for row in rows))

    def test_missing_seed_cannot_be_a_final_report(self):
        data, plan, evaluations = self.fixture()
        del data['scores']['A1-seed1024']
        with self.assertRaises(AssertionError):
            validate_statistics(data, plan, evaluations)

    def test_wrong_difference_or_raw_ledger_is_rejected(self):
        data, plan, evaluations = self.fixture()
        altered = deepcopy(data)
        altered['paired_contrasts']['A1-U']['grounding']['difference'] = .1
        with self.assertRaises(AssertionError):
            validate_statistics(altered, plan, evaluations)
        altered = deepcopy(data)
        altered['ledger_sha256']['A1-seed42'] = 'other-ledger'
        with self.assertRaises(AssertionError):
            validate_statistics(altered, plan, evaluations)

    def test_reporting_budget_or_seed_cannot_drift(self):
        data, plan, evaluations = self.fixture()
        for key, value in [('bootstrap_replicates', 2000), ('bootstrap_seed', 0), ('family_size', 4)]:
            altered = deepcopy(data)
            altered[key] = value
            with self.assertRaises(AssertionError):
                validate_statistics(altered, plan, evaluations)


if __name__ == '__main__':
    unittest.main()
