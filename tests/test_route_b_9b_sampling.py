"""Fixed prompt/update budgets and exclusion provenance for the9B extension."""
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from route_b_9b_loop import select_minimal, arm_rows
from experiment_common import AXES


class NineBSamplingTests(unittest.TestCase):
    def setUp(self):
        self.pool = [{'axis': axis, 'item_id': f'{axis}-{i:04}', 'bootstrap_unit': f'{axis}-unit-{i}'}
                     for axis in AXES for i in range(650)]

    def test_balanced500_is_reproducible_and_from_eligible_pool(self):
        removed = {f'consistency-{i:04}' for i in range(5)}
        eligible = [r for r in self.pool if r['item_id'] not in removed]
        a, b = select_minimal(eligible), select_minimal(list(reversed(eligible)))
        self.assertEqual(a, b)
        self.assertEqual(len({r['item_id'] for r in a}), 500)
        self.assertFalse({r['item_id'] for r in a} & removed)
        self.assertEqual(Counter(r['axis'] for r in a), Counter({axis: 125 for axis in AXES}))

    def test_u_formal1000_updates_are_exactly250_per_axis(self):
        rows = arm_rows(self.pool, 'formal', 'U', 42)
        self.assertEqual(len(rows), 600)
        updates = [rows[i % len(rows)] for i in range(1000)]
        self.assertEqual(Counter(r['axis'] for r in updates), Counter({axis: 250 for axis in AXES}))
        self.assertEqual(rows, arm_rows(self.pool, 'formal', 'U', 42))

    def test_single_axis_arms_have_equal600_prompt_budgets(self):
        for arm, axis in [('A1', 'mmstar'), ('A2', 'consistency')]:
            rows = arm_rows(self.pool, 'formal', arm, 137)
            self.assertEqual(len(rows), 600)
            self.assertEqual({r['axis'] for r in rows}, {axis})
            self.assertNotEqual(rows, arm_rows(self.pool, 'formal', arm, 42))

    def test_n_is_not_a_training_arm(self):
        with self.assertRaises(AssertionError):
            arm_rows(self.pool, 'formal', 'N', 42)


if __name__ == '__main__':
    unittest.main()
