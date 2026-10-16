"""Check requested A1-A2/A1-U/A1-N pairing without conflating partial seeds."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from route_b_completion_reports import paired_intervals
from experiment_common import AXES


class RequestedContrastTest(unittest.TestCase):
    def test_constant_effect_and_single_seed_limit(self):
        def records(bit):
            return {(axis,'synthetic',str(i)):{'correct_bit':bit,'image_group':str(i)} for axis in AXES for i in range(4)}
        complete = {('A1',42):({},records(1)),('A2',42):({},records(0)),
                    ('U',42):({},records(1)),('N',42):({},records(0))}
        result = paired_intervals(complete,100)
        self.assertEqual(set(result),{'A1-A2','A1-U','A1-N'})
        for name,value in [('A1-A2',1.),('A1-U',0.),('A1-N',1.)]:
            self.assertFalse(result[name]['training_seed_uncertainty_included'])
            for axis in AXES:
                self.assertEqual(result[name]['axes'][axis]['difference'],value)
                self.assertEqual(result[name]['axes'][axis]['ci95'],[value,value])
        del complete[('A2',42)]
        result = paired_intervals(complete,50)
        self.assertEqual(result['A1-A2']['status'],'pending_matched_complete_evaluations')
        self.assertEqual(result['A1-N']['status'],'available')


if __name__ == '__main__':
    unittest.main()
