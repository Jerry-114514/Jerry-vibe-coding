from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from precision_bridge_decision import assess,MODELS,AXES


class BridgeDecisionTests(unittest.TestCase):
    def setUp(self):
        self.local={m:{a:.5 for a in AXES} for m in MODELS}
        self.api={m:{a:.5049 for a in AXES} for m in MODELS}

    def test_all_eight_differences_must_pass(self):
        self.assertTrue(assess(self.local,self.api)['score_threshold_passed'])
        self.api['qwen7b']['grounding']=.51
        self.assertFalse(assess(self.local,self.api)['score_threshold_passed'])

    def test_exact_boundary_does_not_pass(self):
        self.api['qwen3b']['pope']=.505
        self.assertFalse(assess(self.local,self.api)['score_threshold_passed'])

    def test_opposite_changes_cannot_cancel_out(self):
        self.api['qwen3b']['pope']=.51
        self.api['qwen3b']['grounding']=.49
        self.assertFalse(assess(self.local,self.api)['score_threshold_passed'])

    def test_missing_checkpoint_and_nonfinite_score_rejected(self):
        with self.assertRaises(ValueError): assess(self.local,{'qwen3b':self.api['qwen3b']})
        self.api['qwen7b']['pope']=float('nan')
        with self.assertRaises(ValueError): assess(self.local,self.api)


if __name__=='__main__': unittest.main()
