from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from protect_9b_disk import allowed, SEVEN, OUT


class DiskScopeTests(unittest.TestCase):
    def test_only_completed7b_eval_scope_can_be_considered(self):
        self.assertTrue(allowed(SEVEN/'formal/A1-seed42/eval.jsonl'))
        for path in [OUT/'formal/A1-seed42/resume.pt',OUT/'native-evaluation-v2/formal/A1-seed42/eval.jsonl',
                     SEVEN/'formal/A1-seed42/loss.jsonl',SEVEN/'minimal/A1-seed42/eval.jsonl',
                     SEVEN/'formal/A1-seed42/adapter/adapter_model.safetensors']:
            self.assertFalse(allowed(path))


if __name__=='__main__':unittest.main()
