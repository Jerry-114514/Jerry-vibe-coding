"""Recovery/delivery safety and paired statistics before any9B GPU formal run."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import analyze_9b_formal as analysis
import run_9b_formal_when_ready as control
from experiment_common import ROOT, AXES, read_json, write_json, sha
from route_b_9b_loop import SEEDS


class FormalControlTests(unittest.TestCase):
    def setUp(self):
        temporary = ROOT / 'tmp'
        temporary.mkdir(exist_ok=True)
        self.scratch = TemporaryDirectory(dir=temporary)
        self.root = Path(self.scratch.name)
        self.assertTrue(self.root.resolve().is_relative_to(ROOT.resolve()))

    def tearDown(self):
        self.scratch.cleanup()

    def fixture(self, mismatch=False):
        folder = self.root / 'formal/A1-seed42'
        (folder / 'adapter').mkdir(parents=True)
        checkpoint = folder / 'resume.pt'
        checkpoint.write_bytes(b'synthetic recovery fixture, not a model')
        (folder / 'adapter/adapter_model.safetensors').write_bytes(b'adapter retained')
        write_json(folder / 'train.meta.json', {'arm': 'A1', 'seed': 42, 'phase': 'formal'})
        write_json(folder / 'complete.json', {'resume_sha256': sha(checkpoint)})
        inventory = folder / 'training-inventory.json'
        write_json(inventory, {'files': [{'path': str(checkpoint.relative_to(ROOT)),
                   'sha256': 'wrong' if mismatch else sha(checkpoint)}]})
        write_json(folder / 'training-delivery.json', {'release_url': 'synthetic',
                   'assets': [{'name': 'fixture-inventory.json', 'sha256': sha(inventory)}]})
        return folder, checkpoint

    def test_server_failure_keeps_local_recovery(self):
        folder, checkpoint = self.fixture()
        with patch.object(control, 'OUT', self.root), patch.object(control, 'verify_training'), \
             patch.object(control, 'verify_receipt', side_effect=AssertionError('server mismatch')):
            with self.assertRaises(AssertionError):
                control.offload_completed_recovery(folder, 'test-only')
        self.assertTrue(checkpoint.exists())
        self.assertFalse((folder / 'resume-offloaded.json').exists())

    def test_mismatched_archive_keeps_local_recovery(self):
        folder, checkpoint = self.fixture(mismatch=True)
        with patch.object(control, 'OUT', self.root), patch.object(control, 'verify_training'), \
             patch.object(control, 'verify_receipt') as server:
            with self.assertRaises(AssertionError):
                control.offload_completed_recovery(folder, 'test-only')
            server.assert_not_called()
        self.assertTrue(checkpoint.exists())

    def test_verified_offload_retains_adapter_and_restore_pointer(self):
        folder, checkpoint = self.fixture()
        with patch.object(control, 'OUT', self.root), patch.object(control, 'verify_training'), \
             patch.object(control, 'verify_receipt', return_value={'status': 'passed'}):
            control.offload_completed_recovery(folder, 'test-only')
        self.assertFalse(checkpoint.exists())
        self.assertTrue((folder / 'adapter/adapter_model.safetensors').exists())
        pointer = read_json(folder / 'resume-offloaded.json')
        self.assertTrue(pointer['local_completed_optimizer_copy_only'])
        self.assertIn('archive_member', pointer)

    def test_incomplete_tail_preserved_and_complete_rows_unchanged(self):
        path = self.root / 'eval.jsonl'
        path.write_bytes(b'{"item":1}\n{"item":2')
        with patch.object(control, 'OUT', self.root):
            control.preserve_incomplete_journal_tail(path)
            control.preserve_incomplete_journal_tail(path)
        self.assertEqual(path.read_bytes(), b'{"item":1}\n')
        tails = list(self.root.glob('eval.jsonl.incomplete-tail-*'))
        self.assertEqual(len(tails), 1)
        self.assertEqual(tails[0].read_bytes(), b'{"item":2')

    def test_wall_budget_does_not_reset(self):
        with patch.object(control, 'OUT', self.root):
            first = control.remaining_budget('test-phase', 1)
            original = (self.root / 'test-phase-wall-budget.json').read_bytes()
            second = control.remaining_budget('test-phase', 1)
        self.assertLessEqual(second, first)
        self.assertEqual(original, (self.root / 'test-phase-wall-budget.json').read_bytes())

    def test_full16_model_statistics_keep_identical_arms_paired(self):
        complete = {}
        for arm, seed in [('N', 42)] + [(a, s) for s in SEEDS for a in ('A1', 'A2', 'U')]:
            records = {(axis, 'synthetic', str(i)): {'image_group': str(i // 2),
                       'correct_bit': (i // 2 + SEEDS.index(seed)) % 2 if arm in ('A1', 'A2') else 0}
                       for axis in AXES for i in range(4)}
            complete[(arm, seed)] = ({'axis_scores': {a: .5 if arm in ('A1', 'A2') else 0. for a in AXES},
                                      'ledger_sha256': 'synthetic'}, records)
        with patch.object(analysis, 'OUT', self.root), \
             patch.object(analysis, 'verify_formal', return_value=({}, complete)):
            result = analysis.analyze(replicates=400)
        self.assertEqual(len(result['scores']), 16)
        self.assertEqual(result['family_size'], 12)
        for axis in AXES:
            self.assertEqual(result['paired_contrasts']['A1-A2'][axis]['difference'], 0.)
            self.assertEqual(result['paired_contrasts']['A1-A2'][axis]['bonferroni_familywise_ci95'], [0., 0.])
            self.assertEqual(result['paired_contrasts']['A1-N'][axis]['difference'], .5)


if __name__ == '__main__':
    unittest.main()
