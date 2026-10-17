"""Restore-chain checks use synthetic bytes, never substitute for model checks."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import audit_9b_student_delivery as audit
from experiment_common import ROOT, sha, write_json


class DeliveryAuditTests(unittest.TestCase):
    def test_canonicalization_only_accepts_known_snapshot_prefixes(self):
        original = 'runs/route-b-9b-122b-20261017/student-experiment/protocol.json'
        self.assertEqual(audit.canonical('runs/9b-student-delivery-20261017/stamp/files/' + original), original)
        self.assertEqual(audit.canonical('runs/9b-teacher-delivery-20261017/stamp/files/' + original), original)
        unknown = 'runs/unrelated/stamp/files/' + original
        self.assertEqual(audit.canonical(unknown), unknown)

    def test_registration_alias_preserves_actual_input_identity(self):
        root = audit.OUT.relative_to(audit.ROOT).as_posix()
        self.assertEqual(audit.canonical(root + '/registration/protocol.json'), root + '/protocol.json')
        expected = (audit.BASE / 'evaluation-input-preparation/formal-manifest.jsonl').relative_to(audit.ROOT).as_posix()
        self.assertEqual(audit.canonical(root + '/registration/eval-input-formal-manifest.jsonl'), expected)
        unknown = root + '/registration/unknown.json'
        self.assertEqual(audit.canonical(unknown), unknown)

    def test_missing_recovery_requires_fresh_archive_coverage_and_restore_chain(self):
        temporary = ROOT / 'tmp'
        temporary.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=temporary) as name:
            folder = Path(name) / 'formal/A1-seed42'
            folder.mkdir(parents=True)
            self.assertTrue(folder.resolve().is_relative_to(ROOT.resolve()))
            recovery = folder / 'resume.pt'
            key = recovery.relative_to(ROOT).as_posix()
            self.assertFalse(audit.archived_recovery(recovery, {}))
            write_json(folder / 'complete.json', {'resume_sha256': 'frozen-recovery-hash'})
            write_json(folder / 'training-verification.json', {
                'status': 'passed_local_recovery_before_offload', 'resume_sha256': 'frozen-recovery-hash'})
            write_json(folder / 'training-delivery.json', {'synthetic': 'receipt'})
            write_json(folder / 'training-inventory.json', {'synthetic': 'inventory'})
            pointer = {'resume_sha256': 'frozen-recovery-hash', 'bytes': 123,
                       'receipt_sha256': sha(folder / 'training-delivery.json'),
                       'inventory_sha256': sha(folder / 'training-inventory.json'), 'archive_member': key}
            write_json(folder / 'resume-offloaded.json', pointer)
            self.assertFalse(audit.archived_recovery(recovery, {}))
            self.assertFalse(audit.archived_recovery(recovery, {key: {(124, 'frozen-recovery-hash')}}))
            self.assertTrue(audit.archived_recovery(recovery, {key: {(123, 'frozen-recovery-hash')}}))
            pointer['receipt_sha256'] = 'altered'
            write_json(folder / 'resume-offloaded.json', pointer)
            with self.assertRaises(AssertionError):
                audit.archived_recovery(recovery, {key: {(123, 'frozen-recovery-hash')}})

    def test_raw_http_evidence_is_required_without_recursive_archive_receipts(self):
        temporary = ROOT / 'tmp'
        temporary.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=temporary) as name:
            base = Path(name)
            evidence = base / 'api400-investigation'
            evidence.mkdir()
            names = ('diagnostic-request.json', 'diagnostic-response.json', 'github-inventory.json', 'github-storage.json')
            for filename in names:
                write_json(evidence / filename, {'synthetic': True})
            with patch.object(audit, 'BASE', base):
                required = audit.requirements()
            self.assertIn(evidence / names[0], required)
            self.assertIn(evidence / names[1], required)
            self.assertNotIn(evidence / names[2], required)
            self.assertNotIn(evidence / names[3], required)


if __name__ == '__main__':
    unittest.main()
