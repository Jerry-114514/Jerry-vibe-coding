from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import resilient_9b_delivery as retry
import cleanup_archived_7b_recovery as cleanup
from watch_9b_transport_recovery import transient


class NightDeliveryTests(unittest.TestCase):
    def test_watchdog_excludes_permanent_errors_and_numeric_failures(self):
        self.assertTrue(transient({'type':'HTTPError','message':'503 Server Error'}))
        self.assertTrue(transient({'type':'ReadTimeout','message':'synthetic'}))
        for row in ({'type':'HTTPError','message':'403 Client Error'},
                    {'type':'AssertionError','message':'N calibration below threshold'},
                    {'type':'OutOfMemoryError','message':'CUDA out of memory'}):
            self.assertFalse(transient(row))

    def test_transient_network_retries_but_validation_does_not(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"tmp") as tmp,patch.object(retry,'OUT',Path(tmp)),patch.object(retry.time,'sleep'):
            values=iter([requests.ConnectionError('synthetic'),42])
            def operation():
                value=next(values)
                if isinstance(value,Exception):raise value
                return value
            self.assertEqual(retry.network_retry(operation,'synthetic'),42)
            self.assertEqual(len((Path(tmp)/'network-retry/events.jsonl').read_text().splitlines()),1)
            with self.assertRaises(AssertionError):retry.network_retry(lambda:(_ for _ in ()).throw(AssertionError('validation')),'synthetic')

    def test_cleanup_scope_excludes9b_and_adapters(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"tmp") as tmp:
            root=Path(tmp)/'7b';(root/'formal/A1-seed42').mkdir(parents=True)
            with patch.object(cleanup,'SEVEN',root):
                self.assertTrue(cleanup.allowed(root/'formal/A1-seed42/resume.pt'))
                self.assertFalse(cleanup.allowed(root/'formal/A1-seed42/adapter/adapter_model.safetensors'))
                self.assertFalse(cleanup.allowed(Path(tmp)/'9b/formal/A1-seed42/resume.pt'))


if __name__=='__main__':unittest.main()
