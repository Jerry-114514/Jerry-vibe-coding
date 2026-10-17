"""Synthetic archive checks; actual recovery health is verified on AutoDL later."""
from io import BytesIO
import hashlib
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import restore_9b_recovery as restore
from experiment_common import ROOT, sha, write_json


class Response:
    def __init__(self,data,total=None):
        self.data=data;self.status_code=206
        self.headers={'Content-Range':f'bytes 0-{len(data)-1}/{total or len(data)}'}
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def raise_for_status(self):pass
    def iter_content(self,*args):yield self.data


class RecoveryRestoreTests(unittest.TestCase):
    def perform(self,corrupt=False):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            out=Path(tmp)
            for arm in ('A1','A2','U'):(out/'formal'/f'{arm}-seed42').mkdir(parents=True)
            for arm in ('A2','U'):(out/'formal'/f'{arm}-seed42'/'resume.pt').write_bytes(b'keep-existing')
            folder=out/'formal/A1-seed42';payload=b'synthetic-optimizer-state'
            member='synthetic/formal/A1-seed42/resume.pt';stream=BytesIO()
            with tarfile.open(fileobj=stream,mode='w:gz') as package:
                item=tarfile.TarInfo(member);item.size=len(payload);package.addfile(item,BytesIO(payload))
            archive=stream.getvalue();digest=hashlib.sha256(archive).hexdigest()
            write_json(folder/'training-delivery.json',{'assets':[{'name':'synthetic.tar.gz','bytes':len(archive),'sha256':digest,'url':'https://example.invalid/never-requested'}]})
            write_json(folder/'training-inventory.json',{'archive':{'bytes':len(archive),'sha256':digest}})
            write_json(folder/'resume-offloaded.json',{'receipt_sha256':sha(folder/'training-delivery.json'),
                       'inventory_sha256':sha(folder/'training-inventory.json'),'archive_member':member,
                       'bytes':len(payload),'resume_sha256':hashlib.sha256(payload).hexdigest()})
            with patch.object(restore,'OUT',out),patch.object(restore,'SEEDS',[42]),\
                 patch.object(restore,'verify_receipt'),\
                 patch.object(restore,'verify_training',return_value={'resume_sha256':hashlib.sha256(payload).hexdigest()}),\
                 patch.object(restore.requests.Session,'get',return_value=Response(b'bad-data' if corrupt else archive)):
                if corrupt:
                    with self.assertRaises(AssertionError):restore.restore_all('synthetic-token-never-sent')
                    self.assertFalse((folder/'resume.pt').exists())
                else:
                    restore.restore_all('synthetic-token-never-sent')
                    self.assertEqual((folder/'resume.pt').read_bytes(),payload)
                    self.assertFalse((folder/'restore-archive.tar.gz').exists())
            for arm in ('A2','U'):self.assertEqual((out/'formal'/f'{arm}-seed42'/'resume.pt').read_bytes(),b'keep-existing')

    def test_exact_archive_restores_and_keeps_existing_recovery(self):self.perform()
    def test_wrong_archive_bytes_never_replace_recovery(self):self.perform(corrupt=True)

    def test_interrupted_range_does_not_duplicate_committed_bytes(self):
        import requests
        class Broken(Response):
            def iter_content(self,*args):
                yield b'partial'
                raise requests.exceptions.ChunkedEncodingError('synthetic interruption')
        payload=b'verified-complete-range';output=BytesIO(b'prefix');output.seek(6)
        with patch.object(restore.requests.Session,'get',side_effect=[Broken(payload),Response(payload)]) as get,patch.object(restore.time,'sleep'):
            restore.stream_asset(restore.requests.Session(),{'name':'synthetic','url':'https://example.invalid/never-requested',
                                 'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()},output)
        self.assertEqual(output.getvalue(),b'prefix'+payload)
        self.assertEqual(get.call_count,2)
        self.assertEqual(get.call_args.kwargs['headers']['Range'],f'bytes=0-{len(payload)-1}')

    def test_parallel_ranges_reassemble_in_exact_order(self):
        payload=b'0123456789abcdefghijklmnopqrstuv';output=BytesIO()
        def response(url,headers,**kwargs):
            start,end=map(int,headers['Range'].removeprefix('bytes=').split('-'))
            result=Response(payload[start:end+1]);result.headers={'Content-Range':f'bytes {start}-{end}/{len(payload)}'}
            return result
        with patch.object(restore,'RANGE_BYTES',5),patch.object(restore.requests.Session,'get',side_effect=response):
            restore.stream_asset(restore.requests.Session(),{'name':'synthetic','url':'https://example.invalid/never-requested',
                                 'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()},output)
        self.assertEqual(output.getvalue(),payload)

    def test_owned_partial_prefix_resumes_and_whole_sha_is_required(self):
        payload=b'prefix-and-remainder';prefix=7
        def response(url,headers,**kwargs):
            start,end=map(int,headers['Range'].removeprefix('bytes=').split('-'))
            result=Response(payload[start:end+1]);result.headers={'Content-Range':f'bytes {start}-{end}/{len(payload)}'}
            return result
        asset={'name':'synthetic','url':'https://example.invalid/never-requested','bytes':len(payload),
               'sha256':hashlib.sha256(payload).hexdigest()}
        with patch.object(restore.requests.Session,'get',side_effect=response):
            output=BytesIO(payload[:prefix]);restore.stream_asset(restore.requests.Session(),asset,output,resume_bytes=prefix)
            self.assertEqual(output.getvalue(),payload)
            with self.assertRaises(AssertionError):restore.stream_asset(restore.requests.Session(),asset,BytesIO(b'x'*prefix),resume_bytes=prefix)


if __name__=='__main__':unittest.main()
