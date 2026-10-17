"""Restore previously offloaded9B recovery; retain every restored/current state."""
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import os
import shlex
import subprocess
import sys
import tarfile
import time

import requests

from experiment_common import ROOT, read_json, sha, write_json
from route_b_9b_loop import OUT, SEEDS
from run_9b_formal_when_ready import verify_receipt
from verify_9b_student_outputs import verify_training


def download_proxies():
    # AutoDL's own accelerator is already used for Git publishing. Read only
    # the two proxy settings privately, never dump env/credentials to output.
    values={k:os.environ.get(k+'_proxy') or os.environ.get(k.upper()+'_PROXY') for k in ('http','https')}
    if Path('/etc/network_turbo').is_file():
        code='import os,json; print(json.dumps({k:os.environ.get(k+"_proxy") or os.environ.get(k.upper()+"_PROXY") for k in ("http","https")}))'
        command='source /etc/network_turbo >/dev/null 2>&1; '+shlex.quote(sys.executable)+' -c '+shlex.quote(code)
        result=subprocess.run(['bash','-c',command],capture_output=True,check=True,timeout=30)
        values=json.loads(result.stdout)
    return {k:v for k,v in values.items() if v}


RANGE_BYTES=4*1024**2
RANGE_WORKERS=4


def fetch_range(session,asset,start,end):
    expected=end-start+1
    for attempt in range(4):
        try:
            with session.get(asset['url'],params={'download':'1','recovery_nonce':str(time.time_ns())},
                             headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache'},
                             stream=True,timeout=(30,60)) as response:
                response.raise_for_status()
                assert response.status_code==206,'Recovery server did not honor byte range'
                assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{asset["bytes"]}','Unexpected recovery byte range'
                buffer=bytearray()
                for chunk in response.iter_content(1024**2):
                    if chunk:buffer.extend(chunk)
                    assert len(buffer)<=expected,'Recovery range body exceeds requested length'
                if len(buffer)!=expected:raise requests.exceptions.ChunkedEncodingError('Short recovery range')
            return buffer
        except requests.RequestException as error:
            status=error.response.status_code if getattr(error,'response',None) is not None else None
            print('Recovery range transient',type(error).__name__,'http_status',status,'offset',start,'attempt',attempt+1,flush=True)
            if attempt==3:raise requests.ConnectionError('Verified GitHub recovery range unavailable after4 bounded GET attempts') from None
            time.sleep(2*(attempt+1))


def stream_asset(session,asset,stream,resume_bytes=0):
    # Large proxy responses can be truncated. Retry only a bounded4MiB range;
    # commit bytes only after exact HTTP range/length checks, then verify the
    # complete asset SHA. A failed segment never enters the archive stream.
    size=0;digest=hashlib.sha256();announced=0
    if resume_bytes:
        assert 0<=resume_bytes<=asset['bytes']
        stream.seek(0)
        while size<resume_bytes:
            chunk=stream.read(min(4*1024**2,resume_bytes-size))
            assert chunk,'Owned scratch prefix unexpectedly short'
            digest.update(chunk);size+=len(chunk)
        stream.seek(size);announced=size
        print('Recovery scratch prefix retained_MiB',round(size/1024**2,1),'whole-asset SHA still mandatory',flush=True)
    # Each worker owns a Session. Bound submitted ranges to4, retaining at
    # most16MiB of completed payload even if an earlier range retries.
    def download(start,end):
        with requests.Session() as worker:
            worker.trust_env=False;worker.headers.update(session.headers)
            worker.proxies.update(session.proxies);worker.verify=session.verify
            return fetch_range(worker,asset,start,end)
    with ThreadPoolExecutor(max_workers=RANGE_WORKERS) as pool:
        while size<asset['bytes']:
            ranges=[(start,min(start+RANGE_BYTES,asset['bytes'])-1)
                    for start in range(size,min(size+RANGE_WORKERS*RANGE_BYTES,asset['bytes']),RANGE_BYTES)]
            futures=[pool.submit(download,start,end) for start,end in ranges]
            for future in futures:
                buffer=future.result()
                stream.write(buffer);digest.update(buffer);size+=len(buffer)
                if size-announced>=64*1024**2 or size==asset['bytes']:
                    print('Recovery archive',asset['name'],'received_MiB',round(size/1024**2,1),flush=True);announced=size
    assert digest.hexdigest()==asset['sha256'],'Downloaded archive SHA mismatch'


def restore_all(token,completed_only=False,models=None):
    if os.name=='nt':return _restore_all(token,completed_only=completed_only,models=models)
    import fcntl
    with (OUT/'recovery-restoration.lock').open('w') as handle:
        # Prefetch may overlap the end-of-training controller. Only one writer
        # can own scratch archives and atomically install recovery states.
        fcntl.flock(handle,fcntl.LOCK_EX)
        return _restore_all(token,completed_only=completed_only,models=models)


def _restore_all(token,completed_only=False,models=None):
    results=[]
    if models is not None:
        assert set(models)<={f'{a}-seed{s}' for s in SEEDS for a in ('A1','A2','U')}
    session=requests.Session();session.trust_env=False
    session.headers.update({'Authorization':'Bearer '+token})
    session.proxies.update(download_proxies())
    # The accelerator exports the system trust bundle; trust_env=False avoids
    # unrelated netrc credentials, so pass that CA bundle explicitly.
    bundle=os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE')
    if not bundle and Path('/etc/ssl/certs/ca-certificates.crt').is_file():
        bundle='/etc/ssl/certs/ca-certificates.crt'
    if bundle:session.verify=bundle
    for seed in SEEDS:
        for arm in ('A1','A2','U'):
            folder=OUT/'formal'/f'{arm}-seed{seed}'
            if models is not None and folder.name not in models:continue
            if completed_only and not (folder/'complete.json').exists():continue
            checkpoint=folder/'resume.pt'
            if checkpoint.exists():continue
            pointer=read_json(folder/'resume-offloaded.json')
            receipt=read_json(folder/'training-delivery.json')
            inventory=read_json(folder/'training-inventory.json')
            assert sha(folder/'training-delivery.json')==pointer['receipt_sha256']
            assert sha(folder/'training-inventory.json')==pointer['inventory_sha256']
            verify_receipt(receipt,token)
            binaries=sorted((a for a in receipt['assets'] if not a['name'].endswith('-inventory.json')),key=lambda a:a['name'])
            archive=folder/'restore-archive.tar.gz'
            assert not archive.is_symlink() and archive.resolve().is_relative_to(OUT.resolve())
            needed=sum(a['bytes'] for a in binaries)+pointer['bytes']+5*1024**3
            assert shutil.disk_usage(OUT).free>=needed, 'Insufficient safe room for recovery restore; never evict another recovery'
            def download():
                if len(binaries)==1 and archive.exists():
                    prefix=archive.stat().st_size
                    assert prefix<=binaries[0]['bytes']
                    with archive.open('r+b') as stream:stream_asset(session,binaries[0],stream,resume_bytes=prefix)
                else:
                    with archive.open('wb') as stream:
                        for asset in binaries:stream_asset(session,asset,stream)
            from resilient_9b_delivery import network_retry
            network_retry(download,'restore9b-archive:'+folder.name)
            assert archive.stat().st_size==inventory['archive']['bytes'] and sha(archive)==inventory['archive']['sha256']
            temporary=folder/'resume.restored.tmp'
            assert not temporary.is_symlink()
            with tarfile.open(archive,'r:gz') as package:
                member=package.getmember(pointer['archive_member'])
                assert member.isfile() and member.size==pointer['bytes']
                with package.extractfile(member) as source,temporary.open('wb') as destination:
                    shutil.copyfileobj(source,destination,4*1024**2)
            assert sha(temporary)==pointer['resume_sha256']
            temporary.replace(checkpoint)
            proof=verify_training(arm,seed,allow_offloaded=False)
            record={'status':'restored_and_actual_recovery_tensor_verified','timestamp':datetime.now(timezone.utc).isoformat(),
                    'resume_sha256':proof['resume_sha256'],'receipt_sha256':pointer['receipt_sha256'],
                    'archive_member':pointer['archive_member'],'checkpoint_retained':True}
            write_json(folder/'restored-from-release.json',record)
            # Only this owned temporary archive is disposable; checkpoint stays.
            assert archive.resolve().is_relative_to(OUT.resolve()) and not archive.is_symlink()
            archive.unlink()
            results.append({'arm':arm,'seed':seed,**record})
    filename=('recovery-restoration-partial-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'
              if completed_only or models is not None else 'recovery-restoration.json')
    write_json(OUT/'native-evaluation-v2'/filename,{'status':'passed','restored':results,
               'scope':'selected completed runs only' if completed_only or models is not None else 'all15 formal runs',
               'policy':'Retain all9B recovery states; only scratch downloads removed after exact verification'})
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completed-only',action='store_true')
    parser.add_argument('--model',action='append')
    args=parser.parse_args()
    token=os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
    assert token,'Credential only through encrypted stdin/inherited environment'
    restore_all(token,completed_only=args.completed_only,models=args.model)
