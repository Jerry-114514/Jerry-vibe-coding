"""Restore previously offloaded9B recovery; retain every restored/current state."""
from datetime import datetime, timezone
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


def stream_asset(session,asset,stream):
    offset=stream.tell()
    for attempt in range(3):
        digest=hashlib.sha256();size=0;announced=0
        try:
            with session.get(asset['url'],stream=True,timeout=(30,60)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(4*1024**2):
                    if not chunk:continue
                    stream.write(chunk);digest.update(chunk);size+=len(chunk)
                    if size-announced>=64*1024**2:
                        print('Recovery archive',asset['name'],'received_MiB',round(size/1024**2,1),flush=True);announced=size
            assert size==asset['bytes'] and digest.hexdigest()==asset['sha256'],'Downloaded archive byte/SHA mismatch'
            return
        except requests.RequestException as error:
            # Retry only this owned scratch segment, never a model call or an
            # optimizer update. A failed/partial download is not a checkpoint.
            stream.seek(offset);stream.truncate()
            print('Recovery download transient',type(error).__name__,'attempt',attempt+1,flush=True)
            if attempt==2:raise RuntimeError('Verified GitHub archive download unavailable after3 bounded GET attempts') from None
            time.sleep(2*(attempt+1))


def restore_all(token,completed_only=False,models=None):
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
            with archive.open('wb') as stream:
                for asset in binaries:
                    stream_asset(session,asset,stream)
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
