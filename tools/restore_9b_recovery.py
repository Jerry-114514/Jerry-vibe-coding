"""Restore previously offloaded9B recovery; retain every restored/current state."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import tarfile

import requests

from experiment_common import ROOT, read_json, sha, write_json
from route_b_9b_loop import OUT, SEEDS
from run_9b_formal_when_ready import verify_receipt
from verify_9b_student_outputs import verify_training


def restore_all(token):
    results=[]
    session=requests.Session();session.trust_env=False
    session.headers.update({'Authorization':'Bearer '+token})
    for seed in SEEDS:
        for arm in ('A1','A2','U'):
            folder=OUT/'formal'/f'{arm}-seed{seed}'
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
                    digest=hashlib.sha256();size=0
                    with session.get(asset['url'],stream=True,timeout=(30,180)) as response:
                        response.raise_for_status()
                        for chunk in response.iter_content(4*1024**2):
                            if chunk:stream.write(chunk);digest.update(chunk);size+=len(chunk)
                    assert size==asset['bytes'] and digest.hexdigest()==asset['sha256']
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
    write_json(OUT/'native-evaluation-v2/recovery-restoration.json',{'status':'passed','restored':results,
               'policy':'Retain all9B recovery states; only scratch downloads removed after exact verification'})
    return results
