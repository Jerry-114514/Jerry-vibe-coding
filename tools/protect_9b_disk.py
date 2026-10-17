"""Low-space guard: only completed, fresh-verified7B raw outputs may be evicted.

Never deletes9B recovery, adapters, source data or model weights. Restore
pointers and exact archive members are recorded before unlinking any file.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import requests

from experiment_common import ROOT, read_json, sha, write_json
from route_b_9b_loop import OUT

SEVEN=ROOT/'runs/route-b-staged-7b-20261016'
FOLDER=OUT/'disk-protection'
TARGET=6*1024**3


def allowed(path):
    path=Path(path)
    return (path.name=='eval.jsonl' and path.parent.parent==SEVEN/'formal' and
            path.resolve().is_relative_to((SEVEN/'formal').resolve()) and not path.is_symlink())


def verified_locations(token,candidates):
    session=requests.Session();session.trust_env=False
    session.headers.update({'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
    endpoint='https://api.github.com/repos/Jerry-114514/Jerry-vibe-coding'
    response=session.get(endpoint+'/releases/tags/arp-protocol-v2-20261016',timeout=30);response.raise_for_status()
    release=response.json();assets={}
    for page in range(1,100):
        response=session.get(endpoint+f'/releases/{release["id"]}/assets',params={'page':page,'per_page':100},timeout=30)
        response.raise_for_status();batch=response.json();assets.update({a['name']:a for a in batch})
        if len(batch)<100:break
    else:raise RuntimeError('Incomplete asset listing')
    wanted={str(p.relative_to(ROOT)):(p.stat().st_size,sha(p)) for p in candidates}
    from audit_route_b_final_delivery import canonical
    folders={p.parent for p in SEVEN.rglob('github-storage.json') if '/files/' not in p.as_posix()}
    folders.update(p.parent for p in (ROOT/'runs/route-b-staged-delivery-20261016').glob('*/github-storage.json'))
    found={}
    for folder in sorted(folders):
        if not (folder/'github-inventory.json').exists():continue
        receipt=read_json(folder/'github-storage.json');inventory=read_json(folder/'github-inventory.json')
        if receipt['release_url']!=release['html_url']:continue
        matching=[entry for entry in inventory['files'] if canonical(entry['path']) in wanted and
                  (entry['bytes'],entry['sha256'])==wanted[canonical(entry['path'])]]
        if not matching:continue
        assert sha(folder/'github-inventory.json')==next(a['sha256'] for a in receipt['assets'] if a['name'].endswith('-inventory.json'))
        for item in receipt['assets']:
            actual=assets[item['name']]
            assert actual['size']==item['bytes'] and actual.get('digest')=='sha256:'+item['sha256']
        assert {a['name']:(a['bytes'],a['sha256']) for a in receipt['assets'] if not a['name'].endswith('-inventory.json')}=={a['name']:(a['bytes'],a['sha256']) for a in inventory['binary_assets']}
        for entry in matching:
            found[canonical(entry['path'])]={'release_url':release['html_url'],'assets':receipt['assets'],
                   'archive_member':entry['path'],'bytes':entry['bytes'],'sha256':entry['sha256'],
                   'fresh_server_size_and_sha256_verified':True}
    return found


def protect(token):
    free=shutil.disk_usage(OUT).free
    if free>=TARGET:return
    assert read_json(SEVEN/'objective-content-verification.json')['status']=='passed'
    candidates=[]
    for path in (SEVEN/'formal').glob('*/eval.jsonl'):
        assert allowed(path)
        marker=read_json(path.parent/'scores.json')
        if marker['rows']!=16900 or marker['ledger_sha256']!=sha(path):continue
        assert path.read_bytes().count(b'\n')==16900
        if path.stat().st_nlink==1:candidates.append(path)
    locations=verified_locations(token,candidates)
    from cleanup_7b_evaluation_disk import ensure_not_open
    removed=[]
    for path in sorted(candidates,key=lambda p:p.stat().st_size,reverse=True):
        if shutil.disk_usage(OUT).free>=TARGET:break
        key=str(path.relative_to(ROOT))
        if key not in locations:continue
        pointer=locations[key]
        assert allowed(path) and sha(path)==pointer['sha256'] and path.stat().st_size==pointer['bytes']
        ensure_not_open([path])
        write_json(FOLDER/'restore-pointers'/(path.parent.name+'.json'),{'original_path':key,**pointer,
                   'row_count':16900,'timestamp':datetime.now(timezone.utc).isoformat(),
                   'policy':'Evict only completed7B raw output; retain all9B recovery',
                   'restore':'Download listed verified archive on AutoDL, extract exact archive_member, verify bytes/SHA and restore original_path'})
        assert allowed(path) and sha(path)==pointer['sha256']
        path.unlink();removed.append(key)
    write_json(FOLDER/'latest-action.json',{'timestamp':datetime.now(timezone.utc).isoformat(),'removed_raw_outputs':removed,
               'free_before_GiB':free/1024**3,'free_after_GiB':shutil.disk_usage(OUT).free/1024**3,
               'all9b_recovery_untouched':True,'if_insufficient':'Keep recovery; registered5GiB workload stop remains active'})
    if removed:
        from publish_artifacts import publish
        publish(FOLDER,'arp-9b-122b-20261017',token=token)


if __name__=='__main__':
    token=os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
    assert token,'Credential through encrypted stdin or inherited environment'
    if '--start' in sys.argv:
        command=('cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/protect_9b_disk.py '
                 '>> runs/route-b-9b-122b-20261017/student-experiment/disk-protection-console.log 2>&1')
        subprocess.run(['screen','-dmS','route-b-9b-disk-protection','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Low-space guard queued: only fresh-verified completed7B raw outputs, never9B recovery')
    else:
        import fcntl
        FOLDER.mkdir(parents=True,exist_ok=True)
        with (FOLDER/'observer.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            write_json(FOLDER/'policy.json',{'user_night_mode':True,'trigger_free_GiB':6,'registered_hard_stop_free_GiB':5,
                       'allowed':'completed7B formal eval.jsonl only, with current SHA and fresh GitHub asset verification',
                       'never_delete':['9B recovery','adapters','model weights','source datasets','training logs']})
            while True:
                try:protect(token)
                except Exception as error:write_json(FOLDER/'error.json',{'type':type(error).__name__,'message':str(error),'deletion_refused_without_proof':True})
                if (OUT/'native-evaluation-v2/finalization/complete.json').exists():break
                time.sleep(45)
