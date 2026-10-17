"""Evict completed7B optimizer copies only after fresh Release SHA checks."""
from datetime import datetime, timezone
import os
import shutil
import sys

from experiment_common import ROOT, read_json, sha, write_json
from protect_9b_disk import SEVEN, verified_locations
from cleanup_7b_evaluation_disk import ensure_not_open
from resilient_9b_delivery import network_retry

DEST=SEVEN/'recovery-eviction-20261017'


def allowed(path):
    return (path.name=='resume.pt' and path.parent.parent in (SEVEN/'minimal',SEVEN/'formal')
            and path.resolve().is_relative_to(SEVEN.resolve()) and not path.is_symlink())


def run(token):
    assert ROOT.resolve().as_posix()=='/root/autodl-tmp/arp'
    assert read_json(SEVEN/'objective-content-verification.json')['status']=='passed'
    candidates=[]
    for phase in ('minimal','formal'):
        for path in (SEVEN/phase).glob('*/resume.pt'):
            assert allowed(path)
            marker=read_json(path.parent/'complete.json')
            assert marker['actual_updates']==(100 if phase=='minimal' else 1000)
            assert marker['adapter_export_exact'] and path.stat().st_nlink==1
            candidates.append(path)
    locations=network_retry(lambda:verified_locations(token,candidates),'verify-old7b-recovery-release')
    assert all(str(p.relative_to(ROOT)) in locations for p in candidates),'Every deletion needs exact archive/member evidence'
    free=shutil.disk_usage(ROOT).free
    manifest={'timestamp':datetime.now(timezone.utc).isoformat(),'status':'planned_fresh_server_verified',
              'free_before_GiB':free/1024**3,'scope':'Completed7B optimizer recovery copies only; adapters, loss logs, raw outputs and all9B states retained',
              'files':[{'original_path':str(p.relative_to(ROOT)),**locations[str(p.relative_to(ROOT))]} for p in candidates]}
    write_json(DEST/'restore-pointers.json',manifest)
    from publish_artifacts import publish
    network_retry(lambda:publish(DEST,'arp-protocol-v2-20261016',token=token),'publish-old7b-restore-pointers-before-delete')
    ensure_not_open(candidates)
    removed=[]
    for path in candidates:
        record=locations[str(path.relative_to(ROOT))]
        assert allowed(path) and path.stat().st_size==record['bytes'] and sha(path)==record['sha256']
        path.unlink();removed.append(str(path.relative_to(ROOT)))
    result={'status':'completed','removed':removed,'bytes_freed':sum(r['bytes'] for r in manifest['files']),
            'free_after_GiB':shutil.disk_usage(ROOT).free/1024**3,'all9b_states_untouched':True,
            'restore_pointers_published_before_delete':True,'timestamp':datetime.now(timezone.utc).isoformat()}
    write_json(DEST/'complete.json',result)
    network_retry(lambda:publish(DEST,'arp-protocol-v2-20261016',token=token),'publish-old7b-cleanup-result')
    print(result,flush=True)


if __name__=='__main__':
    token=os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
    assert token
    run(token)
