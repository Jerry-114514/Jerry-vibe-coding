"""Storage-safe delivery wrapper around the frozen experiment implementation.

Training/evaluation code and frozen configuration remain unchanged. Previously
verified binary checkpoints are referenced, not packed into every new snapshot.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT, read_json, read_rows, sha, write_json, units
import route_b_staged as experiment
import publish_artifacts as publisher

original_publish = publisher.publish
original_run = subprocess.run
DELIVERY = ROOT/'runs/route-b-staged-delivery-20261016'


def prior_inventory(folder):
    # Publisher already checked remote asset sizes/server digests before receipt.
    candidates = [folder,experiment.OUT]
    for candidate in candidates:
        receipt = candidate/'github-storage.json'
        inventory = candidate/'github-inventory.json'
        if receipt.exists() and inventory.exists():
            r = read_json(receipt)
            if r.get('assets'):
                yield read_json(inventory),r


def covered_binaries(folder):
    required = [p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.pt','.safetensors')]
    if not required:
        return True,[]
    covered = {}
    refs = []
    for inventory,receipt in prior_inventory(folder):
        covered.update({r['path']:r['sha256'] for r in inventory['files']})
        refs.append(receipt)
    return all(covered.get(str(p.relative_to(ROOT))) == sha(p) for p in required),refs


def metadata_snapshot(folder,refs):
    target = DELIVERY/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(time.time_ns()))
    target.mkdir(parents=True)
    for path in folder.rglob('*'):
        if not path.is_file() or path.is_symlink() or path.suffix in ('.pt','.safetensors','.tmp'):
            continue
        if 'adapter' in path.relative_to(folder).parts:
            continue
        destination = target/'files'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    write_json(target/'binary-asset-references.json',{'source':str(folder.relative_to(ROOT)),
               'verified_prior_deliveries':refs,'note':'Referenced binary files remain in previous verifiedRelease assets; not omitted data'})
    return target


def incremental_publish(folder,tag,token=None,**kwargs):
    folder = Path(folder).resolve()
    if folder == experiment.CACHE.resolve():
        return original_publish(folder,tag,token=token,**kwargs)
    if not folder.is_relative_to(experiment.OUT.resolve()):
        return original_publish(folder,tag,token=token,**kwargs)
    if folder == experiment.OUT.resolve():
        refs = []
        # Ensure all existing optimizer/adapter binaries have been delivered first.
        for phase in ('minimal','formal'):
            for arm in sorted((folder/phase).glob('*-seed*')):
                if not (arm/'complete.json').exists():
                    continue
                covered,prior = covered_binaries(arm)
                if not covered:
                    original_publish(arm,tag,token=token)
                    _,prior = covered_binaries(arm)
                refs.extend(prior)
        snapshot = metadata_snapshot(folder,refs)
    else:
        covered,refs = covered_binaries(folder)
        if not covered:
            return original_publish(folder,tag,token=token,**kwargs)
        snapshot = metadata_snapshot(folder,refs)
    receipt = original_publish(snapshot,tag,token=token)
    write_json(folder/'latest-metadata-delivery.json',receipt)
    return receipt


def skip_completed_child(args,**kwargs):
    # Re-enter the unchanged pipeline without regenerating completed stage1 work.
    if isinstance(args,list) and str(experiment.__file__) in args:
        index = args.index(str(experiment.__file__))
        action = args[index+1] if len(args)>index+1 else None
        fields = args[index+2:]
        if '--phase' in fields and fields[fields.index('--phase')+1] == 'minimal':
            if action == 'cache' and (experiment.CACHE/'minimal.complete.json').exists():
                marker = read_json(experiment.CACHE/'minimal.complete.json')
                assert marker['items'] == 500
                assert all(sha(experiment.CACHE/name) == digest for name,digest in marker['files'].items())
                return subprocess.CompletedProcess(args,0)
            if action in ('train','eval'):
                arm = fields[fields.index('--arm')+1]
                folder = experiment.OUT/'minimal'/f'{arm}-seed42'
                marker = folder/('complete.json' if action == 'train' else 'scores.json')
                if marker.exists():
                    print('Reused verified stage1',action,arm,flush=True)
                    return subprocess.CompletedProcess(args,0)
    return original_run(args,**kwargs)


def worker():
    from verify_route_b_stage_outputs import verify
    verify('minimal')
    previous = read_json(experiment.OUT/'minimal-acceptance.json')
    assert previous['status'] == 'passed'
    # Numeric source SHA remains the frozen original; this wrapper changes delivery only.
    write_json(experiment.OUT/'delivery-execution-override.json',{'wrapper_sha256':sha(Path(__file__)),
               'numeric_source_sha256':sha(Path(experiment.__file__)),
               'change':'Incremental verified binary references; preserve completed stage1 outputs',
               'training_configuration_changed':False,'reason':'9B weights plus15 optimizer states leave insufficient scratch for a duplicate whole-run archive'})
    publisher.publish = incremental_publish
    experiment.subprocess.run = skip_completed_child
    experiment.minimal_acceptance = lambda:previous
    experiment.pipeline()


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        token = sys.stdin.readline().strip()
        if not token:raise RuntimeError('Backup credential required')
        command = 'cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/run_route_b_storage_safe.py --worker >> runs/route-b-staged-7b-20261016/storage-safe-console.log 2>&1'
        original_run(['screen','-dmS','route-b-storage-safe','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Started storage-safe continuation; completed stage1 and frozen numeric code preserved')
