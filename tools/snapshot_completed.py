"""Freeze checksum-verified, completed ledgers without touching a live shard."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import shutil
from experiment_common import ROOT, MODELS, read_json, read_rows, sha, write_json

def snapshot(run, include_training_artifacts=False):
    run=Path(run).resolve()
    if not run.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Run must be under project runs')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=ROOT/'runs/github-snapshots'/f'{run.name}-{stamp}'
    destination.mkdir(parents=True,exist_ok=False)
    completed=[]
    for key in MODELS:
        for marker in sorted((run/'ledger').glob(f'{key}-[0-9][0-9][0-9].complete.json')):
            ledger=marker.with_suffix('').with_suffix('.jsonl')
            meta=ledger.with_suffix('.meta.json')
            evidence=read_json(marker)
            if sha(ledger)!=evidence['ledger_sha256']:
                raise ValueError('Completed source ledger checksum mismatch: '+key)
            for path in [ledger,meta,marker]:
                target=destination/'ledger'/path.name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
                if sha(target)!=sha(path):
                    raise ValueError('Snapshot copy checksum mismatch')
            completed.append({'checkpoint':key,'rows':evidence['rows'],'ledger_sha256':evidence['ledger_sha256']})
    if not completed:
        raise ValueError('No completed ledgers available')
    frozen_files=list(read_json(ROOT/'data/experiment.lock.json')['files'])+['data/experiment.lock.json']
    for relative in frozen_files:
        source=ROOT/relative
        target=destination/'frozen-inputs'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    # These folders are finalized before D2 starts; live teacher caches and
    # optimizer outputs are deliberately snapshotted separately after completion.
    analysis_gate=run/'analysis/gate.json'
    if analysis_gate.exists():
        gate=read_json(analysis_gate)
        if gate.get('formal') and not gate.get('synthetic_test'):
            for name in ['analysis','analysis-cluster','paper','training']:
                source=run/name
                if source.is_dir():
                    shutil.copytree(source,destination/name)
    teacher_items=0
    training_arms=[]
    if include_training_artifacts:
        teacher=run/'teacher-cache'
        if (teacher/'complete.json').exists():
            evidence=read_json(teacher/'complete.json')
            pool=ROOT/'data/training/pool.jsonl'
            if evidence['manifest_sha256']!=sha(pool) or evidence['items']!=len(read_rows(pool)):
                raise ValueError('Teacher cache coverage/manifest mismatch')
            if len(evidence['files'])!=evidence['items']:
                raise ValueError('Teacher cache inventory incomplete')
            for name,digest in evidence['files'].items():
                if sha(teacher/name)!=digest:
                    raise ValueError('Teacher cache checksum mismatch')
            shutil.copytree(teacher,destination/'teacher-cache')
            teacher_items=evidence['items']
        for arm in ['A1','A2','U','N']:
            folder=run/'arms'/arm
            if not (folder/'complete.json').exists():
                continue
            complete=read_json(folder/'complete.json')
            expected=0 if arm=='N' else 1000
            if complete.get('smoke') or complete['effective_updates']!=expected:
                raise ValueError('Training completion mismatch: '+arm)
            if arm!='N' and [row['step'] for row in read_rows(folder/'loss.jsonl')]!=list(range(1,1001)):
                raise ValueError('Training history incomplete: '+arm)
            shutil.copytree(folder,destination/'arms'/arm)
            training_arms.append(arm)
            for marker in sorted((run/'post'/arm).glob('*.complete.json')):
                ledger=marker.with_suffix('').with_suffix('.jsonl')
                if sha(ledger)!=read_json(marker)['ledger_sha256']:
                    raise ValueError('Post-training ledger checksum mismatch')
                target=destination/'post'/arm
                target.mkdir(parents=True,exist_ok=True)
                for path in [ledger,ledger.with_suffix('.meta.json'),marker]:
                    shutil.copy2(path,target/path.name)
    final_verified=(run/'final-verification.json').exists() and read_json(run/'final-verification.json').get('status')=='passed'
    for name in ['status.json','audit.json','bc-verification.json','final-verification.json']:
        if (run/name).exists():
            shutil.copy2(run/name,destination/name)
    if final_verified:
        shutil.copy2(run/'console.log',destination/'console.log')
    write_json(destination/'snapshot.json',{'source_run':str(run.relative_to(ROOT)),
        'created_utc':stamp,'completed_ledgers':completed,'full_experiment_complete':final_verified,
        'teacher_cache_items':teacher_items,'completed_training_arms':training_arms,
        'live_partial_ledgers_included':False})
    print(str(destination),flush=True)
    return destination

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default=str(ROOT/'runs/formal-20261014'))
    parser.add_argument('--include-training-artifacts',action='store_true')
    args=parser.parse_args()
    snapshot(args.run,args.include_training_artifacts)
