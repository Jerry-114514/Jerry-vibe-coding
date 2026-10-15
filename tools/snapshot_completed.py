"""Freeze checksum-verified, completed ledgers without touching a live shard."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import shutil
from experiment_common import ROOT, MODELS, read_json, sha, write_json

def snapshot(run):
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
    for relative in ['data/manifest.jsonl','data/experiment.lock.json','data/baselines/per_item.jsonl']:
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
    write_json(destination/'snapshot.json',{'source_run':str(run.relative_to(ROOT)),
        'created_utc':stamp,'completed_ledgers':completed,'full_experiment_complete':False,
        'live_partial_ledgers_included':False})
    print(str(destination),flush=True)
    return destination

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default=str(ROOT/'runs/formal-20261014'))
    snapshot(parser.parse_args().run)
