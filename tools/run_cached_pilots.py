"""Serial pilots for cached, 24GB-compatible models; publish completed outputs."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from experiment_common import ROOT, read_json, sha, write_json

RUN=ROOT/'runs/protocol-v2'
TAG='arp-protocol-v2-20261016'
MODELS=['intern2b','intern8b','llava05b']

def snapshot_completed():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target=ROOT/'runs/protocol-v2-snapshots'/stamp
    target.mkdir(parents=True,exist_ok=False)
    included=[]
    if (RUN/'p0/status.json').exists() and read_json(RUN/'p0/status.json')['status']=='complete':
        shutil.copytree(RUN/'p0',target/'p0')
        included.append('P0 exploratory reanalysis')
    for key in ['qwen3b','qwen7b',*MODELS]:
        folder=RUN/'calibration/precision-bridge'/key
        marker=folder/'complete.json'
        if marker.exists():
            complete=read_json(marker)
            if complete['ledger_sha256']!=sha(folder/'ledger.jsonl') or complete['rows']!=800:
                raise ValueError('Completed pilot checksum or coverage failed: '+key)
            shutil.copytree(folder,target/'precision-bridge'/key)
            included.append(key+' BF16 development pilot')
    for size in ['3b','7b']:
        training=RUN/('calibration/training-memory-'+size)
        if (training/'complete.json').exists():
            done=read_json(training/'complete.json')
            verification=training/'verification.json'
            if not verification.exists() or read_json(verification).get('status')!='passed':
                continue
            if done['actual_updates']!=100 or not done['engineering_pilot_only']:
                raise ValueError('Training pilot completion mismatch')
            shutil.copytree(training,target/('training-memory-'+size))
            included.append(size+' 100-update engineering pilot')
    for path in (RUN/'calibration').glob('*.json*'):
        if path.is_file():
            destination=target/'calibration'/path.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
    write_json(target/'snapshot.json',{'created_utc':stamp,'completed_only':True,
        'included':included,'formal_training_or_confirmation':False,
        'legacy_teacher_cache':'arp-formal-20261014 final Release (unchanged)',
        'source_code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'tools/precision_bridge.py',ROOT/'tools/rulers_v2.py',ROOT/'tools/reanalyze_v2.py',ROOT/'tools/training_memory_pilot.py']}})
    return target

def publish_completed():
    from publish_artifacts import publish
    target=snapshot_completed()
    for attempt in range(3):
        try:
            return publish(target,TAG,token=os.environ['GH_TOKEN'])
        except Exception:
            if attempt==2:
                raise
            time.sleep(2**(attempt+1))

def worker():
    status={'status':'running','completed':[],'skipped_completed':[],
            'started_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Development precision pilots; formal C/D remains gated'}
    write_json(RUN/'cached-pilots-status.json',status)
    try:
        reports=sorted((ROOT/'runs/protocol-v2-snapshots').glob('*/github-storage.json'))
        status['backup']=read_json(reports[-1])['release_url'] if reports else publish_completed()['release_url']
        for key in MODELS:
            status.update(current_model=key,updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(RUN/'cached-pilots-status.json',status)
            marker=RUN/'calibration/precision-bridge'/key/'complete.json'
            already_complete=marker.exists()
            if already_complete:
                prior=read_json(marker)
                if prior['rows']!=800 or prior['ledger_sha256']!=sha(marker.parent/'ledger.jsonl'):
                    raise ValueError('Existing completion is not verified: '+key)
                status['skipped_completed'].append(key)
            else:
                subprocess.run([sys.executable,'-u',str(ROOT/'tools/precision_bridge.py'),'--model',key],cwd=ROOT,check=True)
            complete=read_json(RUN/'calibration/precision-bridge'/key/'complete.json')
            if complete['peak_reserved_bytes']>23*1024**3:
                raise RuntimeError('Pilot exceeded the registered memory budget')
            status['completed'].append({'model':key,'rows':complete['rows'],
                'peak_allocated_bytes':complete['peak_allocated_bytes']})
            if not already_complete:
                status['backup']=publish_completed()['release_url']
        status.update(status='complete',current_model=None,completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        status.update(status='failed',error=type(exc).__name__,message=str(exc))
        raise
    finally:
        write_json(RUN/'cached-pilots-status.json',status)
        print(status,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker()
    else:
        token=sys.stdin.readline().strip()
        if not token:
            raise RuntimeError('GitHub backup credential required on stdin')
        sessions=subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout
        if '.arp-v2-cached-pilots' in sessions:
            raise RuntimeError('The cached-pilot worker is already running')
        lock=read_json(ROOT/'configs/models.lock.json')
        for key in MODELS:
            weights=Path('/root/arp-model-cache')/key/'weights.verified.json'
            meta=read_json(weights)
            if meta['revision']!=lock['models'][key]['revision']:
                raise RuntimeError('Cached revision differs: '+key)
            for row in meta['files']:
                path=weights.parent/row['file']
                if not path.is_file() or path.stat().st_size!=row['bytes']:
                    raise RuntimeError('This batch forbids new large downloads: '+key)
        write_json(RUN/'calibration/cached-pilot-design.json',{'models':MODELS,
            'precision':'bf16','scoring_units_per_axis':100,'prompts_per_model':800,
            'max_torch_reserved_gib':23,'GPU_policy':'Shared flock; one model at a time',
            'formal_confirmation':False,'large_downloads':False,
            'auto_backup':'Only completed checksummed stages, AutoDL directly to GitHub Release'})
        environment={**os.environ,'GH_TOKEN':token}
        command='source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/run_cached_pilots.py --worker >> runs/protocol-v2/cached-pilots-console.log 2>&1'
        subprocess.run(['screen','-dmS','arp-v2-cached-pilots','bash','-lc',command],env=environment,check=True)
        print('Started arp-v2-cached-pilots with completion-only GitHub backups',flush=True)
