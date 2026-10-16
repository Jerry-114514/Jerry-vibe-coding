"""Remote process monitor: no inference calls, retries, or scientific-gate overrides."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from experiment_common import ROOT, read_json, write_json

WORKERS = {'api_development_pilot.py', 'precision_bridge.py', 'training_memory_pilot.py',
           'run_cached_pilots.py', 'run_safe_supplements.py', 'paired_gate_v2.py',
           'prepare_independent_data.py', 'run_protocol_v2.py'}
OUT = ROOT / 'runs/protocol-v2-monitor'


def snapshot():
    live = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            script = next((Path(arg).name for arg in args[1:] if Path(arg).name in WORKERS), None)
            if script and ('python' in Path(args[0]).name):
                live.append({'pid': int(entry.name), 'script': script})
        except (OSError, StopIteration):
            continue
    stages = []
    roots = [ROOT / 'runs/protocol-v2', ROOT / 'runs/protocol-v2-data']
    for base in roots:
        for path in sorted(base.rglob('*status.json')):
            try:
                state = read_json(path)
            except (OSError, ValueError):
                continue
            status = state.get('status', 'unknown')
            ledger = path.parent / 'ledger.jsonl'
            activity = max(path.stat().st_mtime, ledger.stat().st_mtime if ledger.exists() else 0)
            age = time.time() - activity
            script = state.get('worker_script')
            if not script and path.parent.name.startswith('api-'):
                script = 'api_development_pilot.py'
            matching = [p for p in live if not script or p['script'] == script]
            issue = None
            if status in ('running', 'queued'):
                if not matching:
                    issue = 'reported_active_but_worker_missing'
                elif age > 1800:
                    issue = 'no_status_or_ledger_progress_for_30_minutes'
            elif status == 'failed':
                issue = 'stage_failed'
            backup = path.parent / 'github-storage.json'
            verification = path.parent / 'verification.json'
            stages.append({'path': str(path.relative_to(ROOT)), 'status': status,
                           'completed_prompts': state.get('completed_prompts', state.get('prompts', state.get('rows'))),
                           'last_progress_age_seconds': round(age), 'issue': issue,
                           'verification_present': verification.exists(), 'backup_receipt_present': backup.exists()})
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return {'checked_utc': datetime.now(timezone.utc).isoformat(), 'interval_seconds': 300,
            'live_workers': live, 'stages': stages,
            'issues': [s for s in stages if s['issue']],
            'gpu_memory_mib_and_utilization_percent': gpu.stdout.strip(),
            'policy': 'Observe only; never restart paid requests or bypass training gate'}


def main(once=False):
    OUT.mkdir(parents=True, exist_ok=True)
    previous = None
    while not (OUT / 'stop').exists():
        state = snapshot()
        write_json(OUT / 'heartbeat.json', state)
        transition = json.dumps({'live': state['live_workers'], 'stages': [
            {k: s[k] for k in ('path', 'status', 'issue', 'verification_present', 'backup_receipt_present')}
            for s in state['stages']]}, sort_keys=True)
        if transition != previous:
            with (OUT / 'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(state, ensure_ascii=False) + '\n')
            print(state['checked_utc'], 'workers', len(state['live_workers']), 'issues', len(state['issues']), flush=True)
            previous = transition
        if once:
            print(json.dumps(state, ensure_ascii=False))
            return
        time.sleep(300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    main(parser.parse_args().once)
