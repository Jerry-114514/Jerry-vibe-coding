"""Run the memory-bounded development pilot under screen with truthful status."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import traceback
from experiment_common import ROOT, read_json, sha, write_json

RUN = ROOT/'runs/protocol-v2'

def utc():
    return datetime.now(timezone.utc).isoformat()

def worker():
    from precision_bridge import make_manifest
    manifest = make_manifest(100)
    status = {'status': 'running', 'started_utc': utc(), 'stage': 'precision_pilot',
              'scope': 'Legacy-development BF16 bridge; not formal training or independent confirmation',
              'manifest_sha256': sha(manifest), 'completed': []}
    try:
        for key in ['qwen3b', 'qwen7b']:
            status.update(current_model=key, updated_utc=utc())
            write_json(RUN/'safe-pilot-status.json', status)
            subprocess.run([sys.executable, '-u', str(ROOT/'tools/precision_bridge.py'), '--model', key], cwd=ROOT, check=True)
            evidence = read_json(RUN/'calibration/precision-bridge'/key/'complete.json')
            assert evidence['rows'] == 800
            assert evidence['peak_reserved_bytes'] <= 23*1024**3
            status['completed'].append({'model': key, 'rows': evidence['rows'],
                                        'peak_allocated_bytes': evidence['peak_allocated_bytes'],
                                        'peak_reserved_bytes': evidence['peak_reserved_bytes']})
        status.update(status='complete', completed_utc=utc(), current_model=None)
    except Exception as exc:
        status.update(status='failed', failed_utc=utc(), error=type(exc).__name__, message=str(exc))
        traceback.print_exc()
        raise
    finally:
        write_json(RUN/'safe-pilot-status.json', status)
        print(status, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker()
    else:
        sessions = subprocess.run(['screen', '-ls'], capture_output=True, text=True).stdout
        if '.arp-v2-pilot' in sessions:
            raise RuntimeError('A pilot screen session already exists')
        RUN.mkdir(parents=True, exist_ok=True)
        design = {'created_utc': utc(), 'models': ['qwen3b', 'qwen7b'],
                  'precision': 'bf16', 'max_torch_reserved_gib': 23,
                  'scoring_units_per_axis': 100, 'prompts_per_model': 800,
                  'generation': {'do_sample': False, 'max_new_tokens': 128},
                  'image_budget': {'min_pixels': 256*28*28, 'max_pixels': 1280*28*28},
                  'training_started': False, 'confirmation': False,
                  'followup_training': 'Only after numerical, data and target-family gate checks',
                  'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/'tools/precision_bridge.py', ROOT/'tools/model_backends.py']}}
        write_json(RUN/'calibration/precision-pilot-design.json', design)
        command = 'source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/run_safe_supplements.py --worker > runs/protocol-v2/safe-pilot-console.log 2>&1'
        subprocess.run(['screen', '-dmS', 'arp-v2-pilot', 'bash', '-lc', command], check=True)
        print('Started arp-v2-pilot; log:', RUN/'safe-pilot-console.log', flush=True)
