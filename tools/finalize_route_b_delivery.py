"""Finalize independent7B evidence only after all evaluation/report jobs finish.

No CUDA, training, inference, or numeric-source changes. Uses the existing full
content audit, incremental publisher, and fresh server-digest delivery audit.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT, read_json, write_json
from route_b_staged import OUT

FINAL = OUT / 'finalization'


def readiness():
    from audit_route_b_completion import readiness as content_readiness
    pending = content_readiness()['missing']
    state = read_json(OUT / 'status.json') if (OUT / 'status.json').exists() else {}
    parent = Path('/proc/7177/cmdline')
    try:
        parent_alive = b'run_route_b_storage_safe.py' in parent.read_bytes()
    except FileNotFoundError:
        parent_alive = False
    return {'ready': not pending and state.get('phase') == 'all_four_stages_complete' and not parent_alive,
            'missing': pending, 'numeric_parent_live': parent_alive,
            'experiment_phase': state.get('phase'), 'pid': os.getpid()}


def finalize(token):
    assert readiness()['ready'], 'Full evaluation/report prerequisites not met'
    from audit_route_b_completion import verify
    from run_route_b_storage_safe import incremental_publish, original_publish
    from audit_route_b_final_delivery import audit
    content = verify()
    assert content['status'] == 'passed', 'Content audit must prove the full objective'
    incremental_publish(OUT, 'arp-protocol-v2-20261016', token=token)
    delivery = audit(token, strict=True)
    assert delivery['status'] == 'passed'
    proof = FINAL / 'verified-evidence'
    proof.mkdir(parents=True, exist_ok=True)
    for name in ('objective-content-verification.json', 'final-delivery-verification.json',
                 'evaluation-aggregate-independent-verification.json'):
        shutil.copyfile(OUT / name, proof / name)
    receipt = original_publish(proof, 'arp-protocol-v2-20261016', token=token)
    result = {'status': 'full_7b_content_and_delivery_verified',
              'timestamp': datetime.now(timezone.utc).isoformat(),
              'verified_files': delivery['verified_files'],
              'required_files': delivery['required_files'],
              'evidence_delivery': receipt, 'no_new_experiments': True,
              'original_arp_gate_unchanged': True}
    write_json(FINAL / 'complete.json', result)
    print(json.dumps(result), flush=True)
    return result


def watch(token):
    import fcntl
    FINAL.mkdir(parents=True, exist_ok=True)
    with (FINAL / 'observer.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if (FINAL / 'complete.json').exists():
                print('Final content and fresh Release delivery already verified', flush=True)
                return
            state = readiness()
            write_json(FINAL / 'queue-status.json', state)
            if state['ready']:
                finalize(token)
                return
            if (OUT / 'failure.json').exists():
                raise RuntimeError('7B experiment failure requires inspection; no completion claim')
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status', 'once', 'watch', 'start'])
    args = parser.parse_args()
    if args.action == 'status':
        print(json.dumps(readiness()), flush=True)
    else:
        secret = os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
        assert secret, 'Credential required through encrypted stdin or inherited environment'
        if args.action == 'start':
            command = ('cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u '
                       'tools/finalize_route_b_delivery.py watch >> '
                       'runs/route-b-staged-7b-20261016/finalization-console.log 2>&1')
            subprocess.run(['screen', '-dmS', 'route-b-finalization', 'bash', '-c', command],
                           env={**os.environ, 'GH_TOKEN': secret}, check=True)
            print('Full-content and fresh-delivery audit queued after all16 models and corrected report')
        else:
            try:
                watch(secret) if args.action == 'watch' else finalize(secret)
            except Exception as error:
                FINAL.mkdir(parents=True, exist_ok=True)
                write_json(FINAL / 'stopped.json', {'type': type(error).__name__, 'message': str(error),
                           'timestamp': datetime.now(timezone.utc).isoformat(), 'completion_unproven': True})
                raise
