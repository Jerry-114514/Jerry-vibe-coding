"""Continue exactly the registered15 trainings; native evaluation is separate.

No frozen numeric source is changed. The latest night-mode instruction retains
every9B recovery state; prior offloaded states are restored by the evaluator.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT, read_json, sha, write_json
from route_b_9b_loop import OUT
from run_9b_formal_when_ready import (register, remaining_budget, preserve_incomplete_journal_tail,
                                     metadata_backup)
from verify_9b_student_outputs import verify_training
from preflight_9b_full import ENV


def run(token):
    import fcntl
    handoff = OUT / 'native-controller-handoff'
    while not (handoff / 'complete.json').exists():
        assert not (handoff / 'stopped.json').exists(), 'Inspect failed controller handoff'
        time.sleep(5)
    with (OUT / 'formal-manager.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = register(token)
        assert read_json(OUT / 'formal-start-verification.json')['formal_plan_sha256'] == sha(OUT / 'formal-plan.json')
        for name, digest in plan['source_sha256'].items():
            assert sha(ROOT / 'tools' / name) == digest
        record = {'status': 'user_native_evaluation_amendment_training_unchanged',
                  'formal_plan_sha256': sha(OUT / 'formal-plan.json'), 'source_sha256': sha(Path(__file__)),
                  'registered_updates': 15000, 'registered_wall_hours': 12,
                  'never_resets_wall_budget': True, 'no_legacy_pixel_formal_evaluation': True,
                  'recovery_offload_disabled_by_user_night_mode': True}
        write_json(OUT / 'training-only-amendment.json', record)
        from publish_artifacts import publish
        registration = OUT / 'training-only-registration'
        registration.mkdir(exist_ok=True)
        shutil.copyfile(OUT / 'training-only-amendment.json', registration / 'amendment.json')
        if not (registration / 'github-storage.json').exists():
            publish(registration, 'arp-9b-122b-20261017', token=token)
        for job in plan['run_order']:
            folder = OUT / 'formal' / f'{job["arm"]}-seed{job["seed"]}'
            if not (folder / 'complete.json').exists():
                assert shutil.disk_usage(OUT).free >= 5*1024**3
                preserve_incomplete_journal_tail(folder / 'loss.jsonl')
                subprocess.run([str(ENV), '-u', str(ROOT / 'tools/route_b_9b_loop.py'), 'train',
                                '--phase', 'formal', '--arm', job['arm'], '--seed', str(job['seed'])],
                               cwd=ROOT, check=True, timeout=remaining_budget('formal-training', 12))
            verify_training(job['arm'], job['seed'], allow_offloaded=True)
            if not (folder / 'training-delivery.json').exists():
                publish(folder, 'arp-9b-122b-20261017', token=token)
                shutil.copyfile(folder / 'github-storage.json', folder / 'training-delivery.json')
                shutil.copyfile(folder / 'github-inventory.json', folder / 'training-inventory.json')
        write_json(OUT / 'formal-training-complete.json', {'runs': 15, 'updates': 15000,
                   'timestamp': datetime.now(timezone.utc).isoformat(), 'evaluation': 'native-evaluation-v2 only'})
        from analyze_9b_formal import training_report
        training_report()
        metadata_backup(OUT, token)
        print('All15 fixed-budget trainings verified and delivered; native evaluation controller must validate its separate gate', flush=True)


if __name__ == '__main__':
    token = os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
    assert token, 'Credential required through encrypted stdin or inherited environment'
    if '--start' in sys.argv:
        command = ('cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u '
                   'tools/continue_9b_training_only.py >> '
                   'runs/route-b-9b-122b-20261017/student-experiment/training-only-console.log 2>&1')
        subprocess.run(['screen', '-dmS', 'route-b-9b-training-only', 'bash', '-c', command],
                       env={**os.environ, 'GH_TOKEN': token}, check=True)
        print('Registered9B training-only continuation started; native evaluator is independent')
    else:
        try:
            run(token)
        except Exception as error:
            write_json(OUT / 'training-only-stopped.json', {'type': type(error).__name__, 'message': str(error),
                       'timestamp': datetime.now(timezone.utc).isoformat()})
            raise
