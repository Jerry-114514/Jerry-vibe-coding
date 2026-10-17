"""Stop only the old scheduler, let its real training child finish unchanged."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

from experiment_common import read_json, write_json
from route_b_9b_loop import OUT

DEST = OUT / 'native-controller-handoff'


def args(pid):
    try:
        return [a.decode() for a in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if a]
    except FileNotFoundError:
        return []


def run():
    parent = 47340
    argv = args(parent)
    assert any(Path(a).name == 'run_9b_formal_when_ready.py' for a in argv)
    DEST.mkdir(parents=True, exist_ok=True)
    assert not (DEST / 'complete.json').exists(), 'Keep completed handoff immutable'
    write_json(DEST / 'plan.json', {'reason': 'User requires independent native0..1000 evaluation and calibrated minimum report',
               'old_scheduler_pid': parent, 'training_child_will_not_be_signalled': True,
               'registered_updates_unchanged': 15000, 'timestamp': datetime.now(timezone.utc).isoformat()})
    os.kill(parent, signal.SIGSTOP)
    children = (Path('/proc') / str(parent) / 'task' / str(parent) / 'children').read_text().split()
    jobs = []
    for child in children:
        command = args(child)
        if not command:
            continue  # exited child awaiting reaping by its stopped parent
        assert any(Path(a).name == 'route_b_9b_loop.py' for a in command) and 'train' in command, command
        jobs.append({'pid': int(child), 'arm': command[command.index('--arm')+1],
                     'seed': int(command[command.index('--seed')+1])})
    write_json(DEST / 'held-scheduler.json', {'status': 'scheduler_only_stopped_existing_train_child_continues', 'jobs': jobs})
    observer = args(48938)
    if any(Path(a).name == 'finalize_9b_delivery.py' for a in observer):
        os.kill(48938, signal.SIGTERM)
    while any(args(job['pid']) for job in jobs):
        print(json.dumps({'waiting_for_original_training_child': [job for job in jobs if args(job['pid'])]}), flush=True)
        time.sleep(30)
    for job in jobs:
        marker = read_json(OUT / 'formal' / f'{job["arm"]}-seed{job["seed"]}' / 'complete.json')
        assert marker['actual_updates'] == 1000
    os.kill(parent, signal.SIGTERM)
    os.kill(parent, signal.SIGCONT)
    for _ in range(100):
        if not args(parent):
            break
        time.sleep(.1)
    assert not args(parent), 'Old scheduler still live; never start a second scheduler'
    write_json(DEST / 'complete.json', {'status': 'old_scheduler_exited_train_child_completed_without_interruption',
               'jobs_finished': jobs, 'registered_updates_unchanged': 15000, 'old_pixel_evaluator_not_started': True,
               'timestamp': datetime.now(timezone.utc).isoformat()})
    print('Controller handoff complete; training-only successor may acquire original scheduler lock', flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        write_json(DEST / 'stopped.json', {'type': type(error).__name__, 'message': str(error),
                   'inspect_actual_processes_before_resuming_scheduler': True})
        raise
