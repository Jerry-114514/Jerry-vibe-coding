"""Observe last3 evals, verify each stored ledger, deliver figures, then resume9B.

The frozen7B parent already compresses/uploads every model immediately. This
observer adds exact independent row/ID/score checks and fresh server coverage.
9B remains paused until7B full content, delivery, and final figures are saved.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import requests

from experiment_common import ROOT, read_json, sha, write_json
from route_b_staged import OUT

CHECKS = OUT / 'evaluation-delivery-checks'
HOLD = ROOT / 'runs/route-b-9b-122b-20261017/user-disk-priority-hold'
MODELS = ('U-seed1024', 'A1-seed1024', 'A2-seed1024')


def check_model(name, token):
    from audit_route_b_evaluation_scores import audit as rescore
    from audit_route_b_final_delivery import audit as delivery_audit, requirements
    result = rescore()
    model = next(r for r in result['verified_models'] if r['model'] == name)
    folder = OUT / 'formal' / name
    required = {p.relative_to(ROOT).as_posix() for p in requirements()}
    keys = [(folder / filename).relative_to(ROOT).as_posix() for filename in ('scores.json', 'eval.jsonl')]
    assert set(keys) <= required
    check = CHECKS / name
    check.mkdir(parents=True, exist_ok=True)
    write_json(check / 'integrity.json', {'status': 'passed_rows_ids_rescore', 'timestamp': datetime.now(timezone.utc).isoformat(),
               'exact_prompt_rows': 16900, 'unique_ids_and_full_manifest_alignment': True, **model})
    # Parent performs per-model compressed delivery before starting the next
    # model. A score marker alone does not prove those upload calls succeeded.
    storage = delivery_audit(token)
    incomplete = set(storage['missing_files']) | set(storage['unpublished_or_changed_files'])
    if set(keys) & incomplete:
        return False
    proof = {'status': 'passed_rows_and_fresh_release_coverage', 'model': name,
             'timestamp': datetime.now(timezone.utc).isoformat(), 'integrity_sha256': sha(check / 'integrity.json'),
             'ledger_sha256': model['ledger_sha256'], 'raw_and_score_files': keys,
             'fresh_server_verified_assets': storage['fresh_server_verified_assets'],
             'per_model_compressed_upload_by_frozen_parent': True}
    write_json(check / 'complete.json', proof)
    return True


def verify_figure_receipt(receipt, token):
    session = requests.Session()
    session.trust_env = False
    session.headers.update({'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json'})
    endpoint = 'https://api.github.com/repos/Jerry-114514/Jerry-vibe-coding'
    response = session.get(endpoint + '/releases/tags/arp-protocol-v2-20261016', timeout=30)
    response.raise_for_status()
    release = response.json()
    assert receipt['release_url'] == release['html_url']
    assets = {}
    for page in range(1, 100):
        response = session.get(endpoint + f'/releases/{release["id"]}/assets', params={'page': page, 'per_page': 100}, timeout=30)
        response.raise_for_status()
        batch = response.json()
        assets.update({a['name']: a for a in batch})
        if len(batch) < 100:
            break
    else:
        raise RuntimeError('Incomplete figure asset listing')
    for entry in receipt['assets']:
        actual = assets[entry['name']]
        assert actual['size'] == entry['bytes'] and actual.get('digest') == 'sha256:' + entry['sha256']
    return {'status': 'passed', 'verified_assets': len(receipt['assets'])}


def resume_9b(token):
    assert read_json(HOLD / 'hold.json')['status'] == '9b_queues_paused_for_7b_delivery'
    assert read_json(OUT / 'finalization/complete.json')['status'] == 'full_7b_content_and_delivery_verified'
    assert read_json(OUT / 'completion-reports/final-figures/delivered.json')['status'] == 'uploaded_and_fresh_server_verified'
    commands = [('preflight', 'tools/preflight_9b_full.py wait'), ('minimal', 'tools/run_9b_minimal_when_ready.py'),
                ('formal', 'tools/run_9b_formal_when_ready.py wait')]
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = proc.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if argv and b'python' in argv[0]:
            assert not any(command.encode() in b' '.join(argv) for _, command in commands), '9B controller already running; inspect before resuming'
    records = []
    for name, command in commands:
        shell = ('cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u ' + command +
                 ' >> runs/route-b-9b-122b-20261017/' + name + '-after-7b-console.log 2>&1')
        subprocess.run(['screen', '-dmS', 'route-b-9b-' + name + '-after-7b', 'bash', '-c', shell],
                       env={**os.environ, 'GH_TOKEN': token}, check=True)
        records.append({'controller': name, 'command': command})
    write_json(HOLD / 'resumed-after-verified-7b.json', {'timestamp': datetime.now(timezone.utc).isoformat(),
               'status': 'queues_resumed_after_7b_full_delivery_and_figures', 'controllers': records})


def worker(token):
    import fcntl
    CHECKS.mkdir(parents=True, exist_ok=True)
    with (CHECKS / 'observer.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            for name in MODELS:
                if (CHECKS / name / 'complete.json').exists():
                    continue
                if (OUT / 'formal' / name / 'scores.json').exists():
                    check_model(name, token)
            if all((CHECKS / name / 'complete.json').exists() for name in MODELS):
                break
            if (OUT / 'failure.json').exists():
                raise RuntimeError('7B pipeline failed; keep9B paused')
            time.sleep(10)
        while not (OUT / 'finalization/complete.json').exists():
            if (OUT / 'finalization/stopped.json').exists():
                raise RuntimeError('7B final delivery audit stopped; keep9B paused')
            time.sleep(10)
        from plot_route_b_final import generate, FOLDER
        if not (FOLDER / 'delivered.json').exists():
            generate()
            for name in MODELS:
                target = FOLDER / 'last-three-evaluations' / name
                target.mkdir(parents=True, exist_ok=True)
                for source in (CHECKS / name).glob('*.json'):
                    shutil.copyfile(source, target / source.name)
            shutil.copyfile(OUT / 'disk-cleanup-20261017/complete.json', FOLDER / 'disk-cleanup-report.json')
            shutil.copyfile(HOLD / 'hold.json', FOLDER / '9b-hold.json')
            from publish_artifacts import publish
            receipt = publish(FOLDER, 'arp-protocol-v2-20261016', token=token)
            verification = verify_figure_receipt(receipt, token)
            write_json(FOLDER / 'delivered.json', {'status': 'uploaded_and_fresh_server_verified',
                       'timestamp': datetime.now(timezone.utc).isoformat(), 'receipt': receipt, 'server_verification': verification})
        if not (HOLD / 'resumed-after-verified-7b.json').exists():
            resume_9b(token)
        print('7B last3 ledgers verified, final figures delivered,9B queue hold released', flush=True)


if __name__ == '__main__':
    token = os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
    assert token, 'Pass credential only through encrypted stdin or inherited environment'
    if '--start' in sys.argv:
        command = ('cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/finish_7b_safely.py '
                   '>> runs/route-b-staged-7b-20261016/safe-finish-console.log 2>&1')
        subprocess.run(['screen', '-dmS', 'route-b-safe-finish', 'bash', '-c', command],
                       env={**os.environ, 'GH_TOKEN': token}, check=True)
        print('Last3 ledger verification and final-figure delivery queued;9B remains paused until all pass')
    else:
        try:
            worker(token)
        except Exception as error:
            write_json(CHECKS / 'stopped.json', {'type': type(error).__name__, 'message': str(error),
                       'timestamp': datetime.now(timezone.utc).isoformat(), '9b_hold_not_intentionally_released': True})
            raise
