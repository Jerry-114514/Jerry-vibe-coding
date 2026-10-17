"""Registered15-run9B extension; wait for allfive real minimum-loop checks.

Completed optimizer recovery stays in verified GitHub assets, retaining local
adapters. Incomplete recovery is never offloaded. No7B artifact is changed.
"""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import requests

from experiment_common import ROOT, read_json, read_rows, sha, write_json, write_rows
from route_b_9b_loop import BASE, OUT, SEEDS, arm_rows, prepare, freeze
import preflight_9b_full as shared
from verify_9b_student_outputs import verify_training, verify_evaluation

TAG = 'arp-9b-122b-20261017'
DELIVERIES = ROOT / 'runs/9b-student-delivery-20261017'


def verify_receipt(receipt, token):
    session = requests.Session()
    session.trust_env = False
    session.headers.update({'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json'})
    base = 'https://api.github.com/repos/Jerry-114514/Jerry-vibe-coding'
    response = session.get(base + '/releases/tags/' + TAG, timeout=30)
    response.raise_for_status()
    release = response.json()
    assert receipt['release_url'] == release['html_url']
    assets = {}
    for page in range(1, 100):
        response = session.get(base + f'/releases/{release["id"]}/assets', params={'page': page, 'per_page': 100}, timeout=30)
        response.raise_for_status()
        batch = response.json()
        assets.update({a['name']: a for a in batch})
        if len(batch) < 100:
            break
    for item in receipt['assets']:
        actual = assets[item['name']]
        assert actual['size'] == item['bytes'] and actual.get('digest') == 'sha256:' + item['sha256']
    return {'status': 'passed', 'server_assets': len(receipt['assets']), 'timestamp': datetime.now(timezone.utc).isoformat()}


def offload_completed_recovery(folder, token):
    folder = Path(folder).resolve()
    assert folder.is_relative_to(OUT.resolve()) and folder.parent.name in ('minimal', 'formal')
    checkpoint = folder / 'resume.pt'
    if not checkpoint.exists():
        assert (folder / 'resume-offloaded.json').exists()
        return
    meta = read_json(folder / 'train.meta.json')
    verify_training(meta['arm'], meta['seed'], phase=meta['phase'], allow_offloaded=False)
    receipt_path, inventory_path = folder / 'training-delivery.json', folder / 'training-inventory.json'
    receipt = read_json(receipt_path)
    inventory = read_json(inventory_path)
    marker = read_json(folder / 'complete.json')
    entry = next(r for r in inventory['files'] if r['path'] == str(checkpoint.relative_to(ROOT)))
    assert entry['sha256'] == marker['resume_sha256'] == sha(checkpoint)
    assert next(a for a in receipt['assets'] if a['name'].endswith('-inventory.json'))['sha256'] == sha(inventory_path)
    server = verify_receipt(receipt, token)
    # Write the restore pointer first; on interruption the local state remains
    # usable, or its exact verified archive location is already recorded.
    write_json(folder / 'resume-offloaded.json', {'resume_sha256': marker['resume_sha256'],
               'bytes': checkpoint.stat().st_size, 'receipt_sha256': sha(receipt_path),
               'inventory_sha256': sha(inventory_path), 'release_url': receipt['release_url'],
               'assets': receipt['assets'], 'archive_member': str(checkpoint.relative_to(ROOT)),
               'local_completed_optimizer_copy_only': True, 'adapter_retained_locally': True,
               'server_verification': server, 'restore': 'Download listed asset(s) directly on AutoDL, concatenate split parts if any, extract archive_member and verify resume_sha256'})
    checkpoint.unlink()


def metadata_backup(folder, token):
    from publish_artifacts import publish
    folder = Path(folder)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(time.time_ns())
    snapshot = DELIVERIES / stamp
    for source in folder.rglob('*'):
        if not source.is_file() or source.is_symlink() or source.suffix in ('.pt', '.safetensors', '.tmp', '.lock'):
            continue
        if 'adapter' in source.relative_to(folder).parts:
            continue
        target = snapshot / 'files' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    references = {p.parent.name: read_json(p) for p in (OUT / 'formal').glob('*/training-delivery.json')}
    write_json(snapshot / 'binary-references.json', references)
    receipt = publish(snapshot, TAG, token=token)
    write_json(folder / 'latest-metadata-delivery.json', receipt)
    return receipt


def register(token):
    prepare()
    pool = read_rows(BASE / 'eligible-pool.jsonl')
    order, manifests = [], {}
    target = OUT / 'formal-registration'
    target.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        arms = ['A1', 'A2', 'U']
        random.Random(seed).shuffle(arms)
        for arm in arms:
            rows = arm_rows(pool, 'formal', arm, seed)
            path = target / f'{arm}-seed{seed}.jsonl'
            if path.exists():
                assert read_rows(path) == rows
            else:
                write_rows(path, rows)
            name = f'{arm}-seed{seed}'
            manifests[name] = {'rows': len(rows), 'sha256': sha(path)}
            order.append({'arm': arm, 'seed': seed, 'steps': 1000})
    sources = ['run_9b_formal_when_ready.py', 'verify_9b_student_outputs.py', 'analyze_9b_formal.py', 'report_route_b_multiplicity.py']
    plan = {'schema': 'independent.9b-formal.fixed15.v1', 'student_protocol_sha256': sha(OUT / 'protocol.json'),
            'source_sha256': {name: sha(ROOT / 'tools' / name) for name in sources},
            'manifests': manifests, 'run_order': order, 'updates': 15000,
            'primary_contrasts': ['A1-A2', 'A1-U', 'A1-N'], 'axes': ['pope', 'grounding', 'consistency', 'mmstar'],
            'bootstrap_replicates': 10000, 'bootstrap_seed': 2026101703, 'family_size': 12,
            'training_wall_hours_cap': 12, 'evaluation_wall_hours_cap': 40,
            'recovery_storage': 'Verify locally and fresh server digests before offloading completed optimizer copies; keep all adapters',
            'not_original_arp': True, 'minimum_loop_pass_required': True,
            'never_extend_budget_or_select_checkpoint_from_scores': True}
    freeze(OUT / 'formal-plan.json', plan)
    shutil.copyfile(OUT / 'formal-plan.json', target / 'formal-plan.json')
    if not (OUT / 'formal-registration-delivered.json').exists():
        from publish_artifacts import publish
        receipt = publish(target, TAG, token=token)
        write_json(OUT / 'formal-registration-delivered.json', receipt)
    return plan


def remaining_budget(phase, hours):
    path = OUT / f'{phase}-wall-budget.json'
    now = datetime.now(timezone.utc)
    if path.exists():
        end = datetime.fromisoformat(read_json(path)['end_utc'])
    else:
        end = now + timedelta(hours=hours)
        write_json(path, {'start_utc': now.isoformat(), 'end_utc': end.isoformat(), 'hours_cap': hours, 'never_reset_on_restart': True})
    remaining = (end - now).total_seconds()
    assert remaining > 0, 'Registered wall budget exhausted for ' + phase
    return remaining


def backup_training(folder, token):
    from publish_artifacts import publish
    if not (folder / 'training-delivery.json').exists():
        publish(folder, TAG, token=token)
        shutil.copyfile(folder / 'github-inventory.json', folder / 'training-inventory.json')
        shutil.copyfile(folder / 'github-storage.json', folder / 'training-delivery.json')
    offload_completed_recovery(folder, token)


def preserve_incomplete_journal_tail(path):
    path = Path(path)
    assert path.resolve().is_relative_to(OUT.resolve())
    if not path.exists():
        return
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        boundary = raw.rfind(b'\n') + 1
        tail = path.with_name(path.name + '.incomplete-tail-' + str(time.time_ns()))
        tail.write_bytes(raw[boundary:])
        with path.open('r+b') as stream:
            stream.truncate(boundary)


def worker(token):
    import fcntl
    with (OUT / 'formal-manager.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = register(token)
        deadline = time.monotonic() + 48 * 3600
        while not (OUT / 'minimal-acceptance.json').exists():
            if (OUT / 'minimal-manager-stopped.json').exists() or (OUT / 'failure.json').exists():
                raise RuntimeError('Minimum loop did not pass; formal9B must not start')
            assert time.monotonic() < deadline, 'Formal queue deadline reached'
            write_json(OUT / 'formal-queue-status.json', {'phase': 'waiting_for_allfive_minimal_checks', 'pid': os.getpid(), 'gpu_used': False})
            time.sleep(30)
        assert read_json(OUT / 'minimal-acceptance.json')['status'] == 'passed'
        for name, expected in plan['source_sha256'].items():
            assert sha(ROOT / 'tools' / name) == expected
        start_proof = OUT / 'formal-start-verification.json'
        if start_proof.exists():
            prior = read_json(start_proof)
            assert prior['minimum_acceptance_sha256'] == sha(OUT / 'minimal-acceptance.json')
            assert prior['formal_plan_sha256'] == sha(OUT / 'formal-plan.json')
        else:
            # Independently rerun the real gate once before any formal work.
            # Later resume relies on its immutable proof after completed
            # minimum optimizer states have moved to verified Release assets.
            from run_9b_minimal_when_ready import verify_minimal
            assert verify_minimal()['status'] == 'passed'
            write_json(start_proof, {'status': 'passed_allfive_real_checks',
                       'minimum_acceptance_sha256': sha(OUT / 'minimal-acceptance.json'),
                       'formal_plan_sha256': sha(OUT / 'formal-plan.json'),
                       'timestamp': datetime.now(timezone.utc).isoformat()})
        engine = ROOT / 'tools/route_b_9b_loop.py'
        def child(action, arm, seed, budget_phase, hours):
            assert shutil.disk_usage(OUT).free >= 5 * 1024 ** 3, 'Registered5GiB disk stop'
            folder = OUT / 'formal' / f'{arm}-seed{seed}'
            preserve_incomplete_journal_tail(folder / ('loss.jsonl' if action == 'train' else 'eval.jsonl'))
            subprocess.run([str(shared.ENV), '-u', str(engine), action, '--phase', 'formal', '--arm', arm, '--seed', str(seed)],
                           cwd=ROOT, check=True, timeout=remaining_budget(budget_phase, hours))
        try:
            for job in plan['run_order']:
                folder = OUT / 'formal' / f'{job["arm"]}-seed{job["seed"]}'
                if not (folder / 'complete.json').exists():
                    child('train', job['arm'], job['seed'], 'formal-training', 12)
                verify_training(job['arm'], job['seed'], allow_offloaded=True)
                backup_training(folder, token)
                # Once the complete minimum report has been delivered, its
                # recovery copies can follow the same verified offload policy.
                if (OUT / 'minimal-report/github-storage.json').exists():
                    for arm in ('A1', 'A2', 'U'):
                        prior = OUT / 'minimal' / f'{arm}-seed42'
                        if (prior / 'resume.pt').exists():
                            if not (prior / 'training-delivery.json').exists():
                                shutil.copyfile(prior / 'github-storage.json', prior / 'training-delivery.json')
                                shutil.copyfile(prior / 'github-inventory.json', prior / 'training-inventory.json')
                            offload_completed_recovery(prior, token)
            if not (OUT / 'formal-training-complete.json').exists():
                write_json(OUT / 'formal-training-complete.json', {'runs': 15, 'updates': 15000, 'timestamp': datetime.now(timezone.utc).isoformat()})
            from analyze_9b_formal import training_report
            if not (OUT / 'formal-report/training-results.json').exists():
                training_report()
            metadata_backup(OUT, token)
            models = [('N', 42)] + [(j['arm'], j['seed']) for j in plan['run_order']]
            for arm, seed in models:
                folder = OUT / 'formal' / f'{arm}-seed{seed}'
                if not (folder / 'scores.json').exists():
                    child('eval', arm, seed, 'formal-evaluation', 40)
                verified, _ = verify_evaluation(arm, seed)
                delivery = folder / 'eval-delivery.json'
                if not delivery.exists() or read_json(delivery)['ledger_sha256'] != verified['ledger_sha256']:
                    receipt = metadata_backup(folder, token)
                    write_json(delivery, {'ledger_sha256': verified['ledger_sha256'], 'receipt': receipt})
            from analyze_9b_formal import analyze
            analyze()
            write_json(OUT / 'status.json', {'phase': 'all9b_formal_stages_complete', 'not_original_arp': True})
        except Exception as error:
            write_json(OUT / 'formal-failure.json', {'type': type(error).__name__, 'message': str(error),
                       'timestamp': datetime.now(timezone.utc).isoformat()})
            raise
        finally:
            metadata_backup(OUT, token)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'wait'])
    args = parser.parse_args()
    token = os.environ.get('GH_TOKEN')
    assert token, 'Pass backup credential only through process environment'
    if args.action == 'register':
        print(json.dumps({'status': 'registered', 'runs': len(register(token)['run_order'])}), flush=True)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        try:
            worker(token)
        except Exception as error:
            write_json(OUT / 'formal-manager-stopped.json', {'type': type(error).__name__, 'message': str(error),
                       'timestamp': datetime.now(timezone.utc).isoformat(),
                       'inspect_stage_markers_and_actual_process_before_any_restart': True})
            raise
