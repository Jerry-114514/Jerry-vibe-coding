"""Queue9B minimum loop after admitted preflight; fixed100 updates, seed42.

This controller does not authorize or launch the15 formal9B runs. It provides
their shared tested engine; formal scheduling requires allfive real checks.
"""
from collections import Counter
from datetime import datetime, timezone, timedelta
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiment_common import ROOT, AXES, read_json, read_rows, sha, write_json, units, score_generation
from route_b_9b_loop import BASE, OUT, prepare
import preflight_9b_full as preflight


def verify_minimal():
    import torch
    from safetensors.torch import load_file
    manifest = read_rows(ROOT / 'runs/protocol-v2/calibration/precision-bridge-manifest.jsonl')
    assert len(manifest) == 800
    lookup = {r['item_id']: r for r in manifest}
    scores, training = {}, {}
    for arm in ('A1', 'A2', 'U', 'N'):
        folder = OUT / 'minimal' / f'{arm}-seed42'
        marker = read_json(folder / 'scores.json')
        ledger = read_rows(folder / 'eval.jsonl')
        assert len(ledger) == marker['rows'] == 800
        assert len({r['item_id'] for r in ledger}) == 800 and {r['item_id'] for r in ledger} == set(lookup)
        assert sha(folder / 'eval.jsonl') == marker['ledger_sha256']
        for record in ledger:
            actual = score_generation(record['raw_generation'], lookup[record['item_id']])
            assert all(actual[k] == record[k] for k in ('correct_bit', 'parsed_answer', 'parse_fail_flag', 'iou'))
        records = units(manifest, ledger)
        computed = {a: sum(r['correct_bit'] for r in records if r['axis'] == a) /
                       sum(r['axis'] == a for r in records) for a in AXES}
        assert computed == marker['axis_scores']
        assert marker['optimizer_loaded'] is False and marker['gradients_enabled'] is False
        scores[arm] = computed
        if arm == 'N':
            assert not (folder / 'adapter').exists() and not (folder / 'resume.pt').exists()
            continue
        complete = read_json(folder / 'complete.json')
        losses = read_rows(folder / 'loss.jsonl')
        assert len(losses) == complete['actual_updates'] == 100
        assert [r['step'] for r in losses] == list(range(1, 101))
        assert all(r['actual_optimizer_update'] and r['grad_norm'] > 0 and
                   all(math.isfinite(r[k]) for k in ('loss', 'ce_gt', 'ce_teacher', 'grad_norm')) for r in losses)
        assert all(abs(r['loss'] - .5 * r['ce_gt'] - .5 * r['ce_teacher']) < 1e-5 for r in losses)
        assert sha(folder / 'loss.jsonl') == complete['loss_sha256']
        assert sha(folder / 'resume.pt') == complete['resume_sha256']
        state = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=True)
        assert state['step'] == 100 and {int(v['step']) for v in state['optimizer']['state'].values()} == {100}
        exported = load_file(str(folder / 'adapter/adapter_model.safetensors'))
        assert sha(folder / 'adapter/adapter_model.safetensors') == complete['adapter_sha256']
        assert set(exported) == set(state['adapter']) and all(torch.equal(v, state['adapter'][k]) for k, v in exported.items())
        assert marker['adapter_reloaded_for_inference']
        training[arm] = {'real_updates': 100, 'axis_update_counts': dict(Counter(r['axis'] for r in losses)),
                         'reserved_peak_bytes': max(r['reserved_peak_bytes'] for r in losses),
                         'step_seconds': sum(r['seconds'] for r in losses)}
        del state, exported
    resume = read_json(OUT / 'minimal/A1-seed42/resume-verification.json')
    assert resume['restored_step'] == 50 and resume['optimizer_restored'] and resume['rng_restored']
    assert training['U']['axis_update_counts'] == {a: 25 for a in AXES}
    first_resumed_loss = read_rows(OUT / 'minimal/A1-seed42/loss.jsonl')[50]['loss']
    proof = {'status': 'passed', 'five_checks': {
        'cache_loading_and_no_silent_truncation': 'Provider stop responses; full student input checks passed before100-update preflight',
        'finite_dual_ce_nonzero_gradients': True, 'adapter_saved_and_reloaded_in_new_eval_process': True,
        'all_four_axis_scores_written_and_rescored': True, 'new_process_resumed50_to100': True},
        'resume': resume, 'first_resumed_loss': first_resumed_loss, 'scores': scores, 'training': training,
        'protocol_sha256': sha(OUT / 'protocol.json'), 'formal9b_runs_started': False,
        'scope': 'Independent9B+122B engineering loop; not originalARP confirmation'}
    write_json(OUT / 'minimal-acceptance.json', proof)
    return proof


def worker():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'minimal-manager.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 48 * 3600
        input_folder = BASE / 'evaluation-input-preparation'
        while not (input_folder / 'complete.json').exists():
            if (input_folder / 'failure.json').exists() or time.monotonic() >= deadline:
                raise RuntimeError('9B CPU evaluation input preparation did not pass; noGPU work started')
            write_json(OUT / 'queue-status.json', {'phase': 'waiting_for_cpu_input_preparation',
                       'pid': os.getpid(), 'gpu_used': False})
            time.sleep(30)
        prepare()
        if not (OUT / 'registration-delivered.json').exists():
            assert os.environ.get('GH_TOKEN'), 'External registration must be backed up before GPU work'
            registration = OUT / 'registration'
            registration.mkdir(exist_ok=True)
            for name in ('protocol.json', 'minimal-pool.jsonl', 'minimal-sample-counts.json'):
                shutil.copyfile(OUT / name, registration / name)
            for name in ('complete.json', 'minimal-manifest.jsonl', 'formal-manifest.jsonl'):
                shutil.copyfile(input_folder / name, registration / ('eval-input-' + name))
            from publish_artifacts import publish
            receipt = publish(registration, 'arp-9b-122b-20261017', token=os.environ['GH_TOKEN'])
            write_json(OUT / 'registration-delivered.json', receipt)
        marker = preflight.OUT / 'complete.json'
        while not marker.exists():
            if (preflight.OUT / 'failure.json').exists():
                raise RuntimeError('9B preflight failed; minimum loop must not launch')
            if time.monotonic() >= deadline:
                raise RuntimeError('Minimum-loop queue deadline reached')
            write_json(OUT / 'queue-status.json', {'phase': 'waiting_for_admitted_9b_preflight', 'pid': os.getpid(), 'gpu_used': False})
            time.sleep(30)
        result = read_json(marker)
        assert result['status'] == 'passed' and result['reserved_peak_bytes'] <= 28_000_000_000
        # Honor the registered combined2h GPU preflight/minimum-loop budget.
        first = read_rows(preflight.OUT / 'nvidia-smi.jsonl')[0]['sample'].split(',')[0].strip()
        began = datetime.strptime(first, '%Y/%m/%d %H:%M:%S.%f').astimezone(timezone.utc)
        finished = datetime.fromisoformat(result['timestamp'])
        consumed = (finished - began).total_seconds() + 60  # conservative loading/monitor reserve
        budget_path = OUT / 'minimal-wall-budget.json'
        if budget_path.exists():
            budget = read_json(budget_path)
            end_utc = datetime.fromisoformat(budget['end_utc'])
        else:
            now = datetime.now(timezone.utc)
            end_utc = min(now + timedelta(seconds=5400), began + timedelta(seconds=7140))
            budget = {'start_utc': now.isoformat(), 'end_utc': end_utc.isoformat(),
                      'preflight_elapsed_seconds_with_reserve': consumed,
                      'combined_preflight_minimal_cap_seconds': 7200,
                      'does_not_reset_on_restart': True}
            write_json(budget_path, budget)
        remaining = (end_utc - datetime.now(timezone.utc)).total_seconds()
        assert remaining > 0, 'Combined2h registered budget exhausted'
        end = time.monotonic() + remaining
        engine = ROOT / 'tools/route_b_9b_loop.py'
        def child(action, arm, stop=None):
            allowance = end - time.monotonic()
            assert allowance > 0, 'Minimum-loop fixed wall budget exhausted'
            command = [str(preflight.ENV), '-u', str(engine), action, '--phase', 'minimal', '--arm', arm, '--seed', '42']
            if stop is not None:
                command += ['--stop-at', str(stop)]
            subprocess.run(command, cwd=ROOT, check=True, timeout=allowance)
        try:
            child('eval', 'N')
            child('train', 'A1', 50)
            child('train', 'A1')
            for arm in ('A2', 'U'):
                child('train', arm)
            for arm in ('A1', 'A2', 'U'):
                child('eval', arm)
            verify_minimal()
        except Exception as error:
            write_json(OUT / 'failure.json', {'type': type(error).__name__, 'message': str(error),
                       'timestamp': datetime.now(timezone.utc).isoformat(), 'formal9b_runs_started': False})
            raise
        finally:
            if os.environ.get('GH_TOKEN'):
                from publish_artifacts import publish
                # Each training arm separately stays below1GiB scratch/split limits.
                for arm in ('A1', 'A2', 'U', 'N'):
                    folder = OUT / 'minimal' / f'{arm}-seed42'
                    if folder.exists():
                        publish(folder, 'arp-9b-122b-20261017', token=os.environ['GH_TOKEN'])
                report = OUT / 'minimal-report'
                report.mkdir(exist_ok=True)
                for name in ('minimal-acceptance.json', 'protocol.json', 'failure.json'):
                    source = OUT / name
                    if source.exists():
                        shutil.copyfile(source, report / name)
                publish(report, 'arp-9b-122b-20261017', token=os.environ['GH_TOKEN'])


if __name__ == '__main__':
    try:
        worker()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / 'minimal-manager-stopped.json', {'type': type(error).__name__, 'message': str(error),
                   'timestamp': datetime.now(timezone.utc).isoformat(), 'formal9b_runs_started': False})
        raise
