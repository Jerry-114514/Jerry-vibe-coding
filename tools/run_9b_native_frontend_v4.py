"""Independent v4 measurement: register, bounded400 pilot, then fixed16 evals.

Keeps v3 failure and all training immutable. Pilot predictions are part of the
same270400 formal ledger, never regenerated after passing admission.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT, AXES, read_json, read_rows, write_json, write_rows, sha, units, score_generation
from route_b_9b_loop import OUT as STUDENT, BASE, SEEDS, MINIMAL_MANIFEST, freeze
from qwen35_native_frontend import native_prompt, score_native_frontend
from processor_pixel_contract import assert_runtime_grid

OUT = STUDENT / 'native-frontend-evaluation-v4'
ENV = ROOT.parent / 'arp-qwen35-venv/bin/python'
TAG = 'arp-9b-122b-20261017'
SOURCES = ('run_9b_native_frontend_v4.py', 'qwen35_native_frontend.py',
           'processor_pixel_contract.py', 'processor_pixel_contract_csv.py',
           'experiment_common.py', 'scoring.py', 'route_b_9b_loop.py',
           'preflight_9b_full.py', 'model_backends.py', 'analyze_9b_formal.py',
           'report_route_b_multiplicity.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def models():
    return [('N', 42)] + [(j['arm'], j['seed']) for j in read_json(STUDENT / 'formal-plan.json')['run_order']]


def prepare():
    path = OUT / 'protocol.json'
    if path.exists():
        plan = read_json(path)
        for name, digest in plan['source_sha256'].items():
            assert sha(ROOT / 'tools' / name) == digest, name
        assert sha(OUT / 'formal-manifest.jsonl') == plan['manifest_sha256']
        assert sha(ROOT / 'data/manifest.jsonl') == plan['original_manifest_sha256']
        assert sha(BASE / 'eligible-pool.jsonl') == plan['training_pool_sha256']
        return plan
    assert read_json(STUDENT / 'formal-training-complete.json')['updates'] == 15000
    assert read_json(ROOT / 'runs/7b-processor-pixel-audit-20261017/complete.json')['models'] == 16
    previous = STUDENT / 'processor-pixel-evaluation-v3'
    assert read_json(previous / 'failed-calibration-report/diagnostic.json')['status'] == 'registered9b_calibration_failed_no_formal_evaluation'
    original = read_rows(ROOT / 'data/manifest.jsonl')
    old = {x['item_id']: x for x in read_rows(previous / 'formal-manifest.jsonl')}
    assert len(original) == len(old) == 16900
    from PIL import Image
    cache = {}

    def rgb(row):
        key = row['image_path']
        if key not in cache:
            p = (ROOT / key).resolve()
            assert p.is_relative_to((ROOT / 'data').resolve())
            with Image.open(p) as image:
                assert image.size == (row['width'], row['height'])
                image = image.convert('RGB')
                digest = hashlib.sha256()
                digest.update(f'{image.width}x{image.height}:RGB:'.encode())
                digest.update(image.tobytes())
                cache[key] = digest.hexdigest()
            if len(cache) % 250 == 0:
                print('CPU RGB identity', len(cache), flush=True)
        return cache[key]

    train_hashes = {rgb(x) for x in read_rows(BASE / 'eligible-pool.jsonl')}
    previous_hashes = {rgb(x) for x in read_rows(MINIMAL_MANIFEST) if x['axis'] == 'grounding'}
    candidates = defaultdict(list)
    excluded = []
    for row in original:
        if row['axis'] != 'grounding':
            continue
        digest = rgb(row)
        reason = 'training_RGB_overlap' if digest in train_hashes else 'old_calibration_RGB_overlap' if digest in previous_hashes else None
        if reason:
            excluded.append({'item_id': row['item_id'], 'rgb_sha256': digest, 'reason': reason})
        else:
            candidates[row['dataset']].append(row)
    for key in candidates:
        candidates[key].sort(key=lambda x: hashlib.sha256(('2026101801:' + x['item_id']).encode()).hexdigest())
    selected, seen_rgb, seen_groups = [], set(), set()
    while len(selected) < 100:
        progressed = False
        for dataset in sorted(candidates):
            while candidates[dataset]:
                row = candidates[dataset].pop(0)
                digest = rgb(row)
                if digest in seen_rgb or row['image_group'] in seen_groups:
                    continue
                selected.append(row)
                seen_rgb.add(digest)
                seen_groups.add(row['image_group'])
                progressed = True
                break
            if len(selected) == 100:
                break
        assert progressed, 'Fewer than100 distinct non-overlapping calibration images; never relax exclusions'
    assert len(seen_rgb) == len(seen_groups) == 100
    from transformers import AutoProcessor
    from route_b_9b_loop import public_prompt_inputs
    import preflight_9b_full as shared
    import torch
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(str(shared.MODEL), local_files_only=True)
    manifest = []
    for index, item in enumerate(original):
        row = native_prompt(item)
        if row['axis'] == 'grounding':
            for key in ('processor_width', 'processor_height', 'processor_grid_thw', 'processor_patch_size'):
                row[key] = old[item['item_id']][key]
            batch = public_prompt_inputs(processor, row)
            assert_runtime_grid(batch, row)
            row['frozen_prompt_tokens'] = int(batch['input_ids'].shape[1])
        else:
            assert row == item
        manifest.append(row)
        if index % 1000 == 0:
            print('CPU input preflight', index, '/16900', flush=True)
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUT / 'formal-manifest.jsonl', manifest)
    write_json(OUT / 'calibration-selection.json', {
        'rule_seed': 2026101801, 'rows': 100, 'distinct_RGB_images': 100, 'distinct_image_groups': 100,
        'item_ids': [r['item_id'] for r in selected], 'adapter_item_ids': [r['item_id'] for r in selected[:20]],
        'excluded': excluded, 'image_path_RGB_sha256': cache,
        'exact_RGB_only_not_a_claim_of_perceptual_dedup_or_pretraining_independence': True})
    adapters = {}
    for arm, seed in models()[1:]:
        folder = STUDENT / 'formal' / f'{arm}-seed{seed}'
        marker = read_json(folder / 'complete.json')
        assert marker['actual_updates'] == 1000
        assert sha(folder / 'loss.jsonl') == marker['loss_sha256']
        digest = sha(folder / 'adapter/adapter_model.safetensors')
        assert digest == marker['adapter_sha256']
        adapters[folder.name] = digest
    plan = {'schema': 'independent.9b.native_frontend.v4', 'timestamp_utc': now(),
        'authorization': 'User2026-10-18: think through the repair plan then execute immediately',
        'superseded_failed_v3_protocol_sha256': sha(previous / 'protocol.json'),
        'original_v3_failure_retained': True, 'no_v3_budget_reset': True,
        'source_sha256': {n: sha(ROOT / 'tools' / n) for n in SOURCES},
        'student_protocol_sha256': sha(STUDENT / 'protocol.json'),
        'training_plan_sha256': sha(STUDENT / 'formal-plan.json'),
        'training_pool_sha256': sha(BASE / 'eligible-pool.jsonl'),
        'original_manifest_sha256': sha(ROOT / 'data/manifest.jsonl'),
        'manifest_sha256': sha(OUT / 'formal-manifest.jsonl'),
        'selection_sha256': sha(OUT / 'calibration-selection.json'),
        'processor_dimensions_source_sha256': sha(previous / 'processor-dimensions.json'),
        'adapters_sha256': adapters, 'student_revision': read_json(STUDENT / 'protocol.json')['revision'],
        'coordinate_contract': 'All9B arms: explicit native0..1000 -> actual processor pixels -> original pixels, IoU>=.5; fixed frontend, no GT/arm/output-magnitude based units',
        'training_unchanged': True, 'training_targets': 'original pixels as frozen',
        'calibration_models': 16, 'calibration_predictions_cap': 400,
        'N_calibration_rows': 100, 'each_adapter_calibration_rows': 20,
        'calibration_wall_seconds_cap': 900, 'N_accuracy_minimum': .80,
        'N_parse_failures_maximum': 5, 'adapter_parse_failures_maximum': 1,
        'calibration_length_capped_maximum': 0,
        'pilot_predictions_reused_in_formal_ledgers': True, 'total_formal_predictions_including_pilot': 270400,
        'formal_models': 16, 'formal_rows_per_model': 16900, 'formal_wall_hours_cap': 40,
        'max_new_tokens': 128, 'reserved_bytes_cap': 30000000000, 'minimum_free_bytes': 5*1024**3,
        'statistics': {'replicates': 10000, 'seed': 2026101703, 'family_size': 12,
                       'contrasts': ['A1-A2', 'A1-U', 'A1-N']},
        'scope': 'Post-result measurement repair and instruction/format adherence; development benchmark, not ARP confirmation or pure capacity/grounding isolation',
        'stop': 'Any admission/hash/memory/truncation/budget failure stops; no prompt retry or unit switching'}
    freeze(path, plan)
    print('Prepared v4 on CPU; no predictions generated', flush=True)
    return plan


def scoring(text, item):
    return score_native_frontend(text, item) if item['axis'] == 'grounding' else score_generation(text, item)


def budget(kind, seconds):
    path = OUT / f'{kind}-wall-budget.json'
    if not path.exists():
        start = time.time()
        write_json(path, {'start_utc': datetime.fromtimestamp(start, timezone.utc).isoformat(),
                         'end_utc': datetime.fromtimestamp(start+seconds, timezone.utc).isoformat(),
                         'seconds_cap': seconds, 'never_reset_on_restart': True})
    end = datetime.fromisoformat(read_json(path)['end_utc']).timestamp()
    assert time.time() < end, 'Registered ' + kind + ' absolute budget exhausted'
    return end


def preserve_tail(path):
    if path.exists():
        raw = path.read_bytes()
        if raw and not raw.endswith(b'\n'):
            boundary = raw.rfind(b'\n')+1
            path.with_name('incomplete-tail-' + str(time.time_ns())).write_bytes(raw[boundary:])
            with path.open('r+b') as stream:
                stream.truncate(boundary)


def evaluate(arm, seed, pilot):
    import torch
    from gpu_lock import gpu_lock
    from route_b_9b_loop import admission, model_backend, public_prompt_inputs, gpu_inputs
    from model_backends import finite_logits_processors
    admission()
    plan = prepare()
    assert (OUT / 'registration/delivery-proof.json').exists(), 'External registration must precede every prediction'
    if not pilot:
        assert read_json(OUT / 'calibration-acceptance.json')['status'] == 'passed'
    manifest = read_rows(OUT / 'formal-manifest.jsonl')
    lookup = {r['item_id']: r for r in manifest}
    selection = read_json(OUT / 'calibration-selection.json')
    assert sha(OUT / 'calibration-selection.json') == plan['selection_sha256']
    target = [lookup[k] for k in selection['item_ids' if arm == 'N' else 'adapter_item_ids']] if pilot else manifest
    folder = OUT / 'formal' / f'{arm}-seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    adapter = None if arm == 'N' else STUDENT / 'formal' / folder.name / 'adapter'
    if adapter is not None:
        assert sha(adapter / 'adapter_model.safetensors') == plan['adapters_sha256'][folder.name]
    freeze(folder / 'eval.meta.json', {'model': folder.name, 'protocol_sha256': sha(OUT / 'protocol.json'),
        'manifest_sha256': plan['manifest_sha256'], 'thinking': False, 'max_new_tokens': 128,
        'pilot_reused_without_regeneration': True})
    ledger = folder / 'eval.jsonl'
    preserve_tail(ledger)
    rows = read_rows(ledger) if ledger.exists() else []
    done = {r['item_id'] for r in rows}
    assert len(done) == len(rows) and done <= {r['item_id'] for r in target}
    marker_path = folder / ('calibration-scores.json' if pilot else 'scores.json')
    if marker_path.exists():
        return
    with gpu_lock():
        end = budget('calibration' if pilot else 'formal', 900 if pilot else 40*3600)
        started = time.monotonic()
        torch.set_num_threads(2)
        torch.set_grad_enabled(False)
        torch.cuda.reset_peak_memory_stats()
        processor, model = model_backend(adapter)
        model.eval().requires_grad_(False)
        assert not any(p.requires_grad for p in model.parameters())
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for item in target:
            if item['item_id'] in done:
                continue
            assert time.time() < end
            assert shutil.disk_usage(OUT).free >= plan['minimum_free_bytes']
            tick = time.monotonic()
            with torch.inference_mode():
                cpu = public_prompt_inputs(processor, item)
                if item['axis'] == 'grounding':
                    assert_runtime_grid(cpu, item)
                    assert cpu['input_ids'].shape[1] == item['frozen_prompt_tokens']
                batch = gpu_inputs(cpu)
                generated = model.generate(**batch, max_new_tokens=128, do_sample=False,
                                           logits_processor=finite_logits_processors())
                ids = generated[0, batch['input_ids'].shape[1]:]
                raw = processor.decode(ids, skip_special_tokens=True).strip()
            torch.cuda.synchronize()
            row = {'item_id': item['item_id'], 'axis': item['axis'],
                'seconds': time.monotonic()-tick, 'generated_token_ids': ids.tolist(),
                'finish_reason': 'eos' if ids.numel() and int(ids[-1]) in eos else 'length',
                'generation_tokens': int(ids.numel()), 'pilot_prediction': pilot, **scoring(raw, item)}
            with ledger.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            done.add(item['item_id'])
            assert torch.cuda.max_memory_reserved() <= plan['reserved_bytes_cap']
            if len(done) % 100 == 0:
                write_json(OUT / 'progress.json', {'phase': 'pilot' if pilot else 'formal',
                    'model': folder.name, 'rows': len(done), 'pid': os.getpid(), 'timestamp_utc': now()})
                print('v4', folder.name, len(done), '/', len(target), flush=True)
        rows = read_rows(ledger)
        assert len(rows) == len(done) == len(target)
        marker = {'rows': len(rows), 'ledger_sha256': sha(ledger),
            'protocol_sha256': sha(OUT / 'protocol.json'), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'optimizer_loaded': False,
            'gradients_enabled': False, 'adapter_reloaded_for_inference': arm != 'N',
            'total_generation_seconds': sum(r['seconds'] for r in rows),
            'active_process_seconds': time.monotonic()-started,
            'length_capped_outputs': sum(r['finish_reason'] == 'length' for r in rows), 'finished_utc': now()}
        if pilot:
            marker.update(accuracy=sum(r['correct_bit'] for r in rows)/len(rows),
                          parse_failures=sum(r['parse_fail_flag'] for r in rows))
        else:
            original = units(manifest, rows)
            marker.update(original_units=len(original), axis_scores={axis: sum(r['correct_bit'] for r in original if r['axis'] == axis)/sum(r['axis'] == axis for r in original) for axis in AXES})
        write_json(marker_path, marker)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


def verify_evaluation(arm, seed):
    folder = OUT / 'formal' / f'{arm}-seed{seed}'
    manifest = read_rows(OUT / 'formal-manifest.jsonl')
    lookup = {r['item_id']: r for r in manifest}
    rows = read_rows(folder / 'eval.jsonl')
    marker = read_json(folder / 'scores.json')
    assert len(rows) == len({r['item_id'] for r in rows}) == marker['rows'] == 16900
    assert {r['item_id'] for r in rows} == set(lookup)
    assert sha(folder / 'eval.jsonl') == marker['ledger_sha256']
    assert marker['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert marker['optimizer_loaded'] is False and marker['gradients_enabled'] is False
    for row in rows:
        value = scoring(row['raw_generation'], lookup[row['item_id']])
        assert all(value[k] == row[k] for k in value)
        assert row['generation_tokens'] == len(row['generated_token_ids'])
    original = units(manifest, rows)
    assert len(original) == marker['original_units'] == 13700
    assert marker['axis_scores'] == {a: sum(r['correct_bit'] for r in original if r['axis'] == a)/sum(r['axis'] == a for r in original) for a in AXES}
    return marker, {(r['axis'], r['dataset'], r['unit']): r for r in original}


def verify_formal(return_evaluations=False):
    evaluations = {(a, s): verify_evaluation(a, s) for a, s in models()}
    result = {'status': 'passed', 'models': 16, 'rows': 270400, 'not_original_arp': True,
              'protocol_sha256': sha(OUT / 'protocol.json')}
    write_json(OUT / 'formal-independent-verification.json', result)
    return (result, evaluations) if return_evaluations else result


def create_figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    palette = {'N': '#666666', 'A1': '#2878b5', 'A2': '#df8f44', 'U': '#35a07b'}
    names = list(result['scores'])
    folder = OUT / 'formal-report'
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for axis, ax in zip(AXES, axes.flat):
        ax.bar(range(16), [100*result['scores'][name][axis] for name in names],
               color=[palette[name.split('-')[0]] for name in names])
        ax.set_xticks(range(16), names, rotation=75, ha='right', fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Original-unit accuracy (%)')
        ax.set_title(axis)
    figure.suptitle('9B +122B: fixed v4 interface, all16 model endpoints')
    for extension in ('png', 'pdf'):
        figure.savefig(folder / ('four-axis-scores.'+extension), dpi=170)
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    labels = []
    for comparison, values in result['paired_contrasts'].items():
        for axis, row in values.items():
            i = len(labels)
            labels.append(comparison+' / '+axis)
            lo, hi = [100*v for v in row['bonferroni_familywise_ci95']]
            ax.plot([lo, hi], [i, i], color='#2878b5', linewidth=2)
            ax.plot(100*row['difference'], i, 'o', color='#222222')
    ax.axvline(0, linestyle='--', color='#777777')
    ax.set_yticks(range(12), labels)
    ax.set_xlabel('Paired difference (percentage points)')
    ax.set_title('Five seeds;10,000 paired seed/image bootstrap; Bonferroni family=12')
    for extension in ('png', 'pdf'):
        figure.savefig(folder / ('paired-intervals.'+extension), dpi=170)
    plt.close(figure)


def deliver(folder, token):
    from publish_artifacts import publish
    from run_9b_formal_when_ready import verify_receipt
    from resilient_9b_delivery import network_retry
    receipt = network_retry(lambda: publish(folder, TAG, token=token), 'v4-publish:'+folder.name)
    proof = network_retry(lambda: verify_receipt(receipt, token), 'v4-server:'+folder.name)
    write_json(folder / 'delivery-proof.json', {'status': 'passed', 'receipt': receipt, 'fresh_server_check': proof})
    return receipt


def child(arm, seed, pilot):
    command = [str(ENV), '-u', str(Path(__file__)), 'eval', '--arm', arm, '--seed', str(seed)]
    if pilot:
        command.append('--pilot')
    with subprocess.Popen(command, cwd=ROOT) as process:
        while True:
            try:
                code = process.wait(timeout=5)
                assert code == 0, ('Evaluation child failed', code, arm, seed)
                return
            except subprocess.TimeoutExpired:
                path = OUT / ('calibration-wall-budget.json' if pilot else 'formal-wall-budget.json')
                if path.exists() and time.time() >= datetime.fromisoformat(read_json(path)['end_utc']).timestamp():
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise RuntimeError('Registered absolute GPU budget exhausted; never reset')


def run(token):
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'manager.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'complete.json').exists():
            return
        plan = prepare()
        registration = OUT / 'registration'
        registration.mkdir(exist_ok=True)
        for path in [OUT/'protocol.json', OUT/'formal-manifest.jsonl', OUT/'calibration-selection.json'] + [ROOT/'tools'/name for name in SOURCES]:
            shutil.copyfile(path, registration / path.name)
        if not (registration / 'delivery-proof.json').exists():
            assert not list((OUT / 'formal').glob('*/eval.jsonl')), 'No predictions before registration'
            deliver(registration, token)
        if not (OUT / 'calibration-acceptance.json').exists():
            reports = {}
            for arm, seed in models():
                folder = OUT / 'formal' / f'{arm}-seed{seed}'
                if not (folder / 'calibration-scores.json').exists():
                    child(arm, seed, True)
                marker = read_json(folder / 'calibration-scores.json')
                expected = 100 if arm == 'N' else 20
                assert marker['rows'] == expected
                assert marker['ledger_sha256'] == sha(folder / 'eval.jsonl')
                proof = OUT / 'calibration-proof' / folder.name
                proof.mkdir(parents=True, exist_ok=True)
                for name in ('calibration-scores.json', 'eval.jsonl', 'eval.meta.json'):
                    shutil.copyfile(folder / name, proof / name)
                reports[folder.name] = marker
                assert marker['length_capped_outputs'] == 0, ('Pilot generation truncated', folder.name)
                assert marker['parse_failures'] <= (5 if arm == 'N' else 1), ('Pilot schema gate failed', folder.name)
                if arm == 'N':
                    assert marker['accuracy'] >= .80, 'N v4 fixed80% gate failed; do not tune or retry'
            assert sum(r['rows'] for r in reports.values()) == 400
            acceptance = {'status': 'passed', 'models': reports, 'predictions': 400,
                          'protocol_sha256': sha(OUT / 'protocol.json'), 'timestamp_utc': now(),
                          'pilot_included_in270400': True, 'training_unchanged': True}
            write_json(OUT / 'calibration-acceptance.json', acceptance)
            write_json(OUT / 'calibration-proof/acceptance.json', acceptance)
            deliver(OUT / 'calibration-proof', token)
        for arm, seed in models():
            folder = OUT / 'formal' / f'{arm}-seed{seed}'
            if not (folder / 'scores.json').exists():
                child(arm, seed, False)
            verify_evaluation(arm, seed)
            if not (folder / 'delivery-proof.json').exists():
                deliver(folder, token)
            write_json(OUT / 'queue-status.json', {'phase': 'formal_model_delivered', 'model': folder.name, 'timestamp_utc': now()})
        import analyze_9b_formal as registered
        registered.OUT = OUT
        registered.verify_formal = verify_formal
        result = registered.analyze(replicates=10000)
        result['measurement_protocol_sha256'] = sha(OUT / 'protocol.json')
        write_json(OUT / 'formal-report/evaluation-results.json', result)
        create_figures(result)
        with (OUT / 'formal-report/evaluation-report.md').open('a', encoding='utf-8') as stream:
            stream.write('\n原生千分比前端固定转换至处理器像素及原图；400校准预测复用于270,400总量。旧v3失败保留。本测量包含定位与指令/格式遵循，不作纯容量或原ARP因果归因。\n')
            stream.write('\n![四轴全部16模型](four-axis-scores.png)\n\n![12项校正配对区间](paired-intervals.png)\n')
        deliver(OUT / 'formal-report', token)
        receipt = deliver(OUT, token)
        completion = {'status': 'all16_v4_evaluations_verified_and_delivered', 'models': 16, 'rows': 270400,
                      'protocol_sha256': sha(OUT / 'protocol.json'), 'timestamp_utc': now(), 'receipt': receipt}
        terminal = OUT / 'terminal-proof'
        write_json(terminal / 'complete.json', completion)
        deliver(terminal, token)
        write_json(OUT / 'complete.json', completion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'start', 'run', 'eval'])
    parser.add_argument('--arm', default='N')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'eval':
        evaluate(args.arm, args.seed, args.pilot)
    else:
        token = os.environ.get('GH_TOKEN') or sys.stdin.readline().strip()
        assert token
        if args.action == 'start':
            command = ('cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u '
                       'tools/run_9b_native_frontend_v4.py run >> runs/9b-native-frontend-v4-console.log 2>&1')
            subprocess.run(['screen', '-dmS', '9b-native-frontend-v4', 'bash', '-c', command],
                           env={**os.environ, 'GH_TOKEN': token}, check=True)
            print('Independent v4 queue started;400-prediction fixed pilot precedes any formal continuation')
        else:
            try:
                run(token)
            except BlockingIOError:
                print('Another v4 manager is already active; no restart or stop marker written', flush=True)
            except Exception as error:
                failure = {'type': type(error).__name__, 'message': str(error), 'timestamp_utc': now(),
                           'no_budget_or_prompt_retry': True, 'training_unchanged': True}
                write_json(OUT / 'stopped.json', failure)
                write_json(OUT / 'calibration-proof/stopped.json', failure)
                try:
                    deliver(OUT / 'calibration-proof', token)
                except Exception as delivery_error:
                    write_json(OUT / 'failure-delivery-error.json', {'type': type(delivery_error).__name__, 'timestamp_utc': now()})
                raise
