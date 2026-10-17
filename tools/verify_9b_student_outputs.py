"""Independent student9B step/checkpoint/coverage/score verification."""
from collections import Counter
import math

from experiment_common import ROOT, AXES, read_json, read_rows, sha, write_json, units, score_generation
from route_b_9b_loop import BASE, OUT, SEEDS, arm_rows


def verify_training(arm, seed, phase='formal', allow_offloaded=False):
    folder = OUT / phase / f'{arm}-seed{seed}'
    marker, meta = read_json(folder / 'complete.json'), read_json(folder / 'train.meta.json')
    protocol = read_json(OUT / 'protocol.json')
    assert sha(ROOT / 'tools/route_b_9b_loop.py') == protocol['source_sha256']
    assert sha(BASE / 'eligible-pool.jsonl') == protocol['eligible_pool_sha256']
    assert sha(BASE / 'teacher.jsonl') == protocol['teacher_ledger_sha256']
    assert sha(ROOT / 'tools/preflight_9b_full.py') == protocol['input_helper_sha256']
    assert meta['protocol_sha256'] == sha(OUT / 'protocol.json')
    target = 1000 if phase == 'formal' else 100
    source = BASE / 'eligible-pool.jsonl' if phase == 'formal' else OUT / 'minimal-pool.jsonl'
    planned = arm_rows(read_rows(source), phase, arm, seed)
    assert meta['sampling_ids'] == [r['item_id'] for r in planned]
    ledger = read_rows(folder / 'loss.jsonl')
    assert len(ledger) == marker['actual_updates'] == target
    assert [r['step'] for r in ledger] == list(range(1, target + 1))
    assert marker['loss_sha256'] == sha(folder / 'loss.jsonl')
    assert marker['adapter_sha256'] == sha(folder / 'adapter/adapter_model.safetensors')
    for i, row in enumerate(ledger):
        planned_row = planned[i % len(planned)]
        assert (row['item_id'], row['axis']) == (planned_row['item_id'], planned_row['axis'])
        assert row['actual_optimizer_update'] and row['grad_norm'] > 0
        assert all(math.isfinite(row[k]) for k in ('loss', 'ce_gt', 'ce_teacher', 'grad_norm', 'seconds'))
        assert abs(row['loss'] - .5 * row['ce_gt'] - .5 * row['ce_teacher']) < 1e-5
    counts = dict(Counter(r['axis'] for r in ledger))
    if arm == 'U':
        assert counts == {a: target // 4 for a in AXES}
    checkpoint = folder / 'resume.pt'
    proof_path = folder / 'training-verification.json'
    if checkpoint.exists():
        import torch
        from safetensors.torch import load_file
        assert sha(checkpoint) == marker['resume_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        exported = load_file(str(folder / 'adapter/adapter_model.safetensors'))
        assert state['step'] == target and state['meta'] == meta
        assert {int(v['step']) for v in state['optimizer']['state'].values()} == {target}
        assert state['rng_cpu'].numel() > 0 and state['rng_cuda'] and state['rng_python']
        assert set(exported) == set(state['adapter'])
        assert all(torch.equal(v, state['adapter'][k]) and torch.isfinite(v).all() for k, v in exported.items())
        assert all(torch.count_nonzero(v) > 0 for k, v in exported.items() if 'lora_B' in k)
        proof = {'status': 'passed_local_recovery_before_offload', 'arm': arm, 'seed': seed, 'phase': phase,
                 'updates': target, 'axis_update_counts': counts, 'resume_sha256': marker['resume_sha256'],
                 'loss_sha256': marker['loss_sha256'], 'adapter_sha256': marker['adapter_sha256'],
                 'optimizer_step_values': [target], 'recovery_adapter_export_exact': True,
                 'protocol_sha256': sha(OUT / 'protocol.json')}
        write_json(proof_path, proof)
        return proof
    assert allow_offloaded, 'Local recovery state required for first independent verification'
    proof = read_json(proof_path)
    offload = read_json(folder / 'resume-offloaded.json')
    assert proof['status'] == 'passed_local_recovery_before_offload'
    assert proof['updates'] == target and proof['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert all(proof[k] == marker[k] for k in ('resume_sha256', 'loss_sha256', 'adapter_sha256'))
    assert offload['resume_sha256'] == marker['resume_sha256']
    receipt = read_json(folder / 'training-delivery.json')
    assert offload['receipt_sha256'] == sha(folder / 'training-delivery.json')
    inventory = read_json(folder / 'training-inventory.json')
    inventory_asset = next(a for a in receipt['assets'] if a['name'].endswith('-inventory.json'))
    assert inventory_asset['sha256'] == sha(folder / 'training-inventory.json')
    entry = next(r for r in inventory['files'] if r['path'] == str(checkpoint.relative_to(ROOT)))
    assert entry['sha256'] == marker['resume_sha256']
    return {**proof, 'current_recovery_location': 'Verified GitHub Release; local completed optimizer offloaded'}


def verify_evaluation(arm, seed, phase='formal'):
    folder = OUT / phase / f'{arm}-seed{seed}'
    path = ROOT / 'data/manifest.jsonl' if phase == 'formal' else ROOT / 'runs/protocol-v2/calibration/precision-bridge-manifest.jsonl'
    manifest = read_rows(path)
    lookup = {r['item_id']: r for r in manifest}
    rows = read_rows(folder / 'eval.jsonl')
    marker = read_json(folder / 'scores.json')
    assert len(rows) == len({r['item_id'] for r in rows}) == len(manifest) == marker['rows']
    assert {r['item_id'] for r in rows} == set(lookup)
    assert sha(folder / 'eval.jsonl') == marker['ledger_sha256']
    assert read_json(folder / 'eval.meta.json')['manifest_sha256'] == sha(path)
    for row in rows:
        actual = score_generation(row['raw_generation'], lookup[row['item_id']])
        assert all(actual[k] == row[k] for k in ('correct_bit', 'parsed_answer', 'parse_fail_flag', 'iou'))
    originals = units(manifest, rows)
    computed = {a: sum(r['correct_bit'] for r in originals if r['axis'] == a) /
                   sum(r['axis'] == a for r in originals) for a in AXES}
    assert computed == marker['axis_scores'] and len(originals) == marker['original_units']
    assert marker['optimizer_loaded'] is False and marker['gradients_enabled'] is False
    if arm == 'N':
        assert not (folder / 'resume.pt').exists() and not (folder / 'adapter').exists()
    else:
        assert marker['adapter_reloaded_for_inference']
    return marker, {(r['axis'], r['dataset'], r['unit']): r for r in originals}


def verify_formal(return_evaluations=False):
    training = [verify_training(a, s, allow_offloaded=True) for s in SEEDS for a in ('A1', 'A2', 'U')]
    models = [('N', 42)] + [(a, s) for s in SEEDS for a in ('A1', 'A2', 'U')]
    evaluations = {(a, s): verify_evaluation(a, s) for a, s in models}
    result = {'status': 'passed', 'training_runs': len(training), 'updates_per_run': 1000,
              'evaluation_models': len(models), 'prompts_per_model': 16900,
              'recovery_policy': 'Every optimizer verified locally before archived offload; fresh server audit separate'}
    write_json(OUT / 'formal-independent-verification.json', result)
    return (result, evaluations) if return_evaluations else result
