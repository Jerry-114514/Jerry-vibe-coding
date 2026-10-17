"""Independent9B+122B training/evaluation engine; original7B code is unchanged.

GPU actions require the admitted100-update9B preflight. Minimum loop uses500
eligible prompts, seed42, three100-update arms and a genuinely untrained N.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT, AXES, read_json, read_rows, write_json, write_rows, sha, units, score_generation
import preflight_9b_full as shared

BASE = shared.BASE
OUT = BASE / 'student-experiment'
SEEDS = [42, 137, 256, 512, 1024]
MINIMAL_MANIFEST = ROOT / 'runs/protocol-v2/calibration/precision-bridge-manifest.jsonl'


def select_minimal(pool):
    rng = random.Random(42)
    result = []
    for axis in AXES:
        candidates = sorted((r for r in pool if r['axis'] == axis), key=lambda r: r['item_id'])
        result.extend(rng.sample(candidates, 125))
    return result


def arm_rows(pool, phase, arm, seed):
    assert arm in ('A1', 'A2', 'U') and seed in SEEDS
    rng = random.Random(seed)
    by_axis = {a: sorted((r for r in pool if r['axis'] == a), key=lambda r: r['item_id']) for a in AXES}
    for rows in by_axis.values():
        rng.shuffle(rows)
    if phase == 'formal':
        n = 150 if arm == 'U' else 600
        assert all(len(rows) >= n for rows in by_axis.values())
        by_axis = {a: rows[:n] for a, rows in by_axis.items()}
    if arm != 'U':
        return by_axis['mmstar' if arm == 'A1' else 'consistency']
    return [by_axis[a][i] for i in range(len(by_axis['pope'])) for a in AXES]


def freeze(path, value):
    if path.exists():
        assert read_json(path) == value, 'Frozen9B inputs changed: ' + str(path)
    else:
        write_json(path, value)


def check_inputs():
    marker = read_json(BASE / 'teacher-eligible-complete.json')
    assert marker['status'] == 'passed_eligible_pool_only'
    assert read_json(BASE / 'teacher-delivery-verification.json')['status'] == 'passed_eligible_teacher_and_release'
    assert sha(BASE / 'eligible-pool.jsonl') == marker['eligible_pool_sha256']
    assert sha(BASE / 'teacher.jsonl') == marker['teacher_ledger_sha256']
    preflight_protocol = read_json(shared.OUT / 'protocol.json')
    assert preflight_protocol['source_sha256'] == sha(Path(shared.__file__))
    return marker


def prepare():
    marker = check_inputs()
    evaluation_inputs = BASE / 'evaluation-input-preparation/complete.json'
    input_proof = read_json(evaluation_inputs)
    assert input_proof['status'] == 'passed_cpu_minimal_and_full_evaluation_inputs'
    assert input_proof['input_engine_sha256'] == sha(Path(__file__))
    pool = read_rows(BASE / 'eligible-pool.jsonl')
    minimal = select_minimal(pool)
    assert Counter(r['axis'] for r in minimal) == Counter({a: 125 for a in AXES})
    config = read_json(ROOT / 'configs/route_b_9b_122b.plan.json')
    design = {'schema': 'independent.9b-122b.student-loop.v1', 'not_original_arp': True,
              'purpose': 'Student/teacher/configuration extension, not an isolated capacity effect',
              'student': config['student'], 'revision': config['revision'], 'teacher': config['teacher'],
              'source_sha256': sha(Path(__file__)), 'input_helper_sha256': sha(Path(shared.__file__)),
              'minimal_controller_sha256': sha(ROOT / 'tools/run_9b_minimal_when_ready.py'),
              'evaluation_input_preflight_sha256': sha(evaluation_inputs),
              'plan_sha256': sha(ROOT / 'configs/route_b_9b_122b.plan.json'),
              'eligible_pool_sha256': marker['eligible_pool_sha256'],
              'teacher_ledger_sha256': marker['teacher_ledger_sha256'],
              'training': config['training'], 'axes': config['axes'], 'seeds': SEEDS,
              'minimal_rows_per_axis': 125, 'minimal_updates': 100, 'formal_updates': 1000,
              'minimal_manifest_sha256': sha(MINIMAL_MANIFEST),
              'formal_manifest_sha256': sha(ROOT / 'data/manifest.jsonl'),
              'endpoint': 'Last fixed-budget checkpoint; never select by evaluation score',
              'no_automatic_budget_extensions': True, 'registered_before_9b_student_training': True,
              'minimal_item_ids': [r['item_id'] for r in minimal]}
    OUT.mkdir(parents=True, exist_ok=True)
    freeze(OUT / 'protocol.json', design)
    if not (OUT / 'minimal-pool.jsonl').exists():
        write_rows(OUT / 'minimal-pool.jsonl', minimal)
    assert read_rows(OUT / 'minimal-pool.jsonl') == minimal
    write_json(OUT / 'minimal-sample-counts.json', {'rows_by_axis': dict(Counter(r['axis'] for r in minimal)),
               'original_units_by_axis': {a: len({r['bootstrap_unit'] for r in minimal if r['axis'] == a}) for a in AXES},
               'note': 'Prompt counts are not independent-original-unit counts'})
    return design


def admission():
    design = prepare()
    result = read_json(shared.OUT / 'complete.json')
    assert result['status'] == 'passed' and result['real_updates'] == 100
    assert result['separate_stress_updates'] == 1 and result['reserved_peak_bytes'] <= 28_000_000_000
    assert result['protocol_sha256'] == sha(shared.OUT / 'protocol.json')
    assert shared.seven_finished(), '7B full evaluation must finish first'
    return design


def gpu_inputs(batch):
    import torch
    return {k: v.to('cuda', dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda') for k, v in batch.items()}


def model_backend(adapter=None):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(shared.MODEL), local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(shared.MODEL), local_files_only=True,
                dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    return processor, model


def public_prompt_inputs(processor, row):
    """Same9B pixel/token recipe as the training-only preflight helper."""
    from PIL import Image
    path = (ROOT / row['image_path']).resolve()
    assert path.is_relative_to((ROOT / 'data').resolve())
    with Image.open(path) as source:
        image = source.convert('RGB')
    if min(image.size) < 32:
        scale = 32 / min(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': row['question']}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    batch = dict(processor(text=[text], images=[image], return_tensors='pt',
                 images_kwargs={'min_pixels': 256 * 32 * 32, 'max_pixels': 1280 * 32 * 32}))
    assert int(batch['image_grid_thw'].prod(dim=1).sum().item() // 4) <= 1280
    assert batch['input_ids'].shape[1] <= 2048, 'Input cap; never silently truncate'
    return batch


def train(phase, arm, seed, stop_at=None):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from train_distill import response_forward
    from gpu_lock import gpu_lock
    design = admission()
    assert arm != 'N' and (phase == 'formal' or seed == 42)
    if phase == 'formal':
        assert read_json(OUT / 'minimal-acceptance.json')['status'] == 'passed', 'Allfive minimum-loop checks must pass first'
    target = 100 if phase == 'minimal' else 1000
    folder = OUT / phase / f'{arm}-seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        marker = read_json(folder / 'complete.json')
        assert marker['actual_updates'] == target and marker['adapter_sha256'] == sha(folder / 'adapter/adapter_model.safetensors')
        return
    source = OUT / 'minimal-pool.jsonl' if phase == 'minimal' else BASE / 'eligible-pool.jsonl'
    rows = arm_rows(read_rows(source), phase, arm, seed)
    teacher = {r['item_id']: r for r in read_rows(BASE / 'teacher.jsonl')}
    meta = {'protocol_sha256': sha(OUT / 'protocol.json'), 'phase': phase, 'arm': arm, 'seed': seed,
            'target_steps': target, 'sampling_ids': [r['item_id'] for r in rows]}
    freeze(folder / 'train.meta.json', meta)
    with gpu_lock():
        assert shutil.disk_usage(OUT).free >= 5 * 1024 ** 3
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.reset_peak_memory_stats()
        processor, base = model_backend()
        base.config.use_cache = False
        config = design['training']
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                    target_modules=config['lora_targets'], bias='none', task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        assert sum(p.numel() for p in params) == config['expected_trainable_parameters']
        assert all('lora_' in n and 'visual' not in n for n, p in model.named_parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0)
        checkpoint, ledger = folder / 'resume.pt', folder / 'loss.jsonl'
        start = 0
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert state['meta'] == meta
            set_peft_model_state_dict(model, state['adapter'])
            optimizer.load_state_dict(state['optimizer'])
            start = state['step']
            assert {int(s['step']) for s in optimizer.state.values()} == {start}
            torch.set_rng_state(state['rng_cpu'])
            torch.cuda.set_rng_state_all(state['rng_cuda'])
            random.setstate(state['rng_python'])
            write_json(folder / 'resume-verification.json', {'restored_step': start, 'optimizer_restored': True,
                       'rng_restored': True, 'checkpoint_sha256': sha(checkpoint), 'new_pid': os.getpid()})
        previous = read_rows(ledger) if ledger.exists() else []
        committed = [r for r in previous if r['step'] <= start]
        assert [r['step'] for r in committed] == list(range(1, start + 1))
        if len(previous) != len(committed):
            write_rows(folder / f'discarded-uncommitted-{time.time_ns()}.jsonl', previous[len(committed):])
        write_rows(ledger, committed)
        for step in range(start + 1, min(target, stop_at or target) + 1):
            assert shutil.disk_usage(OUT).free >= 5 * 1024 ** 3
            row = rows[(step - 1) % len(rows)]
            cached = teacher[row['item_id']]
            assert any(x['sha256'] == sha(ROOT / row['image_path']) for x in cached['image_hashes'])
            started = time.monotonic()
            prompt = shared.prompt_inputs(processor, row)
            length = prompt['input_ids'].shape[1]
            gt_ids = shared.targets(processor, shared.answer_text(row))
            teacher_ids = shared.targets(processor, shared.answer_text(row, cached))
            assert length + max(len(gt_ids), len(teacher_ids)) <= 2048
            gt = gpu_inputs(shared.suffix_inputs(prompt, gt_ids))
            teacher_batch = gpu_inputs(shared.suffix_inputs(prompt, teacher_ids))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, labels = response_forward(model, gt, length, base_model=base)
                ce_gt = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
                logits_t, labels_t = response_forward(model, teacher_batch, length, base_model=base)
                ce_teacher = F.cross_entropy(logits_t.float().reshape(-1, logits_t.shape[-1]), labels_t.reshape(-1))
                loss = .5 * ce_gt + .5 * ce_teacher
            assert torch.isfinite(loss)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1)
            assert torch.isfinite(grad) and grad > 0
            optimizer.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved()
            assert peak <= 30_000_000_000, 'Registered30GB memory stop'
            record = {'step': step, 'item_id': row['item_id'], 'axis': row['axis'], 'loss': float(loss.detach()),
                      'ce_gt': float(ce_gt.detach()), 'ce_teacher': float(ce_teacher.detach()), 'grad_norm': float(grad),
                      'seconds': time.monotonic() - started, 'reserved_peak_bytes': peak,
                      'allocated_peak_bytes': torch.cuda.max_memory_allocated(), 'actual_optimizer_update': True}
            with ledger.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            write_json(OUT / 'status.json', {'phase': phase, 'arm': arm, 'seed': seed, 'step': step, 'pid': os.getpid()})
            if step % 50 == 0 or step == target or step == stop_at:
                state = {'meta': meta, 'step': step,
                         'adapter': {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                         'optimizer': optimizer.state_dict(), 'rng_cpu': torch.get_rng_state(),
                         'rng_cuda': torch.cuda.get_rng_state_all(), 'rng_python': random.getstate()}
                temporary = checkpoint.with_suffix('.tmp')
                torch.save(state, temporary)
                temporary.replace(checkpoint)
                print(phase, arm, seed, step, record['loss'], flush=True)
            del logits, logits_t, loss, ce_gt, ce_teacher, gt, teacher_batch, prompt
        if min(target, stop_at or target) < target:
            return
        model.save_pretrained(folder / 'adapter', safe_serialization=True)
        from safetensors.torch import load_file
        actual = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
        exported = load_file(str(folder / 'adapter/adapter_model.safetensors'))
        assert set(actual) == set(exported) and all(torch.equal(v, exported[k]) for k, v in actual.items())
        assert all(torch.isfinite(v).all() for v in exported.values())
        assert all(torch.count_nonzero(v) > 0 for k, v in actual.items() if 'lora_B' in k)
        assert {int(s['step']) for s in optimizer.state.values()} == {target}
        write_json(folder / 'complete.json', {'actual_updates': target, 'seed': seed, 'arm': arm,
                   'adapter_export_exact': True, 'optimizer_step_values': [target], 'loss_sha256': sha(ledger),
                   'resume_sha256': sha(checkpoint), 'reserved_peak_bytes': torch.cuda.max_memory_reserved(),
                   'adapter_sha256': sha(folder / 'adapter/adapter_model.safetensors')})


def evaluate(phase, arm, seed):
    import torch
    from gpu_lock import gpu_lock
    from model_backends import finite_logits_processors
    admission()
    if phase == 'formal':
        assert read_json(OUT / 'minimal-acceptance.json')['status'] == 'passed'
        assert all(read_json(OUT / 'formal' / f'{a}-seed{s}' / 'complete.json')['actual_updates'] == 1000
                   for s in SEEDS for a in ('A1', 'A2', 'U')), 'Complete all15 fixed-budget training runs before formal evaluation'
    folder = OUT / phase / f'{arm}-seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = MINIMAL_MANIFEST if phase == 'minimal' else ROOT / 'data/manifest.jsonl'
    manifest = read_rows(manifest_path)
    assert len(manifest) == (800 if phase == 'minimal' else 16900)
    freeze(folder / 'eval.meta.json', {'manifest_sha256': sha(manifest_path), 'protocol_sha256': sha(OUT / 'protocol.json'),
                                     'arm': arm, 'seed': seed, 'thinking': False, 'max_new_tokens': 128})
    ledger = folder / 'eval.jsonl'
    previous = read_rows(ledger) if ledger.exists() else []
    done = {r['item_id'] for r in previous}
    assert len(done) == len(previous) and done <= {r['item_id'] for r in manifest}
    if (folder / 'scores.json').exists():
        assert read_json(folder / 'scores.json')['ledger_sha256'] == sha(ledger)
        return
    with gpu_lock():
        torch.cuda.reset_peak_memory_stats()
        processor, model = model_backend(None if arm == 'N' else folder / 'adapter')
        model.eval().requires_grad_(False)
        assert not any(p.requires_grad for p in model.parameters())
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for row in manifest:
            if row['item_id'] in done:
                continue
            assert shutil.disk_usage(OUT).free >= 5 * 1024 ** 3
            started = time.monotonic()
            with torch.inference_mode():
                batch = gpu_inputs(public_prompt_inputs(processor, row))
                generated = model.generate(**batch, max_new_tokens=128, do_sample=False,
                            logits_processor=finite_logits_processors())
                ids = generated[0, batch['input_ids'].shape[1]:]
                raw = processor.decode(ids, skip_special_tokens=True).strip()
            torch.cuda.synchronize()
            record = {'item_id': row['item_id'], 'axis': row['axis'], 'seconds': time.monotonic() - started,
                      'finish_reason': 'eos' if ids.numel() and int(ids[-1]) in eos else 'length',
                      'generation_tokens': ids.numel(), **score_generation(raw, row)}
            with ledger.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            done.add(row['item_id'])
            assert torch.cuda.max_memory_reserved() <= 30_000_000_000
            if len(done) % 100 == 0:
                write_json(OUT / 'status.json', {'phase': 'eval_' + phase, 'arm': arm, 'seed': seed,
                           'rows': len(done), 'pid': os.getpid()})
                print('9B eval', phase, arm, len(done), '/', len(manifest), flush=True)
        records = read_rows(ledger)
        originals = units(manifest, records)
        scores = {a: sum(r['correct_bit'] for r in originals if r['axis'] == a) /
                        sum(r['axis'] == a for r in originals) for a in AXES}
        write_json(folder / 'scores.json', {'rows': len(records), 'original_units': len(originals),
                   'ledger_sha256': sha(ledger), 'axis_scores': scores, 'adapter_reloaded_for_inference': arm != 'N',
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'optimizer_loaded': False, 'gradients_enabled': False,
                   'total_generation_seconds': sum(r['seconds'] for r in records),
                   'length_capped_outputs': sum(r['finish_reason'] == 'length' for r in records)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train', 'eval'])
    parser.add_argument('--phase', choices=['minimal', 'formal'], default='minimal')
    parser.add_argument('--arm', choices=['A1', 'A2', 'U', 'N'], default='N')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare()), flush=True)
    elif args.action == 'train':
        train(args.phase, args.arm, args.seed, args.stop_at)
    else:
        evaluate(args.phase, args.arm, args.seed)
