"""Independent route B. No ARP gate changes; BF16 dual-CE, local 7B teacher.

GPU work is serialized. Teacher provenance checks can run alongside CPU tooling.
The original v1 cache is immutable; this run records EOS and capped input sizes.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from experiment_common import ROOT, AXES, read_json, read_rows, sha, write_json, write_rows, score_generation, units

OUT = ROOT / 'runs/route-b-staged-7b-20261016'
CACHE = ROOT / 'runs/route-b-7b-20261016/teacher-cache-bf16-eos'
MODULES = r'(?:model\.layers|model\.language_model\.layers)\.\d+\.(?:self_attn|mlp)\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
SEEDS = [42, 137, 256, 512, 1024]


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen input changed: ' + str(path))
    else:
        write_json(path, value)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = read_rows(ROOT / 'data/training/pool.jsonl')
    assert len(pool) == 3545 and len({r['item_id'] for r in pool}) == 3545
    rng = random.Random(42)
    sample = []
    for axis in AXES:
        candidates = sorted((r for r in pool if r['axis'] == axis), key=lambda r: r['item_id'])
        sample.extend(rng.sample(candidates, 125))
    spec = {'schema': 'route-b.independent.staged.2', 'purpose': 'Fixed-axis self-distillation intervention; engineering loop first',
            'not_original_arp': True, 'stage_c_bypass': 'User-authorized independent experiment only; C remains pending',
            'teacher': 'Qwen2.5-VL-7B-Instruct', 'student': 'Qwen2.5-VL-7B-Instruct',
            'revision': read_json(ROOT / 'configs/models.lock.json')['models']['qwen7b']['revision'],
            'axes': {'A1': ['mmstar'], 'A2': ['consistency'], 'U': AXES, 'N': []},
            'minimal': {'rows': 500, 'rows_per_axis': 125, 'seeds': [42], 'updates_per_train_arm': 100,
                        'resume_test': 'A1 checkpoint after50; new process resumes50 through100'},
            'formal': {'seeds': SEEDS, 'updates_per_train_arm': 1000, 'runs': 15,
                       'per_arm_pool_rows': 600, 'U_rows_per_axis': 150,
                       'sampling': 'Fixed 600-row budgets with deterministic replacement for A2; cyclic exposure; U exactly250 updates/axis'},
            'training': {'dtype': 'bf16', 'lora_r': 16, 'lora_alpha': 32, 'lora_dropout': 0,
                         'lora_scope_regex': MODULES, 'lr': 1e-4, 'weight_decay': 0,
                         'micro_batch': 1, 'gradient_accumulation': 1, 'gradient_checkpointing': True,
                         'max_grad_norm': 1.0, 'loss': {'ground_truth_ce': 0.5, 'teacher_response_ce': 0.5},
                         'top50_kl_weight': 0, 'max_total_tokens': 2048, 'max_visual_tokens': 1280,
                         'max_response_tokens': 128, 'checkpoint_every_updates': 50},
            'teacher_cache': {'top_k': 50, 'all_pool_rows': 3545, 'generation': 'greedy max128; require raw EOS',
                              'precision': 'bf16', 'image_protocol': 'explicit size1280, original-pixel prompt unchanged',
                              'legacy_cache': 'runs/formal-20261014/teacher-cache; audited, never overwritten',
                              'why_new_version': 'Legacy lacks raw EOS; BF16 and explicit resizing change logits; no silent reuse'},
            'eval': {'minimal_manifest': 'runs/protocol-v2/calibration/precision-bridge-manifest.jsonl',
                     'hardware': 'RTX5090 for training and evaluation; original4090 released', 'formal_manifest': 'data/manifest.jsonl', 'formal_rows_expected': 16900,
                     'interpretation': 'Existing development benchmark, not an untouched confirmation set',
                     'primary': 'Four-axis paired deltas vs N; A1/A2 vs U; equal-weight macro secondary',
                     'uncertainty': 'Report all five seeds; original-unit clustered bootstrap; exploratory, no ARP causal claim'},
            'stops': ['nonfinite loss/gradient', 'zero gradient', 'OOM or reserved>30GB', 'hash mismatch',
                      'teacher generation without EOS', 'input cap exceeded; never silently truncate',
                      'cache or ledger duplicates', 'free disk<5GiB', 'wall clock>8h for training pipeline'],
            'budget': {'max_training_updates': 15300, 'teacher_api_requests': 0,
                       'training_gpu_hours_cap': 8, 'hourly_CNY': 2.70,
                       'training_gpu_CNY_cap': 21.60, 'eval_gpu_hours_cap': 30, 'eval_gpu_CNY_cap': 81.0, 'hourly_eval_CNY': 2.70},
            'pool_sha256': sha(ROOT / 'data/training/pool.jsonl'),
            'source_sha256': sha(Path(__file__)), 'minimal_item_ids': [r['item_id'] for r in sample]}
    freeze(OUT / 'protocol.json', spec)
    if not (OUT / 'minimal-pool.jsonl').exists():
        write_rows(OUT / 'minimal-pool.jsonl', sample)
    assert [r['item_id'] for r in read_rows(OUT / 'minimal-pool.jsonl')] == spec['minimal_item_ids']
    # Ensure seed-level unit independence is visible, not hidden behind row counts.
    write_json(OUT / 'sample-counts.json', {'rows_by_axis': dict(Counter(r['axis'] for r in sample)),
               'original_units_by_axis': {a: len({r['bootstrap_unit'] for r in sample if r['axis'] == a}) for a in AXES}})
    return spec


def inputs(processor, row, response=None):
    import torch
    from PIL import Image
    image = (ROOT / row['image_path']).resolve()
    if not image.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('Image outside frozen project data')
    with Image.open(image) as source:
        rgb = source.convert('RGB')
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': row['question']}]}]
    if response is not None:
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': response}]})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=response is None, add_vision_id=True)
    x = processor(text=[text], images=[rgb], return_tensors='pt', images_kwargs={
        'size': {'shortest_edge': 256*28*28, 'longest_edge': 1280*28*28},
        'min_pixels': 256*28*28, 'max_pixels': 1280*28*28})
    if int(x['image_grid_thw'].prod(dim=1).sum().item() // 4) > 1280:
        raise ValueError('Visual token budget exceeded')
    if x['input_ids'].shape[1] > 2048:
        raise ValueError('Sequence too long; no truncation: ' + row['item_id'])
    return {k: v.to('cuda', dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda') for k, v in x.items()}


def backend(adapter=None):
    from protocol_v2_backend import QwenV2Backend
    b = QwenV2Backend('qwen7b')
    if adapter:
        from peft import PeftModel
        b.model = PeftModel.from_pretrained(b.model, str(adapter)).eval()
    return b


def teacher_cache(phase):
    import torch
    from train_distill import cache_name, response_forward
    from model_backends import finite_logits_processors
    from gpu_lock import gpu_lock
    prepare()
    destination = CACHE
    destination.mkdir(parents=True, exist_ok=True)
    rows = read_rows(OUT / 'minimal-pool.jsonl' if phase == 'minimal' else ROOT / 'data/training/pool.jsonl')
    marker = 'minimal.complete.json' if phase == 'minimal' else 'complete.json'
    with gpu_lock():
        b = backend()
        eos = b.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for i, row in enumerate(rows, 1):
            path = destination / cache_name(row['item_id'])
            if path.exists():
                x = torch.load(path, map_location='cpu', weights_only=True)
                assert x['item_id'] == row['item_id'] and x['finish_reason'] == 'eos'
                continue
            prompt = inputs(b.processor, row)
            p = prompt['input_ids'].shape[1]
            with torch.inference_mode():
                generated = b.model.generate(**prompt, max_new_tokens=128, do_sample=False,
                                            logits_processor=finite_logits_processors())
                response_ids = generated[0, p:]
                if not response_ids.numel() or int(response_ids[-1]) not in eos:
                    raise ValueError('Teacher length cap hit without EOS: ' + row['item_id'])
                response = b.processor.decode(response_ids, skip_special_tokens=True).strip()
                if not response:
                    raise ValueError('Empty teacher response')
                full = inputs(b.processor, row, response)
                assert torch.equal(full['input_ids'][:, :p], prompt['input_ids'])
                logits, targets = response_forward(b.model, full, p)
                logits = logits[0].float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Teacher logits nonfinite')
                values, indices = logits.topk(50, dim=-1)
                entry = {'item_id': row['item_id'], 'response': response, 'finish_reason': 'eos',
                         'generation_ids': response_ids.cpu(), 'prompt_length': p,
                         'input_ids': full['input_ids'].cpu(), 'targets': targets.cpu(),
                         'top_indices': indices.cpu().int(), 'top_logits': values.cpu(),
                         'log_normalizer': logits.logsumexp(-1).cpu(), 'image_sha256': sha(ROOT / row['image_path'])}
            temporary = path.with_suffix('.tmp')
            torch.save(entry, temporary)
            temporary.replace(path)
            write_json(OUT / 'status.json', {'phase': 'teacher_cache', 'rows': i, 'total': len(rows), 'pid': os.getpid()})
            if i % 50 == 0:
                print('teacher cache', i, '/', len(rows), flush=True)
        write_json(destination / marker, {'items': len(rows), 'raw_eos_verified': True,
                   'files': {cache_name(r['item_id']): sha(destination / cache_name(r['item_id'])) for r in rows}})


def arm_rows(phase, arm, seed):
    pool = read_rows(OUT / 'minimal-pool.jsonl' if phase == 'minimal' else ROOT / 'data/training/pool.jsonl')
    rng = random.Random(seed)
    by_axis = {a: sorted([r for r in pool if r['axis'] == a], key=lambda r: r['item_id']) for a in AXES}
    for values in by_axis.values():
        rng.shuffle(values)
    if phase == 'formal':
        # Equal prompt-row budget, not a claim of equal independent-unit budget.
        size = 150 if arm == 'U' else 600
        by_axis = {a: values[:size] for a, values in by_axis.items()}
    if arm != 'U':
        return by_axis['mmstar' if arm == 'A1' else 'consistency']
    return [by_axis[a][i % len(by_axis[a])] for i in range(len(by_axis['pope'])) for a in AXES]


def train(phase, arm, seed, stop_at=None):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from train_distill import cache_name, response_forward
    from gpu_lock import gpu_lock
    spec = prepare()
    if arm == 'N':
        raise ValueError('N must never train')
    target = 100 if phase == 'minimal' else 1000
    output = OUT / phase / f'{arm}-seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        return
    cache = CACHE
    inventory = cache / ('minimal.complete.json' if phase == 'minimal' else 'complete.json')
    cache_complete = read_json(inventory)
    assert cache_complete['items'] == (500 if phase == 'minimal' else 3545) and cache_complete['raw_eos_verified']
    rows = arm_rows(phase, arm, seed)
    meta = {'arm': arm, 'seed': seed, 'phase': phase, 'protocol_sha256': sha(OUT / 'protocol.json'),
            'cache_inventory_sha256': sha(inventory), 'target_steps': target,
            'sampling_ids': [r['item_id'] for r in rows]}
    freeze(output / 'train.meta.json', meta)
    with gpu_lock():
        torch.manual_seed(seed)
        random.seed(seed)
        b = backend()
        base = b.model
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                                              target_modules=MODULES, bias='none', task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        assert params and all('lora_' in n and 'visual' not in n for n, p in model.named_parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0)
        resume = output / 'resume.pt'
        start = 0
        if resume.exists():
            saved = torch.load(resume, map_location='cpu', weights_only=True)
            assert saved['meta'] == meta
            set_peft_model_state_dict(model, saved['adapter'])
            optimizer.load_state_dict(saved['optimizer'])
            start = saved['step']
            torch.set_rng_state(saved['rng_cpu'])
            torch.cuda.set_rng_state_all(saved['rng_cuda'])
            write_json(output / 'resume-verification.json', {'restored_step': start, 'optimizer_restored': True,
                       'rng_restored': True, 'checkpoint_sha256': sha(resume), 'new_pid': os.getpid()})
        ledger = output / 'loss.jsonl'
        old = read_rows(ledger) if ledger.exists() else []
        old = [r for r in old if r['step'] <= start]
        assert [r['step'] for r in old] == list(range(1, start + 1))
        write_rows(ledger, old)
        torch.cuda.reset_peak_memory_stats()
        for step in range(start + 1, min(stop_at or target, target) + 1):
            row = rows[(step - 1) % len(rows)]
            cached_path = cache / cache_name(row['item_id'])
            assert sha(cached_path) == cache_complete['files'][cached_path.name]
            cached = torch.load(cached_path, map_location='cpu', weights_only=True)
            assert cached['finish_reason'] == 'eos' and cached['image_sha256'] == sha(ROOT / row['image_path'])
            gt = json.dumps(row['answer_key']) if row['axis'] == 'grounding' else str(row['answer_key'])
            t0 = time.monotonic()
            prompt = inputs(b.processor, row)
            length = prompt['input_ids'].shape[1]
            gt_inputs = inputs(b.processor, row, gt)
            teacher_inputs = inputs(b.processor, row, cached['response'])
            assert torch.equal(teacher_inputs['input_ids'].cpu(), cached['input_ids'])
            for batch in (gt_inputs, teacher_inputs):
                assert torch.equal(batch['input_ids'][:, :length], prompt['input_ids'])
                assert 0 < batch['input_ids'].shape[1] - length <= 128
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, targets = response_forward(model, gt_inputs, length, base_model=base)
                ce_gt = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
                logits_t, targets_t = response_forward(model, teacher_inputs, length, base_model=base)
                ce_teacher = F.cross_entropy(logits_t.float().reshape(-1, logits_t.shape[-1]), targets_t.reshape(-1))
                loss = 0.5 * ce_gt + 0.5 * ce_teacher
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite dual CE')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError('Nonfinite or zero gradient')
            optimizer.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved()
            if peak > 30_000_000_000:
                raise RuntimeError('Registered30GB memory stop')
            record = {'step': step, 'item_id': row['item_id'], 'axis': row['axis'], 'ce_gt': float(ce_gt.detach()),
                      'ce_teacher': float(ce_teacher.detach()), 'loss': float(loss.detach()), 'grad_norm': float(norm),
                      'seconds': time.monotonic() - t0, 'reserved_peak_bytes': peak, 'actual_optimizer_update': True}
            with ledger.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            write_json(OUT / 'status.json', {'phase': phase, 'arm': arm, 'seed': seed, 'step': step,
                       'target_steps': target, 'pid': os.getpid()})
            if step % 50 == 0 or step == target or step == stop_at:
                adapter = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
                state = {'step': step, 'meta': meta, 'adapter': adapter, 'optimizer': optimizer.state_dict(),
                         'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()}
                temporary = resume.with_suffix('.tmp')
                torch.save(state, temporary)
                temporary.replace(resume)
                print(phase, arm, seed, step, 'loss', record['loss'], flush=True)
            del logits, logits_t, loss, ce_gt, ce_teacher, gt_inputs, teacher_inputs, prompt
        if min(stop_at or target, target) < target:
            return
        model.save_pretrained(output / 'adapter', safe_serialization=True)
        b.processor.save_pretrained(output / 'adapter')
        original = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
        assert all(torch.count_nonzero(v) > 0 for k, v in original.items() if 'lora_B' in k)
        from safetensors.torch import load_file
        exported = load_file(str(output / 'adapter/adapter_model.safetensors'))
        assert set(exported) == set(original) and all(torch.equal(original[k], exported[k]) for k in original)
        write_json(output / 'complete.json', {'actual_updates': target, 'seed': seed, 'arm': arm,
                   'adapter_export_exact': True, 'loss_sha256': sha(ledger), 'reserved_peak_bytes': torch.cuda.max_memory_reserved(),
                   'adapter_sha256': sha(output / 'adapter/adapter_model.safetensors')})


def evaluate(phase, arm, seed):
    from gpu_lock import gpu_lock
    spec = prepare()
    output = OUT / phase / f'{arm}-seed{seed}'
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / spec['eval']['minimal_manifest' if phase == 'minimal' else 'formal_manifest']
    manifest = read_rows(manifest_path)
    if phase == 'minimal':
        assert len(manifest) == 800
    else:
        assert len(manifest) == 16900
    freeze(output / 'eval.meta.json', {'manifest_sha256': sha(manifest_path), 'arm': arm, 'seed': seed,
           'protocol_sha256': sha(OUT / 'protocol.json'), 'precision': 'bf16', 'max_visual_tokens': 1280})
    path = output / 'eval.jsonl'
    previous = read_rows(path) if path.exists() else []
    done = {r['item_id'] for r in previous}
    assert len(done) == len(previous) and done <= {r['item_id'] for r in manifest}
    with gpu_lock():
        b = backend(None if arm == 'N' else output / 'adapter')
        import torch
        torch.cuda.reset_peak_memory_stats()
        from model_backends import finite_logits_processors
        for row in manifest:
            if row['item_id'] in done:
                continue
            t0 = time.monotonic()
            with torch.inference_mode():
                x = inputs(b.processor, row)
                y = b.model.generate(**x, max_new_tokens=128, do_sample=False, logits_processor=finite_logits_processors())
                raw = b.processor.decode(y[0, x['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            record = {'item_id': row['item_id'], 'axis': row['axis'], 'seconds': time.monotonic() - t0,
                      **score_generation(raw, row)}
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            done.add(row['item_id'])
            if len(done) % 100 == 0:
                print('eval', phase, arm, len(done), '/', len(manifest), flush=True)
                write_json(OUT / 'status.json', {'phase': 'eval_' + phase, 'arm': arm, 'rows': len(done), 'pid': os.getpid()})
    scores = units(manifest, read_rows(path))
    axes = {axis: sum(r['correct_bit'] for r in scores if r['axis'] == axis) /
            sum(r['axis'] == axis for r in scores) for axis in AXES}
    write_json(output / 'scores.json', {'axis_scores': axes, 'rows': len(manifest), 'ledger_sha256': sha(path),
               'adapter_reloaded_for_inference': arm != 'N', 'original_units': len(scores), 'gpu': torch.cuda.get_device_name(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'total_generation_seconds': sum(r['seconds'] for r in read_rows(path))})


def minimal_acceptance():
    result = {}
    for arm in ('A1', 'A2', 'U', 'N'):
        folder = OUT / 'minimal' / f'{arm}-seed42'
        scores = read_json(folder / 'scores.json')
        assert set(scores['axis_scores']) == set(AXES) and scores['rows'] == 800
        if arm != 'N':
            complete = read_json(folder / 'complete.json')
            losses = read_rows(folder / 'loss.jsonl')
            assert complete['actual_updates'] == len(losses) == 100
            assert [r['step'] for r in losses] == list(range(1, 101))
            assert all(r['grad_norm'] > 0 and all(__import__('math').isfinite(r[k]) for k in ('loss', 'ce_gt', 'ce_teacher', 'grad_norm')) for r in losses)
            assert complete['adapter_export_exact'] and scores['adapter_reloaded_for_inference']
        result[arm] = scores['axis_scores']
    resume = read_json(OUT / 'minimal/A1-seed42/resume-verification.json')
    assert resume['restored_step'] == 50 and resume['optimizer_restored'] and resume['rng_restored']
    cache = read_json(CACHE / 'minimal.complete.json')
    assert cache['items'] == 500 and cache['raw_eos_verified']
    write_json(OUT / 'minimal-acceptance.json', {'status': 'passed', 'checks': 5, 'scores': result,
               'protocol_sha256': sha(OUT / 'protocol.json'), 'timestamp': datetime.now(timezone.utc).isoformat()})


def pipeline():
    import shutil
    prepare()
    started = time.monotonic()
    def child(*args, timeout=None):
        if time.monotonic() - started > 8*3600 or shutil.disk_usage(OUT).free < 5*1024**3:
            raise RuntimeError('Registered training time/disk budget reached')
        subprocess.run([sys.executable, '-u', str(Path(__file__)), *args], check=True, cwd=ROOT, timeout=timeout)
    def backup(folder, teacher=False):
        if os.environ.get('GH_TOKEN'):
            from publish_artifacts import publish
            publish(folder, 'arp-protocol-v2-20261016', token=os.environ['GH_TOKEN'], include_teacher=teacher)
    try:
        # Stage1 has priority: only500 cache rows needed before real training.
        child('cache', '--phase', 'minimal')
        child('eval', '--phase', 'minimal', '--arm', 'N', '--seed', '42')
        child('train', '--phase', 'minimal', '--arm', 'A1', '--seed', '42', '--stop-at', '50')
        child('train', '--phase', 'minimal', '--arm', 'A1', '--seed', '42')
        for arm in ('A2', 'U'):
            child('train', '--phase', 'minimal', '--arm', arm, '--seed', '42')
        for arm in ('A1', 'A2', 'U'):
            child('eval', '--phase', 'minimal', '--arm', arm, '--seed', '42')
        minimal_acceptance()
        subprocess.run([sys.executable, '-u', str(ROOT/'tools/report_route_b_staged.py')], check=True, cwd=ROOT)
        backup(OUT)
        # Stage2: after all five checks, finish full cache and bind immutable inputs.
        child('cache', '--phase', 'formal')
        order = []
        for seed in SEEDS:
            arms = ['A1', 'A2', 'U']
            random.Random(seed).shuffle(arms)
            order.extend({'arm': arm, 'seed': seed, 'steps': 1000} for arm in arms)
        frozen = {'protocol_sha256': sha(OUT/'protocol.json'), 'cache_sha256': sha(CACHE/'complete.json'),
                  'minimal_acceptance_sha256': sha(OUT/'minimal-acceptance.json'), 'run_order': order,
                  'preflight_source_sha256': '781d0f024aa74a0ec02f4628380c5a5d4176d6bac123f84de7a60e78171d060a'}
        freeze(OUT/'phase2-frozen.json', frozen)
        (OUT/'phase2-frozen.sha256').write_text(sha(OUT/'phase2-frozen.json')+'\n')
        # Preserve a full direct-to-GitHub teacher-cache backup before formal training.
        backup(CACHE, teacher=True)
        backup(OUT)
        # Stage3:15 fixed-budget runs; no evaluation-driven budget changes.
        for job in order:
            child('train', '--phase', 'formal', '--arm', job['arm'], '--seed', str(job['seed']))
            backup(OUT/'formal'/f"{job['arm']}-seed{job['seed']}")
        write_json(OUT/'stage3-complete.json', {'runs': 15, 'updates': 15000, 'formal_eval_complete': False})
        # Stage4:5090 evaluation, no optimizer and no gradients. Separate30h budget.
        eval_started = time.monotonic()
        for arm, seed in [('N', 42)]+[(j['arm'], j['seed']) for j in order]:
            remaining = 30*3600-(time.monotonic()-eval_started)
            if remaining <= 0:
                raise RuntimeError('Registered30h evaluation budget reached')
            subprocess.run([sys.executable, '-u', str(Path(__file__)), 'eval', '--phase', 'formal',
                            '--arm', arm, '--seed', str(seed)], check=True, cwd=ROOT, timeout=remaining)
            backup(OUT/'formal'/f'{arm}-seed{seed}')
        subprocess.run([sys.executable, '-u', str(ROOT/'tools/analyze_route_b_staged.py')], check=True, cwd=ROOT)
        write_json(OUT/'status.json', {'phase': 'all_four_stages_complete', 'independent_experiment_only': True})
    except Exception as error:
        write_json(OUT/'failure.json', {'type': type(error).__name__, 'message': str(error),
                   'time': datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        backup(OUT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'cache', 'train', 'eval', 'pipeline', 'start'])
    p.add_argument('--phase', choices=['minimal', 'formal'], default='minimal')
    p.add_argument('--arm', choices=['A1', 'A2', 'U', 'N'], default='A1')
    p.add_argument('--seed', type=int, choices=SEEDS, default=42)
    p.add_argument('--stop-at', type=int)
    args = p.parse_args()
    if args.action == 'start':
        token = sys.stdin.readline().strip()
        if not token:
            raise RuntimeError('Backup credential required through encrypted stdin')
        OUT.mkdir(parents=True, exist_ok=True)
        if 'route-b-staged' in subprocess.run(['screen', '-ls'], capture_output=True, text=True).stdout:
            raise RuntimeError('Pipeline already present; inspect before resuming')
        command = 'cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/route_b_staged.py pipeline >> runs/route-b-staged-7b-20261016/console.log 2>&1'
        subprocess.run(['screen', '-dmS', 'route-b-staged', 'bash', '-c', command],
                       env={**os.environ, 'GH_TOKEN': token}, check=True)
        print('Started independent route B; ARP stage C unchanged')
    elif args.action == 'prepare':
        prepare()
    elif args.action == 'cache':
        teacher_cache(args.phase)
    elif args.action == 'train':
        train(args.phase, args.arm, args.seed, args.stop_at)
    elif args.action == 'eval':
        evaluate(args.phase, args.arm, args.seed)
    else:
        pipeline()


if __name__ == '__main__':
    main()
