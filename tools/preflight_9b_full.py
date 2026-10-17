"""Independent BF16 9B full language attention+MLP LoRA / dual-CE preflight.

100 distinct real training units, then a separate 2048/128 shape stress update.
GPU execution waits for completion of all7B evaluation and the shared GPU lock.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time

from experiment_common import ROOT, AXES, read_json, read_rows, sha, write_json, write_rows, parse_answer

BASE = ROOT / 'runs/route-b-9b-122b-20261017'
OUT = BASE / 'full-preflight'
MODEL = ROOT / '.model_cache/qwen35_9b'
ENV = ROOT / '../arp-qwen35-venv/bin/python'


def prompt_inputs(processor, row):
    import torch
    from PIL import Image
    path = (ROOT / row['image_path']).resolve()
    assert path.is_relative_to((ROOT / 'data/training').resolve())
    with Image.open(path) as source:
        image = source.convert('RGB')
    if min(image.size) < 32:
        scale = 32 / min(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': row['question']}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    batch = processor(text=[text], images=[image], return_tensors='pt',
                      images_kwargs={'min_pixels': 256 * 32 * 32, 'max_pixels': 1280 * 32 * 32})
    visual = int(batch['image_grid_thw'].prod(dim=1).sum().item() // 4)
    assert visual <= 1280 and batch['input_ids'].shape[1] < 2048
    assert batch['input_ids'].dtype == torch.int64
    return dict(batch)


def answer_text(row, teacher=None):
    if row['axis'] != 'grounding':
        return str(row['answer_key']) if teacher is None else teacher['raw_generation'].strip()
    if teacher is None:
        return json.dumps(row['answer_key'])
    box = parse_answer(teacher['raw_generation'], {'answer_type': 'bbox', 'width': 1000, 'height': 1000})
    assert box is not None
    return json.dumps([round(box[i] * (row['width'] if i % 2 == 0 else row['height']) / 1000, 3) for i in range(4)])


def suffix_inputs(prompt, ids):
    import torch
    result = dict(prompt)
    suffix = torch.tensor([ids], dtype=torch.int64)
    result['input_ids'] = torch.cat([prompt['input_ids'], suffix], dim=1)
    for key in ('attention_mask', 'token_type_ids', 'mm_token_type_ids'):
        if key in prompt:
            extra = torch.ones_like(suffix) if key == 'attention_mask' else torch.zeros_like(suffix)
            result[key] = torch.cat([prompt[key], extra.to(prompt[key].dtype)], dim=1)
    return result


def targets(processor, text):
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    assert processor.tokenizer.eos_token_id is not None
    ids.append(processor.tokenizer.eos_token_id)
    assert 0 < len(ids) <= 128, 'Response cap exceeded; no truncation'
    return ids


def prepare():
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    marker = read_json(BASE / 'teacher-eligible-complete.json')
    assert marker['status'] == 'passed_eligible_pool_only'
    assert sha(BASE / 'eligible-pool.jsonl') == marker['eligible_pool_sha256']
    assert sha(BASE / 'teacher.jsonl') == marker['teacher_ledger_sha256']
    rows = read_rows(BASE / 'eligible-pool.jsonl')
    teacher = {r['item_id']: r for r in read_rows(BASE / 'teacher.jsonl')}
    if (OUT / 'protocol.json').exists():
        protocol = read_json(OUT / 'protocol.json')
        assert protocol['source_sha256'] == sha(Path(__file__))
        assert protocol['eligible_pool_sha256'] == marker['eligible_pool_sha256']
        assert protocol['teacher_ledger_sha256'] == marker['teacher_ledger_sha256']
        assert protocol['plan_sha256'] == sha(ROOT / 'configs/route_b_9b_122b.plan.json')
        selected = read_rows(OUT / 'manifest.jsonl')
        assert [r['item_id'] for r in selected] == protocol['selected_item_ids']
        return processor, selected, teacher, protocol
    groups = {}
    maximums = {}
    for row in rows:
        prompt = prompt_inputs(processor, row)
        length = prompt['input_ids'].shape[1]
        cached = teacher[row['item_id']]
        assert cached['finish_reason'] == 'stop' and cached['returned_model'] == 'qwen3.5-122b-a10b'
        assert any(x['sha256'] == sha(ROOT / row['image_path']) for x in cached['image_hashes'])
        responses = [targets(processor, answer_text(row)), targets(processor, answer_text(row, cached))]
        total = length + max(map(len, responses))
        assert total <= 2048, 'Sequence cap exceeded; no truncation'
        key = (row['axis'], row['bootstrap_unit'])
        if key not in groups or total > groups[key][0]:
            groups[key] = (total, row)
        maximums[row['item_id']] = total
    selected = []
    for axis in AXES:
        candidates = [x for (a, _), x in groups.items() if a == axis]
        longest = max(candidates, key=lambda x: (x[0], x[1]['item_id']))[1]
        remaining = sorted((x[1] for x in candidates if x[1]['item_id'] != longest['item_id']),
                           key=lambda r: hashlib.sha256(('9b-memory:' + r['bootstrap_unit']).encode()).hexdigest())
        selected += [longest] + remaining[:24]
    assert len(selected) == 100 and len({r['bootstrap_unit'] for r in selected}) == 100
    random.Random(2026101701).shuffle(selected)
    plan = read_json(ROOT / 'configs/route_b_9b_122b.plan.json')
    protocol = {'schema': 'independent.9b.full-memory.v1', 'student': plan['student'], 'revision': plan['revision'],
                'teacher': plan['teacher'], 'engineering_only': True, 'real_updates': 100, 'seed': 2026101701,
                'training': plan['training'], 'stress_total_tokens': 2048, 'stress_response_tokens': 128,
                'thresholds': plan['preflight'], 'source_sha256': sha(Path(__file__)),
                'eligible_pool_sha256': marker['eligible_pool_sha256'], 'teacher_ledger_sha256': marker['teacher_ledger_sha256'],
                'plan_sha256': sha(ROOT / 'configs/route_b_9b_122b.plan.json'),
                'grounding': 'Original-pixel student question and GT; normalized1000 teacher converted explicitly to original pixels, rounded0.001pixel',
                'assistant_boundary': 'Nonthinking generation prompt; tokenize response separately and append explicit student EOS',
                'selected_item_ids': [r['item_id'] for r in selected],
                'maximum_real_training_total_tokens': max(maximums.values()), 'eligible_rows_checked': len(rows)}
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        assert read_json(OUT / 'protocol.json') == protocol
    else:
        write_json(OUT / 'protocol.json', protocol)
        write_rows(OUT / 'manifest.jsonl', selected)
    return processor, selected, teacher, protocol


def gpu_run():
    import torch
    import torch.nn.functional as F
    from transformers import Qwen3_5ForConditionalGeneration
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from train_distill import response_forward
    from gpu_lock import gpu_lock
    processor, rows, cached, protocol = prepare()
    assert seven_finished(), '7B full evaluation must finish before9B training preflight'
    plan = protocol['training']
    monitor_stop = threading.Event()

    def monitor():
        while not monitor_stop.wait(1):
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
            with (OUT / 'nvidia-smi.jsonl').open('a') as stream:
                stream.write(json.dumps({'sample': output.strip()}) + '\n')

    def cuda(batch):
        return {k: v.to('cuda', dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda') for k, v in batch.items()}

    with gpu_lock():
        # A lock is necessary but not sufficient: verify no other CUDA model is resident.
        assert torch.cuda.mem_get_info()[0] >= 29 * 1024 ** 3, 'GPU is not idle for an isolated preflight'
        assert shutil.disk_usage(OUT).free >= 5 * 1024 ** 3, 'Registered5GiB disk stop'
        torch.manual_seed(protocol['seed'])
        torch.cuda.reset_peak_memory_stats()
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        try:
            base = Qwen3_5ForConditionalGeneration.from_pretrained(str(MODEL), local_files_only=True,
                        dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
            base.config.use_cache = False
            model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                        target_modules=plan['lora_targets'], bias='none', task_type='CAUSAL_LM'))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.enable_input_require_grads()
            model.train()
            params = [p for p in model.parameters() if p.requires_grad]
            names = [n for n, p in model.named_parameters() if p.requires_grad]
            assert sum(p.numel() for p in params) == plan['expected_trainable_parameters']
            assert all('lora_' in n and 'visual' not in n for n in names)
            assert any('linear_attn' in n for n in names) and any('self_attn' in n for n in names) and any('.mlp.' in n for n in names)
            optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0)
            write_json(OUT / 'trainable-parameters.json', {'count': sum(p.numel() for p in params), 'names': names})

            def update(prompt, gt_ids, teacher_ids):
                length = prompt['input_ids'].shape[1]
                gt, teacher = cuda(suffix_inputs(prompt, gt_ids)), cuda(suffix_inputs(prompt, teacher_ids))
                optimizer.zero_grad(set_to_none=True)
                started = time.monotonic()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, target = response_forward(model, gt, length, base_model=base)
                    ce_gt = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))
                    logits_t, target_t = response_forward(model, teacher, length, base_model=base)
                    ce_teacher = F.cross_entropy(logits_t.float().reshape(-1, logits_t.shape[-1]), target_t.reshape(-1))
                    loss = .5 * ce_gt + .5 * ce_teacher
                assert torch.isfinite(loss)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(params, 1)
                assert torch.isfinite(grad) and grad > 0
                optimizer.step()
                torch.cuda.synchronize()
                result = {'loss': float(loss.detach()), 'ce_gt': float(ce_gt.detach()), 'ce_teacher': float(ce_teacher.detach()),
                          'grad_norm': float(grad), 'seconds': time.monotonic() - started,
                          'reserved_peak_bytes': torch.cuda.max_memory_reserved(), 'actual_optimizer_update': True}
                assert result['reserved_peak_bytes'] <= 30_000_000_000, '30GB memory stop'
                return result

            real = []
            for step, row in enumerate(rows, 1):
                assert shutil.disk_usage(OUT).free >= 5 * 1024 ** 3, 'Registered5GiB disk stop'
                started = time.monotonic()
                assert any(x['sha256'] == sha(ROOT / row['image_path']) for x in cached[row['item_id']]['image_hashes'])
                prompt = prompt_inputs(processor, row)
                result = update(prompt, targets(processor, answer_text(row)), targets(processor, answer_text(row, cached[row['item_id']])))
                result.update(step=step, item_id=row['item_id'], axis=row['axis'], total_step_seconds=time.monotonic() - started)
                real.append(result)
                with (OUT / 'loss.jsonl').open('a') as stream:
                    stream.write(json.dumps(result, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                print('9B preflight', step, '/100', result['loss'], flush=True)
            real_peak = torch.cuda.max_memory_reserved()
            adapter_state = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
            assert all(torch.count_nonzero(v) > 0 for k, v in adapter_state.items() if 'lora_B' in k)
            model.save_pretrained(OUT / 'adapter', safe_serialization=True)
            from safetensors.torch import load_file
            exported = load_file(str(OUT / 'adapter/adapter_model.safetensors'))
            assert set(exported) == set(adapter_state) and all(torch.equal(exported[k], v) for k, v in adapter_state.items())
            torch.save({'real_updates': 100, 'separate_stress_updates': 0, 'adapter': adapter_state,
                        'optimizer': optimizer.state_dict(), 'rng_cpu': torch.get_rng_state(),
                        'rng_cuda': torch.cuda.get_rng_state_all()}, OUT / 'resume.pt')
            del adapter_state, exported
            # Separate shape-only pressure update: preserve all real image tokens,
            # add ordinary text tokens to prompt, append128 response tokens.
            worst = max(rows, key=lambda r: int(prompt_inputs(processor, r)['image_grid_thw'].prod().item()))
            prompt = prompt_inputs(processor, worst)
            fill = processor.tokenizer.encode('x', add_special_tokens=False)[0]
            missing = 2048 - 128 - prompt['input_ids'].shape[1]
            assert missing >= 0
            stress_prompt = suffix_inputs(prompt, [fill] * missing)
            stress = update(stress_prompt, [fill] * 127 + [processor.tokenizer.eos_token_id],
                            [fill] * 127 + [processor.tokenizer.eos_token_id])
            write_json(OUT / 'shape-stress.json', {**stress, 'separate_from_100_real_updates': True, 'total_tokens': 2048, 'response_tokens': 128})
            peak = torch.cuda.max_memory_reserved()
            result = {'status': 'passed' if peak <= 28_000_000_000 else 'not_admitted',
                      'real_updates': 100, 'separate_stress_updates': 1, 'reserved_peak_bytes': peak,
                      'reserved_peak_GB': peak / 1e9, 'reserved_peak_GiB': peak / 1024 ** 3,
                      'real_updates_reserved_peak_bytes': real_peak,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'adapter_saved_after_real_updates_before_stress': True, 'adapter_export_exact': True,
                      'real_mean_seconds_per_step': sum(r['total_step_seconds'] for r in real) / 100,
                      'protocol_sha256': sha(OUT / 'protocol.json'), 'loss_sha256': sha(OUT / 'loss.jsonl'),
                      'formal_training_started': False, 'gpu': torch.cuda.get_device_name(),
                      'timestamp': datetime.now(timezone.utc).isoformat()}
            write_json(OUT / 'complete.json', result)
            if result['status'] == 'passed':
                recipe = {'preflight': result, 'training': plan, 'student_revision': protocol['revision'],
                          'eligible_pool_sha256': protocol['eligible_pool_sha256'],
                          'teacher_ledger_sha256': protocol['teacher_ledger_sha256'],
                          'scope': 'Hardware and numeric recipe frozen; minimal loop acceptance still required before formal9B training'}
                path = ROOT / 'configs/route_b_9b_5090.frozen.json'
                assert not path.exists() or read_json(path) == recipe
                write_json(path, recipe)
                (OUT / 'frozen-config.sha256').write_text(sha(path) + '\n')
        finally:
            monitor_stop.set()
            monitor_thread.join(timeout=5)


def seven_finished():
    out = ROOT / 'runs/route-b-staged-7b-20261016'
    parent = Path('/proc/7177/cmdline')
    if parent.exists() and b'run_route_b_storage_safe.py' in parent.read_bytes():
        return False
    expected = [('N', 42)] + [(a, s) for s in [42, 137, 256, 512, 1024] for a in ('A1', 'A2', 'U')]
    return ((out / 'status.json').exists() and read_json(out / 'status.json').get('phase') == 'all_four_stages_complete'
            and all((out / 'formal' / f'{a}-seed{s}' / 'scores.json').exists() for a, s in expected))


def wait():
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 48 * 3600
    while time.monotonic() < deadline:
        ready = seven_finished()
        teacher = (BASE / 'teacher-eligible-complete.json').exists()
        if teacher and not (OUT / 'protocol.json').exists():
            # Full-pool CPU preprocessing may run alongside7B GPU evaluation.
            # The later GPU process only rechecks immutable hashes and loads100 rows.
            subprocess.run([str(ENV.resolve()), '-u', str(Path(__file__)), 'prepare'], cwd=ROOT, check=True, timeout=1800)
        write_json(OUT / 'queue-status.json', {'waiting_for_7b_complete': not ready,
                   'waiting_for_teacher_eligible_cache': not teacher, 'pid': os.getpid(), 'gpu_used_while_waiting': False})
        if ready and teacher:
            if (OUT / 'complete.json').exists() or (OUT / 'loss.jsonl').exists():
                raise RuntimeError('Existing preflight artifacts require inspection; never repeat100 updates automatically')
            try:
                subprocess.run([str(ENV.resolve()), '-u', str(Path(__file__)), 'run'], cwd=ROOT, check=True, timeout=7200)
            except Exception as error:
                write_json(OUT / 'failure.json', {'type': type(error).__name__, 'message': str(error),
                           'timestamp': datetime.now(timezone.utc).isoformat(), '7b_experiment_affected': False})
                raise
            finally:
                if os.environ.get('GH_TOKEN'):
                    from publish_artifacts import publish
                    publish(OUT, 'arp-protocol-v2-20261016', token=os.environ['GH_TOKEN'])
            return
        time.sleep(30)
    raise RuntimeError('9B preflight queue deadline reached; no unauthorized earlyGPU use')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'wait'])
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare()[3]), flush=True)
    elif args.action == 'run':
        gpu_run()
    else:
        wait()
