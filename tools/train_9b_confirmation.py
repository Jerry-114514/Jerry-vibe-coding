"""Train the paired 9B GT-only confirmation arms with checkpointed sample order."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import time

from experiment_common import AXES, ROOT, read_json, read_rows, sha, write_json
from prepare_9b_confirmation_training import SEEDS

GIB = 1024**3
ENGINEERING_SEED = 2026111200
CONFIG = ROOT / 'docs/results/protocol-v2-20261016/9b-full-preflight-protocol.json'


def load_inputs(run, arm, seed, phase):
    if arm not in ('P-G', 'N-G') or phase not in ('preflight', 'formal'):
        raise ValueError('Unknown arm or phase')
    if (phase == 'formal' and seed not in SEEDS) or (phase == 'preflight' and seed != ENGINEERING_SEED):
        raise ValueError('Seed is not registered for this phase')
    plan_dir = run / 'private/training-plan'
    plan_path = plan_dir / 'training-plan.json'
    plan = read_json(plan_path)
    inputs = plan_dir / f'{arm}.jsonl'
    token_path = run / 'private/training-token-verification.jsonl'
    report_path = run / 'tokenization-verification.json'
    report = read_json(report_path)
    if sha(inputs) != plan['sha256'][inputs.name] or sha(token_path) != report['training_token_rows_sha256']:
        raise ValueError('Training examples or token lock changed')
    rows = read_rows(inputs)
    tokens = [row for row in read_rows(token_path) if row['arm'] == arm]
    lookup = {row['item_id']: row for row in rows}
    by_id = {row['item_id']: row for row in tokens}
    if len(rows) != 600 or len(tokens) != 600 or len(lookup) != 600 or lookup.keys() != by_id.keys():
        raise ValueError('Token lock does not cover the 600 training items')
    if phase == 'formal':
        path = plan_dir / f'seed-{seed}.json'
        if sha(path) != plan['sha256'][path.name]:
            raise ValueError('Seed schedule changed')
        schedule = read_json(path)['updates']
        if Counter(row['axis'] for row in schedule) != dict.fromkeys(AXES, 250):
            raise ValueError('Expected 250 updates per axis')
    else:
        # Same 100 items in both arms, including the longest actual input
        # of either arm and the largest vision grid in the pool.
        paired_tokens = read_rows(token_path)
        largest = [
            max(paired_tokens, key=lambda row: row[metric])['item_id']
            for metric in ('prompt_tokens', 'target_tokens', 'visual_tokens')
        ]
        rest = list(lookup)
        random.Random(seed).shuffle(rest)
        schedule = [{'item_id': key, 'axis': lookup[key]['axis']}
                    for key in list(dict.fromkeys(largest + rest))[:100]]
    if any(lookup[row['item_id']]['axis'] != row['axis'] for row in schedule):
        raise ValueError('Scheduled axis does not match source')
    metadata = {'phase': phase, 'condition': arm, 'seed': seed, 'updates': len(schedule),
                'plan_sha256': sha(plan_path), 'input_sha256': sha(inputs), 'tokens_sha256': sha(token_path),
                'token_report_sha256': sha(report_path), 'model_revision': report['model_revision'],
                'config_sha256': sha(CONFIG), 'trainer_sha256': sha(Path(__file__))}
    return schedule, lookup, by_id, metadata


def reconcile_ledger(path, schedule, step, expected_sha):
    from run_9b_native_frontend_v41 import preserve_tail

    preserve_tail(path)
    raw = path.read_bytes() if path.exists() else b''
    lines = raw.splitlines(keepends=True)
    prefix = b''.join(lines[:step])
    if len(lines) < step or hashlib.sha256(prefix).hexdigest() != expected_sha:
        raise ValueError('Checkpoint and committed loss ledger disagree')
    for index, (line, expected) in enumerate(zip(lines[:step], schedule), 1):
        row = json.loads(line)
        if row['step'] != index or row['item_id'] != expected['item_id'] or row['axis'] != expected['axis']:
            raise ValueError('Committed training ledger differs from schedule')
    if len(lines) > step:
        path.with_name(f'uncommitted-loss-{time.time_ns()}.jsonl').write_bytes(b''.join(lines[step:]))
        path.write_bytes(prefix)


def execute_steps(path, schedule, start, until, update, checkpoint, guard):
    with path.open('a', encoding='utf-8', newline='\n') as ledger:
        for step in range(start + 1, until + 1):
            guard()
            item = schedule[step - 1]
            measurements = update(item)
            if not math.isfinite(measurements['loss']) or not 0 < measurements['grad_norm'] < math.inf:
                raise ValueError('Invalid training loss or gradient')
            record = {**measurements, **item, 'step': step, 'actual_optimizer_update': True}
            ledger.write(json.dumps(record, allow_nan=False) + '\n')
            ledger.flush()
            os.fsync(ledger.fileno())
            if step % 50 == 0 or step == until:
                checkpoint(step, sha(path))
                print(f'update {step}/{until}: loss {record["loss"]:.6f}', flush=True)


def train(run, phase, arm, seed, deadline, until):
    schedule, rows, tokens, metadata = load_inputs(run, arm, seed, phase)
    if until not in (50, len(schedule)):
        raise ValueError('Only the registered 50-step interruption or the fixed endpoint is allowed')
    metadata['deadline_unix'] = deadline
    if phase == 'formal':
        review = read_json(run / 'frozen/review-status.json')
        if (review['status'] != 'reviewed_manifest_frozen'
                or review['evaluation_manifest_sha256'] != sha(run / 'frozen/evaluation-manifest.jsonl')):
            raise ValueError('Formal training requires a frozen human-reviewed confirmation set')
        metadata['review_sha256'] = sha(run / 'frozen/review-status.json')
        for condition in ('P-G', 'N-G'):
            done = read_json(run / 'preflight' / f'{condition}-seed{ENGINEERING_SEED}' / 'complete.json')
            if (done['actual_updates'] != 100 or done['minimum_free_bytes'] < 4 * GIB
                    or done['reserved_peak_bytes'] > 26 * GIB or not done['new_process_resumed_at50']
                    or done['config_sha256'] != metadata['config_sha256']
                    or done['plan_sha256'] != metadata['plan_sha256']
                    or done['tokens_sha256'] != metadata['tokens_sha256']
                    or done['trainer_sha256'] != metadata['trainer_sha256']):
                raise ValueError(f'Preflight has not passed for {condition}')
    folder = run / ('training' if phase == 'formal' else 'preflight') / f'{arm}-seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)

    def guard():
        if time.time() >= deadline:
            raise TimeoutError('Registered training deadline expired')
        if shutil.disk_usage(run).free < 20 * GIB:
            raise OSError('Training storage fell below 20 GiB')

    guard()
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from gpu_lock import gpu_lock
    import preflight_9b_full as shared
    from route_b_9b_loop import model_backend, gpu_inputs
    from train_distill import response_forward

    if read_json(shared.MODEL / 'weights.verified.json')['revision'] != metadata['model_revision']:
        raise ValueError('Model revision differs from the local token lock')
    config = read_json(CONFIG)['training']
    ledger, checkpoint = folder / 'loss.jsonl', folder / 'resume.pt'
    with gpu_lock():
        if (folder / 'complete.json').exists():
            done = read_json(folder / 'complete.json')
            if done['meta'] != metadata or any(sha(folder / name) != digest for name, digest in done['files'].items()):
                raise ValueError('Completed training artifact or metadata changed')
            return done
        guard()
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.reset_peak_memory_stats()
        processor, base = model_backend()
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(
            r=config['lora_r'], lora_alpha=config['lora_alpha'], lora_dropout=config['lora_dropout'],
            target_modules=config['lora_targets'], bias='none', task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        if (sum(p.numel() for p in params) != config['expected_trainable_parameters']
                or any('lora_' not in n or 'visual' in n for n, p in model.named_parameters() if p.requires_grad)):
            raise ValueError('Trainable parameters differ from frozen language-only LoRA')
        initial = hashlib.sha256()
        for key, value in sorted(get_peft_model_state_dict(model).items()):
            initial.update(key.encode() + b'\0')
            initial.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        paired_init = folder.parent / f'seed-{seed}-initial.json'
        identity = {'seed': seed, 'lora_sha256': initial.hexdigest()}
        if paired_init.exists() and read_json(paired_init) != identity:
            raise ValueError('Paired initial LoRA states differ')
        write_json(paired_init, identity)
        optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
        start, prefix_sha, min_free, max_reserved, resumed = 0, hashlib.sha256(b'').hexdigest(), 2**63, 0, False
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if state['meta'] != metadata:
                raise ValueError('Checkpoint metadata differs from run')
            start, prefix_sha = state['step'], state['loss_sha256']
            min_free, max_reserved = state['min_free'], state['max_reserved']
            resumed = state['new_process_resumed_at50'] or (start == 50 and state['pid'] != os.getpid())
            set_peft_model_state_dict(model, state['adapter'])
            optimizer.load_state_dict(state['optimizer'])
            if {int(v['step']) for v in optimizer.state.values()} != {start}:
                raise ValueError('Optimizer cursor differs from checkpoint')
            torch.set_rng_state(state['rng_cpu'])
            torch.cuda.set_rng_state_all(state['rng_cuda'])
            random.setstate(state['rng_python'])
            del state
        if start > until:
            raise ValueError('Requested endpoint precedes the checkpoint')
        reconcile_ledger(ledger, schedule, start, prefix_sha)

        def update(item):
            nonlocal min_free, max_reserved
            tick = time.monotonic()
            row, locked = rows[item['item_id']], tokens[item['item_id']]
            if sha(ROOT / row['image_path']) != row['image_sha256']:
                raise ValueError('Training image changed')
            prompt = shared.prompt_inputs(processor, row)
            ids = prompt['input_ids'][0].tolist()
            target = shared.targets(processor, str(row['target']))
            if (hashlib.sha256(struct.pack(f'<{len(ids)}q', *ids)).hexdigest() != locked['prompt_ids_sha256']
                    or prompt['image_grid_thw'][0].tolist() != locked['image_grid_thw']
                    or target != locked['target_ids'] or len(ids) + len(target) > 2048):
                raise ValueError('Runtime training tokens differ from lock')
            batch = gpu_inputs(shared.suffix_inputs(prompt, target))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, labels = response_forward(model, batch, len(ids), base_model=base)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite GT CE')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, config['clip_grad_norm'], error_if_nonfinite=True)
            if float(grad) <= 0:
                raise ValueError('Zero gradient')
            optimizer.step()
            torch.cuda.synchronize()
            reserved = torch.cuda.max_memory_reserved()
            # At sampling time reserved includes cached peak allocations.
            free, _ = torch.cuda.mem_get_info()
            min_free, max_reserved = min(min_free, free), max(max_reserved, reserved)
            if min_free < 4 * GIB or max_reserved > 26 * GIB:
                raise MemoryError('Training crossed the frozen GPU headroom limits')
            return {'loss': float(loss.detach()), 'grad_norm': float(grad), 'seconds': time.monotonic() - tick,
                    'reserved_peak_bytes': reserved, 'sampled_free_bytes': free,
                    'allocated_peak_bytes': torch.cuda.max_memory_allocated()}

        def save(step, digest):
            state = {'meta': metadata, 'step': step, 'loss_sha256': digest,
                     'adapter': {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                     'optimizer': optimizer.state_dict(), 'rng_cpu': torch.get_rng_state(),
                     'rng_cuda': torch.cuda.get_rng_state_all(), 'rng_python': random.getstate(),
                     'pid': os.getpid(), 'min_free': min_free, 'max_reserved': max_reserved,
                     'new_process_resumed_at50': resumed}
            temp = checkpoint.with_suffix('.tmp')
            torch.save(state, temp)
            temp.replace(checkpoint)

        execute_steps(ledger, schedule, start, until, update, save, guard)
        if until < len(schedule):
            return {'status': 'saved_at50', 'checkpoint_sha256': sha(checkpoint)}
        if {int(v['step']) for v in optimizer.state.values()} != {until}:
            raise ValueError('Optimizer cursor differs from final step')
        model.save_pretrained(folder / 'adapter', safe_serialization=True)
        from safetensors.torch import load_file
        exported = load_file(str(folder / 'adapter/adapter_model.safetensors'), device='cpu')
        final = get_peft_model_state_dict(model)
        if exported.keys() != final.keys() or any(
                not torch.equal(value, final[key].detach().cpu()) or not torch.isfinite(value).all()
                for key, value in exported.items()):
            raise ValueError('Adapter export differs from final model')
        files = {name: sha(folder / name) for name in (
            'adapter/adapter_model.safetensors', 'adapter/adapter_config.json', 'loss.jsonl', 'resume.pt')}
        done = {**metadata, 'meta': metadata, 'actual_updates': until,
                'adapter_sha256': files['adapter/adapter_model.safetensors'], 'files': files,
                'minimum_free_bytes': min_free, 'reserved_peak_bytes': max_reserved,
                'new_process_resumed_at50': resumed, 'initial_lora_sha256': identity['lora_sha256']}
        write_json(folder / 'complete.json', done)
        return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('phase', choices=('preflight', 'formal'))
    parser.add_argument('arm', choices=('P-G', 'N-G'))
    parser.add_argument('seed', type=int)
    parser.add_argument('--until', type=int, required=True)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args()
    end = datetime.fromisoformat(args.deadline_utc)
    if end.tzinfo is None:
        parser.error('--deadline-utc requires an explicit timezone')
    print(json.dumps(train(args.run, args.phase, args.arm, args.seed, end.timestamp(), args.until)))
