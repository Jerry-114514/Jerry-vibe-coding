"""Run the frozen 9B confirmation evaluation with a resumable raw ledger."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time

from experiment_common import ROOT, read_json, read_rows, sha, write_json, write_rows
from prepare_9b_confirmation_training import SEEDS
from score_9b_confirmation import EOS_IDS, MAX_NEW_TOKENS, score_manifest, score_row

GIB = 1024**3
RESERVED_LIMIT = 26 * GIB


def run_ledger(manifest, folder, metadata, generate, guard):
    """Run in manifest order and only resume with matching immutable metadata."""
    from run_9b_native_frontend_v41 import preserve_tail

    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / 'evaluation-meta.json'
    if meta_path.exists():
        if read_json(meta_path) != metadata:
            raise ValueError('Resume metadata differs from the original evaluation')
    else:
        write_json(meta_path, metadata)
    raw_path = folder / 'raw.jsonl'
    preserve_tail(raw_path)
    previous = read_rows(raw_path) if raw_path.exists() else []
    completed = folder / 'scores.json'
    if completed.exists():
        saved = read_json(completed)
        if (len(previous) != len(manifest) or saved['raw_sha256'] != sha(raw_path)
                or saved['scored_sha256'] != sha(folder / 'scored.jsonl')):
            raise ValueError('Completed evaluation artifacts changed')
    if len(previous) > len(manifest):
        raise ValueError('Ledger exceeds frozen manifest')
    for item, row in zip(manifest, previous):
        if (item['item_id'], item['axis']) != (row['item_id'], row['axis']):
            raise ValueError('Prior ledger IDs are not a prefix of the frozen manifest')
        score_row(item, row)
    with raw_path.open('a', encoding='utf-8', newline='\n') as stream:
        for index, item in enumerate(manifest[len(previous):], start=len(previous) + 1):
            guard()
            row = {**generate(item), 'item_id': item['item_id'], 'axis': item['axis']}
            score_row(item, row)
            stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            if index % 100 == 0:
                print(f'{folder.name}: {index}/{len(manifest)}', flush=True)
    rows = read_rows(raw_path)
    scored, scores = score_manifest(manifest, rows)
    write_rows(folder / 'scored.jsonl', scored)
    result = {
        'rows': len(rows), 'scores': scores, 'optimizer_loaded': False, 'gradients_enabled': False,
        'length_capped': dict(Counter(r['axis'] for r in rows if r['finish_reason'] == 'length')),
        'parse_failures': dict(Counter(r['axis'] for r in scored if r['parse_fail_flag'])),
        'generation_seconds': sum(row['seconds'] for row in rows),
        'reserved_peak_bytes': max(row['reserved_peak_bytes'] for row in rows),
        'allocated_peak_bytes': max(row['allocated_peak_bytes'] for row in rows),
        'raw_sha256': sha(raw_path), 'scored_sha256': sha(folder / 'scored.jsonl'),
        'evaluation_meta_sha256': sha(meta_path),
    }
    write_json(completed, result)
    return result


def evaluation_inputs(manifest_path):
    review = read_json(manifest_path.parent / 'review-status.json')
    if (review['status'] != 'reviewed_manifest_frozen'
            or review['evaluation_manifest_sha256'] != sha(manifest_path)):
        raise ValueError('A hash-matched, human-reviewed confirmation manifest is required')
    rows = read_rows(manifest_path)
    if len(rows) != 1320 or len({row['item_id'] for row in rows}) != 1320:
        raise ValueError('Expected 1,320 unique confirmation prompts')
    checked = {}
    for row in rows:
        path = (ROOT / row['image_path']).resolve()
        if not path.is_relative_to(ROOT / 'data'):
            raise ValueError(f'Unexpected evaluation image path: {path}')
        if path not in checked:
            checked[path] = sha(path)
        if checked[path] != row['image_sha256']:
            raise ValueError(f"Evaluation image differs from frozen bytes: {row['item_id']}")
    return review, rows


def evaluate(manifest_path, endpoint, run, deadline):
    review, rows = evaluation_inputs(manifest_path)
    jobs = {'BASE': None, **{f'{arm}-seed{seed}': (arm, seed)
                           for seed in SEEDS for arm in ('P-G', 'N-G')}}
    if endpoint not in jobs:
        raise ValueError(f'Unregistered endpoint: {endpoint}')
    adapter, adapter_hashes = None, None
    if endpoint != 'BASE':
        arm, seed = jobs[endpoint]
        training = run / 'training' / endpoint
        finished = read_json(training / 'complete.json')
        if (finished['condition'], finished['seed'], finished['actual_updates']) != (arm, seed, 1000):
            raise ValueError('Adapter is not a registered 1000-step training endpoint')
        adapter = training / 'adapter'
        adapter_hashes = {name: sha(adapter / name) for name in ('adapter_model.safetensors', 'adapter_config.json')}
        if adapter_hashes['adapter_model.safetensors'] != finished['adapter_sha256']:
            raise ValueError('Adapter weights changed after training')

    def guard():
        if time.time() >= deadline:
            raise TimeoutError('Registered evaluation deadline expired')
        if shutil.disk_usage(run).free < 20 * GIB:
            raise OSError('Evaluation storage fell below the registered 20 GiB reserve')

    guard()
    import torch
    from gpu_lock import gpu_lock
    from model_backends import finite_logits_processors
    from preflight_9b_full import MODEL
    from qwen35_termination_contract import explicit_eos_ids
    from route_b_9b_loop import gpu_inputs, model_backend, public_prompt_inputs

    if read_json(MODEL / 'weights.verified.json')['revision'] != review['model_revision']:
        raise ValueError('Model revision differs from tokenizer and review lock')
    meta = {
        'endpoint': endpoint, 'manifest_sha256': review['evaluation_manifest_sha256'],
        'model_revision': review['model_revision'], 'adapter': adapter_hashes,
        'max_new_tokens': MAX_NEW_TOKENS, 'eos_token_ids': sorted(EOS_IDS),
        'deadline_unix': deadline, 'evaluator_sha256': sha(Path(__file__)),
        'scorer_sha256': sha(ROOT / 'tools/score_9b_confirmation.py'),
    }
    folder = run / 'evaluation' / endpoint
    with gpu_lock():
        if (folder / 'scores.json').exists():
            return run_ledger(rows, folder, meta, None, guard)
        guard()
        torch.cuda.reset_peak_memory_stats()
        processor, model = model_backend(adapter)
        model.eval().requires_grad_(False)
        eos = explicit_eos_ids(processor.tokenizer, model.generation_config)
        if set(eos) != EOS_IDS:
            raise ValueError('Unexpected Qwen termination IDs')

        def generate(item):
            start = time.monotonic()
            with torch.inference_mode():
                cpu = public_prompt_inputs(processor, item)
                ids = cpu['input_ids'][0].tolist()
                digest = hashlib.sha256(struct.pack(f'<{len(ids)}q', *ids)).hexdigest()
                if (len(ids) != item['prompt_tokens'] or digest != item['prompt_ids_sha256']
                        or cpu['image_grid_thw'][0].tolist() != item['image_grid_thw']):
                    raise ValueError(f"Runtime input differs from local token lock: {item['item_id']}")
                batch = gpu_inputs(cpu)
                result = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                        eos_token_id=eos, logits_processor=finite_logits_processors())
                tokens = result[0, len(ids):].tolist()
                raw = processor.decode(tokens, skip_special_tokens=True).strip()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved()
            if peak > RESERVED_LIMIT:
                raise MemoryError('PyTorch reserved exceeded 26 GiB')
            return {
                'raw_generation': raw, 'generated_token_ids': tokens,
                'finish_reason': 'eos' if tokens and tokens[-1] in EOS_IDS else 'length',
                'seconds': time.monotonic() - start,
                'reserved_peak_bytes': peak, 'allocated_peak_bytes': torch.cuda.max_memory_allocated(),
            }

        return run_ledger(rows, folder, meta, generate, guard)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('endpoint')
    parser.add_argument('run', type=Path)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None:
        parser.error('--deadline-utc requires an explicit timezone')
    print(json.dumps(evaluate(args.manifest, args.endpoint, args.run, deadline.timestamp()), ensure_ascii=False))
