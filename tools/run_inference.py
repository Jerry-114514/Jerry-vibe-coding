"""B1/B3: restartable, sharded per-item ledger, with immutable run metadata."""
import argparse
import json
import os
from pathlib import Path
import time
from experiment_common import AXES, MODELS, ROOT, read_json, read_rows, score_generation, sha, write_json
from model_backends import Backend, freeze_models


def inference(key, output, smoke=False, shard=0, shards=1, adapter=None):
    if not 0 <= shard < shards:
        raise ValueError('Invalid shard')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'data/manifest.jsonl'
    manifest = read_rows(manifest_path)
    if sha(manifest_path) != read_json(ROOT / 'data/experiment.lock.json')['files']['data/manifest.jsonl']:
        raise ValueError('Frozen manifest modified')
    selected = [x for i, x in enumerate(manifest) if i % shards == shard]
    if smoke:
        # Whole paraphrase group plus one example from each other axis.
        first_base = next(x['bootstrap_unit'] for x in manifest if x['axis'] == 'consistency')
        selected = [x for x in manifest if x['axis'] == 'consistency' and x['bootstrap_unit'] == first_base]
        selected += [next(x for x in manifest if x['axis'] == axis) for axis in AXES if axis != 'consistency']
    model_lock = freeze_models()
    metadata = {'schema': 'arp.ledger.v1', 'checkpoint': key,
                'manifest_sha256': sha(manifest_path), 'model': model_lock['models'][key],
                'fp16': True, 'do_sample': False, 'num_beams': 1, 'max_new_tokens': 128,
                'parser': 'explicit-json-schema.v2', 'smoke': smoke, 'shard': shard, 'shards': shards,
                'inference_backend_version':'finite-logits-sdpa.v1',
                'adapter': str(adapter) if adapter else None, 'selected_rows': len(selected),
                'qwen_pixels': [256*28*28, 1280*28*28], 'intern_max_tiles': 12}
    meta_path = output / f'{key}-{shard:03d}.meta.json'
    if meta_path.exists() and read_json(meta_path) != metadata:
        if not smoke:
            raise ValueError('Cannot resume a ledger under a changed protocol')
        # Only preflight ledgers may be upgraded by rescoring existing raw output.
        old_path=output/f'{key}-{shard:03d}.jsonl'
        previous=read_json(meta_path)
        changed={k for k in set(previous)|set(metadata) if previous.get(k)!=metadata.get(k)}
        if changed-{'parser'}:
            archive=output/'superseded'/key/str(time.time_ns())
            archive.mkdir(parents=True,exist_ok=True)
            for path in [meta_path,old_path,output/f'{key}-{shard:03d}.complete.json']:
                if path.exists():
                    path.replace(archive/path.name)
        elif old_path.exists():
            by_id={item['item_id']:item for item in selected}
            from experiment_common import write_rows
            old=read_rows(old_path)
            for row in old:
                row.update(score_generation(row['raw_generation'],by_id[row['item_id']]))
            write_rows(old_path,old)
    write_json(meta_path, metadata)
    ledger_path = output / f'{key}-{shard:03d}.jsonl'
    # A killed process can leave a partial last line; remove only that tail.
    if ledger_path.exists():
        data = ledger_path.read_bytes()
        if data and not data.endswith(b'\n'):
            with ledger_path.open('r+b') as f:
                f.truncate(data.rfind(b'\n') + 1)
    existing = read_rows(ledger_path) if ledger_path.exists() else []
    done = {x['item_id'] for x in existing}
    if len(done) != len(existing) or not done <= {x['item_id'] for x in selected}:
        raise ValueError('Duplicate or unrelated ledger rows')
    if len(done) == len(selected):
        prior_path=output/f'{key}-{shard:03d}.complete.json'
        prior=read_json(prior_path) if prior_path.exists() else {}
        write_json(output / f'{key}-{shard:03d}.complete.json',
                   {**prior,'rows':len(done),'ledger_sha256':sha(ledger_path),'resumed':True})
        print(f'{key}: already complete', flush=True)
        return
    backend = Backend(key, adapter)
    import torch
    with ledger_path.open('a', encoding='utf-8', newline='\n') as f:
        for item in selected:
            if item['item_id'] in done:
                continue
            start = time.monotonic()
            # Infrastructure/OOM errors stop the shard; never score them as model failures.
            raw = backend.generate(item)
            row = {'checkpoint': key, 'axis': item['axis'], 'item_id': item['item_id'],
                   'numerics_checked':True,
                   'dataset': item['dataset'], 'bootstrap_unit': item['bootstrap_unit'],
                   'image_group': item['image_group'], **score_generation(raw, item),
                   'elapsed_seconds': round(time.monotonic() - start, 4)}
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
            done.add(item['item_id'])
            if smoke or len(done) % 100 == 0:
                print(f'{key} {len(done)}/{len(selected)} {item["axis"]}: {raw[:160]!r}', flush=True)
    write_json(output / f'{key}-{shard:03d}.complete.json',
               {'rows': len(done), 'ledger_sha256': sha(ledger_path), 'peak_vram_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', choices=MODELS, required=True)
    p.add_argument('--output', default=str(ROOT / 'runs/formal/ledger'))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--adapter')
    a = p.parse_args()
    from gpu_lock import gpu_lock
    with gpu_lock():
        inference(a.checkpoint, a.output, a.smoke, a.shard, a.shards, a.adapter)
