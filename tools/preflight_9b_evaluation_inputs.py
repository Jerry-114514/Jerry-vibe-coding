"""CPU-only9B prompt budget/hash checks for both800 and16900 eval inputs."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import shutil
import statistics
import time

from experiment_common import ROOT, read_json, read_rows, sha, write_json
from route_b_9b_loop import BASE, MINIMAL_MANIFEST, public_prompt_inputs
import preflight_9b_full as shared


def main():
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(str(shared.MODEL), local_files_only=True)
    folder = BASE / 'evaluation-input-preparation'
    folder.mkdir(parents=True, exist_ok=True)
    assets = {r['path']: r for r in read_rows(ROOT / 'data/assets.sha256.jsonl')}
    verified = {}
    results = {}
    started = time.monotonic()
    for label, source, expected in [('minimal', MINIMAL_MANIFEST, 800),
                                     ('formal', ROOT / 'data/manifest.jsonl', 16900)]:
        rows = read_rows(source)
        assert len(rows) == len({r['item_id'] for r in rows}) == expected
        tokens = defaultdict(list)
        for i, row in enumerate(rows, 1):
            path = row['image_path']
            if path not in verified:
                assert path in assets and sha(ROOT / path) == assets[path]['sha256']
                verified[path] = assets[path]['sha256']
            batch = public_prompt_inputs(processor, row)
            tokens[row['axis']].append(int(batch['input_ids'].shape[1]))
            if i % 1000 == 0:
                write_json(folder / 'progress.json', {'phase': label, 'rows': i, 'expected': expected,
                           'cpu_only': True, 'elapsed_seconds': time.monotonic() - started})
        destination = folder / f'{label}-manifest.jsonl'
        if destination.exists():
            assert sha(destination) == sha(source)
        else:
            shutil.copyfile(source, destination)
        results[label] = {'rows': len(rows), 'manifest_sha256': sha(source),
                          'axis_rows': dict(Counter(r['axis'] for r in rows)),
                          'prompt_tokens': {a: {'min': min(v), 'max': max(v), 'mean': statistics.mean(v)}
                                            for a, v in tokens.items()}}
    assert not torch.cuda.is_initialized()
    result = {'status': 'passed_cpu_minimal_and_full_evaluation_inputs', 'cpu_only': True,
              'no_cuda_context': True, 'cpu_threads': 2, 'results': results,
              'unique_images_hash_checked': len(verified), 'max_prompt_tokens': 2048, 'max_visual_tokens': 1280,
              'elapsed_seconds': time.monotonic() - started, 'source_sha256': sha(ROOT / 'tools/preflight_9b_evaluation_inputs.py'),
              'input_engine_sha256': sha(ROOT / 'tools/route_b_9b_loop.py'),
              'note': 'No predictions, model weights, optimizer or gradients loaded; original evaluation manifests preserved byte-for-byte'}
    write_json(folder / 'complete.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        folder = BASE / 'evaluation-input-preparation'
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / 'failure.json', {'type': type(error).__name__, 'message': str(error),
                   'timestamp': datetime.now(timezone.utc).isoformat(), 'gpu_used': False})
        raise
