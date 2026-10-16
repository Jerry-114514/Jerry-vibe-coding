"""Real 24GB-safe BF16 pilot; old data are development, never new confirmation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

from experiment_common import AXES, MODELS, ROOT, read_json, read_rows, score_generation, sha, units, write_json, write_rows

RUN = ROOT/'runs/protocol-v2'

def make_manifest(per_axis=100):
    source = ROOT/'data/manifest.jsonl'
    grouped = defaultdict(list)
    for row in read_rows(source):
        grouped[(row['axis'], row['dataset'], row['bootstrap_unit'])].append(row)
    chosen = []
    for axis in AXES:
        strata = defaultdict(list)
        for key, rows in grouped.items():
            if key[0] == axis:
                strata[(rows[0]['dataset'], rows[0].get('subset', ''))].append((key, rows))
        ordered = sorted(strata)
        for index, stratum in enumerate(ordered):
            count = per_axis//len(ordered) + int(index < per_axis % len(ordered))
            candidates = sorted(strata[stratum], key=lambda x: hashlib.sha256(('20261016:bf16-dev:'+str(x[0])).encode()).hexdigest())
            if len(candidates) < count:
                raise ValueError('Insufficient distinct scoring units')
            for _, rows in candidates[:count]:
                chosen.extend([{**row, 'protocol_v2_role': 'legacy_development'} for row in rows])
    destination = RUN/'calibration/precision-bridge-manifest.jsonl'
    if destination.exists():
        if read_rows(destination) != chosen:
            raise ValueError('Pilot manifest already frozen with different contents')
    else:
        write_rows(destination, chosen)
    write_json(destination.with_suffix('.meta.json'), {
        'role': 'development_only', 'independent_confirmation': False,
        'selection_seed': 20261016, 'scoring_units_per_axis': per_axis,
        'prompt_rows': len(chosen), 'source_manifest_sha256': sha(source),
        'manifest_sha256': sha(destination),
        'reason': 'Historical audit questions reused for numerical calibration; excluded from future independent confirmation'})
    return destination

def memory_worker(stop, path, key):
    while not stop.is_set():
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'model': key,
                                     'query': result.stdout.strip(), 'returncode': result.returncode})+'\n')
        stop.wait(5)

def pilot(key, per_axis=100, limit=None):
    import torch
    from gpu_lock import gpu_lock
    from model_backends import Backend
    manifest_path = make_manifest(per_axis)
    manifest = read_rows(manifest_path)
    if limit:
        manifest = [r for axis in AXES for r in [x for x in manifest if x['axis']==axis][:limit]]
    output = RUN/'calibration/precision-bridge'/key
    output.mkdir(parents=True, exist_ok=True)
    path = output/'ledger.jsonl'
    marker = output/'complete.json'
    metadata = {'model': key, 'precision': 'bf16', 'manifest_sha256': sha(manifest_path),
                'prompt_rows': len(manifest), 'limit_per_axis': limit, 'confirmation': False,
                'model_revision': read_json(ROOT/'configs/models.lock.json')['models'][key]['revision']}
    if (output/'meta.json').exists() and read_json(output/'meta.json') != metadata:
        raise ValueError('Pilot output belongs to another protocol')
    write_json(output/'meta.json', metadata)
    if marker.exists() and read_json(marker).get('ledger_sha256') == sha(path):
        print(key, 'pilot already complete', flush=True)
        return
    if path.exists():
        raw = path.read_bytes()
        if raw and not raw.endswith(b'\n'):
            path.write_bytes(raw[:raw.rfind(b'\n')+1])
    previous = read_rows(path) if path.exists() else []
    done = {r['item_id'] for r in previous}
    if len(done) != len(previous) or not done <= {r['item_id'] for r in manifest}:
        raise ValueError('Pilot coverage or identities invalid')
    with gpu_lock():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < 21*1024**3:
            raise RuntimeError('Pilot needs an otherwise idle 24GB GPU')
        torch.cuda.set_per_process_memory_fraction(min(1.0, 23*1024**3/total_bytes))
        torch.cuda.reset_peak_memory_stats()
        stop = threading.Event()
        worker = threading.Thread(target=memory_worker, args=(stop, output/'nvidia-smi.jsonl', key), daemon=True)
        worker.start()
        start = time.monotonic()
        try:
            backend = Backend(key, precision='bf16')
            devices = {str(v) for v in getattr(backend.model, 'hf_device_map', {}).values()}
            if devices - {'0', 'cuda', 'cuda:0'}:
                raise RuntimeError('This pilot forbids hidden CPU/disk weight offloading')
            with path.open('a', encoding='utf-8', newline='\n') as stream:
                for item in manifest:
                    if item['item_id'] in done:
                        continue
                    tick = time.monotonic()
                    raw = backend.generate(item)
                    row = {'checkpoint': key, 'axis': item['axis'], 'item_id': item['item_id'],
                           'backend': backend.backend_version, 'numerics_checked': True,
                           **score_generation(raw, item), 'elapsed_seconds': time.monotonic()-tick,
                           'usage': getattr(backend, 'last_usage', {})}
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush()
                    done.add(item['item_id'])
                    if len(done)%25 == 0 or len(done)==len(manifest):
                        print(key, len(done), '/', len(manifest), flush=True)
            rows = read_rows(path)
            if limit is None:
                scored = units(manifest, rows)
                def means(records):
                    result = {}
                    for axis in AXES:
                        strata = sorted({r['stratum'] for r in records if r['axis']==axis})
                        result[axis] = sum(sum(r['correct_bit'] for r in records if r['axis']==axis and r['stratum']==s)/sum(r['axis']==axis and r['stratum']==s for r in records) for s in strata)/len(strata)
                    return result
                axis_scores = means(scored)
                old = [r for p in sorted((ROOT/'runs/formal-20261014/ledger').glob(key+'-[0-9][0-9][0-9].jsonl')) for r in read_rows(p) if r['item_id'] in done]
                legacy_units = units(manifest, old)
                legacy_scores = means(legacy_units)
                comparison = {'legacy_scores_on_same_items': legacy_scores,
                              'bf16_minus_legacy': {a: axis_scores[a]-legacy_scores[a] for a in AXES},
                              'changed_scoring_units': sum(a['correct_bit']!=b['correct_bit'] for a,b in zip(scored,legacy_units)),
                              'parse_failures_by_axis': {a: sum(r['parse_fail_flag'] for r in scored if r['axis']==a) for a in AXES}}
            else:
                axis_scores = None
                comparison = None
            prior_peak = read_json(output/'failed.json').get('peak_allocated_bytes', 0) if (output/'failed.json').exists() else 0
            write_json(marker, {'status': 'complete', **metadata, 'rows': len(rows), 'axis_scores': axis_scores,
                                'ledger_sha256': sha(path), 'elapsed_this_invocation_seconds': time.monotonic()-start,
                                'peak_allocated_bytes': max(prior_peak, torch.cuda.max_memory_allocated()),
                                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                                'comparison': comparison,
                                'actual_language_dtype': str(next(backend.model.language_model.parameters()).dtype) if hasattr(backend.model, 'language_model') else str(next(backend.model.parameters()).dtype),
                                'actual_visual_dtype': str(next((backend.model.visual if hasattr(backend.model,'visual') else backend.model.vision_model if hasattr(backend.model,'vision_model') else backend.model.vision_tower).parameters()).dtype),
                                'gpu': torch.cuda.get_device_name(), 'total_memory_bytes': total_bytes,
                                'is_formal_training_or_confirmation': False})
        except Exception as exc:
            write_json(output/'failed.json', {'status': 'failed', 'error': type(exc).__name__,
                                             'message': str(exc), 'completed_rows': len(done),
                                             'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
            raise
        finally:
            stop.set()
            worker.join(timeout=6)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS))
    parser.add_argument('--per-axis', type=int, default=100)
    parser.add_argument('--limit-per-axis', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        print(make_manifest(args.per_axis))
    elif args.model:
        pilot(args.model, args.per_axis, args.limit_per_axis)
    else:
        parser.error('--model or --prepare-only is required')
