"""Independent 9B+122B response cache; CPU/API only, bounded and resumable.

Never reuses 7B token IDs/logits. Reuses earlier API responses only after exact
prompt and transmitted-image provenance checks. Ambiguous paid calls stop.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
import sys
import time

import requests

from experiment_common import ROOT, read_json, read_rows, sha, write_json
from run_teacher_role_calibration import content

OUT = ROOT / 'runs/route-b-9b-122b-20261017'
MODEL = 'qwen3.5-122b-a10b'
MAX_REQUESTS = 3545
MAX_TOKENS = 5_000_000


def prompt_for(row):
    prompt = dict(row)
    if row['axis'] == 'grounding':
        expression = row['question'].split('Locate the object referred to by: ', 1)[1].split('. Return only', 1)[0]
        prompt['question'] = ('Locate the object referred to by: ' + expression +
            '. Return only a JSON array containing one object with bbox_2d and label. '
            'Normalize [x_min,y_min,x_max,y_max] coordinates to [0,1000] relative to the full image.')
    return prompt


def valid(entry, row, prompt, hashes):
    assert entry['item_id'] == row['item_id'] and entry['axis'] == row['axis']
    assert entry['actual_prompt'] == prompt['question'] and entry['image_hashes'] == hashes
    assert entry['returned_model'] == MODEL and entry['finish_reason'] == 'stop'
    assert entry['raw_generation'].strip() and entry['usage']['total_tokens'] > 0
    if row['axis'] == 'grounding':
        from experiment_common import parse_answer
        box = parse_answer(entry['raw_generation'], {'answer_type': 'bbox', 'width': 1000, 'height': 1000})
        assert box is not None and len(box) == 4
        assert all(0 <= x <= 1000 for x in box) and box[0] < box[2] and box[1] < box[3]


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def worker():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'teacher-worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read_json(ROOT / 'configs/api_teacher.json')
        assert config['model'] == MODEL
        base = os.environ.get('base_url') or config['base_url']
        assert base.rstrip('/') == config['base_url'].rstrip('/')
        key = os.environ.get(config['api_key_env'])
        if not key:
            raise RuntimeError('Teacher API credential missing; do not print or copy it')
        pool = read_rows(ROOT / 'data/training/pool.jsonl')
        assert len(pool) == 3545 and len({r['item_id'] for r in pool}) == 3545
        design = {'purpose': 'Independent 9B student +122B API teacher response cache',
                  'not_original_arp': True, 'student': 'Qwen/Qwen3.5-9B', 'teacher': MODEL,
                  'max_new_requests': MAX_REQUESTS, 'max_new_usage_tokens': MAX_TOKENS,
                  'max_response_tokens': 128, 'temperature': 0, 'enable_thinking': False,
                  'pool_sha256': sha(ROOT / 'data/training/pool.jsonl'),
                  'source_sha256': sha(ROOT / 'tools/prepare_9b_api_teacher_cache.py'),
                  'coordinate_contract': 'Teacher normalized1000; student targets explicitly rescaled to original image pixels',
                  'reuse': 'Only exact prompt/model/transmitted image bytes; no 7B token/logit reuse',
                  'ambiguous_paid_call': 'Stop without retry; preserve inflight marker'}
        design_path = OUT / 'teacher-protocol.json'
        if design_path.exists():
            assert read_json(design_path) == design
        else:
            write_json(design_path, design)
        ledger, inflight = OUT / 'teacher.jsonl', OUT / 'teacher-api-inflight.json'
        previous = read_rows(ledger) if ledger.exists() else []
        cached = {r['item_id']: r for r in previous}
        assert len(cached) == len(previous)
        if inflight.exists():
            if read_json(inflight)['item_id'] not in cached:
                raise RuntimeError('Ambiguous paid request pending; no automatic paid retry')
            inflight.unlink()
        source = ROOT / 'runs/protocol-v2/calibration/p14-full-7b-v2/teacher.jsonl'
        reuse = {r['item_id']: r for r in read_rows(source)} if source.exists() else {}
        session = requests.Session()
        session.trust_env = False
        session.headers['Authorization'] = 'Bearer ' + key
        spent = sum(r['usage']['total_tokens'] for r in previous if not r.get('reused_source'))
        requests_made = sum(not r.get('reused_source') for r in previous)
        for row in pool:
            prompt = prompt_for(row)
            blocks, hashes = content(prompt)
            item_id = row['item_id']
            if item_id in cached:
                valid(cached[item_id], row, prompt, hashes)
                continue
            entry = reuse.get(item_id)
            if entry and entry['actual_prompt'] == prompt['question'] and entry['image_hashes'] == hashes:
                valid(entry, row, prompt, hashes)
                entry = {**entry, 'reused_source': str(source.relative_to(ROOT)),
                         'reused_source_sha256': sha(source)}
            else:
                if requests_made >= MAX_REQUESTS or spent + 16384 > MAX_TOKENS:
                    raise RuntimeError('Frozen teacher API request/token cap reached')
                started = datetime.now(timezone.utc).isoformat()
                write_json(inflight, {'item_id': item_id, 'started_utc': started})
                t0 = time.monotonic()
                response = session.post(base.rstrip('/') + '/chat/completions',
                    json={'model': MODEL, 'max_tokens': 128, 'temperature': 0,
                          'enable_thinking': False, 'messages': [{'role': 'user', 'content': blocks}]}, timeout=120)
                if response.status_code != 200:
                    write_json(OUT / 'teacher-api-failure.json', {'item_id': item_id,
                               'http_status': response.status_code, 'no_automatic_retry': True})
                    raise RuntimeError('Teacher API request failed; preserved inflight marker')
                body = response.json()
                choice = body['choices'][0]
                entry = {'item_id': item_id, 'axis': row['axis'], 'actual_prompt': prompt['question'],
                         'raw_generation': choice['message'].get('content') or '',
                         'returned_model': body.get('model'), 'finish_reason': choice.get('finish_reason'),
                         'usage': body['usage'], 'image_hashes': hashes, 'request_id': body.get('id'),
                         'started_utc': started, 'seconds': time.monotonic() - t0}
                # Preserve even truncated/invalid answers, so failures do not lead
                # to duplicate billing or selective retries.
                append(ledger, entry)
                cached[item_id] = entry
                requests_made += 1
                spent += entry['usage']['total_tokens']
                inflight.unlink()
                valid(entry, row, prompt, hashes)
                entry = None
            if entry is not None:
                append(ledger, entry)
                cached[item_id] = entry
            write_json(OUT / 'teacher-status.json', {'rows': len(cached), 'expected': 3545,
                       'new_requests': requests_made, 'new_usage_tokens': spent,
                       'pid': os.getpid(), 'gpu_used': False})
            if len(cached) % 50 == 0:
                print('9B teacher responses', len(cached), '/3545', flush=True)
        assert len(cached) == 3545
        write_json(OUT / 'teacher-complete.json', {'status': 'passed', 'rows': len(cached),
                   'new_requests': requests_made, 'new_usage_tokens': spent,
                   'reused_rows': len(cached) - requests_made, 'ledger_sha256': sha(ledger),
                   'protocol_sha256': sha(design_path), 'student_tokenization_pending': True})


if __name__ == '__main__':
    try:
        worker()
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / 'teacher-stopped.json', {'type': type(error).__name__,
                   'message': str(error), 'timestamp': datetime.now(timezone.utc).isoformat(),
                   '7b_experiment_affected': False})
        raise
