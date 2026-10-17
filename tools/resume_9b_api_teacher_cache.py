"""User-authorized diagnostic/resume; never request an already cached item.

The original cache generator and protocol remain immutable. Records a versioned
amendment, exact diagnostic request and raw response; no blind retry loop.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import time

import requests

import prepare_9b_api_teacher_cache as original
from experiment_common import ROOT, read_json, read_rows, sha, write_json

OUT = original.OUT


def worker():
    with (OUT / 'teacher-worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read_json(OUT / 'teacher-protocol.json')
        assert protocol['source_sha256'] == sha(ROOT / 'tools/prepare_9b_api_teacher_cache.py')
        config = read_json(ROOT / 'configs/api_teacher.json')
        base = os.environ.get('base_url') or config['base_url']
        assert base.rstrip('/') == config['base_url'].rstrip('/')
        assert config['model'] == original.MODEL
        token = os.environ.get(config['api_key_env'])
        if not token:
            raise RuntimeError('API credential absent; do not print credentials')
        pool = read_rows(ROOT / 'data/training/pool.jsonl')
        assert sha(ROOT / 'data/training/pool.jsonl') == protocol['pool_sha256']
        ledger, inflight = OUT / 'teacher.jsonl', OUT / 'teacher-api-inflight.json'
        previous = read_rows(ledger)
        cached = {r['item_id']: r for r in previous}
        assert len(cached) == len(previous)
        amendment = OUT / 'resume-v2'
        amendment.mkdir(exist_ok=True)
        design = {'authorization': 'User requested HTTP400 diagnosis, request/response preservation and resume without duplicate cached calls',
                  'original_protocol_sha256': sha(OUT / 'teacher-protocol.json'),
                  'source_sha256': sha(ROOT / 'tools/resume_9b_api_teacher_cache.py'),
                  'initial_cached_rows': 1307, 'no_retry_of_cached_items': True,
                  'failure_policy': 'One diagnostic attempt for the known HTTP400 rejected item; stop on any new HTTP/transport error',
                  'original_response_body_available': False,
                  'original_evidence': 'HTTP400 status, item_id and timestamp only; exact request can be reconstructed, original response cannot',
                  'max_new_requests_including_historical_failure': original.MAX_REQUESTS,
                  'max_new_usage_tokens': original.MAX_TOKENS}
        path = amendment / 'protocol.json'
        if path.exists():
            assert read_json(path) == design
        else:
            write_json(path, design)
        known_failure = read_json(OUT / 'teacher-api-failure.json')
        if inflight.exists():
            pending = read_json(inflight)
            if pending['item_id'] not in cached:
                assert known_failure['http_status'] == 400 and pending['item_id'] == known_failure['item_id']
                if (amendment / 'diagnostic-attempt-started.json').exists():
                    raise RuntimeError('Diagnostic already attempted; inspect raw records before any further call')
                write_json(amendment / 'original-inflight.json', pending)
            inflight.unlink()
        used = sum(r['usage']['total_tokens'] for r in previous if not r.get('reused_source'))
        calls = sum(not r.get('reused_source') for r in previous) + 1  # historical HTTP400 attempt
        session = requests.Session()
        session.trust_env = False
        session.headers['Authorization'] = 'Bearer ' + token
        reused_source = ROOT / 'runs/protocol-v2/calibration/p14-full-7b-v2/teacher.jsonl'
        reusable = {r['item_id']: r for r in read_rows(reused_source)}
        for row in pool:
            item_id = row['item_id']
            prompt = original.prompt_for(row)
            if item_id in cached:
                # Existing provenance was checked at creation. Verify source
                # image bytes again without constructing a new API request.
                entry = cached[item_id]
                assert entry['actual_prompt'] == prompt['question']
                assert entry['finish_reason'] == 'stop' and entry['returned_model'] == original.MODEL
                assert all(sha(ROOT / x['path']) == x['sha256'] for x in entry['image_hashes'])
                continue
            blocks, hashes = original.content(prompt)
            candidate = reusable.get(item_id)
            if candidate and candidate['actual_prompt'] == prompt['question'] and candidate['image_hashes'] == hashes:
                original.valid(candidate, row, prompt, hashes)
                entry = {**candidate, 'reused_source': str(reused_source.relative_to(ROOT)),
                         'reused_source_sha256': sha(reused_source)}
                original.append(ledger, entry)
                cached[item_id] = entry
                continue
            if calls >= original.MAX_REQUESTS or used + 16384 > original.MAX_TOKENS:
                raise RuntimeError('Original cumulative request/token cap reached')
            request = {'model': original.MODEL, 'max_tokens': 128, 'temperature': 0,
                       'enable_thinking': False, 'messages': [{'role': 'user', 'content': blocks}]}
            digest = hashlib.sha256(item_id.encode()).hexdigest()
            folder = amendment / 'requests' / digest
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / 'request.json').exists():
                raise RuntimeError('Existing uncached attempt requires inspection; not blindly retried')
            started = datetime.now(timezone.utc).isoformat()
            write_json(folder / 'request.json', {'item_id': item_id, 'url': base.rstrip('/') + '/chat/completions',
                       'started_utc': started, 'payload': request, 'image_hashes': hashes,
                       'authorization_header_omitted': True})
            write_json(inflight, {'item_id': item_id, 'started_utc': started, 'request_record': str(folder.relative_to(ROOT))})
            if item_id == known_failure['item_id']:
                write_json(amendment / 'diagnostic-attempt-started.json', {'item_id': item_id, 'started_utc': started})
            calls += 1
            t0 = time.monotonic()
            response = session.post(base.rstrip('/') + '/chat/completions', json=request, timeout=120)
            write_json(folder / 'response.json', {'http_status': response.status_code,
                       'headers': {k: v for k, v in response.headers.items()
                                   if k.lower() in ('content-type', 'x-request-id', 'request-id', 'x-dashscope-request-id')},
                       'body': response.text, 'seconds': time.monotonic() - t0})
            if response.status_code != 200:
                write_json(amendment / 'failure.json', {'item_id': item_id, 'http_status': response.status_code,
                           'request_record': str(folder.relative_to(ROOT)), 'cached_rows': len(cached),
                           'duplicate_cached_requests': 0})
                raise RuntimeError('Diagnostic/resume HTTP failure; exact raw request and response preserved')
            body = response.json()
            choice = body['choices'][0]
            entry = {'item_id': item_id, 'axis': row['axis'], 'actual_prompt': prompt['question'],
                     'raw_generation': choice['message'].get('content') or '', 'returned_model': body.get('model'),
                     'finish_reason': choice.get('finish_reason'), 'usage': body['usage'], 'image_hashes': hashes,
                     'request_id': body.get('id'), 'started_utc': started, 'seconds': time.monotonic() - t0,
                     'request_record': str(folder.relative_to(ROOT))}
            original.append(ledger, entry)
            cached[item_id] = entry
            used += entry['usage']['total_tokens']
            inflight.unlink()
            original.valid(entry, row, prompt, hashes)
            write_json(OUT / 'teacher-status.json', {'rows': len(cached), 'expected': 3545,
                       'new_requests_including_historical_failure': calls, 'new_usage_tokens': used,
                       'pid': os.getpid(), 'gpu_used': False, 'duplicate_cached_requests': 0,
                       'resume_protocol': str(path.relative_to(ROOT))})
            if len(cached) % 50 == 0:
                print('Teacher resume', len(cached), '/3545', flush=True)
        assert len(cached) == 3545
        write_json(OUT / 'teacher-complete.json', {'status': 'passed', 'rows': 3545,
                   'new_usage_tokens': used, 'ledger_sha256': sha(ledger),
                   'protocol_sha256': sha(OUT / 'teacher-protocol.json'),
                   'resume_protocol_sha256': sha(path), 'duplicate_cached_requests': 0,
                   'student_tokenization_pending': True})


if __name__ == '__main__':
    try:
        worker()
    except Exception as error:
        write_json(OUT / 'resume-v2-stopped.json', {'type': type(error).__name__, 'message': str(error),
                   'timestamp': datetime.now(timezone.utc).isoformat(), '7b_experiment_affected': False})
        raise
