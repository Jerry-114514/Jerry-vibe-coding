"""Resume uncached teacher inputs; quarantine provider-refused source groups.

No prompt/image modification, provider switching or retries of refused images.
Original pool and all previous responses remain intact. Eligibility is frozen
before 9B training and disclosed as a configuration difference from 7B.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import time

import requests

import prepare_9b_api_teacher_cache as old
from experiment_common import ROOT, read_json, read_rows, sha, write_json, write_rows

OUT = old.OUT
RUN = OUT / 'resume-v3'


def worker():
    RUN.mkdir(parents=True, exist_ok=True)
    with (OUT / 'teacher-worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read_json(ROOT / 'configs/api_teacher.json')
        base = os.environ.get('base_url') or config['base_url']
        assert base.rstrip('/') == config['base_url'].rstrip('/') and config['model'] == old.MODEL
        token = os.environ.get(config['api_key_env'])
        assert token, 'API credential missing; never print credentials'
        pool = read_rows(ROOT / 'data/training/pool.jsonl')
        assert sha(ROOT / 'data/training/pool.jsonl') == read_json(OUT / 'teacher-protocol.json')['pool_sha256']
        by_id = {r['item_id']: r for r in pool}
        source = ROOT / 'runs/protocol-v2/calibration/p14-full-7b-v2/teacher.jsonl'
        reusable = {r['item_id']: r for r in read_rows(source)}
        cached_rows = read_rows(OUT / 'teacher.jsonl')
        cached = {r['item_id']: r for r in cached_rows}
        assert len(cached) == len(cached_rows)
        plan = {'source_sha256': sha(ROOT / 'tools/resume_9b_teacher_eligible.py'),
                'scope': 'Provider-availability eligibility amendment, not a performance-based exclusion',
                'quarantine': 'Every training prompt sharing a refused source image SHA256 or original bootstrap unit',
                'no_refused_image_retries_or_prompt_changes': True, 'skip_all_cached_item_ids': True,
                'historical_api_attempts': 1256, 'max_total_api_attempts': old.MAX_REQUESTS,
                'max_new_usage_tokens': old.MAX_TOKENS,
                'original_pool_sha256': sha(ROOT / 'data/training/pool.jsonl'),
                '7b_inputs_or_results_changed': False, '9b_sampling_must_be_refrozen_from_eligible_pool': True}
        if (RUN / 'protocol.json').exists():
            assert read_json(RUN / 'protocol.json') == plan
        else:
            write_json(RUN / 'protocol.json', plan)
        refusals = read_rows(RUN / 'refusals.jsonl') if (RUN / 'refusals.jsonl').exists() else []
        known_id = read_json(OUT / 'teacher-api-failure.json')['item_id']
        prior_response = next((OUT / 'resume-v2/requests').glob('*/response.json'))
        error = read_json(prior_response)
        assert error['http_status'] == 400 and json.loads(error['body'])['error']['code'] == 'data_inspection_failed'
        if not any(r['item_id'] == known_id for r in refusals):
            refusals.append({'item_id': known_id, 'source_image_sha256': sha(ROOT / by_id[known_id]['image_path']),
                             'bootstrap_unit': by_id[known_id]['bootstrap_unit'],
                             'response_record': str(prior_response.relative_to(ROOT)), 'code': 'data_inspection_failed'})
            write_rows(RUN / 'refusals.jsonl', refusals)

        def excluded(row):
            return any(row['bootstrap_unit'] == r['bootstrap_unit'] or
                       image_hashes[row['image_path']] == r['source_image_sha256'] for r in refusals)

        image_hashes = {r['image_path']: sha(ROOT / r['image_path']) for r in pool}
        inflight = OUT / 'teacher-api-inflight.json'
        if inflight.exists():
            pending = read_json(inflight)
            if pending['item_id'] not in cached and not excluded(by_id[pending['item_id']]):
                raise RuntimeError('Unresolved paid request; do not retry blindly')
            write_json(RUN / 'previous-inflight.json', pending)
            inflight.unlink()
        session = requests.Session()
        session.trust_env = False
        session.headers['Authorization'] = 'Bearer ' + token
        attempts_path = RUN / 'attempts.jsonl'
        attempts = read_rows(attempts_path) if attempts_path.exists() else []
        used = sum(r['usage']['total_tokens'] for r in cached_rows if not r.get('reused_source'))
        for row in pool:
            item_id = row['item_id']
            if excluded(row) or item_id in cached:
                continue
            prompt = old.prompt_for(row)
            blocks, hashes = old.content(prompt)
            prior = reusable.get(item_id)
            if prior and prior['actual_prompt'] == prompt['question'] and prior['image_hashes'] == hashes:
                old.valid(prior, row, prompt, hashes)
                entry = {**prior, 'reused_source': str(source.relative_to(ROOT)), 'reused_source_sha256': sha(source)}
            else:
                if plan['historical_api_attempts'] + len(attempts) >= old.MAX_REQUESTS or used + 16384 > old.MAX_TOKENS:
                    raise RuntimeError('Cumulative request/token cap reached')
                request = {'model': old.MODEL, 'max_tokens': 128, 'temperature': 0,
                           'enable_thinking': False, 'messages': [{'role': 'user', 'content': blocks}]}
                folder = RUN / 'requests' / hashlib.sha256(item_id.encode()).hexdigest()
                folder.mkdir(parents=True, exist_ok=True)
                if (folder / 'request.json').exists():
                    raise RuntimeError('An uncached request already exists; inspect instead of retrying')
                started = datetime.now(timezone.utc).isoformat()
                write_json(folder / 'request.json', {'item_id': item_id, 'started_utc': started,
                           'url': base.rstrip('/') + '/chat/completions', 'payload': request,
                           'image_hashes': hashes, 'authorization_header_omitted': True})
                attempt = {'item_id': item_id, 'started_utc': started, 'request_record': str(folder.relative_to(ROOT))}
                old.append(attempts_path, attempt)
                attempts.append(attempt)
                write_json(inflight, attempt)
                t0 = time.monotonic()
                response = session.post(base.rstrip('/') + '/chat/completions', json=request, timeout=120)
                write_json(folder / 'response.json', {'http_status': response.status_code,
                           'request_id': response.headers.get('x-request-id'), 'body': response.text,
                           'seconds': time.monotonic() - t0})
                body = response.json()
                if response.status_code == 400 and body.get('error', {}).get('code') == 'data_inspection_failed':
                    refusal = {'item_id': item_id, 'source_image_sha256': image_hashes[row['image_path']],
                               'bootstrap_unit': row['bootstrap_unit'], 'code': 'data_inspection_failed',
                               'response_record': str((folder / 'response.json').relative_to(ROOT))}
                    old.append(RUN / 'refusals.jsonl', refusal)
                    refusals.append(refusal)
                    inflight.unlink()
                    continue
                if response.status_code != 200:
                    raise RuntimeError('Unexpected API failure; full request/response preserved, no automatic retry')
                choice = body['choices'][0]
                entry = {'item_id': item_id, 'axis': row['axis'], 'actual_prompt': prompt['question'],
                         'raw_generation': choice['message'].get('content') or '', 'returned_model': body.get('model'),
                         'finish_reason': choice.get('finish_reason'), 'usage': body['usage'], 'image_hashes': hashes,
                         'request_id': body.get('id'), 'started_utc': started, 'seconds': time.monotonic() - t0,
                         'request_record': str(folder.relative_to(ROOT))}
                used += entry['usage']['total_tokens']
                old.append(OUT / 'teacher.jsonl', entry)
                cached[item_id] = entry
                inflight.unlink()
                old.valid(entry, row, prompt, hashes)
                entry = None
            if entry is not None:
                old.append(OUT / 'teacher.jsonl', entry)
                cached[item_id] = entry
            eligible = [r for r in pool if not excluded(r)]
            write_json(OUT / 'teacher-status.json', {'rows_retained': len(cached), 'original_pool_rows': 3545,
                       'eligible_expected': len(eligible), 'eligible_cached': sum(r['item_id'] in cached for r in eligible),
                       'quarantined_rows': 3545 - len(eligible), 'new_usage_tokens': used,
                       'total_attempts': plan['historical_api_attempts'] + len(attempts),
                       'duplicate_cached_requests': 0, 'gpu_used': False, 'pid': os.getpid(), 'version': 'v3_eligible'})
            if len(cached) % 50 == 0:
                print('Eligible teacher cache retained', len(cached), 'eligible target', len(eligible), flush=True)
        eligible = [r for r in pool if not excluded(r)]
        assert all(r['item_id'] in cached for r in eligible)
        for row in eligible:
            prompt = old.prompt_for(row)
            _, hashes = old.content(prompt)
            old.valid(cached[row['item_id']], row, prompt, hashes)
        write_rows(OUT / 'eligible-pool.jsonl', eligible)
        write_rows(OUT / 'quarantined-pool.jsonl', [r for r in pool if excluded(r)])
        write_json(OUT / 'teacher-eligible-complete.json', {'status': 'passed_eligible_pool_only',
                   'original_rows': 3545, 'eligible_rows': len(eligible), 'excluded_rows': 3545 - len(eligible),
                   'cache_rows_retained': len(cached), 'duplicate_cached_requests': 0,
                   'eligible_pool_sha256': sha(OUT / 'eligible-pool.jsonl'),
                   'teacher_ledger_sha256': sha(OUT / 'teacher.jsonl'),
                   'protocol_sha256': sha(RUN / 'protocol.json'), 'new_usage_tokens': used,
                   'does_not_claim_all3545_teachers_available': True, 'student_tokenization_pending': True})


if __name__ == '__main__':
    try:
        worker()
    except Exception as error:
        write_json(OUT / 'resume-v3-stopped.json', {'type': type(error).__name__, 'message': str(error),
                   'timestamp': datetime.now(timezone.utc).isoformat(), '7b_experiment_affected': False})
        raise
