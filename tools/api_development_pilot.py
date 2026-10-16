"""Bounded API development experiment; historical questions never become training data."""
import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests
from PIL import Image
from experiment_common import AXES, ROOT, read_json, read_rows, score_generation, sha, units, write_json, write_rows

OUT = ROOT / 'runs/protocol-v2/calibration/api-teacher-development'
NATIVE = False
VALIDATE_NATIVE = False


def freeze():
    if NATIVE:
        return freeze_native()
    source = ROOT / 'runs/protocol-v2/calibration/precision-bridge-manifest.jsonl'
    groups = defaultdict(list)
    for row in read_rows(source):
        groups[(row['axis'], row['dataset'], row['bootstrap_unit'])].append(row)
    selected = []
    for axis in AXES:
        strata = defaultdict(list)
        for key, rows in groups.items():
            if key[0] == axis:
                strata[(rows[0]['dataset'], rows[0].get('subset', ''))].append((key, rows))
        for index, stratum in enumerate(sorted(strata)):
            count = 8 // len(strata) + int(index < 8 % len(strata))
            candidates = sorted(strata[stratum], key=lambda pair: hashlib.sha256(
                ('20261016:api-development:' + str(pair[0])).encode()).hexdigest())
            if len(candidates) < count:
                raise ValueError('Insufficient scoring units')
            for _, rows in candidates[:count]:
                selected.extend(sorted(rows, key=lambda row: row['item_id']))
    if len(selected) != 64:
        raise ValueError('Expected 32 scoring units and 64 prompts')
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'manifest.jsonl'
    if manifest.exists() and read_rows(manifest) != selected:
        raise ValueError('Existing development manifest differs')
    if not manifest.exists():
        write_rows(manifest, selected)
    protocol = {'purpose': 'historical_development_only', 'confirmation': False,
                'training_data': False, 'formal_training_allowed': False,
                'scoring_units_per_axis': 8, 'prompts': 64,
                'model': 'qwen3-vl-plus', 'immutable_model_version_confirmed': False,
                'temperature': 0, 'max_output_tokens_per_request': 128,
                'max_requests': 64, 'max_cumulative_reported_tokens': 200000,
                'image_policy': 'Original pixels encoded losslessly as PNG; no manual resizing',
                'max_image_pixels': 4000000, 'max_encoded_image_bytes': 8000000,
                'top_logprobs': 5, 'top50_kd_compatible': False,
                'retry_policy': 'No automatic retries of paid calls; ambiguous in-flight calls stop resume',
                'source_manifest_sha256': sha(source), 'manifest_sha256': sha(manifest),
                'scoring': 'Frozen v1 strict parser; same prompts; equal-weight dataset strata',
                'interpretation': 'Commercial teacher development comparison, not a published checkpoint scaling ladder'}
    if (OUT / 'protocol.json').exists() and read_json(OUT / 'protocol.json') != protocol:
        raise ValueError('Frozen API protocol differs')
    write_json(OUT / 'protocol.json', protocol)
    return selected, protocol


def freeze_native():
    development = ROOT / 'runs/protocol-v2/calibration/api-teacher-development/manifest.jsonl'
    source = ROOT / 'runs/protocol-v2/calibration/precision-bridge-manifest.jsonl' if VALIDATE_NATIVE else development
    excluded = {row['item_id'] for row in read_rows(development) if row['axis'] == 'grounding'} if VALIDATE_NATIVE else set()
    original = [row for row in read_rows(source) if row['axis'] == 'grounding' and row['item_id'] not in excluded]
    selected = []
    for row in original:
        expression = row['question'].split('Locate the object referred to by: ', 1)[1].split('. Return only its bounding box', 1)[0]
        selected.append({**row, 'question': 'Locate the object referred to by: ' + expression +
                        '. Return only one JSON array [x_min, y_min, x_max, y_max], '
                        'with x and y coordinates normalized to [0, 1000] relative to the full image. '
                        'Use 0 for the left/top edge and 1000 for the right/bottom edge. Do not use pixel coordinates.'})
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.jsonl').exists() and read_rows(OUT / 'manifest.jsonl') != selected:
        raise ValueError('Native calibration manifest changed')
    expected = 92 if VALIDATE_NATIVE else 8
    if len(selected) != expected:
        raise ValueError('Native coordinate experiment has unexpected coverage')
    write_rows(OUT / 'manifest.jsonl', selected)
    protocol = {**read_json(development.parent / 'protocol.json'), 'prompts': expected, 'max_requests': expected,
                'scoring_units_per_axis': expected,
                'max_cumulative_reported_tokens': 100000 if VALIDATE_NATIVE else 30000,
                'purpose': 'held_out_native_contract_development_validation' if VALIDATE_NATIVE else 'native_coordinate_development_calibration',
                'source_manifest_sha256': sha(source), 'manifest_sha256': sha(OUT / 'manifest.jsonl'),
                'coordinate_contract': 'normalized_1000',
                'contract_source': 'https://arxiv.org/html/2511.21631v1#S3.SS2.SSS4',
                'scoring': 'Strict JSON normalized box in [0,1000], fixed x*width/1000 and y*height/1000 mapping; unconditional pixel IoU >= 0.5',
                'interpretation': 'Remaining 92 historical development units, excluding all eight API calibration units; not independent final confirmation' if VALIDATE_NATIVE else 'Same eight development units, explicit native prompt; no confirmation or scale claim'}
    if (OUT / 'protocol.json').exists() and read_json(OUT / 'protocol.json') != protocol:
        raise ValueError('Native coordinate protocol changed')
    write_json(OUT / 'protocol.json', protocol)
    write_json(OUT / 'public-provenance.json', {
        'all_inputs_from_public_benchmarks': True, 'private_user_files_included': False,
        'parent_provenance_sha256': sha(development.parent / 'public-provenance.json'),
        'public_sources': ['https://github.com/lichengunc/refer'],
        'answer_keys_sent': False})
    original_manifest = {row['item_id']: row for row in read_rows(ROOT / 'data/manifest.jsonl')}
    for row in original:
        if row['dataset'] not in ('refcoco', 'refcoco+', 'refcocog'):
            raise ValueError('Native coordinate data must come from public REFER benchmarks')
        if {key: value for key, value in row.items() if key != 'protocol_v2_role'} != original_manifest[row['item_id']]:
            raise ValueError('Public-source input does not match original frozen manifest')
    return selected, protocol


def axis_scores(records):
    scores = {}
    for axis in AXES:
        strata = defaultdict(list)
        for row in records:
            if row['axis'] == axis:
                strata[row['stratum']].append(row['correct_bit'])
        if strata:
            scores[axis] = sum(sum(values)/len(values) for values in strata.values())/len(strata)
    return scores


def worker():
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            work()
        except Exception as error:
            # Never persist exception strings: HTTP libraries can embed request details.
            write_json(OUT / 'status.json', {'status': 'failed', 'error_type': type(error).__name__,
                       'formal_training_allowed': False, 'updated_utc': datetime.now(timezone.utc).isoformat()})
            raise


def work():
    manifest, protocol = freeze()
    config = read_json(ROOT / 'configs/api_teacher.json')
    key = os.environ.get(config['api_key_env'])
    if not key:
        raise RuntimeError('Missing API key')
    base = os.environ.get('base_url') or config['base_url']
    if base.rstrip('/') != config['base_url'].rstrip('/'):
        raise ValueError('Base URL changed')
    ledger = OUT / 'ledger.jsonl'
    previous = read_rows(ledger) if ledger.exists() else []
    ids = {row['item_id'] for row in previous}
    if len(ids) != len(previous) or not ids <= {row['item_id'] for row in manifest}:
        raise ValueError('Invalid resumed ledger coverage')
    inflight = OUT / 'inflight.json'
    if inflight.exists():
        if read_json(inflight)['item_id'] not in ids:
            raise RuntimeError('Unresolved paid request; manual reconciliation needed before resume')
        inflight.unlink()
    used = sum(row['usage']['total_tokens'] for row in previous)
    session = requests.Session()
    session.trust_env = False
    session.headers['Authorization'] = 'Bearer ' + key
    for item in manifest:
        if item['item_id'] in ids:
            continue
        # A conservative allowance below the cumulative stop threshold.
        if used + 16384 > protocol['max_cumulative_reported_tokens']:
            raise RuntimeError('Development token stop threshold reached')
        path = (ROOT / item['image_path']).resolve()
        if not path.is_relative_to((ROOT / 'data').resolve()):
            raise ValueError('Image outside project data')
        with Image.open(path) as image:
            if image.width * image.height > protocol['max_image_pixels']:
                raise ValueError('Image pixel budget exceeded')
            buffer = io.BytesIO()
            image.convert('RGB').save(buffer, format='PNG')
        encoded = buffer.getvalue()
        if len(encoded) > protocol['max_encoded_image_bytes']:
            raise ValueError('Encoded image budget exceeded')
        body = {'model': protocol['model'], 'temperature': 0, 'max_tokens': 128,
                'enable_thinking': False, 'logprobs': True, 'top_logprobs': 5,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(encoded).decode()}},
                    {'type': 'text', 'text': item['question']}]}]}
        write_json(inflight, {'item_id': item['item_id'], 'started_utc': datetime.now(timezone.utc).isoformat()})
        tick = time.monotonic()
        response = session.post(base.rstrip('/') + '/chat/completions', json=body, timeout=120)
        if response.status_code != 200:
            write_json(OUT / 'http_failure.json', {'http_status': response.status_code,
                       'item_id': item['item_id'], 'automatic_retry': False})
            raise RuntimeError('API request failed; not automatically retried')
        answer = response.json()
        choice = answer['choices'][0]
        raw = choice['message']['content']
        if NATIVE:
            from coordinate_contracts import score_normalized_grounding
            scored_output = score_normalized_grounding(raw, item)
        else:
            scored_output = score_generation(raw, item)
        row = {'checkpoint': protocol['model'], 'axis': item['axis'], 'item_id': item['item_id'],
               **scored_output,
               'usage': answer['usage'], 'request_id': answer.get('id'),
               'returned_model': answer.get('model'), 'finish_reason': choice.get('finish_reason'),
               'token_logprobs': choice.get('logprobs'), 'image_file_sha256': sha(path),
               'elapsed_seconds': time.monotonic() - tick}
        with ledger.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        inflight.unlink()
        used += row['usage']['total_tokens']
        ids.add(item['item_id'])
        write_json(OUT / 'status.json', {'status': 'running', 'completed_prompts': len(ids),
                   'expected_prompts': len(manifest), 'total_reported_tokens': used,
                   'formal_training_allowed': False, 'updated_utc': datetime.now(timezone.utc).isoformat()})
        print('API development', len(ids), '/', len(manifest), 'tokens', used, flush=True)
    rows = read_rows(ledger)
    scored = units(manifest, rows)
    comparisons = {}
    for model in ('qwen3b', 'qwen7b'):
        baseline = read_rows(ROOT / 'runs/protocol-v2/calibration/precision-bridge' / model / 'ledger.jsonl')
        subset = [row for row in baseline if row['item_id'] in ids]
        comparisons[model] = axis_scores(units(manifest, subset))
    summary = {'status': 'complete', 'prompts': len(rows), 'scoring_units': len(scored),
               'axis_scores': axis_scores(scored), 'local_bf16_scores_same_items': comparisons,
               'parse_failures_by_axis': {axis: sum(row['parse_fail_flag'] for row in scored if row['axis'] == axis) for axis in AXES},
               'total_reported_tokens': used, 'ledger_sha256': sha(ledger),
               'manifest_sha256': sha(OUT / 'manifest.jsonl'), 'formal_training_allowed': False,
               'confirmation': False, 'completed_utc': datetime.now(timezone.utc).isoformat()}
    write_json(OUT / 'complete.json', summary)
    write_json(OUT / 'status.json', summary)
    from verify_api_pilot import verify
    verify(OUT)
    if os.environ.get('GH_TOKEN'):
        from publish_artifacts import publish
        publish(OUT, 'arp-protocol-v2-20261016', token=os.environ['GH_TOKEN'])
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--native-grounding', action='store_true')
    parser.add_argument('--validate-native-contract', action='store_true')
    args = parser.parse_args()
    VALIDATE_NATIVE = args.validate_native_contract
    if args.native_grounding or VALIDATE_NATIVE:
        NATIVE = True
        OUT = ROOT / 'runs/protocol-v2/calibration' / ('api-native-grounding-validation' if VALIDATE_NATIVE else 'api-native-grounding')
    if args.worker:
        worker()
    else:
        token = sys.stdin.readline().strip()
        if not token or not os.environ.get('DASHSCOPE_API_KEY'):
            raise SystemExit('Both API and backup credentials are required')
        freeze()
        sessions = subprocess.run(['screen', '-ls'], capture_output=True, text=True).stdout
        screen_name = 'arp-v2-api-native-validation' if VALIDATE_NATIVE else 'arp-v2-api-native-grounding' if NATIVE else 'arp-v2-api-development'
        if '.' + screen_name in sessions:
            raise SystemExit('Development API worker already exists')
        command = ('cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/api_development_pilot.py --worker' +
                   (' --validate-native-contract' if VALIDATE_NATIVE else ' --native-grounding' if NATIVE else '') + ' >> ' + str(OUT / 'console.log') + ' 2>&1')
        subprocess.run(['screen', '-dmS', screen_name, 'bash', '-c', command],
                       env={**os.environ, 'GH_TOKEN': token}, check=True)
        print('Started bounded ' + str(len(read_rows(OUT / 'manifest.jsonl'))) + '-prompt API development experiment with completion backup', flush=True)
