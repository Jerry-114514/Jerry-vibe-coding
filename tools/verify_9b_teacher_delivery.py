"""Independently verify eligible122B teacher coverage and current Release bytes."""
import hashlib
import json
import sys

import requests

from experiment_common import ROOT, read_json, read_rows, sha, write_json

BASE = ROOT / 'runs/route-b-9b-122b-20261017'


def verify(token):
    marker = read_json(BASE / 'teacher-eligible-complete.json')
    assert marker['status'] == 'passed_eligible_pool_only'
    assert sha(BASE / 'teacher.jsonl') == marker['teacher_ledger_sha256']
    assert sha(BASE / 'eligible-pool.jsonl') == marker['eligible_pool_sha256']
    pool = read_rows(ROOT / 'data/training/pool.jsonl')
    eligible, quarantine = read_rows(BASE / 'eligible-pool.jsonl'), read_rows(BASE / 'quarantined-pool.jsonl')
    allowed, excluded = {r['item_id'] for r in eligible}, {r['item_id'] for r in quarantine}
    assert len(allowed) == len(eligible) == marker['eligible_rows']
    assert len(excluded) == len(quarantine) == marker['excluded_rows']
    assert not allowed & excluded and allowed | excluded == {r['item_id'] for r in pool}
    cached = read_rows(BASE / 'teacher.jsonl')
    ids = [r['item_id'] for r in cached]
    assert len(ids) == len(set(ids)) and allowed <= set(ids)
    assert all(r['returned_model'] == 'qwen3.5-122b-a10b' and r['finish_reason'] == 'stop'
               and r['raw_generation'].strip() for r in cached)
    initial = read_json(BASE / 'api400-investigation/investigation.json')
    prefix = (BASE / 'teacher.jsonl').read_bytes().splitlines(keepends=True)[:1307]
    assert hashlib.sha256(b''.join(prefix)).hexdigest() == initial['initial_cached_prefix_sha256']
    initial_ids = {json.loads(line)['item_id'] for line in prefix}
    attempts = read_rows(BASE / 'resume-v3/attempts.jsonl')
    attempt_ids = [r['item_id'] for r in attempts]
    assert len(attempt_ids) == len(set(attempt_ids)) and not initial_ids & set(attempt_ids)
    receipt = read_json(BASE / 'teacher-delivery.json')
    snapshot = ROOT / receipt['snapshot']
    inventory_path = snapshot / 'github-inventory.json'
    inventory = read_json(inventory_path)
    assert sha(inventory_path) == next(a['sha256'] for a in receipt['assets'] if a['name'].endswith('-inventory.json'))
    for entry in inventory['files']:
        path = ROOT / entry['path']
        assert path.resolve().is_relative_to(ROOT.resolve())
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    session = requests.Session()
    session.trust_env = False
    session.headers.update({'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json'})
    base = 'https://api.github.com/repos/Jerry-114514/Jerry-vibe-coding'
    response = session.get(base + '/releases/tags/arp-9b-122b-20261017', timeout=30)
    response.raise_for_status()
    release = response.json()
    assets = {}
    for page in range(1, 30):
        response = session.get(base + f'/releases/{release["id"]}/assets', params={'page': page, 'per_page': 100}, timeout=30)
        response.raise_for_status()
        batch = response.json()
        assets.update({a['name']: a for a in batch})
        if len(batch) < 100:
            break
    for item in receipt['assets']:
        remote = assets[item['name']]
        assert remote['size'] == item['bytes'] and remote['digest'] == 'sha256:' + item['sha256']
    required = [BASE / name for name in ('teacher.jsonl', 'eligible-pool.jsonl', 'quarantined-pool.jsonl',
                                       'teacher-eligible-complete.json', 'resume-v3/refusals.jsonl')]
    coverage = {r['path']: r['sha256'] for r in inventory['files']}
    for path in required:
        archived = snapshot / 'files' / path.relative_to(ROOT)
        assert coverage[str(archived.relative_to(ROOT))] == sha(path)
    result = {'status': 'passed_eligible_teacher_and_release', 'original_rows': len(pool),
              'eligible_rows': len(eligible), 'quarantined_rows': len(quarantine),
              'teacher_rows_retained': len(cached), 'initial1307_prefix_unchanged': True,
              'duplicate_successful_or_initial_cached_requests': 0,
              'archive_files_verified': len(inventory['files']), 'fresh_server_assets_verified': len(receipt['assets']),
              'release_url': release['html_url'], 'teacher_ledger_sha256': marker['teacher_ledger_sha256'],
              'eligible_pool_sha256': marker['eligible_pool_sha256'], 'large_payloads_downloaded': False,
              'student_full_pool_tokenization_separate': True}
    write_json(BASE / 'teacher-delivery-verification.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    secret = sys.stdin.readline().strip()
    assert secret, 'Credential required through encrypted stdin'
    verify(secret)
