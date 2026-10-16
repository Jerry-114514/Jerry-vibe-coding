"""Verify current datasets are recoverable from the historical GitHub release.

CPU only; receives GitHub credentials through encrypted stdin. No large downloads.
"""
import json
import sys
from pathlib import PurePosixPath

import requests

from experiment_common import ROOT, read_json, read_rows, sha, write_json


def main():
    token = sys.stdin.readline().strip()
    if not token:
        raise RuntimeError('GitHub credential required through encrypted stdin')
    snapshot = ROOT / 'runs/github-snapshots/formal-20261014-20261016T003911Z'
    inventory_path = snapshot / 'github-inventory.json'
    receipt = read_json(snapshot / 'github-storage.json')
    inventory = read_json(inventory_path)
    assert sha(inventory_path) == next(
        a['sha256'] for a in receipt['assets'] if a['name'].endswith('-inventory.json'))
    indexed = {}
    for entry in inventory['files']:
        path = entry['path']
        if '/frozen-inputs/' in path:
            path = path.split('/frozen-inputs/', 1)[1]
        if path.startswith('data/'):
            indexed.setdefault(path, []).append(entry)
    required = {
        'data/manifest.jsonl', 'data/assets.sha256.jsonl', 'data/experiment.lock.json',
        'data/training/pool.jsonl', 'data/training/pool.lock.json',
        'data/training/assets.jsonl',
    }
    counts = {}
    for name, path in [('training', 'data/training/pool.jsonl'),
                       ('evaluation', 'data/manifest.jsonl')]:
        rows = read_rows(ROOT / path)
        images = {r['image_path'] for r in rows}
        for image in images:
            relative = PurePosixPath(image)
            assert not relative.is_absolute() and '..' not in relative.parts
        required.update(images)
        counts[name] = {'rows': len(rows), 'referenced_images': len(images)}
    missing, mismatched = [], []
    for relative in sorted(required):
        if relative not in indexed:
            missing.append(relative)
            continue
        current = ROOT / relative
        size, digest = current.stat().st_size, sha(current)
        if not any(e['bytes'] == size and e['sha256'] == digest for e in indexed[relative]):
            mismatched.append(relative)
    assert not missing and not mismatched, {'missing': missing, 'mismatched': mismatched}
    session = requests.Session()
    session.trust_env = False
    session.headers.update({'Authorization': 'Bearer ' + token,
                            'Accept': 'application/vnd.github+json'})
    base = 'https://api.github.com/repos/Jerry-114514/Jerry-vibe-coding'
    response = session.get(base + '/releases/tags/arp-formal-20261014', timeout=30)
    response.raise_for_status()
    release = response.json()
    assets = {}
    for page in range(1, 30):
        response = session.get(base + f'/releases/{release["id"]}/assets',
                               params={'per_page': 100, 'page': page}, timeout=30)
        response.raise_for_status()
        batch = response.json()
        assets.update({a['name']: a for a in batch})
        if len(batch) < 100:
            break
    verified = []
    for entry in receipt['assets']:
        remote = assets[entry['name']]
        assert remote['size'] == entry['bytes']
        assert remote.get('digest') == 'sha256:' + entry['sha256']
        verified.append({'name': entry['name'], 'bytes': entry['bytes'],
                         'sha256': entry['sha256'], 'server_digest_verified': True})
    result = {'status': 'passed', 'scope': 'Current training/evaluation inputs only',
              'datasets': counts, 'required_files_verified': len(required),
              'all_current_file_bytes_and_sha256_match_archived_inventory': True,
              'archived_inventory_sha256': sha(inventory_path),
              'release_url': release['html_url'], 'assets': verified,
              'large_payloads_downloaded': False,
              'note': 'Model weights and new EOS caches have separate delivery records.'}
    write_json(ROOT / 'runs/route-b-staged-7b-20261016/dataset-delivery-verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
