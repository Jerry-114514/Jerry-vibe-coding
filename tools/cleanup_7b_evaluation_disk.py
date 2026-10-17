"""User-requested cleanup: unused3B shards and server-verified old preflight copies.

Never selects route-B checkpoints, input data, 7B weights, or 9B weights.
Only explicit files are unlinked; all provenance and restore pointers are kept.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import requests

from experiment_common import ROOT, read_json, sha, write_json

OLD = ROOT / 'runs/protocol-v2-snapshots'
CALIBRATION = ROOT / 'runs/protocol-v2/calibration'
DESTINATION = ROOT / 'runs/route-b-staged-7b-20261016/disk-cleanup-20261017'


def ensure_not_open(paths):
    selected = {str(p.resolve()) for p in paths}
    for process in Path('/proc').glob('[0-9]*'):
        try:
            mapped = (process / 'maps').read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            mapped = ''
        mapped_paths = {line.split(maxsplit=5)[5] for line in mapped.splitlines() if len(line.split(maxsplit=5)) == 6}
        assert not selected & mapped_paths, 'Selected cache is mapped in process ' + process.name
        try:
            descriptors = list((process / 'fd').iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError):
                continue
            assert target not in selected, 'Selected cache is open in process ' + process.name


def plan(token):
    assert ROOT.resolve() == Path('/root/autodl-tmp/arp').resolve(), 'Remote project only'
    session = requests.Session()
    session.trust_env = False
    session.headers.update({'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json'})
    endpoint = 'https://api.github.com/repos/Jerry-114514/Jerry-vibe-coding'
    releases, covered = {}, {}
    receipts = list(OLD.glob('*/github-storage.json'))
    receipts += [CALIBRATION / name / 'github-storage.json'
                 for name in ('p14-full-7b-v2', 'p14-full-7b-5090-v1')]
    for path in receipts:
        if not path.exists():
            continue
        receipt = read_json(path)
        url = receipt['release_url']
        tag = url.rsplit('/', 1)[-1]
        assert url.startswith('https://github.com/Jerry-114514/Jerry-vibe-coding/releases/tag/')
        if url not in releases:
            response = session.get(endpoint + '/releases/tags/' + tag, timeout=30)
            response.raise_for_status()
            release = response.json()
            assert release['html_url'] == url
            assets = {}
            for page in range(1, 100):
                response = session.get(endpoint + f'/releases/{release["id"]}/assets',
                                       params={'page': page, 'per_page': 100}, timeout=30)
                response.raise_for_status()
                batch = response.json()
                assets.update({a['name']: a for a in batch})
                if len(batch) < 100:
                    break
            else:
                raise RuntimeError('Incomplete server asset listing')
            releases[url] = assets
        inventory_path = path.with_name('github-inventory.json')
        inventory = read_json(inventory_path)
        assert sha(inventory_path) == next(a['sha256'] for a in receipt['assets'] if a['name'].endswith('-inventory.json'))
        for item in receipt['assets']:
            actual = releases[url][item['name']]
            assert actual['size'] == item['bytes'] and actual.get('digest') == 'sha256:' + item['sha256']
        for entry in inventory['files']:
            covered.setdefault((entry['bytes'], entry['sha256']), {'release_url': url,
                'assets': receipt['assets'], 'archive_member': entry['path'], 'server_digest_verified': True})
    records = []
    weights = ROOT / '.model_cache/qwen3b'
    lock = read_json(weights / 'weights.verified.json')
    for entry in lock['files']:
        path = weights / entry['file']
        if not path.exists():
            continue
        assert path.suffix == '.safetensors' and path.parent.resolve() == weights.resolve()
        assert path.stat().st_nlink == 1, 'Shared inode requires separate accounting'
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        records.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': entry['bytes'], 'sha256': entry['sha256'],
            'allocated_bytes': path.stat().st_blocks * 512,
            'reason': 'Unused public3B weight cache; all7B/9B weights retained',
            'restore': {'repository': 'Qwen/Qwen2.5-VL-3B-Instruct', 'revision': lock['revision'],
                        'file': entry['file'], 'verify_sha256': entry['sha256'],
                        'source': entry['source']}})
    candidates = [p for parent in (OLD, CALIBRATION) for p in parent.rglob('*')
                  if p.is_file() and p.suffix in ('.pt', '.safetensors')]
    for path in sorted(candidates):
        assert not path.is_symlink() and path.resolve().is_relative_to(ROOT.resolve())
        if path.stat().st_nlink != 1:
            continue
        digest, size = sha(path), path.stat().st_size
        if (size, digest) not in covered:
            continue
        records.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': size, 'sha256': digest,
                        'allocated_bytes': path.stat().st_blocks * 512,
                        'reason': 'Old preflight binary or duplicate snapshot; fresh server digest verified',
                        'restore': covered[(size, digest)]})
    ensure_not_open([ROOT / row['path'] for row in records])
    before = shutil.disk_usage(ROOT).free
    return {'status': 'verified_cleanup_plan', 'timestamp': datetime.now(timezone.utc).isoformat(),
            'free_before_bytes': before, 'recoverable_bytes': sum(row['allocated_bytes'] for row in records),
            'expected_free_GiB': (before + sum(row['allocated_bytes'] for row in records)) / 1024 ** 3,
            'files': records, 'kept': ['7B/9B weights', 'all route-B checkpoints/adapters/caches/raw evaluations',
                                     'all source datasets and manifests', 'old preflight metadata and SHA records']}


def cleanup(token, apply=False):
    destination = DESTINATION
    assert not (destination / 'complete.json').exists(), 'Keep previous restore record immutable; this action is already complete'
    result = plan(token)
    assert result['expected_free_GiB'] >= 20, 'Insufficient verified candidates; inspect before any deletion'
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'plan.json', result)
    if apply:
        removed = []
        for entry in result['files']:
            path = ROOT / entry['path']
            # Revalidate exact scope and bytes immediately before unlinking.
            permitted = path.parent == ROOT / '.model_cache/qwen3b' or any(path.is_relative_to(p) for p in (OLD, CALIBRATION))
            assert permitted and path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink()
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
            path.unlink()
            removed.append(entry['path'])
            with (destination / 'removed.jsonl').open('a') as journal:
                journal.write(json.dumps({'path': entry['path'], 'restore_plan': 'plan.json'}) + '\n')
        write_json(destination / 'removed.json', {'files': removed, 'restore_plan': 'plan.json'})
        free = shutil.disk_usage(ROOT).free
        result.update(status='completed', removed_files=len(removed), free_after_bytes=free, free_after_GiB=free / 1024 ** 3)
        write_json(destination / 'complete.json', result)
        assert free >= 20 * 1024 ** 3, 'Free-space target not met; do not claim completion'
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    secret = sys.stdin.readline().strip()
    assert secret, 'GitHub token required through encrypted stdin'
    cleanup(secret, apply=args.apply)
