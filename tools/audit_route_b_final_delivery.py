"""Verify the complete route-B artifact set across incremental Release assets.

This audits published inventories and fresh GitHub digests, not just receipts.
It never uploads artifacts or starts experiments. Credentials arrive on stdin.
"""
import argparse
from datetime import datetime, timezone
import json
import sys

import requests

from experiment_common import ROOT, read_json, sha, write_json
from route_b_staged import OUT, CACHE, SEEDS

TAG = 'arp-protocol-v2-20261016'
DELIVERY = ROOT / 'runs/route-b-staged-delivery-20261016'
NINE = ROOT / 'runs/qwen35-9b-infrastructure-20261016'


def requirements():
    paths = [OUT / name for name in (
        'protocol.json', 'phase2-frozen.json', 'minimal-acceptance.json',
        'minimal-independent-verification.json', 'minimal-report.json',
        'full-eos-cache-independent-verification.json',
        'eval-input-independent-preflight.json', 'dataset-delivery-verification.json',
        'evaluation-aggregate-independent-verification.json',
        'requested-report-plan.json', 'stage3-complete.json',
        'formal-independent-verification.json', 'formal-paired-analysis.json',
        'objective-content-verification.json')]
    paths += [CACHE / 'complete.json']
    if (CACHE / 'complete.json').exists():
        paths += [CACHE / name for name in read_json(CACHE / 'complete.json')['files']]
    for phase, seeds in [('minimal', [42]), ('formal', SEEDS)]:
        for seed in seeds:
            for arm in ('A1', 'A2', 'U'):
                folder = OUT / phase / f'{arm}-seed{seed}'
                paths += [folder / name for name in (
                    'complete.json', 'train.meta.json', 'loss.jsonl', 'resume.pt',
                    'adapter/adapter_config.json', 'adapter/adapter_model.safetensors',
                    'scores.json', 'eval.jsonl')]
                if phase == 'formal':
                    paths.append(folder / 'independent-training-verification.json')
        paths += [OUT / phase / 'N-seed42' / name for name in ('scores.json', 'eval.jsonl')]
    paths += [OUT / 'completion-reports' / name for name in (
        'training/training-results.json', 'training/training-report.md',
        'training/loss-curves.png', 'evaluation/evaluation-results.json',
        'evaluation/evaluation-report.md', 'multiplicity/reporting-amendment.json',
        'multiplicity/multiplicity-results.json', 'multiplicity/multiplicity-report.md')]
    paths += [NINE / name for name in (
        'model.lock.json', 'weights-ready.json', 'environment.lock.json',
        'environment-ready.json', 'architecture-inventory.json',
        'gpu-smoke.json', 'infrastructure-complete.json')]
    return paths


def canonical(path):
    # Incremental snapshots retain original paths under their /files/ subtree.
    prefix = str(DELIVERY.relative_to(ROOT)) + '/'
    if path.startswith(prefix) and '/files/' in path:
        return path.split('/files/', 1)[1]
    return path


def audit(token, strict=False):
    session = requests.Session()
    session.trust_env = False
    session.headers.update({'Authorization': 'Bearer ' + token,
                            'Accept': 'application/vnd.github+json'})
    base = 'https://api.github.com/repos/Jerry-114514/Jerry-vibe-coding'
    response = session.get(base + '/releases/tags/' + TAG, timeout=30)
    response.raise_for_status()
    release = response.json()
    assets = {}
    for page in range(1, 100):
        response = session.get(base + f'/releases/{release["id"]}/assets',
                               params={'per_page': 100, 'page': page}, timeout=30)
        response.raise_for_status()
        batch = response.json()
        assets.update({a['name']: a for a in batch})
        if len(batch) < 100:
            break
    folders = {OUT, CACHE, NINE}
    folders.update(p.parent for p in OUT.rglob('github-storage.json'))
    if DELIVERY.exists():
        # Ignore receipts copied *inside* snapshot files: the real receipt is
        # the direct child of each delivery directory.
        folders.update(p.parent for p in DELIVERY.glob('*/github-storage.json'))
    covered, checked = {}, {}
    for folder in sorted(folders):
        receipt_path, inventory_path = folder / 'github-storage.json', folder / 'github-inventory.json'
        if not receipt_path.exists() or not inventory_path.exists():
            continue
        receipt = read_json(receipt_path)
        if receipt['release_url'] != release['html_url']:
            continue
        inventory_sha = sha(inventory_path)
        inventory_asset = next(a for a in receipt['assets'] if a['name'].endswith('-inventory.json'))
        assert inventory_asset['sha256'] == inventory_sha
        for item in receipt['assets']:
            remote = assets[item['name']]
            assert remote['size'] == item['bytes']
            assert remote.get('digest') == 'sha256:' + item['sha256']
            checked[item['name']] = {'name': item['name'], 'sha256': item['sha256'],
                                     'bytes': item['bytes'], 'server_digest_verified': True}
        inventory = read_json(inventory_path)
        actual_parts = {a['name']: (a['bytes'], a['sha256']) for a in receipt['assets']
                        if not a['name'].endswith('-inventory.json')}
        assert actual_parts == {a['name']: (a['bytes'], a['sha256'])
                                for a in inventory['binary_assets']}
        for entry in inventory['files']:
            key = canonical(entry['path'])
            covered.setdefault(key, set()).add((entry['bytes'], entry['sha256']))
    missing, unpublished, verified = [], [], []
    for path in requirements():
        relative = str(path.relative_to(ROOT))
        if not path.is_file():
            missing.append(relative)
            continue
        if (path.stat().st_size, sha(path)) not in covered.get(relative, set()):
            unpublished.append(relative)
        else:
            verified.append(relative)
    content = OUT / 'objective-content-verification.json'
    complete = content.exists() and read_json(content).get('status') == 'passed'
    result = {'status': 'passed' if complete and not missing and not unpublished else 'incomplete',
              'timestamp': datetime.now(timezone.utc).isoformat(),
              'scope': 'Minimum loop, full EOS cache, all 15 training/recovery artifacts, '
                       '16 full evaluations, requested reports and 9B infrastructure records',
              'release_url': release['html_url'], 'verified_files': len(verified),
              'required_files': len(requirements()), 'missing_files': missing,
              'unpublished_or_changed_files': unpublished,
              'objective_content_verified': complete,
              'fresh_server_verified_assets': list(checked.values()),
              'large_payloads_downloaded': False,
              'dataset_release_audit': 'dataset-delivery-verification.json (separate historical release)'}
    write_json(OUT / 'final-delivery-verification.json', result)
    print(json.dumps({**result, 'missing_files': missing[:12],
                      'unpublished_or_changed_files': unpublished[:12],
                      'fresh_server_verified_assets': len(checked)}), flush=True)
    if strict and result['status'] != 'passed':
        raise RuntimeError('Complete objective delivery has not been proven; inspect audit JSON')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    secret = sys.stdin.readline().strip()
    if not secret:
        raise RuntimeError('GitHub credential required through encrypted stdin')
    audit(secret, strict=args.strict)
