"""Fresh Release coverage audit, including offloaded9B optimizer recovery.

Read-only to experiment payloads. Writes an audit record; never fetches weights,
changes a registered source, or starts a GPU job. Not a training permission gate.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import PurePosixPath
import sys

import requests

from experiment_common import ROOT, read_json, sha, write_json
from route_b_9b_loop import BASE, OUT, SEEDS

DELIVERY_ROOTS = ('runs/9b-student-delivery-20261017/', 'runs/9b-teacher-delivery-20261017/')


def canonical(path):
    path = PurePosixPath(path).as_posix()
    if any(path.startswith(prefix) for prefix in DELIVERY_ROOTS) and '/files/' in path:
        return path.split('/files/', 1)[1]
    prefix = OUT.relative_to(ROOT).as_posix() + '/'
    for subdir in ('registration/', 'formal-registration/', 'minimal-report/'):
        if path.startswith(prefix + subdir):
            name = path[len(prefix + subdir):]
            if subdir == 'registration/' and name.startswith('eval-input-'):
                return (BASE / 'evaluation-input-preparation' / name[len('eval-input-'):]).relative_to(ROOT).as_posix()
            if name in ('protocol.json', 'minimal-pool.jsonl', 'minimal-sample-counts.json',
                        'formal-plan.json', 'minimal-acceptance.json'):
                return prefix + name
    return path


def requirements():
    paths = [ROOT / 'configs/route_b_9b_5090.frozen.json']
    paths += [BASE / name for name in ('teacher.jsonl', 'eligible-pool.jsonl', 'quarantined-pool.jsonl',
             'teacher-eligible-complete.json', 'resume-v3/refusals.jsonl')]
    paths += [BASE / 'full-preflight' / name for name in ('protocol.json', 'manifest.jsonl', 'complete.json',
              'loss.jsonl', 'nvidia-smi.jsonl', 'shape-stress.json', 'trainable-parameters.json',
              'resume.pt', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'frozen-config.sha256')]
    paths += [BASE / 'evaluation-input-preparation' / name for name in ('complete.json', 'minimal-manifest.jsonl', 'formal-manifest.jsonl')]
    paths += [OUT / name for name in ('protocol.json', 'minimal-pool.jsonl', 'minimal-sample-counts.json',
              'minimal-acceptance.json', 'minimal/A1-seed42/resume-verification.json',
              'formal-plan.json', 'formal-start-verification.json', 'formal-training-complete.json',
              'minimal-wall-budget.json', 'formal-training-wall-budget.json', 'formal-evaluation-wall-budget.json',
              'formal-independent-verification.json', 'status.json',
              'formal-report/training-results.json', 'formal-report/training-report.md',
              'formal-report/loss-curves.png', 'formal-report/evaluation-results.json', 'formal-report/evaluation-report.md')]
    for phase, seeds in (('minimal', [42]), ('formal', SEEDS)):
        for seed in seeds:
            for arm in ('A1', 'A2', 'U'):
                folder = OUT / phase / f'{arm}-seed{seed}'
                if phase == 'formal':
                    paths.append(OUT / 'formal-registration' / f'{arm}-seed{seed}.jsonl')
                paths += [folder / name for name in ('complete.json', 'train.meta.json', 'loss.jsonl',
                          'resume.pt', 'adapter/adapter_model.safetensors', 'adapter/adapter_config.json',
                          'training-verification.json', 'scores.json', 'eval.jsonl', 'eval.meta.json')]
        paths += [OUT / phase / 'N-seed42' / name for name in ('scores.json', 'eval.jsonl', 'eval.meta.json')]
    # Receipt/inventory bytes are checked through their own server assets above;
    # requiring those same bookkeeping files as archive members is recursive.
    paths += sorted(p for p in (BASE / 'api400-investigation').rglob('*') if p.is_file()
                    and p.name not in ('github-inventory.json', 'github-storage.json'))
    return paths


def archived_recovery(path, covered):
    """A missing local recovery is acceptable only with an exact restore chain."""
    folder = path.parent
    pointer_path = folder / 'resume-offloaded.json'
    if path.name != 'resume.pt' or not pointer_path.exists():
        return False
    marker, pointer = read_json(folder / 'complete.json'), read_json(pointer_path)
    proof = read_json(folder / 'training-verification.json')
    assert proof['status'] == 'passed_local_recovery_before_offload'
    assert proof['resume_sha256'] == pointer['resume_sha256'] == marker['resume_sha256']
    assert pointer['receipt_sha256'] == sha(folder / 'training-delivery.json')
    assert pointer['inventory_sha256'] == sha(folder / 'training-inventory.json')
    key = path.relative_to(ROOT).as_posix()
    assert pointer['archive_member'] == key
    return (pointer['bytes'], marker['resume_sha256']) in covered.get(key, set())


def collect_coverage(token):
    session = requests.Session()
    session.trust_env = False
    session.headers.update({'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json'})
    endpoint = 'https://api.github.com/repos/Jerry-114514/Jerry-vibe-coding'
    releases = {}

    def release_assets(url):
        if url in releases:
            return releases[url]
        tag = url.rsplit('/', 1)[-1]
        assert tag in ('arp-9b-122b-20261017', 'arp-protocol-v2-20261016')
        response = session.get(endpoint + '/releases/tags/' + tag, timeout=30)
        response.raise_for_status()
        release = response.json()
        assert release['html_url'] == url
        assets = {}
        for page in range(1, 100):
            response = session.get(endpoint + f'/releases/{release["id"]}/assets',
                                   params={'per_page': 100, 'page': page}, timeout=30)
            response.raise_for_status()
            batch = response.json()
            assets.update({a['name']: a for a in batch})
            if len(batch) < 100:
                releases[url] = assets
                return assets
        raise RuntimeError('Asset pagination exhausted; partial coverage is insufficient')

    folders = {p.parent for p in BASE.rglob('github-storage.json') if '/files/' not in p.as_posix()}
    for prefix in DELIVERY_ROOTS:
        folders.update(p.parent for p in (ROOT / prefix).glob('*/github-storage.json'))
    teacher_receipt = BASE / 'teacher-delivery.json'
    if teacher_receipt.exists():
        folders.add(ROOT / read_json(teacher_receipt)['snapshot'])
    covered, checked = {}, {}
    for folder in sorted(folders):
        receipt_path, inventory_path = folder / 'github-storage.json', folder / 'github-inventory.json'
        if not receipt_path.exists() or not inventory_path.exists():
            continue
        receipt, inventory = read_json(receipt_path), read_json(inventory_path)
        remote_assets = release_assets(receipt['release_url'])
        inventory_asset = next(a for a in receipt['assets'] if a['name'].endswith('-inventory.json'))
        assert inventory_asset['sha256'] == sha(inventory_path)
        for entry in receipt['assets']:
            actual = remote_assets[entry['name']]
            assert actual['size'] == entry['bytes'] and actual.get('digest') == 'sha256:' + entry['sha256']
            checked[entry['name']] = {'release_url': receipt['release_url'], **entry}
        binary = {a['name']: (a['bytes'], a['sha256']) for a in receipt['assets'] if not a['name'].endswith('-inventory.json')}
        assert binary == {a['name']: (a['bytes'], a['sha256']) for a in inventory['binary_assets']}
        for entry in inventory['files']:
            covered.setdefault(canonical(entry['path']), set()).add((entry['bytes'], entry['sha256']))
    return covered, checked


def audit(token, strict=False):
    covered, checked = collect_coverage(token)
    missing, unpublished, verified, offloaded = [], [], [], []
    for path in requirements():
        key = path.relative_to(ROOT).as_posix()
        if not path.exists():
            if archived_recovery(path, covered):
                verified.append(key)
                offloaded.append(key)
            else:
                missing.append(key)
        elif (path.stat().st_size, sha(path)) in covered.get(key, set()):
            verified.append(key)
        else:
            unpublished.append(key)
    result = {'status': 'passed_delivery_coverage' if not missing and not unpublished else 'incomplete',
              'timestamp': datetime.now(timezone.utc).isoformat(), 'required_files': len(requirements()),
              'verified_files': len(verified), 'offloaded_recovery_files': offloaded,
              'missing_files': missing, 'unpublished_or_changed_files': unpublished,
              'fresh_server_verified_assets': list(checked.values()), 'large_payloads_downloaded': False,
              'scope': '9B extension artifact coverage only; independent scientific/content verification remains required'}
    write_json(OUT / 'final-delivery-verification.json', result)
    print(json.dumps({**result, 'fresh_server_verified_assets': len(checked),
                     'missing_files': missing[:10], 'unpublished_or_changed_files': unpublished[:10]}), flush=True)
    if strict and result['status'] != 'passed_delivery_coverage':
        raise RuntimeError('Full9B Release delivery not proven')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    secret = sys.stdin.readline().strip()
    assert secret, 'Credential required through encrypted stdin'
    audit(secret, strict=args.strict)
