"""Read-only CPU verification of cloned audit ledger and pinned model weights."""
import json
import time
from experiment_common import ROOT, read_json, read_rows, sha, write_json


def main():
    out = ROOT / 'runs/route-b-7b-20261016'
    started = time.monotonic()
    ledger = ROOT / 'runs/protocol-v2/audit/qwen-clean-v2/qwen7b/ledger.jsonl'
    data = ledger.read_bytes()
    assert data.endswith(b'\n'), 'Partial final audit row'
    rows = read_rows(ledger)
    assert len(rows) == 7400 and len({r['item_id'] for r in rows}) == 7400
    manifest = read_rows(ROOT / 'runs/protocol-v2-data/clean-role-manifests-v4/audit.jsonl')
    by_id = {r['item_id']: r for r in manifest}
    assert {r['item_id'] for r in rows} <= set(by_id)
    from run_clean_audit_v2 import score
    for row in rows:
        actual = score(row['raw_generation'], by_id[row['item_id']], row['usage'])
        assert all(actual[k] == row[k] for k in ('correct_bit', 'parse_fail_flag', 'parsed_answer', 'iou'))
    weights = {}
    lock = read_json(ROOT / 'configs/models.lock.json')
    for receipt in sorted((ROOT / '.model_cache').glob('*/weights.verified.json')):
        key = receipt.parent.name
        if key not in ('qwen3b', 'qwen7b'):
            continue
        verified = read_json(receipt)
        assert verified['revision'] == lock['models'][key]['revision']
        files = []
        for entry in verified['files']:
            path = receipt.parent / entry['file']
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
            files.append({'file': entry['file'], 'sha256': entry['sha256'], 'bytes': entry['bytes']})
        weights[key] = {'revision': verified['revision'], 'files': files}
    assert 'qwen7b' in weights
    result = {'status': 'passed', 'audit_rows': 7400, 'audit_unique_coverage_verified': True,
              'audit_independent_rescore_verified': True, 'audit_ledger_sha256': sha(ledger),
              'source_host_byte_comparison': 'Source4090 released; current SHA recorded, original-copy identity not independently provable without preclone digest',
              'teacher_cache_integrity': read_json(out / 'cache-audit-complete.json'),
              'weights': weights, 'elapsed_seconds': time.monotonic() - started}
    write_json(out / 'migration-verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
