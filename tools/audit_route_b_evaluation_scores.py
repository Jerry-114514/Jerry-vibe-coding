"""Rescore every completed evaluation and verify reported four-axis aggregates.

Incomplete models are explicitly listed, never treated as failed or complete.
No GPU, model inference, training or API calls.
"""
from datetime import datetime, timezone
import json

from experiment_common import ROOT, AXES, read_json, read_rows, sha, write_json, units, score_generation
from route_b_staged import OUT, SEEDS


def audit(require_all=False):
    manifest_path = ROOT / 'data/manifest.jsonl'
    manifest = read_rows(manifest_path)
    assert len(manifest) == 16900
    lookup = {r['item_id']: r for r in manifest}
    assert len(lookup) == len(manifest)
    expected = [('N', 42)] + [(a, s) for s in SEEDS for a in ('A1', 'A2', 'U')]
    verified, pending = [], []
    for arm, seed in expected:
        folder = OUT / 'formal' / f'{arm}-seed{seed}'
        if not (folder / 'scores.json').exists():
            pending.append(folder.name)
            continue
        marker = read_json(folder / 'scores.json')
        records = read_rows(folder / 'eval.jsonl')
        assert len(records) == marker['rows'] == len(manifest)
        assert {r['item_id'] for r in records} == set(lookup)
        assert len({r['item_id'] for r in records}) == len(records)
        assert marker['ledger_sha256'] == sha(folder / 'eval.jsonl')
        assert read_json(folder / 'eval.meta.json')['manifest_sha256'] == sha(manifest_path)
        for row in records:
            rescored = score_generation(row['raw_generation'], lookup[row['item_id']])
            assert all(row[k] == rescored[k] for k in ('correct_bit', 'parsed_answer', 'parse_fail_flag', 'iou'))
        originals = units(manifest, records)
        computed = {}
        for axis in AXES:
            axis_rows = [r for r in originals if r['axis'] == axis]
            computed[axis] = sum(r['correct_bit'] for r in axis_rows) / len(axis_rows)
        assert marker['axis_scores'] == computed
        assert marker['original_units'] == len(originals)
        verified.append({'model': folder.name, 'prompts': len(records), 'original_units': len(originals),
                         'axis_scores': computed, 'ledger_sha256': marker['ledger_sha256'],
                         'score_marker_sha256': sha(folder / 'scores.json')})
    result = {'status': 'all16_verified' if len(verified) == 16 else 'completed_subset_verified',
              'timestamp': datetime.now(timezone.utc).isoformat(), 'verified_models': verified,
              'pending_models': pending, 'expected_models': 16, 'scope': 'Raw output rescore, exact coverage and four-axis aggregate equality',
              'no_new_inference': True, 'manifest_sha256': sha(manifest_path)}
    write_json(OUT / 'evaluation-aggregate-independent-verification.json', result)
    if require_all:
        assert len(verified) == 16, 'Full evaluation is not yet complete'
    print(json.dumps({'status': result['status'], 'verified_models': len(verified), 'pending_models': pending}), flush=True)
    return result


if __name__ == '__main__':
    audit()
