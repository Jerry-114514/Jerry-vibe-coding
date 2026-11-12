from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from evaluate_9b_confirmation import run_ledger
from experiment_common import read_rows


def test_resume_keeps_valid_prefix_and_preserves_torn_tail(tmp_path):
    manifest = [
        {'item_id': key, 'bootstrap_unit': key, 'axis': 'pope', 'dataset': 'fixture',
         'image_group': key, 'answer_type': 'yes_no', 'answer_key': 'yes'}
        for key in ('a', 'b', 'c')
    ]
    visited = []

    def generate(row):
        visited.append(row['item_id'])
        return {'raw_generation': 'yes', 'generated_token_ids': [1, 248046], 'finish_reason': 'eos',
                'seconds': 0.1, 'reserved_peak_bytes': 10, 'allocated_peak_bytes': 5}

    def interrupt():
        if len(visited) == 1:
            raise TimeoutError('fixture interruption')

    with pytest.raises(TimeoutError):
        run_ledger(manifest, tmp_path, {'manifest': 'fixed'}, generate, interrupt)
    with (tmp_path / 'raw.jsonl').open('ab') as raw:
        raw.write(b'{"item_id":"b"')
    with pytest.raises(ValueError, match='Resume metadata'):
        run_ledger(manifest, tmp_path, {'manifest': 'different'}, generate, lambda: None)
    result = run_ledger(manifest, tmp_path, {'manifest': 'fixed'}, generate, lambda: None)
    assert visited == ['a', 'b', 'c']
    assert [r['item_id'] for r in read_rows(tmp_path / 'scored.jsonl')] == visited
    assert result['scores']['native_1000']['accuracy']['pope'] == 1
    assert next(tmp_path.glob('incomplete-tail-*')).read_bytes() == b'{"item_id":"b"'
    assert run_ledger(manifest, tmp_path, {'manifest': 'fixed'}, None, lambda: None) == result
    with (tmp_path / 'scored.jsonl').open('ab') as scored:
        scored.write(b'\n')
    with pytest.raises(ValueError, match='artifacts changed'):
        run_ledger(manifest, tmp_path, {'manifest': 'fixed'}, None, lambda: None)
