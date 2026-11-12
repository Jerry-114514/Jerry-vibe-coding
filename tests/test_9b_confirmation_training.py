import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from experiment_common import read_rows
from train_9b_confirmation import execute_steps, reconcile_ledger


def test_resume_rolls_back_uncommitted_updates_and_respects_schedule(tmp_path):
    schedule = [{'item_id': f'item-{step}', 'axis': 'grounding'} for step in range(60)]
    ledger = tmp_path / 'loss.jsonl'
    saved = {}
    visited = []

    def update(item):
        visited.append(item['item_id'])
        return {'loss': 0.25, 'grad_norm': 0.1}

    def guard():
        if len(visited) == 53:
            raise TimeoutError('Fixture interruption')

    with pytest.raises(TimeoutError):
        execute_steps(ledger, schedule, 0, 60, update, saved.__setitem__, guard)
    with (tmp_path / 'loss.jsonl').open('ab') as stream:
        stream.write(b'{"step":54')
    assert len(visited) == 53
    with pytest.raises(ValueError, match='committed loss ledger'):
        reconcile_ledger(ledger, schedule, 50, hashlib.sha256(b'wrong').hexdigest())
    reconcile_ledger(ledger, schedule, 50, saved[50])
    assert len(read_rows(ledger)) == 50
    assert next(tmp_path.glob('incomplete-tail-*')).read_bytes() == b'{"step":54'
    assert len(read_rows(next(tmp_path.glob('uncommitted-loss-*')))) == 3
    execute_steps(ledger, schedule, 50, 60, update, saved.__setitem__, lambda: None)
    assert [row['item_id'] for row in read_rows(ledger)] == [item['item_id'] for item in schedule]
    assert visited[53:] == [item['item_id'] for item in schedule[50:]]
