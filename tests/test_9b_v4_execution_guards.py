"""Verify resume/budget invariants without starting a GPU or network action."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_9b_native_frontend_v4 as v4


def test_absolute_budget_is_not_extended_on_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(v4, 'OUT', tmp_path)
    monkeypatch.setattr(v4.time, 'time', lambda: 1000.)
    end = v4.budget('calibration', 900)
    original = (tmp_path / 'calibration-wall-budget.json').read_bytes()
    monkeypatch.setattr(v4.time, 'time', lambda: 1500.)
    assert v4.budget('calibration', 900) == end == 1900.
    assert (tmp_path / 'calibration-wall-budget.json').read_bytes() == original
    monkeypatch.setattr(v4.time, 'time', lambda: 1900.)
    with pytest.raises(AssertionError):
        v4.budget('calibration', 900)
    assert (tmp_path / 'calibration-wall-budget.json').read_bytes() == original


def test_interrupted_journal_tail_is_preserved_before_truncation(tmp_path):
    path = tmp_path / 'eval.jsonl'
    prefix, tail = b'{"item_id":"done"}\n', b'{"item_id":"partial'
    path.write_bytes(prefix+tail)
    v4.preserve_tail(path)
    assert path.read_bytes() == prefix
    fragments = list(tmp_path.glob('incomplete-tail-*'))
    assert len(fragments) == 1 and fragments[0].read_bytes() == tail
    v4.preserve_tail(path)
    assert len(list(tmp_path.glob('incomplete-tail-*'))) == 1


def test_pilot_and_formal_use_same_fixed_scoring():
    item = {'axis': 'grounding', 'width': 640, 'height': 480,
            'processor_width': 704, 'processor_height': 512, 'answer_key': [64, 96, 256, 288]}
    a = v4.scoring('[100,200,400,600]', item)
    assert a['correct_bit'] == 1 and a['iou'] == pytest.approx(1.)
    assert a['coordinate_system'] == 'fixed_native1000_to_processor_pixels_to_original_pixels'
