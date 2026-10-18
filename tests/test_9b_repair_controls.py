import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_9b_repair as runtime
from experiment_common import write_json
from report_9b_repair import EFFECTS


def test_new_absolute_budget_never_resets_or_touches_old_budget(monkeypatch,tmp_path):
    monkeypatch.setattr(runtime,'OUT',tmp_path)
    write_json(tmp_path/'protocol.json',{'training_wall_seconds_cap':100,'evaluation_wall_seconds_cap':200})
    old=tmp_path/'old-calibration-wall-budget.json';old.write_bytes(b'old frozen budget')
    monkeypatch.setattr(runtime.time,'time',lambda:1000.)
    with pytest.raises(AssertionError):runtime.phase_budget('training')
    assert runtime.phase_budget('training',create=True)==1100
    before=(tmp_path/'training-wall-budget.json').read_bytes()
    monkeypatch.setattr(runtime.time,'time',lambda:1050.)
    assert runtime.phase_budget('training',create=True)==1100
    monkeypatch.setattr(runtime.time,'time',lambda:1100.)
    with pytest.raises(AssertionError):runtime.phase_budget('training',create=True)
    assert (tmp_path/'training-wall-budget.json').read_bytes()==before
    assert old.read_bytes()==b'old frozen budget'


def test_missing_old_queue_verification_blocks_before_new_budget(monkeypatch,tmp_path):
    monkeypatch.setattr(runtime,'OUT',tmp_path)
    monkeypatch.setattr(runtime,'verify_protocol',lambda:{})
    with pytest.raises(FileNotFoundError):runtime.admission()
    assert not (tmp_path/'training-wall-budget.json').exists()
    assert not (tmp_path/'evaluation-wall-budget.json').exists()


def test_factorial_coefficients_are_paired_and_correct():
    # Order is PD,PG,ND,NG; a constant shift must cancel in all effects.
    point=np.array([.5,.4,.7,.5])
    assert point@EFFECTS['coordinate']==pytest.approx(.15)
    assert point@EFFECTS['teacher']==pytest.approx(.15)
    assert point@EFFECTS['interaction']==pytest.approx(.1)
    assert all(sum(c)==0 for c in EFFECTS.values())
