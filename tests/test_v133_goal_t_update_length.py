"""New safety boundaries for Goal T's continuous training run."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import run_v133_goal_t_update_length as goal


def test_checkpoint_positions_are_fixed_to_pass_ends():
    assert not goal.checkpoint_due(31, 1)
    assert goal.checkpoint_due(32, 1)
    assert not goal.checkpoint_due(63, 2)
    assert goal.checkpoint_due(64, 2)
    with pytest.raises(RuntimeError): goal.checkpoint_due(32, 2)
    with pytest.raises(RuntimeError): goal.checkpoint_due(64, 1)


def test_rng_snapshot_round_trip_preserves_cpu_state():
    before = goal.rng_snapshot(torch, np)
    torch.manual_seed(894)
    np.random.seed(132)
    goal.restore_rng(torch, np, before)
    assert goal.rng_equal(torch, np, before, goal.rng_snapshot(torch, np))


def test_product_gate_applies_to_each_short_checkpoint():
    passing = {"total": 20, "regression": 7, "invalid": 0}
    assert goal.eligible(passing)
    assert not goal.eligible({**passing, "total": 19})
    assert not goal.eligible({**passing, "regression": 6})
    assert not goal.eligible({**passing, "invalid": 1})


def test_stage_has_no_elapsed_time_cap(monkeypatch):
    calls, writes = [], []
    monkeypatch.setattr(goal.subprocess, "run", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(goal.time, "monotonic", iter((0., 100000.)).__next__)
    monkeypatch.setattr(goal, "write", lambda path, value: writes.append(value.copy()))
    state = {"gpu_seconds": 0., "stages": []}
    goal.run_stage("train42", state)
    assert calls and "timeout" not in calls[0]
    assert state["gpu_seconds"] == 100000.
    assert state["stages"][0]["name"] == "train42"
