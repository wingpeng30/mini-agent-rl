"""Risk-focused tests for the bounded preference comparison."""
from __future__ import annotations

import torch

import run_v133_goal_u_v_evidence_preference as uv


def test_dpo_prefers_chosen_relative_to_frozen_reference():
    neutral = uv.dpo_loss(torch, torch.tensor(-4.), torch.tensor(-6.), -4., -6.)
    better = uv.dpo_loss(torch, torch.tensor(-3.), torch.tensor(-6.), -4., -6.)
    worse = uv.dpo_loss(torch, torch.tensor(-5.), torch.tensor(-6.), -4., -6.)
    assert better < neutral < worse
    assert torch.isclose(neutral, torch.tensor(0.6931472), atol=1e-6)


def test_old_gate_needs_both_seeds_and_factual_gain(monkeypatch):
    values = {
        "incumbent": {"total": 20, "regression": 7, "invalid": 0},
        "42-a": {"total": 21, "regression": 7, "invalid": 0},
        "43-a": {"total": 21, "regression": 8, "invalid": 0},
        "42-b": {"total": 22, "regression": 7, "invalid": 0},
        "43-b": {"total": 22, "regression": 8, "invalid": 0},
    }
    monkeypatch.setattr(uv, "count_stats", lambda name: values[name])
    baseline = {"sufficient": 23, "unsupported_or_wrong": 6, "one_sided_error": 5,
                "missing_qualifier": 3}
    better = {"sufficient": 24, "unsupported_or_wrong": 5, "one_sided_error": 4,
              "missing_qualifier": 3}
    review = {"models": {k: (baseline if k == "incumbent" else dict(better)) for k in values}}
    assert uv.old_gate(review) == "b"
    review["models"]["43-b"]["unsupported_or_wrong"] = 6
    assert uv.old_gate(review) == "a"
    review["models"]["43-a"]["sufficient"] = 23
    assert uv.old_gate(review) is None


def test_pair_input_and_label_isolation(monkeypatch):
    case_id = "training-only"
    monkeypatch.setattr(uv.q, "train_cases", lambda: [{"case_id": case_id}])
    monkeypatch.setattr(uv.r, "OUT", uv.OUT / "unused-r")
    monkeypatch.setattr(uv.s, "LABELS", uv.OUT / "unused-labels.json")
    def fake_read(path):
        if path == uv.PAIRS:
            return {"pairs": [{"case_id": case_id, "chosen": "writer", "rejected": "poet",
                                "input_ids": [1, 2], "input_mask": [1, 1]}] * 8}
        return {"cases": [{"case_id": case_id, "revised_answer": "writer"}]}
    monkeypatch.setattr(uv, "read", fake_read)
    monkeypatch.setattr(uv, "rows", lambda path: [{"case_id": case_id,
                                                    "input_ids": [1, 2], "input_mask": [1, 1]}])
    try:
        uv.pairs()
    except RuntimeError as error:
        assert "case ids" in str(error)
    else:
        raise AssertionError("Duplicate pair IDs must fail")


def test_fresh_gate_checks_both_seeds_and_both_strata(monkeypatch):
    baseline = {"total": 18, "invalid": 0}
    good = {"total": 19, "invalid": 0}
    values = {"42-b": dict(good), "43-b": dict(good)}
    monkeypatch.setattr(uv, "count_stats", lambda label, fresh=False: values[label])
    base_sem = {"sufficient": 22, "unsupported_or_wrong": 6, "one_sided_error": 4,
                "missing_qualifier": 4,
                "strata": {"one_sided": {"sufficient": 10}, "ordinary": {"sufficient": 12}}}
    new_sem = {"sufficient": 23, "unsupported_or_wrong": 6, "one_sided_error": 4,
               "missing_qualifier": 4,
               "strata": {"one_sided": {"sufficient": 11}, "ordinary": {"sufficient": 12}}}
    review = {"models": {"42-b": dict(new_sem), "43-b": dict(new_sem)}}
    assert uv.fresh_gate("b", review, baseline, base_sem)
    values["43-b"]["total"] = 18
    assert not uv.fresh_gate("b", review, baseline, base_sem)
    values["43-b"]["total"] = 19
    review["models"]["43-b"] = {**new_sem, "strata": {"one_sided": {"sufficient": 11},
                                                      "ordinary": {"sufficient": 11}}}
    assert not uv.fresh_gate("b", review, baseline, base_sem)
