"""Checks for Goal S's label-only intervention and product gates."""
from __future__ import annotations

import json

import prepare_v133_goal_s_labels as labels
import run_v133_goal_s_label_granularity as goal


def test_label_audit_uses_only_frozen_training_inputs():
    audit = labels.build()
    cases = audit["cases"]
    assert len(cases) == 16
    assert audit["changes"] == 2
    modified = {row["case_id"]: (row["original_answer"], row["revised_answer"]) for row in cases if row["changed"]}
    assert modified == {
        "5a8b8eec55429949d91db5e0": ("band", "rock band"),
        "5ab3a3c4554299233954ff6e": ("director", "film director"),
    }
    assert all(row["visible_documents"] and row["question"] for row in cases)
    assert all("development" not in json.dumps(row).lower() for row in cases)


def test_persisted_document_order_is_restored_without_changing_values():
    document = {"id": "1", "score": 1., "text": "T", "title": "Title"}
    trace = [{"observation": {"result": [document], "ok": True}}]
    rebuilt = goal.restored_trace(trace)
    actual = rebuilt[0]["observation"]["result"][0]
    assert list(actual) == ["id", "title", "text", "score"]
    assert actual == document
    assert list(document) == ["id", "score", "text", "title"]


def test_product_upgrade_requires_both_seeds():
    passing = {"total": 20, "regression": 7, "invalid": 0}
    assert goal.eligible_product(passing)
    assert not goal.eligible_product({**passing, "total": 19})
    assert not goal.eligible_product({**passing, "regression": 6})
    assert not goal.eligible_product({**passing, "invalid": 1})


def test_replication_product_gate(monkeypatch):
    items = {"42-a": {"total": 15, "regression": 6},
             "42-b": {"total": 16, "regression": 6}}
    monkeypatch.setattr(goal, "stats", lambda name: items[name])
    assert goal.replication_gate() == {"replicate": True, "reason": "product_better_than_control"}
    items["42-b"] = {"total": 16, "regression": 5}
    monkeypatch.setattr(goal.Path, "exists", lambda self: False)
    assert goal.replication_gate()["replicate"] is None
