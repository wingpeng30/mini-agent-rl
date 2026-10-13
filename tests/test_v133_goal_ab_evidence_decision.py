"""Focused gates for Goal AB's new evidence-decision review."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("goal_ab", SCRIPTS / "run_v133_goal_ab_evidence_decision.py")
ab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ab)


def test_prepared_packet_covers_actual_step_prompts():
    path = ab.OUT / "diagnostic-packet.jsonl"
    if not path.exists():
        pytest.skip("Goal AB preparation has not run in this checkout")
    rows = ab.z.rows(path)
    assert len(rows) == 208
    assert {split: sum(x["split"] == split for x in rows) for split in ab.SPLITS} == {
        "dev": 64, "old": 32, "y": 48, "aa_regression": 64}
    for case in rows:
        assert case["steps"]
        for step in case["steps"]:
            assert case["question"].strip().casefold() in step["actual_prompt"].casefold()
            assert len(step["prompt_sha256"]) == 64


def test_diagnostic_gate_rejects_missing_or_unsupported_citations(tmp_path, monkeypatch):
    packet = [{"split": split, "case_id": f"{split}-{n}",
               "steps": [{"visible_document_ids": ["a"]}]}
              for split, count in (("dev", 64), ("old", 32), ("y", 48), ("aa_regression", 64))
              for n in range(count)]
    review = [{"split": x["split"], "case_id": x["case_id"], "steps": [{
        "evidence_state": "sufficient", "evidence_ids": ["b"] if i == 0 else ["a"],
        "reason": "claim"}], "events": ["premature_answer"] if i < 8 else [],
        "primary_failure": "premature_answer" if i < 8 else "none", "reason": "claim"}
              for i, x in enumerate(packet)]
    monkeypatch.setattr(ab, "OUT", tmp_path)
    monkeypatch.setattr(ab, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(ab, "verify", lambda: {"min_dev_stop_errors": 8})
    (tmp_path / "state.json").write_text(json.dumps({"status": "awaiting_diagnostic_review", "stages": []}))
    (tmp_path / "diagnostic-packet.jsonl").write_text("".join(json.dumps(x) + "\n" for x in packet))
    (tmp_path / "diagnostic-review.jsonl").write_text("".join(json.dumps(x) + "\n" for x in review))
    with pytest.raises(RuntimeError, match="Invalid step evidence"):
        ab.diagnostic_gate()
    review[0]["steps"][0]["evidence_ids"] = ["a"]
    (tmp_path / "diagnostic-review.jsonl").write_text("".join(json.dumps(x) + "\n" for x in review))
    assert ab.diagnostic_gate()["passed"]


def test_stopping_error_taxonomy_is_frozen():
    assert set(("premature_answer", "unnecessary_search")).issubset(ab.EVENTS)
    assert "wrong_bridge_query" not in ("premature_answer", "unnecessary_search")


def test_decision_proposals_replay_actual_tokens_and_supports():
    path = ab.OUT / "pair-proposals.jsonl"
    if not path.exists():
        pytest.skip("Decision proposals are not yet materialized here")
    rows = ab.z.rows(path)
    assert 32 <= len(rows) <= 64
    assert len({x["case_id"] for x in rows}) == len(rows)
    for case in rows:
        assert case["teacher_searches"] <= 3
        assert case["teacher_model_calls"] <= 4
        first = case["teacher_steps"]["insufficient"]
        last = case["teacher_steps"]["sufficient"]
        assert first["action"] == "search" and last["action"] == "answer"
        assert len(first["support_titles_visible"]) < 2
        assert set(last["support_titles_visible"]) == set(case["teacher_support_titles"])
        for action in (first, last):
            assert action["action_token_ids"]
            assert len(action["prompt_token_ids"]) == len(action["prompt_attention_mask"])


def test_confirmation_cannot_start_before_development_gate(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"status": "awaiting_development_review"}))
    monkeypatch.setattr(ab, "STATE", state_path)
    with pytest.raises(RuntimeError, match="sealed"):
        ab.evaluate_confirmation()
