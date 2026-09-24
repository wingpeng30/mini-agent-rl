"""Fail-closed checks for v114 review citations and data gates."""
from __future__ import annotations

import sys
import tempfile
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from unittest.mock import patch

from finalize_v114_reviews import (gate, load_decisions, normalize_luna_reviews,
                                   normalized_risks, select_cohort, valid_review)


def packet(action: str = "search") -> dict:
    return {
        "case_id": "example", "question": "Which person completed X?", "packet_sha256": "fixed", "preflight_error": None,
        "proposed_action": {"action": action, "query": "find X"} if action == "search" else {"action": "answer", "answer": "X"},
        "visible_evidence": [{"id": "v", "text": "The subject has a missing relation."}],
        "retrieval_top3": [{"id": "n", "text": "X completed the relation."}],
    }


def review(action: str = "search") -> dict:
    return {
        "reviewer_id": "luna-a", "packet_sha256": "fixed", "approved": True,
        "reason": "The visible evidence leaves a relation gap.", "issue_tags": [],
        "risk_labels": ["relation"], "missing_fact": "Who completed the relation?",
        "citations": [{"stage": "retrieved" if action == "search" else "visible",
                       "document_id": "n" if action == "search" else "v",
                       "quote": "X completed the relation." if action == "search" else "missing relation"}],
    }


def test_valid_search_requires_verbatim_new_top3_and_gap():
    p, r = packet(), review()
    assert valid_review(r, p, "a") == []
    r["missing_fact"] = None
    assert "search_requires_missing_fact" in valid_review(r, p, "a")
    r["missing_fact"] = "Who?"
    r["citations"][0]["quote"] = "paraphrased"
    assert "illegal_verbatim_citation" in valid_review(r, p, "a")


def test_answer_cannot_use_future_retrieval():
    p, r = packet("answer"), review("answer")
    assert valid_review(r, p, "a") == []
    r["citations"] = [{"stage": "retrieved", "document_id": "n", "quote": "X completed the relation."}]
    assert "answer_requires_visible_only_citations" in valid_review(r, p, "a")


def test_retrieved_duplicate_is_legal_but_not_new_evidence():
    p, r = packet(), review()
    p["retrieval_top3"].append({"id": "v", "text": "The subject has a missing relation."})
    r["citations"] = [{"stage": "retrieved", "document_id": "v", "quote": "missing relation"}]
    assert "illegal_verbatim_citation" not in valid_review(r, p, "a")
    assert "search_requires_new_top3_citation" in valid_review(r, p, "a")


def test_entity_identity_risk_maps_to_frozen_relation_bucket():
    assert normalized_risks(["entity", "time"]) == {"relation", "time"}


def test_incomplete_coverage_fails_even_with_enough_states():
    summary = {"states": 100, "questions": 49, "answer_fraction": .5,
               "nonempty_search_fraction": .4, "empty_search_fraction": .1,
               "risk_state_counts": {"relation": 100, "measurement": 10, "time": 10, "scope": 10}}
    assert "internal_frozen_question_coverage_incomplete" in gate(summary, "internal", set(map(str, range(50))))


def test_duplicate_review_is_quarantined():
    p = packet("answer")
    r = {**review("answer"), "case_id": "example"}
    with patch("finalize_v114_reviews.load", return_value=[r, r]):
        decisions, errors = load_decisions("a", {"example": p})
    assert "duplicate_decision" in decisions["example"]["_invalid_errors"]
    assert any(item["error"] == "duplicate_decision" for item in errors)


def test_reused_reason_is_quarantined_only_across_unrelated_questions():
    first = {**packet("answer"), "case_id": "one"}
    second = {**packet("answer"), "case_id": "two"}
    template = review("answer")
    rows = [{**template, "case_id": "one"}, {**template, "case_id": "two"}]
    with patch("finalize_v114_reviews.load", return_value=rows):
        decisions, _ = load_decisions("a", {"one": first, "two": second})
    assert all("non_specific_repeated_approval_reason" not in r["_invalid_errors"] for r in decisions.values())

    second["question"] = "Which year did Y begin?"
    with patch("finalize_v114_reviews.load", return_value=rows):
        decisions, _ = load_decisions("a", {"one": first, "two": second})
    assert all("non_specific_repeated_approval_reason" in r["_invalid_errors"] for r in decisions.values())


def approved_row(case_id: str, source_id: str, action: str) -> dict:
    p = packet("answer" if action == "answer" else "search")
    if action == "answer":
        p["visible_evidence"] = [{"id": "v", "text": "supported answer"}]
        p["proposed_action"] = {"action": "answer", "answer": "X"}
    else:
        p["proposed_action"] = {"action": "search", "query": "find X"}
    return {"case_id": case_id, "source_id": source_id, "split": "internal" if source_id.startswith("i") else "train",
            "packet": p,
            "review_a": {"risk_labels": ["relation", "measurement", "time", "scope"]},
            "review_b": {"risk_labels": ["relation", "measurement", "time", "scope"]}}


def test_cohort_selector_returns_minimum_gate_passing_sets():
    train_rows = [approved_row(f"t{i}", f"tq{i}", "answer" if i % 2 == 0 else "search")
                  for i in range(300)]
    selected_train = select_cohort(train_rows, "train", set())
    assert selected_train is not None
    assert len(selected_train) == 300
    assert len({row["source_id"] for row in selected_train}) >= 150

    internal_rows = [approved_row(f"i{source}-{step}", f"i{source}", "answer" if step == 0 else "search")
                     for source in range(50) for step in range(2)]
    selected_internal = select_cohort(internal_rows, "internal", {f"i{i}" for i in range(50)})
    assert selected_internal is not None
    assert len(selected_internal) == 100
    assert {row["source_id"] for row in selected_internal} == {f"i{i}" for i in range(50)}


def test_external_luna_review_schema_normalizes_with_bound_addenda():
    from finalize_v114_reviews import WORK

    answer_packet = {"case_id": "a", "packet_sha256": "pa", "question": "How many?",
                     "preflight_error": None, "proposed_action": {"action": "answer", "answer": "4"},
                     "visible_evidence": [{"id": "v", "title": "Counts", "text": "The total was 4."}],
                     "retrieval_top3": []}
    search_packet = {"case_id": "s", "packet_sha256": "ps", "question": "Which year?",
                     "preflight_error": None, "proposed_action": {"action": "search", "query": "year"},
                     "visible_evidence": [{"id": "v", "title": "Event", "text": "The event occurred."}],
                     "retrieval_top3": [{"id": "n", "title": "Date", "text": "The year was 1999."}]}
    raw = [
        {"case_id": "a", "packet_sha256": "pa", "verdict": "approve", "reason": "Direct evidence.",
         "citations": [{"evidence_id": "v", "title": "Counts", "quote": "The total was 4."}], "issue_tags": []},
        {"case_id": "s", "packet_sha256": "ps", "verdict": "approve", "reason": "The year is absent.",
         "citations": [{"evidence_id": "n", "title": "Date", "quote": "The year was 1999."}], "issue_tags": []},
    ]
    addenda = [
        {"case_id": "a", "packet_sha256": "pa", "risk_labels": ["measurement"]},
        {"case_id": "s", "packet_sha256": "ps", "risk_labels": ["time"], "missing_fact": "The event's year."},
    ]
    with tempfile.TemporaryDirectory(prefix=".v114-review-test-", dir=Path.cwd()) as temp_dir:
        temp_path = Path(temp_dir)
        with patch("finalize_v114_reviews.WORK", temp_path):
            (temp_path / "probe-b-decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8")
            (temp_path / "probe-b-addenda.jsonl").write_text("".join(json.dumps(r) + "\n" for r in addenda), encoding="utf-8")
            path = normalize_luna_reviews("b", "probe", {"a": answer_packet, "s": search_packet}, ["a", "s"])
            normalized = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert normalized[0]["reviewer_id"] == "luna-b"
    assert normalized[0]["citations"][0]["stage"] == "visible"
    assert normalized[1]["citations"][0]["stage"] == "retrieved"
    assert normalized[1]["missing_fact"] == "The event's year."
    assert valid_review(normalized[0], answer_packet, "b") == []
    assert valid_review(normalized[1], search_packet, "b") == []
