import json

import pytest

from scripts.review_v116_api import validate_citations
from scripts.prepare_v116_data import WORK_PATH
from scripts.finalize_v116 import choose_train_supplements, valid_vote


def test_candidate_citation_cannot_borrow_other_alias_or_future_doc():
    packet = {"kind": "state", "candidates": {
        "A": {"before_action": [{"id": "a", "text": "Ada founded Beta"}],
              "retrieved_after_search": [{"id": "after", "text": "Ada joined Gamma"}]},
        "B": {"before_action": [{"id": "b", "text": "Bea founded Delta"}],
              "retrieved_after_search": None}}}
    validate_citations(packet, {"citations": [{"stage": "before_action", "step": None,
                                                "document_id": "a", "quote": "Ada founded Beta"}]}, "A")
    with pytest.raises(ValueError, match="candidate_step_visible"):
        validate_citations(packet, {"citations": [{"stage": "before_action", "step": None,
                                                    "document_id": "b", "quote": "Bea founded Delta"}]}, "A")
    with pytest.raises(ValueError, match="candidate_step_visible"):
        validate_citations(packet, {"citations": [{"stage": "before_action", "step": None,
                                                    "document_id": "after", "quote": "Ada joined Gamma"}]}, "A")


def test_step_bound_citation_cannot_use_later_visible_text():
    packet = {"kind": "changed_task", "candidates": {"A": {"steps": [
        {"step": 0, "before_action": [{"id": "old", "text": "Known fact"}],
         "retrieved_after_search": [{"id": "new", "text": "New fact"}]},
        {"step": 1, "before_action": [{"id": "new", "text": "New fact"}],
         "retrieved_after_search": None}]}}}
    with pytest.raises(ValueError, match="candidate_step_visible"):
        validate_citations(packet, {"citations": [{"stage": "before_action", "step": 0,
                                                    "document_id": "new", "quote": "New fact"}]}, "A")
    validate_citations(packet, {"citations": [{"stage": "before_action", "step": 1,
                                                "document_id": "new", "quote": "New fact"}]}, "A")


def test_current_run_outputs_are_not_historical_exposure():
    from pathlib import Path
    assert WORK_PATH(Path("D:/AgentLightning/mini-agent-rl/data/evidence-sft-v116/internal-candidate-private.jsonl"))
    assert not WORK_PATH(Path("D:/AgentLightning/mini-agent-rl/data/evidence-sft-v114-review/accepted.jsonl"))


def test_partial_or_future_only_data_review_cannot_pass():
    packet = {"proposed_action": {"action": "answer"},
              "before_action": [{"id": "old", "text": "Old evidence"}],
              "retrieved_after_search": [{"id": "future", "text": "Future answer"}]}
    assert not valid_vote(packet, {"approved": True, "reason": "looks good", "risk_labels": []})[0]
    vote = {"approved": True, "reason": "uses later evidence", "risk_labels": ["relation"],
            "citations": [{"stage": "retrieved_after_search", "step": None,
                           "document_id": "future", "quote": "Future answer"}]}
    assert valid_vote(packet, vote) == (False, "answer_citation_not_visible")


def test_search_requires_grounded_query_new_evidence_and_valid_quote():
    packet = {"proposed_action": {"action": "search"},
              "before_action": [{"id": "old", "text": "The event took place in Boston."}],
              "retrieved_after_search": [{"id": "new", "text": "The year was 1924."}]}
    approved = {"approved": True, "reason": "year is absent before search",
                "risk_labels": ["time"], "missing_fact": "year",
                "search_needed": "yes", "query_grounded": "yes", "retrieval_relevant": "yes",
                "citations": [{"stage": "before_action", "step": None,
                               "document_id": "old", "quote": "event took place in Boston"},
                              {"stage": "retrieved_after_search", "step": None,
                               "document_id": "new", "quote": "The year was 1924"}]}
    assert valid_vote(packet, approved) == (True, "valid")
    assert valid_vote(packet, {**approved, "query_grounded": "no"}) == (False, "search_semantics_not_approved")
    assert valid_vote(packet, {**approved, "retrieval_relevant": "not_available"}) == (False, "search_semantics_not_approved")
    bad = {**approved, "citations": [{"stage": "retrieved_after_search", "step": None,
                                      "document_id": "new", "quote": "The year was 1925"}]}
    assert valid_vote(packet, bad) == (False, "invalid_citation")


def test_duplicate_and_partial_reviews_are_not_silently_deduplicated(tmp_path, monkeypatch):
    from scripts import finalize_v116
    monkeypatch.setattr(finalize_v116, "REVIEW", tmp_path)
    row = {"case_id": "same", "approved": False, "reason": "missing"}
    (tmp_path / "luna-train-batch-01.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate_or_missing_luna_id"):
        finalize_v116.agent_reviews("train")


def test_minimal_supplement_meets_pair_and_action_balance():
    base = []
    for index in range(154):
        base.append(({"proposed_action": {"action": "answer"}, "before_action": [{"id": str(index)}]},
                     f"q{index}", []))
    for index in range(166):
        base.append(({"proposed_action": {"action": "search"}, "before_action": [{"id": str(index)}]},
                     f"q{index}" if index < 62 else f"s{index}", []))
    for index in range(12):
        base.append(({"proposed_action": {"action": "search"}, "before_action": []}, f"e{index}", []))
    approved, ids = [], {}
    for index in range(20):
        case = f"a{index}"; ids[case] = f"s{index + 62}"
        approved.append(({"case_id": case, "proposed_action": {"action": "answer"}, "before_action": [{"id": case}]}, []))
    for index in range(20):
        case = f"b{index}"; ids[case] = f"q{index + 62}"
        approved.append(({"case_id": case, "proposed_action": {"action": "search"}, "before_action": [{"id": case}]}, []))
    chosen = choose_train_supplements(base, approved, ids)
    assert len(chosen) == 13
    assert sum(row[0]["proposed_action"]["action"] == "answer" for row in chosen) >= 2


def test_packet_hash_mismatch_and_missing_second_reviewer_fail_closed(tmp_path, monkeypatch):
    from scripts import finalize_v116
    monkeypatch.setattr(finalize_v116, "REVIEW", tmp_path)
    packet = {"case_id": "p", "packet_sha256": "expected", "proposed_action": {"action": "answer"},
              "before_action": [{"id": "d", "text": "Ada founded Beta."}]}
    (tmp_path / "train-packets.jsonl").write_text(json.dumps(packet) + "\n", encoding="utf-8")
    vote = {"case_id": "p", "packet_sha256": "wrong", "approved": True, "reason": "supported",
            "risk_labels": ["relation"], "citations": [{"stage": "before_action", "step": None,
                                                      "document_id": "d", "quote": "Ada founded Beta"}]}
    (tmp_path / "luna-train-batch-01.jsonl").write_text(json.dumps(vote) + "\n", encoding="utf-8")
    accepted, audit = finalize_v116.review_pool("train")
    assert not accepted
    assert audit[0]["luna_status"] == "luna_packet_hash_mismatch"
    assert audit[0]["api_status"] == "incomplete_vote"
