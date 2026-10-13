from __future__ import annotations

import pytest

from scripts.v126_review_protocol import SYSTEM, valid_vote


def _search_packet():
    return {
        "case_id": "zero-context-search",
        "question": "Which program appears on WLAX's syndicated list?",
        "proposed_action": {"action": "search", "query": "WLAX"},
        "before_action": [],
        "retrieved_after_search": [{"id": "wlax", "text": "The list includes Judge Judy."}],
    }


def _search_vote(packet):
    return {
        "case_id": packet["case_id"], "approved": True,
        "reason": "The question names WLAX but provides no list; retrieval supplies it.",
        "risk_labels": [],
        "missing_fact": "The program included on WLAX's syndicated list.",
        "search_needed": "yes", "query_grounded": "yes", "retrieval_relevant": "yes",
        "citations": [
            {"stage": "question", "step": None, "document_id": None,
             "quote": "Which program appears on WLAX's syndicated list?"},
            {"stage": "retrieved_after_search", "step": None, "document_id": "wlax",
             "quote": "The list includes Judge Judy."},
        ],
    }


def test_question_citation_can_support_zero_context_search_gap():
    packet = _search_packet()
    assert valid_vote(packet, _search_vote(packet))[0]
    assert "not evidence for the answer" in SYSTEM


def test_question_citation_must_be_exact_and_documentless():
    packet = _search_packet()
    vote = _search_vote(packet)
    vote["citations"][0]["quote"] = "The WLAX list is absent."
    assert valid_vote(packet, vote) == (False, "question_citation_not_visible")
    vote = _search_vote(packet)
    vote["citations"][0]["document_id"] = "wlax"
    assert valid_vote(packet, vote) == (False, "question_citation_not_visible")


def test_question_citation_cannot_support_an_answer():
    packet = _search_packet()
    packet["proposed_action"] = {"action": "answer", "answer": "Judge Judy"}
    vote = _search_vote(packet)
    vote["citations"] = [vote["citations"][0]]
    assert valid_vote(packet, vote) == (False, "approved_answer_missing_prior_support")


def test_invalid_reviewer_defined_risk_label_rejected():
    packet = _search_packet()
    vote = _search_vote(packet)
    vote["risk_labels"] = ["uncitable_prior_gap"]
    assert valid_vote(packet, vote) == (False, "decision_risk_labels_invalid")


@pytest.mark.parametrize("missing_fact", ["none", "unknown", "not specified"])
def test_question_citation_does_not_relax_missing_fact_gate(missing_fact):
    packet = _search_packet()
    vote = _search_vote(packet)
    vote["missing_fact"] = missing_fact
    assert valid_vote(packet, vote) == (False, "approved_search_missing_specific_support")
