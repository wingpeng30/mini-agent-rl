from __future__ import annotations

import pytest

from scripts.v125_review_protocol import SYSTEM, validate, valid_vote


def _packet(action="answer"):
    return {
        "case_id": "sentinel-1",
        "proposed_action": {"action": action},
        "before_action": [
            {"id": "person", "text": "Mira was born in Lyon."},
            {"id": "venue", "text": "The concert took place at the Hall of Mirrors."},
        ],
        "retrieved_after_search": [
            {"id": "retrieved", "text": "The Hall of Mirrors is in Lyon."},
        ],
    }


def test_multidocument_join_accepts_separate_exact_citations():
    packet = _packet()
    packet["before_action"].append(
        {"id": "city", "text": "The Hall of Mirrors is in Lyon."})
    vote = {
        "case_id": packet["case_id"], "approved": True,
        "reason": "The venue document places the event at the Hall of Mirrors; the venue document separately locates that hall in Lyon.",
        "risk_labels": ["relation"], "missing_fact": "",
        "search_needed": "not_search", "query_grounded": "not_search",
        "retrieval_relevant": "not_search",
        "citations": [
            {"stage": "before_action", "step": None, "document_id": "venue",
             "quote": "The concert took place at the Hall of Mirrors."},
            {"stage": "before_action", "step": None, "document_id": "city",
             "quote": "The Hall of Mirrors is in Lyon."},
        ],
    }
    validate(packet, vote)
    assert "separate documents" in SYSTEM


def test_composite_quote_cannot_be_fabricated_across_documents():
    packet = _packet()
    vote = {
        "case_id": packet["case_id"], "approved": True, "reason": "joined",
        "risk_labels": ["relation"], "missing_fact": "",
        "search_needed": "not_search", "query_grounded": "not_search",
        "retrieval_relevant": "not_search",
        "citations": [{"stage": "before_action", "step": None,
                       "document_id": "venue",
                       "quote": "The concert took place at the Hall of Mirrors. The Hall is in Lyon."}],
    }
    with pytest.raises(ValueError, match="citation_not_visible_in_one_document"):
        validate(packet, vote)


@pytest.mark.parametrize("missing_fact", ["none", "unknown", "not specified", " ", "?"])
def test_approved_search_rejects_nonspecific_missing_fact(missing_fact):
    packet = _packet("search")
    vote = {
        "case_id": packet["case_id"], "approved": True, "reason": "retrieval helps",
        "risk_labels": ["relation"], "missing_fact": missing_fact,
        "search_needed": "yes", "query_grounded": "yes", "retrieval_relevant": "yes",
        "citations": [
            {"stage": "before_action", "step": None, "document_id": "venue",
             "quote": "The concert took place at the Hall of Mirrors."},
            {"stage": "retrieved_after_search", "step": None, "document_id": "retrieved",
             "quote": "The Hall of Mirrors is in Lyon."},
        ],
    }
    assert valid_vote(packet, vote) == (False, "approved_search_missing_specific_support")


def test_search_missing_fact_must_identify_the_gap():
    packet = _packet("search")
    vote = {
        "case_id": packet["case_id"], "approved": True, "reason": "needed city link",
        "risk_labels": ["relation"],
        "missing_fact": "the city where the Hall of Mirrors is located",
        "search_needed": "yes", "query_grounded": "yes", "retrieval_relevant": "yes",
        "citations": [
            {"stage": "before_action", "step": None, "document_id": "venue",
             "quote": "The concert took place at the Hall of Mirrors."},
            {"stage": "retrieved_after_search", "step": None, "document_id": "retrieved",
             "quote": "The Hall of Mirrors is in Lyon."},
        ],
    }
    assert valid_vote(packet, vote)[0]


def test_answer_cannot_use_future_retrieval_as_support():
    packet = _packet()
    vote = {
        "case_id": packet["case_id"], "approved": True, "reason": "future result",
        "risk_labels": ["relation"], "missing_fact": "",
        "search_needed": "not_search", "query_grounded": "not_search",
        "retrieval_relevant": "not_search",
        "citations": [{"stage": "retrieved_after_search", "step": None,
                       "document_id": "retrieved", "quote": "The Hall of Mirrors is in Lyon."}],
    }
    with pytest.raises(ValueError, match="approved_answer_missing_prior_support"):
        validate(packet, vote)
