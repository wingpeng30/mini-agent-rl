from __future__ import annotations

import unittest

from scripts.v126_review_protocol import valid_vote as old_valid_vote
from scripts.v128_review_protocol import (
    PROTOCOL_VERSION, contract_identity, valid_vote,
)


def packet(*, before=None, retrieved=None, action="search"):
    return {
        "case_id": "case-1",
        "question": "Who formed first, The Get Up Kids or Moonbabies?",
        "proposed_action": {"action": action,
                            "query": "Moonbabies" if action == "search" else None,
                            "answer": "The Get Up Kids" if action == "answer" else None},
        "before_action": before or [],
        "retrieved_after_search": retrieved or [],
    }


def citation(stage, quote, doc_id=None):
    return {"stage": stage, "step": None, "document_id": doc_id, "quote": quote}


def vote(*citations, approved=True, **extra):
    return {
        "case_id": "case-1", "approved": approved, "reason": "Evidence was checked.",
        "risk_labels": [], "missing_fact": "Moonbabies formation year",
        "search_needed": "yes", "query_grounded": "yes", "retrieval_relevant": "yes",
        "citations": list(citations), **extra,
    }


class ReviewProtocolV128Tests(unittest.TestCase):
    def test_old_protocol_accepts_question_only_prior_citation_even_when_documents_exist(self):
        p = packet(
            before=[{"id": "known", "text": "The Get Up Kids formed in 1995."}],
            retrieved=[{"id": "missing", "text": "Moonbabies formed in 1997."}],
        )
        v = vote(
            citation("question", p["question"]),
            citation("retrieved_after_search", "Moonbabies formed in 1997.", "missing"),
        )
        self.assertTrue(old_valid_vote(p, v)[0])
        self.assertEqual(valid_vote(p, v),
                         (False, "approved_search_missing_specific_temporal_support"))

    def test_search_with_prior_documents_cites_gap_and_new_evidence(self):
        p = packet(
            before=[{"id": "known", "text": "The Get Up Kids formed in 1995."}],
            retrieved=[{"id": "missing", "text": "Moonbabies formed in 1997."}],
        )
        v = vote(
            citation("before_action", "The Get Up Kids formed in 1995.", "known"),
            citation("retrieved_after_search", "Moonbabies formed in 1997.", "missing"),
        )
        self.assertTrue(valid_vote(p, v)[0])

    def test_zero_context_search_can_use_question_as_prior_context(self):
        p = packet(retrieved=[{"id": "r", "text": "Moonbabies formed in 1997."}])
        v = vote(citation("question", p["question"]),
                 citation("retrieved_after_search", "Moonbabies formed in 1997.", "r"))
        self.assertTrue(valid_vote(p, v)[0])

    def test_whitespace_quote_that_old_validator_accepts_is_rejected(self):
        p = packet(before=[{"id": "d", "text": "Prior evidence is visible. "}], action="answer")
        v = vote(citation("before_action", " ", "d"), search_needed="no",
                 query_grounded="no", retrieval_relevant="no")
        self.assertTrue(old_valid_vote(p, v)[0])
        self.assertEqual(valid_vote(p, v), (False, "citation_quote_empty"))

    def test_answer_support_can_span_documents_without_splicing_quotes(self):
        p = packet(before=[
            {"id": "a", "text": "The event happened in 1995."},
            {"id": "b", "text": "That event was The Get Up Kids forming."},
        ], action="answer")
        v = vote(
            citation("before_action", "The event happened in 1995.", "a"),
            citation("before_action", "That event was The Get Up Kids forming.", "b"),
        )
        self.assertTrue(valid_vote(p, v)[0])
        v["citations"] = [citation("before_action",
                                   "The event happened in 1995. That event was The Get Up Kids forming.", "a")]
        self.assertEqual(valid_vote(p, v),
                         (False, "citation_not_visible_in_one_document_at_stage"))

    def test_answer_cannot_cite_future_retrieval(self):
        p = packet(retrieved=[{"id": "later", "text": "The answer appears later."}], action="answer")
        v = vote(citation("retrieved_after_search", "The answer appears later.", "later"),
                 search_needed="no", query_grounded="no", retrieval_relevant="no")
        self.assertEqual(valid_vote(p, v), (False, "answer_citation_outside_action_time"))

    def test_conflicting_duplicate_document_id_is_rejected(self):
        p = packet(before=[{"id": "same", "text": "Version A."},
                           {"id": "same", "text": "Version B."}],
                   retrieved=[{"id": "r", "text": "New evidence."}])
        v = vote(citation("before_action", "Version A.", "same"),
                 citation("retrieved_after_search", "New evidence.", "r"))
        self.assertEqual(valid_vote(p, v),
                         (False, "duplicate_document_id_conflicting_text"))

    def test_duplicate_risk_labels_are_not_a_valid_vote(self):
        p = packet(retrieved=[{"id": "r", "text": "New evidence."}])
        v = vote(citation("question", p["question"]),
                 citation("retrieved_after_search", "New evidence.", "r"),
                 risk_labels=["relation", "relation"])
        self.assertEqual(valid_vote(p, v), (False, "decision_risk_labels_invalid"))

    def test_contract_identity_is_versioned_and_fingerprinted(self):
        identity = contract_identity()
        self.assertEqual(identity["version"], PROTOCOL_VERSION)
        self.assertEqual(set(identity), {"version", "system_sha256", "validator_sha256"})
        self.assertTrue(all(len(identity[k]) == 64 for k in ("system_sha256", "validator_sha256")))


if __name__ == "__main__":
    unittest.main()
