from scripts.v130_review_protocol import PROTOCOL_VERSION, SYSTEM, valid_vote


def base_vote(case_id, citations):
    return {
        "case_id": case_id,
        "approved": False,
        "reason": "The proposed action lacks support in the visible evidence.",
        "risk_labels": [],
        "missing_fact": "",
        "search_needed": "no",
        "query_grounded": "no",
        "retrieval_relevant": "no",
        "citations": citations,
    }


def test_answer_citations_must_be_document_evidence_not_question_text():
    packet = {
        "case_id": "answer-1",
        "question": "What kind of music do they make?",
        "proposed_action": {"action": "answer", "answer": "rock"},
        "before_action": [{"id": "doc-1", "text": "The band makes rock music."}],
        "retrieved_after_search": [],
    }
    question_cited = base_vote("answer-1", [
        {"stage": "question", "step": None, "document_id": None,
         "quote": "What kind of music do they make?"},
        {"stage": "before_action", "step": None, "document_id": "doc-1",
         "quote": "The band makes rock music."},
    ])
    valid, reason = valid_vote(packet, question_cited)
    assert not valid
    assert reason == "question_citation_not_visible_or_wrong_action"

    document_only = base_vote("answer-1", [
        {"stage": "before_action", "step": None, "document_id": "doc-1",
         "quote": "The band makes rock music."},
    ])
    assert valid_vote(packet, document_only)[0]


def test_search_quotes_must_match_one_document_exactly():
    packet = {
        "case_id": "search-1",
        "question": "Which genre do both artists make?",
        "proposed_action": {"action": "search", "query": "Artist genre"},
        "before_action": [{"id": "prior", "text": "The source names both artists."}],
        "retrieved_after_search": [{"id": "result", "text": "One exact ASCII fact."}],
    }
    malformed = base_vote("search-1", [
        {"stage": "before_action", "step": None, "document_id": "prior",
         "quote": "The source names both artists."},
        {"stage": "retrieved_after_search", "step": None, "document_id": "result",
         "quote": "One exact ASCII fact. plus an invented Unicode suffix"},
    ])
    assert not valid_vote(packet, malformed)[0]

    exact = base_vote("search-1", [
        {"stage": "before_action", "step": None, "document_id": "prior",
         "quote": "names both artists"},
        {"stage": "retrieved_after_search", "step": None, "document_id": "result",
         "quote": "One exact ASCII fact."},
    ])
    assert valid_vote(packet, exact)[0]


def test_protocol_identity_and_prompt_bind_explicit_citation_rules():
    prompt = " ".join(SYSTEM.split())
    assert PROTOCOL_VERSION == "v130-evidence-review-citations-v1"
    assert "Never cite the question itself" in prompt
    assert "short, contiguous, verbatim substring" in prompt
    assert "Do not paraphrase, normalize punctuation, repair encoding" in prompt
