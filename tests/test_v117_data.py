"""Focused checks for immutable-corpus review and fail-closed pair selection."""
from scripts.finalize_v117 import select_pairs, valid_v117_vote
from scripts.prepare_v117_review import structural_issue


def _packet(action, before, after=None, answer="A"):
    return {"proposed_action": {"action": action, "query": "A"} if action == "search"
            else {"action": "answer", "answer": answer},
            "question": "Where was A born?", "before_action": before,
            "retrieved_after_search": after}


def _doc(identity, text):
    return {"id": identity, "title": identity, "text": text}


def test_search_must_add_missing_fact():
    first = _doc("A", "A was a person.")
    second = _doc("B", "A was born in B.")
    source = {"context": {"title": ["A", "B"], "sentences": [[first["text"]], [second["text"]]]},
              "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]}}
    search = _packet("search", [first], [first])
    answer = _packet("answer", [first, second], answer="B")
    assert structural_issue(search, answer, source) == "no_new_document"
    search["retrieved_after_search"] = [second]
    assert structural_issue(search, answer, source) is None


def test_truncation_blocks_lost_answer_fact():
    first = _doc("A", "A was a person.")
    second = _doc("B", "A was born in B.")
    source = {"context": {"title": ["A", "B"], "sentences": [[first["text"]], [second["text"]]]},
              "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]}}
    search = _packet("search", [first], [second])
    answer = _packet("answer", [first], answer="B")
    assert structural_issue(search, answer, source) == "answer_facts_lost_after_truncation"


def test_selector_requires_complete_risk_coverage():
    pairs = []
    for i in range(55):
        label = ["scope"] if i < 2 else ["relation", "time", "measurement"]
        pairs.append((f"id{i:02}", [(None, label), (None, label)]))
    assert select_pairs(pairs) is None
    for i in range(2, 5):
        pairs[i] = (pairs[i][0], [(None, ["scope"]), (None, ["scope"])])
    assert len(select_pairs(pairs)) == 50


def test_search_vote_needs_pre_action_and_retrieval_quotes():
    before = _doc("A", "A was a person.")
    after = _doc("B", "A was born in B.")
    packet = {"proposed_action": {"action": "search", "query": "A"},
              "before_action": [before], "retrieved_after_search": [after]}
    vote = {"approved": True, "reason": "The birthplace is missing before search.",
            "risk_labels": ["relation"], "missing_fact": "birthplace",
            "search_needed": "yes", "query_grounded": "yes", "retrieval_relevant": "yes",
            "citations": [{"stage": "retrieved_after_search", "step": None,
                           "document_id": "B", "quote": "A was born in B."}]}
    assert valid_v117_vote(packet, vote) == (False, "search_needs_before_and_after_quotes")
    vote["citations"].insert(0, {"stage": "before_action", "step": None,
                                 "document_id": "A", "quote": "A was a person."})
    assert valid_v117_vote(packet, vote) == (True, "valid")
