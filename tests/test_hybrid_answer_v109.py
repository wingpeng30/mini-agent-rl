import json

from mini_agent_rl.hybrid_answer import (answer_metrics, anonymize_candidates,
    apply_selection, conservative_boundary_guard, merge_candidates, parse_candidates)

DOCS = [{"id": "d", "text": "The film was directed by Michael Dowse."}]


def test_candidates_allow_composed_answer_but_require_verbatim_quote():
    raw = json.dumps({"candidates": [{"answer": "Dowse", "citations": [
        {"document_id": "d", "quote": "directed by Michael Dowse"}]}]})
    rows, error = parse_candidates(raw, DOCS)
    assert error is None and rows[0]["answer"] == "Dowse"
    bad, _ = parse_candidates(json.dumps({"candidates": [{"answer": "x", "citations": [
        {"document_id": "d", "quote": "not visible"}]}]}), DOCS)
    assert bad == []


def test_anonymous_order_is_deterministic_and_selection_maps_back():
    merged = merge_candidates("wrong", [[{"answer": "Michael Dowse", "citations": [
        {"document_id": "d", "quote": "Michael Dowse"}]}]])
    ordered = anonymize_candidates("task", merged)
    correct = next(item for item in ordered if item["answer"] == "Michael Dowse")
    raw = json.dumps({"decision": "select", "candidate_id": correct["candidate_id"],
                      "citations": [{"document_id": "d", "quote": "Michael Dowse"}]})
    answer, status, error = apply_selection(raw, ordered, "wrong", DOCS)
    assert (answer, status, error) == ("Michael Dowse", "selected", None)


def test_strict_and_normalized_metrics_are_distinct():
    metrics = answer_metrics("The Beatles", "Beatles")
    assert metrics["strict_em"] == 0 and metrics["normalized_em"] == 1


def test_boundary_guard_compresses_explanation_and_preserves_alias_baseline():
    candidates=[{"answer":"Pablo Trapero"},{"answer":"Aleksander Ford"},{"answer":"Aleksander Ford was born first."}]
    assert conservative_boundary_guard("Aleksander Ford was born first.","Pablo Trapero",candidates)[0]=="Aleksander Ford"
    assert conservative_boundary_guard("Glenn Hughes is older.","Glenn Hughes",[{"answer":"Glenn Hughes"},{"answer":"Glenn Hughes is older."}])[0]=="Glenn Hughes"
