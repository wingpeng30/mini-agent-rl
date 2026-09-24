import json

from mini_agent_rl.answer_selector import (apply_label, correct_option_labels,
    dedupe_key, ordered_options, parse_candidates)

DOCS = [{"id": "d1", "title": "t", "text": "Alpha was founded in 1901. Beta began in 1908."}]

def test_candidate_spans_are_exact_unique_and_exclude_original():
    raw = json.dumps({"candidates": ["Beta", "Beta", "Gamma", " 1901"]})
    rows, error = parse_candidates(raw, "Alpha", DOCS)
    assert error is None
    assert rows == [{"answer": "Beta", "document_id": "d1", "start": 27, "end": 31}]

def test_bad_json_produces_no_candidates():
    rows, error = parse_candidates("not json", "Alpha", DOCS)
    assert rows == [] and error.startswith("invalid_json")

def test_deterministic_permutation_and_none_target():
    alternatives = [{"answer": "Beta", "document_id": "d1", "start": 27, "end": 31}]
    assert ordered_options("x", "Alpha", alternatives) == ordered_options("x", "Alpha", alternatives)
    options = ordered_options("x", "Alpha", alternatives)
    assert correct_option_labels(options, "missing") == ["NONE"]
    assert correct_option_labels(options, "beta") != ["NONE"]

def test_apply_label_falls_back_and_abstains():
    options = [{"answer": "Alpha"}, {"answer": "Beta"}]
    assert apply_label("NONE", options, "Alpha").answer == "Alpha"
    assert apply_label("F", options, "Alpha").status == "fallback_invalid_label"
    assert apply_label("B", options, "Alpha").answer == "Beta"

def test_dedupe_key_normalizes_candidate():
    assert dedupe_key("t", "h", "The Alpha", "keep") == dedupe_key("t", "h", "alpha", "keep")
