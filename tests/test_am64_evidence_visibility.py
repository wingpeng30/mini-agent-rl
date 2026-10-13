"""Regression protection for the offline audit's important measurement errors."""
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import analyze_am_evidence_visibility as audit


def test_unknown_answer_is_not_coerced_to_incorrect():
    assert audit.classify(True, None) is None
    assert audit.classify(False, None) is None
    with pytest.raises(ValueError):
        audit.classify(True, 0)


def test_pending_complete_evidence_stays_in_visibility_denominator():
    rows = [
        dict(case_id="correct", classification=audit.LABELS[0], all_necessary_evidence_visible=True,
             answer_correct=True, strict_exact_match=0, normalized_exact_match=0, review_origin="new"),
        dict(case_id="pending", classification=None, all_necessary_evidence_visible=True,
             answer_correct=None, strict_exact_match=0, normalized_exact_match=0, review_origin="new"),
        dict(case_id="incomplete", classification=audit.LABELS[3], all_necessary_evidence_visible=False,
             answer_correct=False, strict_exact_match=0, normalized_exact_match=0, review_origin="new"),
    ]
    summary = audit.summarize(rows)
    assert summary["all_necessary_evidence_visibility_rate"] == 2 / 3
    assert summary["classified"] == 2 and summary["pending_count"] == 1
    complete = summary["conditional_answer_accuracy"]["complete"]
    assert complete["conditional_accuracy"] == 1
    assert complete["full_group_accuracy_bounds"] == [0.5, 1]


def test_quote_is_mapped_to_its_document_body_not_question_or_other_document():
    # A hostile-looking repeated quotation must not cause a question citation,
    # wrong-document citation, or incorrectly unescaped character offsets.
    quote = 'He said "yes".\nBorn in Montréal.'
    prompt = "Question: " + quote + "\n<tool_response>" + json.dumps({"result": [
        {"id": "wrong", "title": "Other", "text": quote},
        {"id": "target", "title": "Target", "text": "prefix " + quote + " suffix"},
    ]}, ensure_ascii=False) + "</tool_response>"
    # Character-sized synthetic offsets make the escaping invariant explicit.
    offsets = [(i, i + 1) for i in range(len(prompt))]
    found = audit.citation(prompt, offsets, audit.documents(prompt), "target", quote)
    assert found["title"] == "Target"
    assert found["body_char_span"] == [7, 7 + len(quote)]
    a, b = found["input_char_span"]
    assert prompt[a:b] == json.dumps(quote, ensure_ascii=False)[1:-1]
    assert a > prompt.index('"id": "target"')
    assert audit.citation(prompt, offsets, audit.documents(prompt), "target", "invented quote") is None


def test_identical_bodies_keep_their_own_document_offsets():
    prompt = "<tool_response>" + json.dumps({"result": [
        {"id": "first", "title": "First", "text": "identical body"},
        {"id": "second", "title": "Second", "text": "identical body"},
    ]}) + "</tool_response>"
    offsets = [(i, i+1) for i in range(len(prompt))]
    found = audit.citation(prompt, offsets, audit.documents(prompt), "second", "identical body")
    assert found["input_char_span"][0] > prompt.index('"id": "second"')


def test_a_returned_after_last_decision_document_is_not_final_visible():
    prompt = '<tool_response>{"result":[{"id":"d1","title":"One","text":"visible"}]}</tool_response>'
    docs = audit.documents(prompt)
    assert {d["id"] for d in docs} == {"d1"}
    assert audit.citation(prompt, [(i, i+1) for i in range(len(prompt))], docs, "d2", "unconsumed retrieval") is None


def test_normalization_matches_frozen_accent_and_article_policy():
    assert audit.normalize("The Kraków!") == "krakow"
    assert audit.normalize("Rabbi Dovber Schneuri") != audit.normalize("Dovber Schneuri")
    # Honorific equivalence is semantic adjudication, never silently added to EM.


@pytest.mark.skipif(not (audit.OUT / "per-case.jsonl").exists(), reason="Local archived evaluation artifacts are optional in CI")
def test_all64_saved_results_keep_pending_and_original_scores():
    results = audit.rows(audit.OUT / "per-case.jsonl")
    originals = {r["case_id"]: r for r in audit.rows(audit.AM / "eval-42-A.jsonl")}
    assert len(results) == len({r["case_id"] for r in results}) == 64
    assert {r["case_id"] for r in results} == set(originals)
    for r in results:
        source = originals[r["case_id"]]
        assert r["answer"] == source["answer"]
        assert r["historical_em_unchanged"] == source["em"]
        assert r["historical_f1_unchanged"] == source["f1"]
        assert r["classification"] == audit.classify(r["all_necessary_evidence_visible"], r["answer_correct"])
    # These body relations must stay incomplete despite perfect answer matching.
    by_id = {r["case_id"]: r for r in results}
    for cid in ("2wiki:66772eb80bde11eba7f7acde48001122", "2wiki:d8c06f220bdc11eba7f7acde48001122"):
        assert by_id[cid]["strict_exact_match"] == 1
        assert by_id[cid]["classification"] == audit.LABELS[2]
