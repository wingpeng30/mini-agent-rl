from __future__ import annotations

import json
from pathlib import Path
import uuid

import pytest

from mini_agent_rl.evaluation import answer_metrics
from scripts.evaluate_v133_hybrid_model_pair import (
    ADAPTERS,
    aggregate,
    digest,
    engineering_acceptance,
    paired_delta,
    resume_rows,
    support_coverage,
    validate_citation,
    validate_complete_pair,
    validate_ledger,
    validate_vote,
    visible_documents_from_messages,
    model_visible_messages,
)


class CharacterTokenizer:
    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        value = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            value["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return value


def frozen_case(case_id="c1"):
    return {"id": case_id, "question": "Who founded it?", "stratum": "development"}


def ledger_row(case, arm="goal-z"):
    return {"case_id": case["id"], "question": case["question"], "stratum": case["stratum"],
            "arm": arm, "adapter_sha256": ADAPTERS[arm]["sha256"]}


def test_paired_ledger_requires_exact_complete_question_pairing():
    cases = [frozen_case("c1"), frozen_case("c2")]
    left = [ledger_row(case) for case in cases]
    right = [ledger_row(case, "am-research") for case in cases]
    paired = validate_complete_pair(cases, left, right)
    assert set(paired[0]) == set(paired[1]) == {"c1", "c2"}
    with pytest.raises(ValueError, match="Incomplete ledger"):
        validate_complete_pair(cases, left, right[:1])


def test_identity_drift_duplicate_records_and_resume_are_detected():
    cases = [frozen_case("c1"), frozen_case("c2")]
    one = ledger_row(cases[0])
    assert validate_ledger(cases, [one], "goal-z") == {"c1"}
    assert [case["id"] for case in cases if case["id"] not in validate_ledger(cases, [one], "goal-z")] == ["c2"]
    changed = dict(one, adapter_sha256="wrong")
    with pytest.raises(ValueError, match="identity drift"):
        validate_ledger(cases, [changed], "goal-z")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_ledger(cases, [one, one], "goal-z")


def test_review_deduplication_ignores_nonrendered_random_tool_call_ids():
    left = [{"role": "assistant", "content": "", "tool_calls": [{"id": "call-a", "name": "search"}]},
            {"role": "tool", "content": "result", "tool_call_id": "call-a"}]
    right = [{"role": "assistant", "content": "", "tool_calls": [{"id": "call-b", "name": "search"}]},
             {"role": "tool", "content": "result", "tool_call_id": "call-b"}]
    assert model_visible_messages(left) == model_visible_messages(right)
    assert model_visible_messages(left) == [{"role": "assistant", "content": ""},
                                              {"role": "tool", "content": "result"}]


def test_resume_discards_only_an_incomplete_trailing_jsonl_record():
    temporary = Path(__file__).resolve().parents[1] / "runs" / f"test-resume-tail-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    path = temporary / "ledger.jsonl"
    first = {"case_id": "c1", "answer": "durable"}
    try:
        path.write_text(json.dumps(first) + "\n{\"case_id\": \"c2\", \"answer\":", encoding="utf-8")
        assert resume_rows(path) == [first]
        assert path.read_text(encoding="utf-8") == json.dumps(first) + "\n"
    finally:
        path.unlink(missing_ok=True)
        temporary.rmdir()


def test_first_two_engineering_acceptance_checks_capture_and_retrieval_identity():
    cases = [frozen_case("c1"), frozen_case("c2"), frozen_case("c3")]
    ledger = []
    for case in cases[:2]:
        token_ids = [10, 11, 12]
        ledger.append({"case_id": case["id"], "model_calls": 1, "searches": 1, "over_budget": False,
            "retrieval": {"retriever": "hybrid-rrf-v1", "candidate_k": 20, "rrf_k": 60},
            "trace": [{"raw": "{\"action\":\"answer\"}", "prompt_token_ids": token_ids,
                "actual_prompt_sha256": digest(token_ids),
                "actual_input_messages": [{"role": "user", "content": "Who founded it?"},
                                          {"role": "assistant", "content": "{}"}],
                "context_usage": {"actual_prompt_tokens": 3}}]})
    result = engineering_acceptance(ledger, cases)
    assert result["passed"] and result["acceptance_questions_counted_in_formal_sample"]
    ledger[1]["over_budget"] = True
    with pytest.raises(RuntimeError, match="acceptance failed"):
        engineering_acceptance(ledger, cases)


def test_alias_scoring_keeps_historical_em_f1_and_empty_answers_fail_nonempty_gold():
    alias_score = answer_metrics("beta", "Beta")
    primary_score = answer_metrics("beta", "Alpha")
    assert max(alias_score["normalized_exact_match"], primary_score["normalized_exact_match"]) == 1
    assert max(alias_score["token_f1"], primary_score["token_f1"]) == 1
    empty = answer_metrics(None, "Alpha")
    assert empty == {"strict_exact_match": 0.0, "normalized_exact_match": 0.0, "token_f1": 0.0}


def citation_packet():
    body = "The company was founded by Ada Lovelace in 1843."
    tool_content = json.dumps({"ok": True, "result": [{"id": "doc-1", "title": "Company", "text": body}]},
                              ensure_ascii=False)
    messages = [{"role": "system", "content": "Answer from evidence."},
                {"role": "tool", "content": tool_content}]
    prompt = "Answer from evidence.\n" + tool_content + "\n<assistant>"
    ids = [ord(char) for char in prompt]
    docs = visible_documents_from_messages(messages)
    packet = {"review_id": "r1", "source_status": "confirmed", "termination_reason": "answer",
              "actual_prompt": prompt, "prompt_token_ids": ids, "actual_input_messages": messages,
              "visible_documents": docs, "required_edges": [{"edge_id": "e1", "status": "pass",
                  "source_evidence": {"document_id": "doc-1", "quote": "founded by Ada Lovelace"}}]}
    return packet


def test_citation_must_be_verbatim_visible_and_gets_actual_input_token_span():
    packet = citation_packet()
    tokenizer = CharacterTokenizer()
    quote = "founded by Ada Lovelace"
    citation = {"edge_id": "e1", "document_id": "doc-1", "quote": quote}
    found = validate_citation(packet, citation, tokenizer)
    assert found is not None
    lo, hi = found["input_token_span"]
    assert "".join(chr(token) for token in packet["prompt_token_ids"][lo:hi]) == quote
    vote = {"semantic_category": "correct", "covered_edge_ids": ["e1"], "citations": [citation]}
    assert validate_vote(packet, vote, tokenizer, tokenizer(packet["actual_prompt"], add_special_tokens=False,
                                                            return_offsets_mapping=True))["evidence_supported"]
    bad_quote = dict(citation, quote="invented quotation")
    assert validate_citation(packet, bad_quote, tokenizer) is None


def test_retrieved_but_not_visible_document_does_not_count_as_support():
    packet = citation_packet()
    packet["actual_input_messages"] = [{"role": "system", "content": "Answer from evidence."},
                                       {"role": "tool", "content": json.dumps({"result": []})}]
    assert visible_documents_from_messages(packet["actual_input_messages"]) == []
    coverage = support_coverage({"doc-1"}, {"doc-1"}, set())
    assert coverage["retrieved_support_count"] == 1
    assert coverage["final_visible_support_count"] == 0
    assert coverage["complete_chain_visible"] is False


def test_unresolved_source_is_indeterminate_not_scored_as_wrong():
    packet = citation_packet()
    packet["source_status"] = "unresolved"
    vote = {"semantic_category": "indeterminate", "covered_edge_ids": [], "citations": []}
    checked = validate_vote(packet, vote, CharacterTokenizer(), None)
    assert checked["semantic_category"] == "indeterminate"
    assert checked["evidence_supported"] is False
    with pytest.raises(ValueError, match="Unresolved source"):
        validate_vote(packet, {"semantic_category": "factual_error", "covered_edge_ids": [], "citations": []},
                      CharacterTokenizer(), None)


def test_summary_counts_and_seeded_paired_bootstrap_obey_metric_invariants():
    rows = [
        {"strict_exact_match": 1.0, "normalized_exact_match": 1.0, "token_f1": 1.0,
         "reason": "answer", "semantic_category": "correct", "evidence_supported": True,
         "complete_chain_visible": True, "support_documents_expected": 2,
         "retrieved_support_count": 2, "final_visible_support_count": 2,
         "invalid_actions": 0, "repeated_actions": 0, "over_budget": False,
         "model_calls": 2, "searches": 1, "elapsed_seconds": 1.0, "gpu_question_seconds": 1.0,
         "gpu_peak_allocated_bytes": 100, "gpu_peak_reserved_bytes": 200},
        {"strict_exact_match": 0.0, "normalized_exact_match": 0.0, "token_f1": 0.25,
         "reason": "invalid_action", "semantic_category": "indeterminate", "evidence_supported": False,
         "complete_chain_visible": False, "support_documents_expected": 2,
         "retrieved_support_count": 1, "final_visible_support_count": 0,
         "invalid_actions": 1, "repeated_actions": 0, "over_budget": False,
         "model_calls": 1, "searches": 0, "elapsed_seconds": 3.0, "gpu_question_seconds": 3.0,
         "gpu_peak_allocated_bytes": 100, "gpu_peak_reserved_bytes": 200},
    ]
    summary = aggregate(rows)
    assert summary["semantic_correct"] == 1
    assert summary["semantic_indeterminate"] == 1
    assert summary["evidence_supported_success"] == 1 <= summary["semantic_correct"]
    assert summary["answer_completion_rate"] == 0.5
    before = {"c1": {"metric": 0.0}, "c2": {"metric": 1.0}}
    after = {"c1": {"metric": 1.0}, "c2": {"metric": 0.0}}
    result = paired_delta(before, after, "metric", ["c1", "c2"])
    assert result["pairs"] == 2 and result["improved"] == result["regressed"] == 1
    assert result["mean_delta"] == 0.0 and len(result["bootstrap_95"]) == 2
