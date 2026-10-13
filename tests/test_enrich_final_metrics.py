"""CPU-only checks for existing-label metrics and saved-evidence denominators."""
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "enrich_final_metrics.py"
SPEC = importlib.util.spec_from_file_location("enrich_final_metrics", SCRIPT)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v) + "\n" for v in values), encoding="utf-8")


def preserved_context():
    return dict(observation_encoding="legacy", truncation_stage="none", prompt_truncated=False,
                lossy_body_trimmed=False, history_dropped=False, context_budget_error=False,
                documents_deleted=0, document_changes=[], presentation={"observation_presentation": "legacy-json"})


def test_alias_precision_and_recall_share_best_f1_reference():
    scores = reporter.alias_scores({"answer": "alpha beta gamma"},
        {"answer": "alpha", "answer_aliases": ["alpha beta gamma delta"]}, {})
    assert scores["selected_reference"] == "alpha beta gamma delta"
    assert (scores["precision"], scores["recall"], scores["f1"]) == (1, .75, 6 / 7)
    assert reporter.token_scores("The New-York!", "newyork") == dict(precision=1, recall=1, f1=1)


def test_final_call_newly_returned_observation_is_not_visible():
    row = {"trace": [
        {"observation": {"ok": True, "result": [{"id": "a", "text": "First support."}]}},
        {"context_usage": preserved_context(), "observation": {"ok": True, "result": [{"id": "b", "text": "Second support."}]}}]}
    audit = dict(status="confirmed", support_documents=[
        {"document_id": "a", "quote": "First support."}, {"document_id": "b", "quote": "Second support."}])
    coverage = reporter.source_coverage(row, audit)
    assert coverage["all_required_documents_retrieved"]
    assert coverage["all_required_quotes_retrieved"]
    assert coverage["last_visible_document_recall"] == .5
    assert coverage["last_visible_quote_recall"] == .5
    assert not coverage["all_required_documents_last_visible"]


def test_unknown_annotations_are_na_and_not_part_of_source_denominator():
    known = reporter.case_metrics({"case_id": "a", "answer": "x", "gold": "x", "trace": []}, {},
        dict(status="confirmed", support_documents=[{"document_id": "a", "quote": "x"}]))
    unknown = reporter.case_metrics({"case_id": "b", "answer": "x", "gold": "x", "trace": []}, {},
        dict(status="unresolved", support_documents=[{"document_id": "b", "quote": "x"}]))
    assert unknown["source"]["all_required_documents_retrieved"] is None
    aggregate = reporter.aggregate([known, unknown])
    assert aggregate["source"]["measurable_cases"] == 1
    assert aggregate["source"]["unknown_cases"] == 1
    assert aggregate["source"]["all_required_documents_retrieved"]["count"] == 1
    assert aggregate["source"]["last_visible_document_recall"]["missing"] == 1


def test_visible_segments_preserve_only_recorded_quote_span():
    metadata = {"document_changes": [dict(document_id="a", original_text="first. second.", visible=True,
                                          visible_segments=[[0, 6]])]}
    row = {"trace": [{"context_usage": metadata}]}
    audit = dict(status="confirmed", support_documents=[{"document_id": "a", "quote": "second."}])
    coverage = reporter.source_coverage(row, audit)
    assert coverage["all_required_documents_last_visible"]
    assert coverage["last_visible_quote_recall"] == 0
    assert reporter.last_visible_documents([{"context_usage": {}}])[0] is None


def test_training_rewards_entropy_physical_attempts_and_kl(tmp_path):
    write_json(tmp_path / "summary.json", {"updates": 2, "training_seconds": 100, "status": "COMPLETED"})
    write_rows(tmp_path / "group-metrics.jsonl", [
        dict(group=1, rewards=[0, 0, 0, 0], reward_mean=0, reward_std=0, updated=False, optimizer_steps=0),
        dict(group=2, rewards=[0, 1, 0, 1], reward_mean=.5, reward_std=.5, updated=True, optimizer_steps=2),
        dict(group=3, rewards=[1, 1, 1, 1], reward_mean=1, reward_std=0, updated=False, optimizer_steps=0)])
    write_rows(tmp_path / "updates.jsonl", [dict(update=1, kl=.1, loss=.2, grad=2, elapsed_seconds=10),
                                             dict(update=2, kl=.3, loss=.4, grad=.5, elapsed_seconds=20)])
    write_rows(tmp_path / "generation-attempts.jsonl", [dict(event="started", rows=4),
        dict(event="completed", rows=4, elapsed_seconds=8), dict(event="started", rows=4)])
    metrics = reporter.training_metrics(tmp_path, [])
    group = metrics["logical_groups"]
    assert (group["mixed"], group["all_correct"], group["all_wrong"], group["effective_groups"]) == (1, 1, 1, 1)
    assert group["outcome_entropy_bits"]["mean"] == pytest.approx(1 / 3)
    assert metrics["policy_entropy"] is None
    assert metrics["optimizer"]["metrics"]["kl"]["mean"] == pytest.approx(.2)
    assert metrics["optimizer"]["gradient_clipping_updates"] == 1
    assert metrics["physical_ledger"]["uncompleted_batch_attempts"] == 1
    assert metrics["cost"]["completed_generation_batch_seconds"]["sum"] == 8


def test_report_supports_bare_and_missing_source_audit_without_model(tmp_path):
    cases_path = tmp_path / "existing-cases.jsonl"
    write_rows(cases_path, [{"id": "a", "answer": "alpha", "answer_aliases": ["a beta"], "stratum": "development"}])
    write_json(tmp_path / "manifest.json", {"identity": {"eval_cases": str(cases_path)}})
    for branch in reporter.BRANCHES:
        folder = tmp_path / "branches" / branch
        write_rows(folder / "evaluation.jsonl", [dict(case_id="a", answer="alpha", gold="alpha", stratum="development",
            normalized_exact_match=1, reason="answer", input_tokens=20, output_tokens=5, model_calls=1, searches=0, trace=[])])
        write_json(folder / "evaluation-summary.json", dict(status="COMPLETED", evaluation_wall_seconds=2,
            questions_per_second=.5, current_process_cases=1, resumed=branch == "grpo42"))
    result = reporter.enrich(tmp_path, audit_path=tmp_path / "missing-source-audit.jsonl")
    assert result["complete"]
    assert not result["new_semantic_review"] and result["new_model_calls"] == 0
    assert result["inputs"]["cases"]["sha256"]
    assert result["evaluation"]["base-untrained"]["groups"]["all128"]["source"]["unknown_cases"] == 1
    assert result["evaluation"]["grpo42"]["cost"]["output_tokens_per_second"] is None
    assert result["evaluation"]["sft42"]["cost"]["output_tokens_per_second"] == 2.5
    assert not result["paired_bootstrap"]["recomputed"]
    assert (tmp_path / "comprehensive-metrics.md").is_file()


def test_percentile_missing_and_duplicate_ids_are_explicit():
    stats = reporter.distribution([0, 10, None])
    assert (stats["p50"], stats["p90"], stats["p95"], stats["p99"]) == (5, 9, 9.5, 9.9)
    assert (stats["count"], stats["missing"]) == (2, 1)
    with pytest.raises(ValueError, match="duplicate"):
        reporter.index([{"case_id": "a"}, {"case_id": "a"}], "test")


def test_group_diversity_and_sft_process_nll_do_not_require_forward(tmp_path):
    group = dict(group=1, case_id="a", policy_version="p1", rewards=[0, 0, 1, 1])
    rollouts = [dict(group=1, case_id="a", policy_version="p1", candidate=i, answer=answer,
                     trace=[dict(action="search", query="q" if i < 2 else "other"), dict(action="answer", answer=answer)])
                for i, answer in enumerate([None, "", "The Paris", "Paris"])]
    diversity = reporter.group_diversity(group, rollouts)
    assert diversity["unique_normalized_answers_including_empty"] == 2
    assert diversity["unique_nonempty_normalized_answers"] == 1
    assert diversity["empty_candidate_count"] == 2
    assert diversity["unique_query_sequences"] == diversity["unique_action_sequences"] == 2
    assert diversity["nonempty_answer_pair_token_f1"]["mean"] == 1
    write_rows(tmp_path / "updates.jsonl", [dict(update=1, loss=2, supervised_tokens=1),
                                             dict(update=2, loss=4, supervised_tokens=3)])
    result = reporter.training_metrics(tmp_path, [])
    assert result["sft_process_nll"]["token_weighted_nll"] == 3.5
    assert result["sft_process_nll"]["exp_process_nll"] == pytest.approx(33.1154519587)
