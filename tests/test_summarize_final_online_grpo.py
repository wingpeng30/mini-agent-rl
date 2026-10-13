"""Verify result denominators and paired/resource metrics, without model calls."""
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_final_online_grpo.py"
SPEC = importlib.util.spec_from_file_location("final_grpo_summary", SCRIPT)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def row(case, correct=0, f1=0.0, stratum="development"):
    return dict(case_id=case, normalized_exact_match=correct, exact_match=correct,
                f1=f1, stratum=stratum, reason="answer", answer=str(correct),
                elapsed_seconds=1, input_tokens=100, output_tokens=10,
                model_calls=2, searches=1)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_pairs_match_ids_and_retain_both_gains_and_losses():
    before = [row("a", 0), row("b", 1, 1)]
    after = [row("b", 0), row("a", 1, 1)]
    result = reporter.paired(before, after, draws=200)
    assert result["identical_case_sets"]
    assert (result["gains"], result["losses"], result["normalized_em_delta"]) == (1, 1, 0)
    assert {change["change"] for change in result["changed_cases"]} == {"gain", "loss"}
    partial = reporter.paired(before, [after[0]])
    assert not partial["identical_case_sets"]
    assert partial["before_only"] == ["a"]


def test_duplicates_cannot_silently_replace_results():
    with pytest.raises(ValueError, match="Duplicate"):
        reporter.aggregate([row("a"), row("a", 1)], expected=2)


def test_partial_run_does_not_invent_final_resume_claim(tmp_path):
    branch = tmp_path / "branches" / "am-serial"
    branch.mkdir(parents=True)
    (branch / "evaluation.jsonl").write_text(json.dumps(row("a")) + "\n", encoding="utf-8")
    result = reporter.summarize(tmp_path, plot=False)
    assert not result["complete"]
    assert result["groups"]["natural104"]["branches"]["am-serial"]["count"] == 1
    assert "不生成最终模型提升表述" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert not result["new_semantic_review"]


def test_full_data_keeps_natural_and_constructed_denominators_separate(tmp_path):
    data = [row(str(i), correct=i % 2, f1=i % 2,
                stratum="constructed_singlehop" if i >= 104 else "development") for i in range(128)]
    for name in (*reporter.BRANCHES, "am-batch", reporter.BARE_BRANCH):
        folder = tmp_path / "branches" / name
        folder.mkdir(parents=True)
        (folder / "evaluation.jsonl").write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
        write(folder / "evaluation-summary.json", dict(evaluation_wall_seconds=128, current_process_cases=128,
                                                        gpu_memory=dict(peak_reserved_gb=3.5)))
    result = reporter.summarize(tmp_path, plot=False)
    assert result["complete"]
    assert result["groups"]["natural104"]["branches"]["grpo43"]["normalized_em_count"] == 52
    assert result["groups"]["single24"]["branches"]["sft42"]["normalized_em_count"] == 12
    assert result["efficiency"]["batch_throughput_ratio"] == 1
    assert result["training"]["grpo42"]["planned_groups"] == 128
    assert result["training"]["sft42"]["planned_updates"] == 128
    assert result["bare_model"]["groups"]["natural104"]["metrics"]["count"] == 104
    assert result["bare_model"]["groups"]["all128"]["paired_to_branches"]["grpo43"]["pairs"] == 128
    assert reporter.BARE_BRANCH not in result["training"]
    assert (tmp_path / "REPORT.md").read_text(encoding="utf-8").index("## natural104") < (tmp_path / "REPORT.md").read_text(encoding="utf-8").index("## all128")


def test_bare_is_required_and_compares_actual_counts_without_selecting_seed(tmp_path):
    data = [row(str(i), correct=i % 2, f1=i % 2,
                stratum="constructed_singlehop" if i >= 104 else "development") for i in range(128)]
    for name in reporter.BRANCHES:
        folder = tmp_path / "branches" / name
        folder.mkdir(parents=True)
        (folder / "evaluation.jsonl").write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
    assert not reporter.summarize(tmp_path, plot=False)["complete"]
    folder = tmp_path / "branches" / reporter.BARE_BRANCH
    folder.mkdir(parents=True)
    bare = [dict(r, exact_match=0, normalized_exact_match=0, f1=0) for r in data]
    (folder / "evaluation.jsonl").write_text("\n".join(json.dumps(r) for r in bare), encoding="utf-8")
    result = reporter.summarize(tmp_path, plot=False)
    assert result["complete"]
    comparison = result["bare_model"]["groups"]["natural104"]["paired_to_branches"]
    assert set(comparison) == set(reporter.BRANCHES)
    assert comparison["grpo42"]["gains"] == 52
    assert comparison["grpo43"]["normalized_em_delta"] == .5
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "裸模型归一化EM 0/104" in text
    assert "seed42 52/104; seed43 52/104" in text


def test_resume_efficiency_uses_only_timed_segment(tmp_path):
    write(tmp_path / "evaluation-summary.json", dict(evaluation_wall_seconds=2, current_process_cases=1,
                                                     resumed=True, output_tokens=9999,
                                                     gpu_memory=dict(peak_reserved_gb=4)))
    data = [row("old"), dict(row("new"), output_tokens=20)]
    result = reporter.efficiency_metrics(tmp_path, data)
    assert result["questions_per_minute"] == 30
    assert result["output_tokens_per_second"] == 10
    assert result["peak_reserved_gib"] == 4
    assert result["scope"] == "current process segment"


def test_crash_tail_is_visible_and_middle_corruption_is_rejected(tmp_path):
    path = tmp_path / "evaluation.jsonl"
    path.write_text(json.dumps(row("a")) + '\n{"case_id":', encoding="utf-8")
    warnings = []
    assert len(reporter.read_rows(path, warnings)) == 1
    assert warnings
    path.write_text('{bad}\n' + json.dumps(row("b")), encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        reporter.read_rows(path, [])
