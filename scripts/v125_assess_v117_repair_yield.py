"""Bound the presently replayable v117 repair candidates without promoting labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_v114_sft_candidates import norm
from scripts.v123_cpu_core import ROOT, readl, write
from scripts.v124_forensic import OUT


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    replay_states_path = OUT / "repair-draft-states-cpu-v2-private.jsonl"
    trajectories_path = OUT / "repair-draft-trajectories-cpu-private.jsonl"
    trajectory_report_path = OUT / "repair-draft-trajectory-replay-report-private.json"
    core_path = OUT / "train-core-candidate-private.jsonl"
    backlog_path = OUT / "v117-remaining-repair-backlog-private.json"
    repair_states = readl(replay_states_path)
    trajectories = readl(trajectories_path)
    trajectory_report = json.loads(trajectory_report_path.read_text(encoding="utf-8"))
    strict_link_path = OUT / "v125-v117-repair-trajectory-link-audit-private.json"
    strict_links = json.loads(strict_link_path.read_text(encoding="utf-8"))
    core_rows = readl(core_path)
    backlog = json.loads(backlog_path.read_text(encoding="utf-8"))
    if trajectory_report["all_original_rows_unchanged"] is not True:
        raise ValueError("original_v117_rows_changed")
    if trajectory_report["training_allowed"] is not False:
        raise ValueError("repair_replay_must_remain_nontraining")

    core_questions = {norm(row["messages"][1]["content"]) for row in core_rows}
    paired_trajectory_groups = [group for group in trajectories
                                if group["contains_search_and_answer"]]
    overlap_groups = [group for group in trajectories
                      if norm(group["question"]) in core_questions]
    novel_questions = {norm(group["question"]) for group in paired_trajectory_groups
                       if norm(group["question"]) not in core_questions}
    action_mixed_v2 = [group for group in repair_states
                       if {state["proposed_action"]["action"] for state in group["states"]}
                       == {"search", "answer"}]
    review_manifest = json.loads((OUT / "repair-draft-review-manifest.json").read_text(
        encoding="utf-8"))
    api_journal = readl(OUT / "repair-deepseek-review-journal-private.jsonl")
    v6_journal = readl(OUT / "repair-v6-deepseek-review-journal-private.jsonl")
    v7_journal = readl(OUT / "repair-v7-deepseek-review-journal-private.jsonl")
    v8_status = json.loads((OUT / "repair-v8-audit-status-private.json").read_text(
        encoding="utf-8"))

    report = {
        "version": "v125-v117-repair-yield-bound-v1",
        "inputs": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
                   for path in (replay_states_path, trajectories_path, trajectory_report_path,
                                strict_link_path, core_path, backlog_path)},
        "existing_repair_backlog": {
            "flagged_states": backlog["unresolved_flagged_states"],
            "normalized_question_groups": backlog["unresolved_normalized_question_groups"],
            "current_cpu_replays": {"groups": len(repair_states),
                "states": sum(len(group["states"]) for group in repair_states),
                "groups_with_search_and_answer_actions": len(action_mixed_v2),
                "all_group_token_limits_pass": all(g["token_limits_passed"] for g in repair_states)},
            "rebuilt_trajectory_replays": {"groups": len(trajectories),
                "states": sum(len(group["states"]) for group in trajectories),
                "groups_with_search_and_answer": len(paired_trajectory_groups),
                "question_groups_overlapping_retained_core": len(overlap_groups),
                "new_question_candidates_upper_before_strict_pair_recheck": len(novel_questions),
                "strict_link_candidates": strict_links["strict_action_result_citation_links"],
                "strict_new_repair_pair_candidates_before_double_review": strict_links[
                    "new_repair_pair_candidates_before_double_review"],
                "quarantined_strict_link_candidates": strict_links[
                    "strict_link_candidates_still_quarantined"],
                "strict_original_record_bindings_pass": strict_links[
                    "original_record_bindings_pass"],
                "all_token_limits_pass": all(g["token_limits_passed"] for g in trajectories),
                "strict_pairing_semantics_and_double_review": "not_run"},
        },
        "review_status": {
            "current_anonymous_review_packet": {"groups": review_manifest["groups"],
                "states": review_manifest["states"], "luna_reviews": "not_recorded"},
            "deepseek_current_packet_final_status": api_journal[-1]["status"],
            "deepseek_v6_packet_final_status": v6_journal[-1]["status"],
            "deepseek_v7_packet_final_status": v7_journal[-1]["status"],
            "deepseek_v8_status": v8_status["deepseek_vote_status"],
            "v8_luna_status": v8_status["luna_vote_status"],
            "unknown_outcomes_retried": False,
        },
        "interpretation": [
            "CPU/token replay establishes neither semantic approval nor a training label.",
            "The upper-bound count is the number of rebuilt paired-question groups disjoint from the retained core; strict action/result/citation linkage still must be checked.",
            "Outcome-unknown DeepSeek requests remain unsettled and are not resent automatically.",
            "No new query or answer is created from gold, and no review vote is changed.",
        ],
        "training_allowed": False,
    }
    write(OUT / "v125-v117-repair-yield-bound-private.json", report)
    print(json.dumps({"backlog_states": report["existing_repair_backlog"]["flagged_states"],
        "cpu_replay_groups": len(repair_states),
        "rebuilt_trajectory_groups": len(trajectories),
        "paired_trajectory_groups": len(paired_trajectory_groups),
        "new_question_candidates_upper_before_strict_pair_recheck": len(novel_questions),
        "strict_action_result_citation_links": strict_links["strict_action_result_citation_links"],
        "new_repair_pair_candidates_before_double_review": strict_links[
            "new_repair_pair_candidates_before_double_review"],
        "review_status": report["review_status"], "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
