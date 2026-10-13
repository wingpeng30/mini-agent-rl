"""Bound the best-case impact of currently evidenced repair/extraction candidates."""
from __future__ import annotations

import hashlib
import json

from scripts.build_v114_sft_candidates import norm
from scripts.prepare_v116_data import full_rows
from scripts.v123_cpu_core import ROOT, readl, write
from scripts.v124_forensic import OUT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = OUT / "v126-recovery-upper-bound-v4-private.json"
    if out.exists():
        raise FileExistsError(out)
    snapshot_path = OUT / "current-trainability-snapshot-private.json"
    core_path = OUT / "train-core-candidate-private.jsonl"
    preflight_path = OUT / "train-core-candidate-preflight-private.json"
    supp_validation_path = OUT / "v122-train-luna-review-validation-private.json"
    supplement_manifest_path = OUT / "double-reviewed-v122-supplement-manifest-private.json"
    repair_links_path = OUT / "v125-v117-repair-trajectory-link-audit-private.json"
    quarantine_replay_path = OUT / "v126-quarantined-repair-replay-candidates-private.jsonl"
    repair_replay_path = OUT / "v126-deduped-pair-replay-candidates-private.jsonl"
    extraction_report_path = OUT / "v126-retained-linked-search-extraction-report-private.json"
    extraction_path = OUT / "v126-retained-linked-search-state-candidates-private.jsonl"
    source_path = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
    paths = [source_path, snapshot_path, core_path, preflight_path, supp_validation_path,
             supplement_manifest_path, repair_links_path, quarantine_replay_path,
             repair_replay_path, extraction_report_path, extraction_path]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    supp_validation = json.loads(supp_validation_path.read_text(encoding="utf-8"))
    core_rows = readl(core_path)
    source_rows = readl(ROOT / "data/evidence-sft-v117/frozen/train.jsonl")
    by_record_id = {f"v117_train:{i:04d}": row for i, row in enumerate(source_rows, 1)}
    core_signatures = {(json.dumps(row["messages"], ensure_ascii=False, separators=(",", ":")),
                       row["target_assistant_index"]) for row in core_rows}
    def is_in_core(record_id):
        row = by_record_id[record_id]
        signature = (json.dumps(row["messages"], ensure_ascii=False, separators=(",", ":")),
                     row["target_assistant_index"])
        return signature in core_signatures
    selected_ids = supp_validation["selected_source_ids_private"]
    selected_rows = full_rows(selected_ids)
    core_questions = {norm(row["messages"][1]["content"]) for row in core_rows}
    supplement_questions = {norm(selected_rows[source_id]["question"]) for source_id in selected_ids}
    if len(core_questions) != 150 or len(supplement_questions) != 25:
        raise ValueError("base_question_count_changed")

    # Candidate questions already in core/supplement do not increase distinct-question count.
    extraction = readl(extraction_path)
    extraction_question_set = {norm(r["question"]) for r in extraction}
    extraction_source_ids = {r["source_search_record_id_private"] for r in extraction}
    extraction_answer_ids = {r["source_answer_record_id_private"] for r in extraction}
    if (len(extraction_question_set) != len(extraction) or
            not extraction_question_set <= core_questions or
            not all(is_in_core(rid) for rid in extraction_source_ids | extraction_answer_ids)):
        raise ValueError("extracted_questions_not_all_in_core")
    repair_pairs = readl(quarantine_replay_path)
    trajectory_groups = readl(OUT / "repair-draft-trajectories-cpu-private.jsonl")
    strict = json.loads(repair_links_path.read_text(encoding="utf-8"))
    repair_ids = {rid for f in strict["findings"]
                  for link in f["strict_action_result_citation_links"]
                  if link["search_disposition"] == "repair_candidate" and
                     link["answer_disposition"] == "repair_candidate"
                  for rid in (link["search_record_id"], link["answer_record_id"])}
    repair_questions = {norm(s["question"]) for group in trajectory_groups
                        for s in group["states"] if s["record_id"] in repair_ids}
    quarantine_questions = {norm(g["question"]) for g in repair_pairs}
    repair_new_questions = repair_questions | quarantine_questions
    if len(repair_questions) != 3 or len(quarantine_questions) != 2 or len(repair_new_questions) != 5:
        raise ValueError("strict_repair_question_count_or_uniqueness_changed")
    if repair_new_questions & (core_questions | supplement_questions):
        raise ValueError("strict_repair_questions_overlap_existing_training")
    repair_candidate_ids = {rid for f in strict["findings"]
                            for link in f["strict_action_result_citation_links"]
                            if link["new_question_vs_retained_core"] and
                            link["search_disposition"] == link["answer_disposition"] and
                            link["search_disposition"] in {"repair_candidate", "quarantine_pending_repair"}
                            for rid in (link["search_record_id"], link["answer_record_id"])}
    if any(is_in_core(rid) for rid in repair_candidate_ids):
        raise ValueError("repair_pair_candidate_already_in_core")

    dedupe_pairs = readl(repair_replay_path)
    dedupe_questions = [norm(g["question"]) for g in dedupe_pairs]
    if len(dedupe_pairs) != 2 or len(set(dedupe_questions)) != 2:
        raise ValueError("dedupe_pair_count_changed")
    if sum(q in core_questions for q in dedupe_questions) != 1:
        raise ValueError("expected_one_core_replacement_and_one_new_question")
    dedupe_new_questions = {q for q in dedupe_questions
                            if q not in core_questions | supplement_questions}
    if len(dedupe_new_questions) != 1 or dedupe_new_questions & repair_new_questions:
        raise ValueError("dedupe_new_question_overlap")
    duplicate_dedupe_questions = set(dedupe_questions) & extraction_question_set
    unique_dedupe_pairs = len(set(dedupe_questions) - extraction_question_set)
    if len(duplicate_dedupe_questions) != 1 or unique_dedupe_pairs != 1:
        raise ValueError("dedupe_extraction_overlap_not_reconciled")
    if not is_in_core("v117_train:0036") or is_in_core("v117_train:0284"):
        raise ValueError("taylor_candidate_replacement_binding_changed")
    if is_in_core("v117_train:0306") or is_in_core("v117_train:0308"):
        raise ValueError("ridl_candidate_expected_outside_core")
    if not is_in_core("v117_train:0282"):
        raise ValueError("taylor_existing_core_duplicate_not_found")

    base = snapshot["v122_train_supplement"]["clean_combination_with_v117_retained_core"]
    state_delta = 10 + 2  # five new repair pairs plus one new dedupe pair; extracted/Taylor rows replace core rows
    question_delta = len(repair_new_questions | dedupe_new_questions)
    pair_delta = len(extraction) + len(repair_new_questions) + unique_dedupe_pairs
    upper = {
        "states": base["states"] + state_delta,
        "questions": base["questions"] + question_delta,
        "strict_complete_pairs": base["strict_complete_pairs"] + pair_delta,
    }
    required = preflight["full_user_plan_gate"]
    deficits = {
        "states": max(0, required["required_states"] - upper["states"]),
        "questions": max(0, required["required_questions"] - upper["questions"]),
        "strict_complete_pairs": max(0, required["required_complete_pairs"] - upper["strict_complete_pairs"]),
    }
    internal = json.loads((ROOT / "data/evidence-sft-v123/internal/insufficiency-certificate.json").read_text(
        encoding="utf-8"))

    report = {
        "version": "v126-recovery-upper-bound-v4",
        "corrects": "v126-recovery-upper-bound-v3-private.json",
        "correction_reason": "v3 double-counted the Taylor McLam pair, which is the same trajectory as one retained-search extraction candidate.",
        "inputs": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths},
        "replacement_bindings_private": {
            "retained_extracted_search_records_in_core": len(extraction_source_ids),
            "retained_extracted_answer_records_in_core": len(extraction_answer_ids),
            "taylor_search_record_to_replace_private": "v117_train:0282",
            "taylor_exact_prefix_search_candidate_private": "v117_train:0284",
            "taylor_answer_record_remains_in_core_private": "v117_train:0036",
        },
        "base_current_combination": {
            "states": base["states"], "questions": base["questions"],
            "strict_complete_pairs": base["strict_complete_pairs"],
        },
        "candidate_contributions_if_all_future_reviews_pass": {
            "retained_trajectory_search_states": {
                "candidate_states": len(extraction),
                "treatment": "replace the ten retained search rows with exact observed-prefix instances; zero state/question delta",
                "strict_pair_delta": len(extraction),
            },
            "v117_strict_repair_pairs": {
                "groups": len(repair_new_questions), "states": 2 * len(repair_new_questions),
                "questions": len(repair_new_questions), "strict_pair_delta": len(repair_new_questions),
            },
            "dedupe_pair_candidates": {
                "groups": len(dedupe_pairs), "new_question_groups": len(dedupe_new_questions),
                "new_states": 2 * len(dedupe_new_questions),
                "strict_pair_delta": unique_dedupe_pairs,
                "overlap_with_retained_trajectory_extraction_groups": len(duplicate_dedupe_questions),
                "one_existing_question_requires_core_search_replacement": True,
            },
        },
        "best_case_structural_ceiling_from_these_candidates": upper,
        "remaining_deficits_even_if_all_candidates_pass": deficits,
        "internal_set_gate": {
            "current_complete_pairs": internal["current_complete_double_approved_pairs"],
            "possible_pairs_upper": internal["possible_complete_pairs_upper"],
            "required_pairs": internal["required_complete_pairs"],
            "possible_risk_state_upper": internal["possible_risk_state_upper"],
            "required_risk_states_per_label": internal["required_risk_states_per_label"],
            "passes": False,
        },
        "actual_new_independently_double_approved_states": 0,
        "training_allowed": False,
    }
    write(out, report)
    print(json.dumps({
        "candidate_questions": upper["questions"],
        "candidate_states": upper["states"],
        "candidate_strict_pairs": upper["strict_complete_pairs"],
        "remaining_deficits": deficits,
        "internal_set_gate_passes": False,
        "training_allowed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
