"""Create a conservative current-export disposition overlay for the G2 inventory."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / "data/evidence-sft-v124-forensic/goal-v128"
OUTPUT = GOAL / "g2-disposition-closure-private.json"


def read(name: str):
    return json.loads((GOAL / name).read_text(encoding="utf-8"))


def classify(repair_ids: set[str], materialized_ids: set[str], unmaterialized_ids: set[str],
             changed_query_ids: set[str], normalized_overlap_ids: set[str],
             coverage_by_id: dict[str, dict]) -> dict:
    if len(repair_ids) != 107 or len(unmaterialized_ids) != 61:
        raise ValueError("frozen_repair_inventory_counts_changed")
    if not unmaterialized_ids <= repair_ids:
        raise ValueError("unmaterialized_ids_outside_repair_inventory")
    if repair_ids != materialized_ids | unmaterialized_ids:
        raise ValueError("materialized_unmaterialized_partition_mismatch")
    if not changed_query_ids <= materialized_ids:
        raise ValueError("changed_query_without_materialized_revised_state")
    if not normalized_overlap_ids <= materialized_ids:
        raise ValueError("question_overlap_without_materialized_revised_state")
    excluded_context = changed_query_ids | normalized_overlap_ids
    if len(materialized_ids) != 46 or len(changed_query_ids) != 9:
        raise ValueError("materialized_changed_query_counts_changed")

    rows = []
    for record_id in sorted(repair_ids):
        if record_id in unmaterialized_ids:
            disposition, reasons = "excluded_from_current_repair_export", ["no_materialized_revision_record"]
        elif record_id in changed_query_ids:
            disposition, reasons = "excluded_from_current_repair_export", [
                "changed_query_answer_context_not_reconstructed"]
        elif record_id in normalized_overlap_ids:
            disposition, reasons = "excluded_from_current_repair_export", [
                "normalized_question_overlaps_current_training_pool"]
        else:
            row = coverage_by_id[record_id]
            required = ("source_row_sha256", "revised_row_sha256", "revised_visible_prefix_sha256",
                        "action_target_sha256")
            if (not row.get("within_limits") or row.get("exact_state_duplicate_in_revised_pool")
                    or row.get("exact_state_duplicate_in_current_training_pool")
                    or any(not row.get(key) for key in required)):
                raise ValueError("remaining_candidate_missing_mechanical_binding:" + record_id)
            if row.get("semantic_review") != "not_run":
                raise ValueError("unexpected_semantic_review_state:" + record_id)
            disposition, reasons = "mechanically_prepared_for_future_semantic_review", [
                "source_action_time_hash_and_token_replay_present",
                "not_a_complete_pair_or_semantic_approval"]
        rows.append({"record_id": record_id, "disposition": disposition, "reasons": reasons,
                     "export_eligible": False, "training_allowed": False})
    remaining = [r for r in rows if r["disposition"] == "mechanically_prepared_for_future_semantic_review"]
    if len(remaining) != 37:
        raise ValueError("unexpected_future_review_candidate_count")
    return {
        "version": "v130-g2-export-disposition-overlay-v1",
        "interpretation": "Overlay only; historical dispositions and original records are unchanged.",
        "repair_candidate_count": len(rows),
        "unmaterialized_excluded": len(unmaterialized_ids),
        "changed_query_context_excluded": len(changed_query_ids),
        "normalized_overlap_excluded": len(normalized_overlap_ids),
        "exclusion_intersection_count": len((changed_query_ids & normalized_overlap_ids)),
        "future_semantic_review_only": len(remaining),
        "semantic_double_approvals_added": 0,
        "strict_complete_pairs_added": 0,
        "rows": rows,
        "quarantine_policy": "All 120 quarantined states remain held and excluded; this overlay does not resolve or relabel them.",
        "unpaired_policy": "All 35 unpaired source questions remain diagnostic-only; no trajectory is synthesized.",
        "training_allowed": False,
        "data_ready_for_experiment": False,
    }


def build() -> dict:
    inventory = read("g2-repair-inventory-private.json")
    unmaterialized = read("g2-unmaterialized-repair-dispositions-private.json")
    coverage = read("g2-revised-state-coverage-private.json")
    changed = read("g2-changed-query-context-feasibility-private.json")
    by_id = {r["record_id"]: r for r in coverage["revised_candidate_states"]}
    return classify(
        set(inventory["v117_repair_candidate_record_ids"]), set(by_id),
        {r["record_id"] for r in unmaterialized["dispositions"]},
        {r["search_record_id"] for r in changed["changed_actions"]
         if r["is_v117_repair_candidate"]},
        {r["record_id"] for r in coverage["revised_candidate_states"]
         if r["normalized_question_overlaps_current_training_pool"]},
        by_id)


def main() -> None:
    report = build()
    if OUTPUT.exists():
        current = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if current != report:
            raise RuntimeError("existing_disposition_overlay_differs_refuse_overwrite")
    else:
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"version": report["version"], "repair_candidates": report["repair_candidate_count"],
                      "excluded": report["unmaterialized_excluded"] + report["changed_query_context_excluded"]
                                  + report["normalized_overlap_excluded"] - report["exclusion_intersection_count"],
                      "future_review_only": report["future_semantic_review_only"],
                      "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
