"""Reconcile named v128 repair/quarantine/unpaired ledgers without model calls."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v124-forensic"
GOAL = BASE / "goal-v128"
SOURCES = {
    "quarantine": BASE / "quarantine-processing-audit-private.json",
    "backlog": BASE / "v117-remaining-repair-backlog-private.json",
    "reconciliation": BASE / "repair-draft-reconciliation-cpu-v2-private.json",
    "replay": BASE / "repair-draft-state-replay-report-v2-private.json",
    "replay_inputs": BASE / "repair-draft-replay-inputs-v2-private.jsonl",
    "replayed_states": BASE / "repair-draft-states-cpu-v2-private.jsonl",
    "evidence_visibility": BASE / "repair-evidence-visibility-private.json",
    "trainability_snapshot": BASE / "current-trainability-snapshot-private.json",
    "repair_candidates": BASE / "v117-repair-candidates-cpu-private.json",
    "trajectory_audit": BASE / "v125-v117-repair-trajectory-link-audit-private.json",
    "dedup_replay": BASE / "v126-deduped-pair-replay-report-private.json",
    "quarantine_replay": BASE / "v126-quarantined-repair-replay-report-private.json",
    "search_extraction": BASE / "v126-retained-linked-search-extraction-report-private.json",
    "fred_meyer": BASE / "v127-fred-meyer-quarantine-replay-report-private.json",
    "unpaired_diagnostic": BASE / "remaining-unpaired-diagnostic-private.jsonl",
    "train_unpaired_triage": BASE / "v122-train-unpaired-semantic-triage-private.jsonl",
    "internal_unpaired_triage": BASE / "v123-unpaired-semantic-triage-private.jsonl",
}
OUT = GOAL / "g2-repair-inventory-private.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def audit() -> dict:
    hashes = {key: sha(path) for key, path in SOURCES.items()}
    quarantine = read_json(SOURCES["quarantine"])["records"]
    backlog_doc = read_json(SOURCES["backlog"])
    backlog = [state for group in backlog_doc["groups"]
               for state in group["flagged_states"]]
    reconciliation = read_json(SOURCES["reconciliation"])["groups"]
    replay_report = read_json(SOURCES["replay"])["summary"]
    visibility_report = read_json(SOURCES["evidence_visibility"])
    trainability_snapshot = read_json(SOURCES["trainability_snapshot"])

    by_id: dict[str, dict] = {}

    def item(record_id: str) -> dict:
        return by_id.setdefault(record_id, {
            "record_id": record_id, "source_ledgers": [], "issues": [],
            "current_dispositions": [], "replay_evidence": [],
            "semantic_double_review": "not_verified_or_not_run",
            "training_allowed": False,
        })

    for row in quarantine:
        x = item(row["record_id"])
        x["source_ledgers"].append("quarantine-processing-audit")
        x["source_sha256"] = row.get("record_sha256")
        x["cohorts"] = [row.get("cohort")]
        x["issues"].extend(row.get("issues", []))
        x["current_dispositions"].append(row.get("effective_primary_disposition"))
        x["processing_status"] = row.get("processing_status")
        x["repair_proposal_present"] = bool(row.get("repair_proposal"))
        x["approved_training_label"] = bool(row.get("approved_training_label"))

    for row in backlog:
        x = item(row["record_id"])
        x["source_ledgers"].append("v117-unresolved-backlog")
        x["source_sha256"] = row.get("record_sha256")
        x.setdefault("cohorts", []).append("v117_train")
        x["issues"].extend(row.get("primary_issues", []))
        x["current_dispositions"].append(row.get("effective_primary_disposition"))
        x["backlog_retrieval_binding"] = row.get("retrieval_binding")
        x["covered_by_repair_reconciliation_claim"] = row.get(
            "covered_by_repair_reconciliation")
        x["repair_proposal_present"] = bool(row.get("primary_repair_proposal"))

    backlog_repair_ids = {row["record_id"] for row in backlog
                          if row.get("effective_primary_disposition") == "repair_candidate"}

    for group in reconciliation:
        for record_id in group["record_ids"]:
            x = item(record_id)
            x["source_ledgers"].append("cpu-reconciliation-v2")
            x.setdefault("current_dispositions", []).append(
                group.get("source_dispositions", {}).get(record_id))
            x["replay_evidence"].append({
                "group_status": group.get("status"),
                "group_state_count": group.get("state_count"),
                "token_limits_passed": group.get("token_limits_passed"),
                "search_states": group.get("search_states"),
                "query_changes": group.get("query_changes"),
                "baseline_unchanged_query_replays_exact": group.get(
                    "baseline_unchanged_query_replays_exact"),
                "downstream_context_rebuild_required": group.get(
                    "downstream_context_rebuild_required"),
            })

    replay_repair_ids = {record_id for group in reconciliation
                         for record_id, disposition in group.get(
                             "source_dispositions", {}).items()
                         if disposition == "repair_candidate"}
    snapshot_v117 = trainability_snapshot["dispositions_by_cohort"]["v117_train"]
    v117_repair_ids = backlog_repair_ids | replay_repair_ids

    replay_by_id = {}
    for group in replay_report:
        for record_id in group.get("record_ids", []):
            replay_by_id[record_id] = group
    visibility_by_id = {}
    for group in visibility_report.get("findings", []):
        for record_id in group.get("record_ids", []):
            visibility_by_id[record_id] = group
    for record_id, x in by_id.items():
        if record_id in replay_by_id:
            y = replay_by_id[record_id]
            x["state_replay"] = {
                "token_limits_passed": y.get("token_limits_passed"),
                "query_edits": y.get("query_edits"),
                "search_results_replayed_per_state": y.get("search_results_replayed_per_state"),
                "contains_search_and_answer": y.get("contains_search_and_answer"),
                "status": y.get("status"),
            }
        if record_id in visibility_by_id:
            y = visibility_by_id[record_id]
            x["evidence_visibility"] = {
                "answer_exact_string_visible": y.get(
                    "answer_states_with_exact_answer_substring_in_one_visible_document"),
                "search_has_new_top3_document": y.get(
                    "search_states_with_at_least_one_new_top3_document"),
                "finding": y.get("finding"),
            }
        x["source_ledgers"] = sorted(set(x["source_ledgers"]))
        x["issues"] = sorted(set(x["issues"]))
        x["current_dispositions"] = sorted(set(filter(None, x["current_dispositions"])))
        x["cohorts"] = sorted(set(x.get("cohorts", [])))

    unpaired = read_jsonl(SOURCES["unpaired_diagnostic"])
    triage_rows = (read_jsonl(SOURCES["train_unpaired_triage"])
                   + read_jsonl(SOURCES["internal_unpaired_triage"]))
    triage_by_source = {row["source_id"]: row for row in triage_rows}
    unpaired_items = []
    for row in unpaired:
        triage = triage_by_source.get(row["source_id"], {})
        unpaired_items.append({
            "inventory_id": row["source_id"],
            "inventory_namespace": row.get("cohort"),
            "triage_disposition": triage.get("disposition"),
            "triage_status": triage.get("status"),
            "structural_status": row.get("status"),
            "frozen_rejection_reason": row.get("frozen_rejection_reason"),
            "diagnostic_reason": row.get("reason"),
            "facts_present_in_frozen_corpus": row.get("facts_present_in_frozen_corpus"),
            "annotated_fact_count": row.get("annotated_fact_count"),
            "generated_search_count": len(row.get("answer_path_search_counts", [])),
            "generated_supported_answers": row.get("generated_supported_answers"),
            "training_allowed": False,
        })

    counts = {
        "quarantine_states": len({r["record_id"] for r in quarantine}),
        "unresolved_v117_backlog_states": len({r["record_id"] for r in backlog}),
        "cpu_reconciliation_states": len({rid for group in reconciliation
                                            for rid in group["record_ids"]}),
        "union_distinct_record_ids": len(by_id),
        "unpaired_questions": len(unpaired),
        "unpaired_unique_source_ids": len({r["source_id"] for r in unpaired}),
        "quarantine_backlog_overlap": len({r["record_id"] for r in quarantine}
                                           & {r["record_id"] for r in backlog}),
        "quarantine_replay_overlap": len({r["record_id"] for r in quarantine}
                                          & {rid for group in reconciliation
                                             for rid in group["record_ids"]}),
        "backlog_replay_overlap": len({r["record_id"] for r in backlog}
                                       & {rid for group in reconciliation
                                          for rid in group["record_ids"]}),
        "v117_repair_candidates_backlog_component": len(backlog_repair_ids),
        "v117_repair_candidates_cpu_reconciliation_component": len(replay_repair_ids),
        "v117_repair_candidate_components_overlap": len(backlog_repair_ids & replay_repair_ids),
        "v117_repair_candidates_reconstructed": len(v117_repair_ids),
        "v117_repair_candidate_snapshot_count": snapshot_v117.get("repair_candidate"),
        "v117_repair_candidate_count_matches_snapshot": (
            len(v117_repair_ids) == snapshot_v117.get("repair_candidate")),
        "legacy_107_count_reconciled": (
            len(v117_repair_ids) == snapshot_v117.get("repair_candidate") == 107),
        "semantic_double_approved_repair_states": 0,
        "training_allowed": False,
    }
    return {
        "version": "v128-repair-inventory-audit-v1",
        "inputs_sha256": hashes,
        "counts": counts,
        "quarantine_cohorts": dict(Counter(r["cohort"] for r in quarantine)),
        "quarantine_statuses": dict(Counter(r["processing_status"] for r in quarantine)),
        "cpu_reconciliation_statuses": dict(Counter(r["status"] for r in reconciliation)),
        "cpu_reconciliation_dispositions": dict(Counter(
            d for group in reconciliation for d in group["source_dispositions"].values())),
        "v117_repair_candidate_id_sha256": hashlib.sha256(
            "\n".join(sorted(v117_repair_ids)).encode("utf-8")).hexdigest(),
        "v117_repair_candidate_record_ids": sorted(v117_repair_ids),
        "unpaired_triage_dispositions": dict(Counter(
            row.get("disposition", "unmatched") for row in unpaired_items)),
        "unpaired_structural_reasons": dict(Counter(
            row.get("diagnostic_reason") for row in unpaired_items)),
        "states": sorted(by_id.values(), key=lambda x: x["record_id"]),
        "unpaired": unpaired_items,
        "limitations": [
            "The 107 v117 repair candidates are reconstructed from 61 repair_candidate IDs in the backlog plus 46 in CPU reconciliation; their ID intersection is zero and the 107 total matches the named trainability snapshot. The two quarantined backlog states and 15 non-repair CPU replay states are excluded from this total.",
            "v124 CPU replay coverage is not semantic review; it leaves every repair candidate pending independent double review.",
            "The 35 unpaired source questions are separate source-ID inventory entries and are not assumed disjoint from state IDs without explicit lineage.",
            "A relevant answer substring or a new top-three document is not proof of entailment or correct action grounding.",
            "No source question was repaired or exported; no reviewer API was called.",
        ],
        "training_allowed": False,
        "data_ready_for_experiment": False,
    }


if __name__ == "__main__":
    result = audit()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({"output": str(OUT), "counts": result["counts"],
                      "training_allowed": False}, ensure_ascii=False))
