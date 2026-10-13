"""Report exact-vs-similar repeated actions in the remaining v117 backlog."""
from __future__ import annotations

import json
from collections import defaultdict

from scripts.v123_cpu_core import digest, readl, write
from scripts.v124_forensic import OUT


CLASSES = {"deduplicate_at_export_after_pair_check",
           "repair_question_or_target_then_deduplicate"}


def main() -> None:
    inventory = {r["record_id"]: r for r in readl(OUT / "inventory-private.jsonl")}
    backlog = json.loads((OUT / "v117-remaining-repair-backlog-private.json").read_text(
        encoding="utf-8"))
    findings = []
    for group in backlog["groups"]:
        if group["work_class"] not in CLASSES:
            continue
        rows = [inventory[x["record_id"]] for x in group["all_same_normalized_question_states"]]
        by_action_query = defaultdict(list)
        for row in rows:
            action = row["proposed_action"]
            if action.get("action") == "search":
                by_action_query[digest(action)].append(row)
        repeated = []
        for action_sha, same_query in by_action_query.items():
            if len(same_query) < 2:
                continue
            by_exact_input = defaultdict(list)
            for row in same_query:
                by_exact_input[digest([row["visible_messages"], row["proposed_action"]])].append(row)
            repeated.append({
                "action_sha256": action_sha,
                "record_ids": [r["record_id"] for r in same_query],
                "query": same_query[0]["proposed_action"].get("query"),
                "exact_prompt_action_duplicate_clusters": [
                    [r["record_id"] for r in cluster]
                    for cluster in by_exact_input.values() if len(cluster) > 1],
                "same_query_distinct_input_ids": [
                    r["record_id"] for r in same_query],
                "retrieval_lineage_status": {
                    r["record_id"]: r["search_retrieval_binding"] for r in same_query},
            })
        findings.append({
            "normalized_question_sha256": group["normalized_question_sha256"],
            "question": group["normalized_question_for_audit"],
            "work_class": group["work_class"],
            "all_group_states": [{"record_id": r["record_id"],
                                  "action": r["proposed_action"].get("action")}
                                 for r in rows],
            "repeated_query_actions": repeated,
            "export_dedup_ready": False,
            "training_allowed": False,
        })
    report = {
        "version": "v124-duplicate-repair-audit-v1",
        "backlog_sha256": digest(backlog),
        "groups_examined": len(findings),
        "groups_with_repeated_search_query": sum(bool(x["repeated_query_actions"]) for x in findings),
        "exact_prompt_action_duplicate_clusters": sum(
            len(q["exact_prompt_action_duplicate_clusters"])
            for x in findings for q in x["repeated_query_actions"]),
        "findings": findings,
        "limitations": [
            "Repeated query text is not by itself an exact duplicate training state.",
            "No answer state, semantic approval, or export removal is authorized by this report.",
            "Retrieval lineage and pair evidence must be checked before any export-level deduplication.",
        ],
        "training_allowed": False,
    }
    write(OUT / "duplicate-repair-groups-audit-private.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
