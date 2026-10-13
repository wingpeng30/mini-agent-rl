"""Check whether repeated v117 search states are true export-level duplicates."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from scripts.v123_cpu_core import digest, readl, write
from scripts.v124_forensic import OUT


def exact_case_ids(lineage_row):
    if not lineage_row["exact_prompt_action_match"]:
        return []
    return sorted({match["source_case_id"] for match in lineage_row["matches"]})


def retrieval_digest(lineage_row):
    if not lineage_row["exact_prompt_action_match"] or len(lineage_row["matches"]) != 1:
        return None
    docs = lineage_row["matches"][0]["retrieved_after_search"]
    return digest([(d["id"], d["title"], d["text"]) for d in docs])


def main() -> None:
    inventory_rows = readl(OUT / "inventory-private.jsonl")
    inventory = {r["record_id"]: r for r in inventory_rows}
    lineage_rows = readl(OUT / "lineage-private.jsonl")
    lineage = {r["record_id"]: r for r in lineage_rows}
    reviews = [r for p in sorted(OUT.glob("primary-batch-*.jsonl")) for r in readl(p)]
    effective = {r["record_id"]: r for r in reviews}
    for p in sorted(OUT.glob("primary-amendments*-private.jsonl")):
        for change in readl(p):
            effective[change["record_id"]] = {
                **effective[change["record_id"]],
                "disposition": change["effective_disposition"],
            }
    for change in readl(OUT / "primary-adjudication-overrides-private.jsonl"):
        effective[change["record_id"]] = {
            **effective[change["record_id"]],
            "disposition": change["effective_disposition"],
        }
    legacy = {r["record_id"]: r for r in json.loads(
        (OUT / "v117-legacy-review-reconciliation-private.json").read_text(
            encoding="utf-8"))["records"]}
    backlog_path = OUT / "v117-remaining-repair-backlog-private.json"
    backlog = json.loads(backlog_path.read_text(encoding="utf-8"))

    findings = []
    for group in backlog["groups"]:
        if group["work_class"] != "deduplicate_at_export_after_pair_check":
            continue
        states = group["all_same_normalized_question_states"]
        prepared = []
        for state in states:
            rid = state["record_id"]
            action = inventory[rid]["proposed_action"]
            prepared.append({
                "record_id": rid,
                "action": action,
                "action_sha256": digest(action),
                "source_case_ids": exact_case_ids(lineage[rid]),
                "retrieval_sha256": retrieval_digest(lineage[rid])
                if action.get("action") == "search" else None,
                "effective_primary_disposition": effective[rid]["disposition"],
                "historical_dual_vote_exact": bool(
                    legacy.get(rid, {}).get("valid_dual_approval")
                    and legacy.get(rid, {}).get("original_content_exact")),
            })
        retained_searches = [r for r in prepared
                             if r["action"].get("action") == "search"
                             and r["effective_primary_disposition"] == "retain_candidate"]
        flagged = [r for r in prepared
                   if r["effective_primary_disposition"] == "repair_candidate"
                   and r["action"].get("action") == "search"]
        for row in flagged:
            row["matching_retained_peer_ids"] = [
                peer["record_id"] for peer in retained_searches
                if row["action_sha256"] == peer["action_sha256"]
                and row["retrieval_sha256"] is not None
                and row["retrieval_sha256"] == peer["retrieval_sha256"]
            ]
            row["same_source_case_retained_peer_ids"] = [
                peer["record_id"] for peer in retained_searches
                if set(row["source_case_ids"]) & set(peer["source_case_ids"])
            ]
            row["same_source_case_answer_peer_ids"] = [
                peer["record_id"] for peer in prepared
                if peer["action"].get("action") == "answer"
                and set(row["source_case_ids"]) & set(peer["source_case_ids"])
            ]
        by_exact_input = defaultdict(list)
        for rid in (r["record_id"] for r in prepared if r["action"].get("action") == "search"):
            source = inventory[rid]
            by_exact_input[digest([source["visible_messages"], source["proposed_action"]])].append(rid)
        findings.append({
            "normalized_question_sha256": group["normalized_question_sha256"],
            "question": group["normalized_question_for_audit"],
            "states": prepared,
            "flagged_search_count": len(flagged),
            "flagged_searches_with_action_and_top3_match": sum(
                bool(r["matching_retained_peer_ids"]) for r in flagged),
            "flagged_searches_with_same_case_retained_peer": sum(
                bool(r["same_source_case_retained_peer_ids"]) for r in flagged),
            "flagged_searches_with_same_case_answer_peer": sum(
                bool(r["same_source_case_answer_peer_ids"]) for r in flagged),
            "exact_prompt_action_duplicate_clusters": sum(len(ids) > 1
                                                          for ids in by_exact_input.values()),
            "export_dedup_ready": False,
            "training_allowed": False,
        })
    result = {
        "version": "v124-duplicate-pair-lineage-audit-v1",
        "backlog_payload_sha256": digest(backlog),
        "groups": len(findings),
        "flagged_search_states": sum(r["flagged_search_count"] for r in findings),
        "flagged_searches_with_action_and_top3_match": sum(
            r["flagged_searches_with_action_and_top3_match"] for r in findings),
        "flagged_searches_with_same_case_retained_peer": sum(
            r["flagged_searches_with_same_case_retained_peer"] for r in findings),
        "flagged_searches_with_same_case_answer_peer": sum(
            r["flagged_searches_with_same_case_answer_peer"] for r in findings),
        "exact_prompt_action_duplicate_clusters": sum(
            r["exact_prompt_action_duplicate_clusters"] for r in findings),
        "findings": findings,
        "interpretation": [
            "Same query and same retrieved top-3 across different source cases are not the same training state.",
            "Only same-case search/answer lineage can support trajectory pairing; this report creates no pair.",
            "Repair-candidate rows remain excluded from the retained core unless a separate valid amendment changes them.",
        ],
        "training_allowed": False,
    }
    write(OUT / "duplicate-pair-lineage-audit-private.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
