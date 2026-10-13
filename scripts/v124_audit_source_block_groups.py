"""Check whether v117 evidence-blocked backlog groups have true case-level peers."""
from __future__ import annotations

import json
import hashlib
from collections import defaultdict

from scripts.v123_cpu_core import digest, readl, write
from scripts.v124_forensic import OUT


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    inventory_path = OUT / "inventory-private.jsonl"
    lineage_path = OUT / "lineage-private.jsonl"
    inventory_rows = readl(inventory_path)
    inventory = {r["record_id"]: r for r in inventory_rows}
    lineage_rows = readl(lineage_path)
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
    backlog_path = OUT / "v117-remaining-repair-backlog-private.json"
    backlog = json.loads(backlog_path.read_text(encoding="utf-8"))
    findings = []
    for group in backlog["groups"]:
        if group["work_class"] != "evidence_or_target_block_requires_source_resolution":
            continue
        states = []
        by_case = defaultdict(list)
        for state in group["all_same_normalized_question_states"]:
            rid = state["record_id"]
            lineage_entry = lineage[rid]
            matches = lineage_entry["matches"]
            case_ids = sorted({m["source_case_id"] for m in matches}) \
                if lineage_entry["exact_prompt_action_match"] else []
            action = inventory[rid]["proposed_action"]
            row = {
                "record_id": rid,
                "record_sha256": inventory[rid]["record_sha256"],
                "action": action,
                "effective_primary_disposition": effective[rid]["disposition"],
                "exact_lineage_case_ids": case_ids,
                "lineage_match_count": len(matches),
            }
            states.append(row)
            for case_id in case_ids:
                by_case[case_id].append(rid)
        shared_case_ids = {case_id: ids for case_id, ids in by_case.items()
                           if len(set(ids)) > 1}
        findings.append({
            "normalized_question_sha256": group["normalized_question_sha256"],
            "question": group["normalized_question_for_audit"],
            "flagged_record_ids": [r["record_id"] for r in group["flagged_states"]],
            "primary_issues": sorted({issue for r in group["flagged_states"]
                                       for issue in r["primary_issues"]}),
            "states": states,
            "shared_exact_case_ids": shared_case_ids,
            "case_level_search_answer_pair_exists": any(
                len({s["action"].get("action") for s in states
                     if s["record_id"] in ids}) > 1
                for ids in shared_case_ids.values()
            ),
            "training_allowed": False,
        })
    result = {
        "version": "v124-source-block-case-audit-v1",
        "backlog_payload_sha256": digest(backlog),
        "inventory_file_sha256": file_sha256(inventory_path),
        "lineage_file_sha256": file_sha256(lineage_path),
        "groups": len(findings),
        "groups_with_shared_exact_case_id": sum(bool(r["shared_exact_case_ids"])
                                                 for r in findings),
        "groups_with_case_level_search_answer_pair": sum(
            r["case_level_search_answer_pair_exists"] for r in findings),
        "findings": findings,
        "limitations": [
            "Normalized question similarity does not establish a trajectory pair.",
            "Only an exact source-case lineage match is reported as a possible peer; it is not semantic approval.",
            "No new query, answer, repair vote, or training label is created.",
        ],
        "training_allowed": False,
    }
    write(OUT / "source-block-case-audit-private.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
