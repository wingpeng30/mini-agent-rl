"""Build a source-bound backlog for unresolved v117 train review cases.

This is a CPU-only diagnostic. It reads only the v117 train source and the
explicit v124 review/reconciliation files. It never touches sealed eval data,
does not call a model or reviewer, and cannot authorize training.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from scripts.v123_cpu_core import digest, norm

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v124-forensic"
SOURCE = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
OUT = BASE / "v117-remaining-repair-backlog-private.json"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    source_rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines()
                   if line.strip()]
    inventory = {r["record_id"]: r for r in read_jsonl(BASE / "inventory-private.jsonl")}
    primary = [r for p in sorted(BASE.glob("primary-batch-*.jsonl"))
               for r in read_jsonl(p)]
    primary_by_id = {r["record_id"]: r for r in primary}
    effective_reviews = {rid: dict(row) for rid, row in primary_by_id.items()}
    amendment_paths = sorted(BASE.glob("primary-amendments*-private.jsonl"))
    amendments = [r for p in amendment_paths for r in read_jsonl(p)]
    if len({r["record_id"] for r in amendments}) != len(amendments):
        raise RuntimeError("duplicate_primary_amendment")
    for amendment in amendments:
        rid = amendment["record_id"]
        original = primary_by_id.get(rid)
        if (original is None or amendment["record_sha256"] != inventory.get(rid, {}).get("record_sha256")
                or amendment["superseded_primary_review_sha256"] != digest(original)):
            raise RuntimeError(f"primary_amendment_binding_mismatch:{rid}")
        effective_reviews[rid]["disposition"] = amendment["effective_disposition"]
        if amendment.get("new_issue"):
            effective_reviews[rid]["issues"] = [amendment["new_issue"]]
        effective_reviews[rid]["repair_proposal"] = amendment.get("proposed_question")
        if amendment.get("reason"):
            effective_reviews[rid]["rationale"] = amendment["reason"]
    overrides = read_jsonl(BASE / "primary-adjudication-overrides-private.jsonl")
    reconciliation = json.loads(
        (BASE / "repair-draft-reconciliation-cpu-v2-private.json").read_text(encoding="utf-8"))
    covered = {rid for group in reconciliation["groups"] for rid in group["record_ids"]}

    override_by_id = {r["record_id"]: r for r in overrides}
    if len(override_by_id) != len(overrides):
        raise RuntimeError("duplicate_primary_adjudication_override")
    for rid, override in override_by_id.items():
        current = effective_reviews.get(rid)
        if (current is None or override["record_sha256"] != inventory.get(rid, {}).get("record_sha256")
                or override["supersedes_effective_disposition"] != current["disposition"]):
            raise RuntimeError(f"primary_adjudication_chain_mismatch:{rid}")
        current["disposition"] = override["effective_disposition"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for ordinal, row in enumerate(source_rows, start=1):
        rid = f"v117_train:{ordinal:04d}"
        inv = inventory.get(rid)
        if not inv:
            raise RuntimeError(f"missing_inventory:{rid}")
        if inv.get("question") != row["messages"][1]["content"]:
            raise RuntimeError(f"source_question_mismatch:{rid}")
        qkey = norm(inv["question"])
        groups[qkey].append({"record_id": rid, "row": row, "inventory": inv})

    backlog: dict[str, list[dict]] = defaultdict(list)
    dispositions = Counter()
    for qkey, members in groups.items():
        for member in members:
            rid = member["record_id"]
            review = effective_reviews.get(rid)
            if not review:
                raise RuntimeError(f"missing_primary_review:{rid}")
            effective = review["disposition"]
            if rid in covered or not rid.startswith("v117_train:"):
                continue
            if effective not in ("repair_candidate", "quarantine_pending_repair"):
                continue
            action = member["inventory"].get("proposed_action", {})
            if isinstance(action, dict):
                action_name = action.get("action") or action.get("type") or "unknown"
            else:
                action_name = "unknown"
            backlog[qkey].append({
                "record_id": rid,
                "record_sha256": member["inventory"]["record_sha256"],
                "source_row_sha256": member["inventory"]["source_row_sha256"],
                "effective_primary_disposition": effective,
                "original_primary_disposition": review["disposition"],
                "primary_issues": review["issues"],
                "primary_repair_proposal": review["repair_proposal"],
                "action": action_name,
                "retrieval_binding": member["inventory"].get("search_retrieval_binding"),
                "covered_by_repair_reconciliation": False,
            })
            dispositions[effective] += 1

    result_groups = []
    for qkey, flagged in sorted(backlog.items(), key=lambda kv: min(
            int(r["record_id"].split(":")[1]) for r in kv[1])):
        all_members = groups[qkey]
        all_actions = []
        for member in all_members:
            action = member["inventory"].get("proposed_action", {})
            if isinstance(action, dict):
                action_name = action.get("action") or action.get("type") or "unknown"
            else:
                action_name = "unknown"
            all_actions.append({"record_id": member["record_id"], "action": action_name})
        issue_set = sorted({issue for member in flagged for issue in
                            effective_reviews[member["record_id"]]["issues"]})
        duplicate_action = any(issue == "duplicate_training_action" or
                               issue == "duplicate_training_action_group" for issue in issue_set)
        source_resolution_issues = {
            "non_unique_comune_in_latium", "same_name_park_ambiguity",
            "cross_park_evidence_splice", "query_targets_holiday_before_person_birth_known",
            "retrieval_hit_not_relation_support", "longest_reign_link_not_retrieved",
            "screenwriter_link_not_retrieved", "generic_query_without_event_link",
            "generic_geographic_query", "ranking_time_unanchored",
            "ambiguous_plural_brand_scope", "group_operator_vs_brand_confusion",
            "personnel_as_of_date_not_explicitly_bound",
        }
        duplicate_only_issues = {
            "duplicate_training_action", "duplicate_training_action_group",
            "query_copies_visible_context",
        }
        if source_resolution_issues.intersection(issue_set):
            work_class = "evidence_or_target_block_requires_source_resolution"
        elif duplicate_action and set(issue_set).issubset(duplicate_only_issues):
            work_class = "deduplicate_at_export_after_pair_check"
        elif duplicate_action:
            work_class = "repair_question_or_target_then_deduplicate"
        else:
            work_class = "question_or_query_revision_candidate"
        result_groups.append({
            "normalized_question_sha256": sha(qkey.encode("utf-8")),
            "normalized_question_for_audit": qkey,
            "exact_question_variants": sorted({m["inventory"]["question"] for m in all_members}),
            "all_same_normalized_question_states": all_actions,
            "flagged_states": flagged,
            "state_count_flagged": len(flagged),
            "state_count_total_in_v117": len(all_members),
            "issue_union": issue_set,
            "work_class": work_class,
            "training_allowed": False,
        })

    report = {
        "version": "v124-v117-remaining-repair-backlog-v1",
        "source_path": "data/evidence-sft-v117/frozen/train.jsonl",
        "source_sha256": sha(source_bytes),
        "primary_ledger_sha256": sha("".join(p.name + sha(p.read_bytes())
                                               for p in sorted(BASE.glob("primary-batch-*.jsonl"))).encode()),
        "primary_amendments_sha256": sha("".join(p.name + sha(p.read_bytes())
                                                    for p in amendment_paths).encode()),
        "effective_overrides_sha256": sha((BASE / "primary-adjudication-overrides-private.jsonl").read_bytes()),
        "repair_reconciliation_sha256": sha((BASE / "repair-draft-reconciliation-cpu-v2-private.json").read_bytes()),
        "covered_repair_states": len(covered),
        "primary_amendments_applied": len(amendments),
        "primary_adjudication_overrides_applied": len(overrides),
        "unresolved_flagged_states": sum(dispositions.values()),
        "unresolved_normalized_question_groups": len(result_groups),
        "effective_dispositions": dict(dispositions),
        "work_classes": dict(Counter(g["work_class"] for g in result_groups)),
        "normalization": "NFKC + casefold and collapse non-alphanumeric runs using scripts.v123_cpu_core.norm; groups are diagnostic only and do not replace source IDs",
        "status_note": "Primary-review backlog only; no new semantic decision, independent vote, repair approval or training authorization is created.",
        "groups": result_groups,
        "training_allowed": False,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "unresolved_flagged_states", "unresolved_normalized_question_groups",
        "effective_dispositions", "work_classes", "training_allowed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
