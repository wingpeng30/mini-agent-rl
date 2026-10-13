"""Prepare the bounded Wardle repair proposal for independent review.

This edits no frozen source row and creates no training label. It changes the
question to a source-scoped country relation and replaces the three redundant,
underspecified search queries with one question-grounded query.
"""
from __future__ import annotations

import copy
import json

from scripts.v123_cpu_core import ROOT, digest, readl, writel
from scripts.v124_forensic import OUT

TRAIN = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
QUESTION = (
    "According to the Wardle article, what country is named for the village "
    "described as being in Greater Manchester and among the foothills of the "
    "South Pennines?"
)
QUERY = "Wardle village Greater Manchester South Pennines country"
ORDINALS = (55, 92, 216, 217)


def target(row: dict) -> dict:
    return json.loads(row["messages"][row["target_assistant_index"]]["content"])


def main() -> None:
    source_rows = readl(TRAIN)
    if len(source_rows) != 345:
        raise RuntimeError("unexpected_v117_train_length")
    by_id = {f"v117_train:{i:04d}": row for i, row in enumerate(source_rows, 1)}
    inventory_rows = readl(OUT / "inventory-private.jsonl")
    inventory = {r["record_id"]: r for r in inventory_rows}
    primary = {r["record_id"]: r for p in sorted(OUT.glob("primary-batch-*.jsonl")) for r in readl(p)}
    if len(inventory) != 1229 or len(primary) != len(inventory):
        raise RuntimeError("inventory_or_primary_coverage_mismatch")

    effective = {rid: dict(row) for rid, row in primary.items()}
    for path in sorted(OUT.glob("primary-amendments*-private.jsonl")):
        for change in readl(path):
            original = primary[change["record_id"]]
            if change["superseded_primary_review_sha256"] != digest(original):
                raise RuntimeError("primary_amendment_hash_mismatch:" + change["record_id"])
            effective[change["record_id"]]["disposition"] = change["effective_disposition"]
    for change in readl(OUT / "primary-adjudication-overrides-private.jsonl"):
        rid = change["record_id"]
        if effective[rid]["disposition"] != change["supersedes_effective_disposition"]:
            raise RuntimeError("primary_override_chain_mismatch:" + rid)
        effective[rid]["disposition"] = change["effective_disposition"]

    ids = [f"v117_train:{ordinal:04d}" for ordinal in ORDINALS]
    old_questions = {by_id[rid]["messages"][1]["content"] for rid in ids}
    actual_group = sorted(rid for rid, row in by_id.items()
                          if row["messages"][1]["content"] in old_questions)
    if len(old_questions) != 1 or actual_group != ids:
        raise RuntimeError("wardle_exact_question_group_mismatch")
    if {effective[rid]["disposition"] for rid in ids} != {"retain_candidate", "repair_candidate"}:
        raise RuntimeError("wardle_effective_dispositions_changed")

    drafts = []
    for rid in ids:
        revised = copy.deepcopy(by_id[rid])
        revised["messages"][1]["content"] = QUESTION
        action = target(revised)
        if action.get("action") == "search":
            action["query"] = QUERY
            revised["messages"][revised["target_assistant_index"]]["content"] = json.dumps(
                action, ensure_ascii=False, separators=(",", ":"))
        elif action.get("action") != "answer" or action.get("answer") != "England":
            raise RuntimeError("wardle_unexpected_answer_action:" + rid)
        drafts.append({
            "record_id": rid,
            "original_row_sha256": digest(by_id[rid]),
            "primary_review_sha256": digest(primary[rid]),
            "original_question": next(m["content"] for m in by_id[rid]["messages"] if m["role"] == "user"),
            "same_question_record_ids": ids,
            "same_question_all_states_reviewed": True,
            "revised_row": revised,
            "amendment_version": "v7",
            "status": "draft_only_requires_semantic_double_review",
            "approved_training_label": False,
            "training_allowed": False,
        })
    output = OUT / "repair-drafts-v7-private.jsonl"
    if output.exists() and readl(output) != drafts:
        raise RuntimeError("existing_v7_drafts_differ_refuse_overwrite")
    if not output.exists():
        writel(output, drafts)
    print(json.dumps({"revision_groups": 1, "states": len(drafts),
                      "search_states": sum(target(r["revised_row"])["action"] == "search" for r in drafts),
                      "status": "draft_only_requires_semantic_double_review",
                      "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
