"""Prepare a bounded v8 repair for the NXT TakeOver: Rival question group."""
from __future__ import annotations

import copy
import json

from scripts.v123_cpu_core import ROOT, digest, readl, writel
from scripts.v124_forensic import OUT

TRAIN = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
IDS = ["v117_train:0027", "v117_train:0251"]
QUESTION = "NXT TakeOver: Rival was held at a private, for-profit university. In what year was that university founded?"
QUERY = "NXT TakeOver Rival event venue university"


def target(row: dict) -> dict:
    return json.loads(row["messages"][row["target_assistant_index"]]["content"])


def main() -> None:
    source = readl(TRAIN)
    if len(source) != 345:
        raise RuntimeError("unexpected_v117_train_length")
    rows = {f"v117_train:{i:04d}": row for i, row in enumerate(source, 1)}
    primary_rows = [r for p in sorted(OUT.glob("primary-batch-*.jsonl")) for r in readl(p)]
    primary = {r["record_id"]: r for r in primary_rows}
    if len(primary) != 1229:
        raise RuntimeError("primary_coverage_mismatch")
    effective = {rid: dict(r) for rid, r in primary.items()}
    for path in sorted(OUT.glob("primary-amendments*-private.jsonl")):
        for change in readl(path):
            if digest(primary[change["record_id"]]) != change["superseded_primary_review_sha256"]:
                raise RuntimeError("primary_amendment_hash_mismatch")
            effective[change["record_id"]]["disposition"] = change["effective_disposition"]
    for change in readl(OUT / "primary-adjudication-overrides-private.jsonl"):
        rid = change["record_id"]
        if effective[rid]["disposition"] != change["supersedes_effective_disposition"]:
            raise RuntimeError("primary_override_chain_mismatch")
        effective[rid]["disposition"] = change["effective_disposition"]
    original_questions = {rows[rid]["messages"][1]["content"] for rid in IDS}
    actual_group = sorted(rid for rid, row in rows.items() if row["messages"][1]["content"] in original_questions)
    if len(original_questions) != 1 or actual_group != IDS:
        raise RuntimeError("nxt_exact_question_group_mismatch")
    if effective[IDS[0]]["disposition"] != "retain_candidate" or effective[IDS[1]]["disposition"] != "repair_candidate":
        raise RuntimeError("nxt_effective_disposition_changed")
    answer = target(rows[IDS[0]])
    search = target(rows[IDS[1]])
    if answer != {"action": "answer", "answer": "1979"} or search.get("action") != "search":
        raise RuntimeError("nxt_unexpected_actions")

    drafts = []
    for rid in IDS:
        revised = copy.deepcopy(rows[rid])
        revised["messages"][1]["content"] = QUESTION
        action = target(revised)
        if action["action"] == "search":
            action["query"] = QUERY
            revised["messages"][revised["target_assistant_index"]]["content"] = json.dumps(
                action, ensure_ascii=False, separators=(",", ":"))
        drafts.append({
            "record_id": rid,
            "original_row_sha256": digest(rows[rid]),
            "primary_review_sha256": digest(primary[rid]),
            "original_question": rows[rid]["messages"][1]["content"],
            "same_question_record_ids": IDS,
            "same_question_all_states_reviewed": True,
            "revised_row": revised,
            "amendment_version": "v8",
            "status": "draft_only_requires_semantic_double_review",
            "approved_training_label": False,
            "training_allowed": False,
        })
    path = OUT / "repair-drafts-v8-private.jsonl"
    if path.exists() and readl(path) != drafts:
        raise RuntimeError("existing_v8_drafts_differ_refuse_overwrite")
    if not path.exists():
        writel(path, drafts)
    print(json.dumps({"revision_groups": 1, "states": len(drafts),
                      "search_states": 1, "answer_states": 1,
                      "status": "draft_only_requires_semantic_double_review",
                      "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
