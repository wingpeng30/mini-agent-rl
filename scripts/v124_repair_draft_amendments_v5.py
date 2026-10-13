"""Append four source-bound repair revisions and retire one ambiguous query draft."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.v123_cpu_core import ROOT, digest, readl, writel, write
from scripts.v124_forensic import OUT

TRAIN = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
SPEC = {
    61: {"question": "According to the cited account, how many fish-and-chip shops were there in the UK in the 1930s?"},
    65: {"question": "The Great Serpent Mound, a 1,348-foot effigy along Ohio Brush Creek in Adams County, Ohio, lies on the plateau of what crater?"},
    83: {"question": "What is the name of Juan Figer's organisation that represents Palmeiras left wing back and midfielder Zé Roberto?",
         "query": "Juan Figer organisation represents Palmeiras left wing back midfielder Zé Roberto"},
    136: {"question": "Of which city was Sid Hatfield police chief before Charles Everett Lively assassinated him?",
          "query": "Sid Hatfield police chief city"},
}


def main() -> None:
    source_rows = readl(TRAIN)
    by_id = {f"v117_train:{i:04d}": row for i, row in enumerate(source_rows, 1)}
    inventory = {r["record_id"]: r for r in readl(OUT / "inventory-private.jsonl")}
    primary = {r["record_id"]: r for path in sorted(OUT.glob("primary-batch-*.jsonl"))
               for r in readl(path)}
    existing = [r for r in readl(OUT / "repair-drafts-v3-private.jsonl")]
    ambiguous = next(r for r in existing if r["record_id"] == "v117_train:0117")
    superseded_query = next(r for r in existing if r["record_id"] == "v117_train:0083")
    superseded_q = next(r for r in existing if r["record_id"] == "v117_train:0112")
    amendments = [{"record_id": "v117_train:0117",
        "superseded_draft_sha256": digest(ambiguous),
        "status": "retire_repair_query_candidate_keep_source_quarantined",
        "reason": "The pre-action context and question do not uniquely identify Percy Jones; replayed top three contains a boxer, rugby player, and musician. The rewrite query still permits an ambiguous person and cannot be rescued with post-retrieval identity clues.",
        "independent_review": False, "approved_training_label": False, "training_allowed": False},
        {"record_id": "v117_train:0083",
        "superseded_draft_sha256": digest(superseded_query),
        "status": "superseded_by_v5_group_revision",
        "reason": "The v5 group-level wording/query supersedes the earlier draft with a query that uses only relations and entities visible in the revised question.",
        "independent_review": False, "approved_training_label": False, "training_allowed": False},
        {"record_id": "v117_train:0112",
        "superseded_draft_sha256": digest(superseded_q),
        "status": "superseded_by_v5_group_revision",
        "reason": "The v5 group-level wording removes an unsupported creation premise and fixes the corrupted player-name spelling; it applies to both answer and search states in this same-question group.",
        "independent_review": False, "approved_training_label": False, "training_allowed": False}]
    write(OUT / "repair-draft-amendments-v5-private.json", {
        "version": "v124-repair-draft-amendments-v5", "amendments": amendments,
        "training_allowed": False})
    rows = []
    for ordinal, spec in SPEC.items():
        rid = f"v117_train:{ordinal:04d}"
        original = by_id[rid]
        revised = copy.deepcopy(original)
        revised["messages"][1]["content"] = spec["question"]
        if "query" in spec:
            action = json.loads(revised["messages"][revised["target_assistant_index"]]["content"])
            if action.get("action") != "search":
                raise RuntimeError("expected_search_action:" + rid)
            action["query"] = spec["query"]
            revised["messages"][revised["target_assistant_index"]]["content"] = json.dumps(
                action, ensure_ascii=False, separators=(",", ":"))
        source_question = inventory[rid]["question"]
        peers = sorted(r["record_id"] for r in inventory.values()
                       if r["cohort"] == "v117_train" and r["question"] == source_question)
        if primary[rid]["disposition"] not in {"repair_candidate", "retain_candidate"}:
            raise RuntimeError("unexpected_source_disposition:" + rid)
        rows.append({"record_id": rid, "original_row_sha256": digest(original),
            "primary_review_sha256": digest(primary[rid]), "original_question": source_question,
            "same_question_record_ids": peers, "same_question_all_states_reviewed": True,
            "revised_row": revised, "amendment_version": "v5",
            "status": "draft_only_requires_semantic_double_review",
            "approved_training_label": False, "training_allowed": False})
    path = OUT / "repair-drafts-v5-private.jsonl"
    if path.exists() and readl(path) != rows:
        raise RuntimeError("existing_v5_repairs_differ_refuse_overwrite")
    if not path.exists():
        writel(path, rows)
    print(json.dumps({"new_repair_drafts": len(rows), "superseded_candidate_drafts": len(amendments),
                      "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
