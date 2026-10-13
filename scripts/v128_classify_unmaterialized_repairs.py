"""Expand the 61 unrevised v117 repair candidates into evidence-bound holds.

This is bookkeeping only: it never rewrites source rows, creates revised
training examples, approves semantics, or invokes a reviewer/model.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.v123_cpu_core import digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v124-forensic"
GOAL = BASE / "goal-v128"
BACKLOG = BASE / "v117-remaining-repair-backlog-private.json"
INVENTORY = GOAL / "g2-repair-inventory-private.json"
REPLAY_INPUTS = BASE / "repair-draft-replay-inputs-v2-private.jsonl"
REPLAYED = BASE / "repair-draft-states-cpu-v2-private.jsonl"
TRAIN = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
OUTPUT = GOAL / "g2-unmaterialized-repair-dispositions-private.json"

WORK_CLASS_TO_HOLD = {
    "evidence_or_target_block_requires_source_resolution": "hold_for_source_or_target_resolution",
    "repair_question_or_target_then_deduplicate": "hold_for_question_or_target_revision",
    "deduplicate_at_export_after_pair_check": "hold_for_pair_check_then_export_deduplication",
}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def audit() -> dict:
    backlog = read_json(BACKLOG)
    inventory = read_json(INVENTORY)
    groups_by_state = {}
    for group in backlog["groups"]:
        for state in group["flagged_states"]:
            rid = state["record_id"]
            if rid in groups_by_state:
                raise ValueError("duplicate_state_in_backlog")
            groups_by_state[rid] = (group, state)

    repair_ids = set(inventory["v117_repair_candidate_record_ids"])
    backlog_repair_ids = {
        rid for rid, (_, state) in groups_by_state.items()
        if state.get("effective_primary_disposition") == "repair_candidate"
    }
    if len(repair_ids) != 107 or len(backlog_repair_ids) != 61:
        raise ValueError("repair_candidate_inventory_count_mismatch")
    if not backlog_repair_ids <= repair_ids:
        raise ValueError("backlog_repair_ids_missing_from_frozen_inventory")

    replay_input_ids = set()
    for group in read_jsonl(REPLAY_INPUTS):
        replay_input_ids.update(state["record_id"] for state in group["states"])
    replayed_ids = set()
    for group in read_jsonl(REPLAYED):
        replayed_ids.update(state["record_id"] for state in group["states"])
    if replay_input_ids != replayed_ids:
        raise ValueError("repair_replay_input_materialization_mismatch")
    if backlog_repair_ids & replay_input_ids:
        raise ValueError("unmaterialized_backlog_candidate_is_in_replay")

    state_inventory = {state["record_id"]: state for state in inventory["states"]}
    train_rows = read_jsonl(TRAIN)
    rows_by_id = {f"v117_train:{i:04d}": row for i, row in enumerate(train_rows, 1)}
    if not backlog_repair_ids <= rows_by_id.keys():
        raise ValueError("backlog_candidate_not_in_frozen_train_source")

    disposition_rows = []
    for rid in sorted(backlog_repair_ids):
        group, source_state = groups_by_state[rid]
        inv = state_inventory[rid]
        work_class = group["work_class"]
        if work_class not in WORK_CLASS_TO_HOLD:
            raise ValueError("unknown_backlog_work_class")
        source_row = rows_by_id[rid]
        source_row_sha = digest(source_row)
        ledger_record_sha = source_state.get("record_sha256")
        inventory_record_sha = inv.get("source_sha256")
        # These hashes have different documented scopes. Verify the frozen row
        # against its own replay lineage separately; never conflate ledger hash.
        replay_source_hashes = {
            state.get("original_row_sha256")
            for group_row in read_jsonl(REPLAYED)
            for state in group_row.get("states", [])
            if state.get("record_id") == rid
        }
        if replay_source_hashes:
            raise ValueError("candidate_claimed_unmaterialized_but_found_in_replay")
        disposition_rows.append({
            "record_id": rid,
            "normalized_question_group_sha256": group["normalized_question_sha256"],
            "work_class": work_class,
            "current_hold_reason": WORK_CLASS_TO_HOLD[work_class],
            "ledger_issues": sorted(set(source_state.get("primary_issues", []))),
            "retrieval_binding_status": source_state.get("retrieval_binding"),
            "covered_by_existing_repair_reconciliation": bool(
                source_state.get("covered_by_repair_reconciliation")),
            "repair_proposal_present_but_unvalidated": bool(
                source_state.get("primary_repair_proposal")),
            "source_ledger_record_sha256": ledger_record_sha,
            "inventory_source_sha256": inventory_record_sha,
            "frozen_train_row_canonical_sha256": source_row_sha,
            "in_existing_materialized_revision": False,
            "in_existing_replay_input": False,
            "export_eligible": False,
            "semantic_review": "not_run_or_not_verified",
            "training_allowed": False,
        })

    if len(disposition_rows) != 61 or len({r["record_id"] for r in disposition_rows}) != 61:
        raise ValueError("expanded_disposition_rows_not_one_to_one")
    class_counts = Counter(row["work_class"] for row in disposition_rows)
    per_class_groups = {}
    for cls in WORK_CLASS_TO_HOLD:
        per_class_groups[cls] = len({r["normalized_question_group_sha256"]
                                     for r in disposition_rows if r["work_class"] == cls})

    result = {
        "version": "v128-unmaterialized-repair-dispositions-v1",
        "scope": "61 backlog repair-candidate states without a materialized revision",
        "inputs_sha256": {
            "backlog": sha_bytes(BACKLOG.read_bytes()),
            "repair_inventory": sha_bytes(INVENTORY.read_bytes()),
            "replay_inputs": sha_bytes(REPLAY_INPUTS.read_bytes()),
            "materialized_replays": sha_bytes(REPLAYED.read_bytes()),
            "frozen_v117_train": sha_bytes(TRAIN.read_bytes()),
        },
        "counts": {
            "backlog_repair_candidates": len(disposition_rows),
            "unique_record_ids": len({r["record_id"] for r in disposition_rows}),
            "unique_normalized_question_groups": len({r["normalized_question_group_sha256"]
                                                       for r in disposition_rows}),
            "work_class_state_counts": dict(sorted(class_counts.items())),
            "work_class_question_group_counts": dict(sorted(per_class_groups.items())),
            "materialized_revisions": sum(r["in_existing_materialized_revision"]
                                           for r in disposition_rows),
            "semantic_approvals": 0,
            "export_eligible": 0,
            "training_allowed": False,
        },
        "dispositions": disposition_rows,
        "interpretation": [
            "Every unresolved backlog repair-candidate ID has an evidence-linked hold reason and is explicitly excluded from export.",
            "The 61 holds are not 61 completed revisions and do not meet the all-revisions G2.2 criterion.",
            "Primary repair proposals are only recorded as present/absent; they are not adopted as valid query edits or training targets.",
            "The listed source hashes preserve provenance but do not establish correctness or semantic approval.",
        ],
        "training_allowed": False,
        "data_ready_for_experiment": False,
    }
    return result


if __name__ == "__main__":
    result = audit()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "training_allowed": False},
                     ensure_ascii=False))
