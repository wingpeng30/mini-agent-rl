"""Find repair groups where a changed search can mechanically occupy an existing answer slot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v124-forensic"
GOAL = BASE / "goal-v128"
REPLAY_INPUT = BASE / "repair-draft-replay-inputs-v2-private.jsonl"
REPLAYED = BASE / "repair-draft-states-cpu-v2-private.jsonl"
REVISED_COVERAGE = GOAL / "g2-revised-state-coverage-private.json"
REPAIR_INVENTORY = GOAL / "g2-repair-inventory-private.json"
OUTPUT = GOAL / "g2-changed-query-context-feasibility-private.json"


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict], bytes]:
    raw = path.read_bytes()
    return [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()], raw


def audit() -> dict:
    inputs, input_bytes = read_jsonl(REPLAY_INPUT)
    replayed, replayed_bytes = read_jsonl(REPLAYED)
    coverage_bytes = REVISED_COVERAGE.read_bytes()
    coverage = json.loads(coverage_bytes.decode("utf-8-sig"))
    inventory_bytes = REPAIR_INVENTORY.read_bytes()
    inventory = json.loads(inventory_bytes.decode("utf-8-sig"))
    if len(inputs) != len(replayed):
        raise ValueError("repair_replay_group_count_mismatch")
    repair_ids = set(inventory["v117_repair_candidate_record_ids"])
    coverage_replays = {row["record_id"]: row for row in coverage["fixed_corpus_replays"]}
    changed_actions, potential_links = [], []
    for source_group, replay_group in zip(inputs, replayed):
        if source_group["record_ids"] != replay_group["record_ids"]:
            raise ValueError("repair_replay_group_identity_mismatch")
        states = {row["record_id"]: row for row in replay_group["states"]}
        changed = [row for row in source_group.get("search_replays", [])
                   if row.get("query_changed")]
        for search_replay in changed:
            search_id = search_replay["record_id"]
            search = states.get(search_id)
            checked_retrieval = coverage_replays.get(search_id)
            if (search is None or search["proposed_action"].get("action") != "search"
                    or search["proposed_action"].get("query") != search_replay.get("query")
                    or checked_retrieval is None
                    or checked_retrieval.get("query_changed_from_source") is not True
                    or checked_retrieval.get("top3_matches_fixed_v114_training_corpus") is not True):
                raise ValueError("changed_search_not_bound_to_fixed_corpus_replay")
            changed_actions.append({
                "search_record_id": search_id,
                "question_group_record_ids": source_group["record_ids"],
                "is_v117_repair_candidate": search_id in repair_ids,
                "query_sha256": checked_retrieval["query_sha256"],
                "retrieved_top3_ids": checked_retrieval["top3_ids"],
                "search_visible_prefix_sha256": sha(json.dumps(
                    search["visible_messages"], ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode("utf-8")),
            })
            n = len(search["visible_messages"])
            for answer in replay_group["states"]:
                if answer["proposed_action"].get("action") != "answer":
                    continue
                answer_messages = answer["visible_messages"]
                if (len(answer_messages) == n + 2
                        and answer_messages[:n] == search["visible_messages"]):
                    potential_links.append({
                        "search_record_id": search_id,
                        "answer_record_id": answer["record_id"],
                        "both_are_v117_repair_candidates": (search_id in repair_ids
                                                              and answer["record_id"] in repair_ids),
                        "same_action_time_prefix_exact": True,
                        "answer_target_sha256": sha(json.dumps(
                            answer["proposed_action"], ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")),
                        "answer_context_rebuild_status": "structural_slot_only_not_materialized",
                    })
    if len(changed_actions) != 10:
        raise ValueError("changed_query_action_count_mismatch")
    return {
        "version": "v128-changed-query-context-feasibility-v1",
        "inputs_sha256": {
            "repair_replay_inputs": sha(input_bytes),
            "repair_replayed_states": sha(replayed_bytes),
            "revised_state_coverage": sha(coverage_bytes),
            "repair_inventory": sha(inventory_bytes),
        },
        "changed_query_actions": len(changed_actions),
        "changed_queries_in_v117_repair_candidate_set": sum(
            row["is_v117_repair_candidate"] for row in changed_actions),
        "changed_query_groups": len({tuple(row["question_group_record_ids"])
                                      for row in changed_actions}),
        "same_group_exact_prefix_immediate_answer_slots": len(potential_links),
        "candidate_rows_materialized": 0,
        "answer_targets_reused_without_edit": True,
        "model_loaded": False,
        "generation_or_api_call": False,
        "training_allowed": False,
        "interpretation": [
            "The structural check identifies whether an existing answer state begins with the exact prefix preceding a changed search and ends immediately after that search result slot.",
            "A match only makes deterministic context reconstruction plausible; retrieval serialization, tokenizer replay and independent semantic approval remain required.",
            "No answer state or tool result was changed or exported by this audit.",
        ],
        "changed_actions": changed_actions,
        "potential_links": potential_links,
    }


if __name__ == "__main__":
    result = audit()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "changed_query_actions", "changed_queries_in_v117_repair_candidate_set",
        "changed_query_groups", "same_group_exact_prefix_immediate_answer_slots",
        "candidate_rows_materialized", "training_allowed")}, ensure_ascii=False))
