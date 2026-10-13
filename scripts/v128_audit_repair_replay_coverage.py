"""Verify bindings and replay coverage for already-authored v117 revisions."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from mini_agent_rl.tools.search import _BM25Index
from scripts.v123_cpu_core import digest, norm
from scripts.v128_audit_repairs import BASE, GOAL, audit as audit_inventory

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
INVENTORY = BASE / "inventory-private.jsonl"
CORE = BASE / "train-core-candidate-private.jsonl"
SUPPLEMENT = BASE / "double-reviewed-v122-supplement-candidate-private.jsonl"
CORPUS = ROOT / "data/evidence-sft-v114-clean/training-corpus.jsonl"
CORPUS_MANIFEST = ROOT / "data/evidence-sft-v114-clean/manifest.json"
REPLAY_INPUT = BASE / "repair-draft-replay-inputs-v2-private.jsonl"
REPLAYED = BASE / "repair-draft-states-cpu-v2-private.jsonl"
REPORT = BASE / "repair-draft-reconciliation-cpu-v2-private.json"
OUTPUT = GOAL / "g2-revised-state-coverage-private.json"


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def action_at(row: dict) -> dict:
    return json.loads(row["messages"][row["target_assistant_index"]]["content"])


def state_key(messages: list[dict], action: dict) -> str:
    return digest({"visible_messages": messages, "proposed_action": action})


def project(docs: list[dict] | None) -> list[dict]:
    return [{k: doc.get(k) for k in ("id", "title", "text")}
            for doc in (docs or []) if isinstance(doc, dict)]


def exact_prefix_link(search_state: dict, answer_state: dict) -> bool:
    prefix = search_state["visible_messages"]
    answer_messages = answer_state["visible_messages"]
    n = len(prefix)
    # answer_state.visible_messages ends before its proposed answer action.
    # The preceding search action and tool result add exactly two messages.
    if len(answer_messages) < n + 2 or answer_messages[:n] != prefix:
        return False
    try:
        action = json.loads(answer_messages[n].get("content", ""))
        tool = json.loads(answer_messages[n + 1].get("content", ""))
    except (TypeError, json.JSONDecodeError):
        return False
    return (answer_messages[n].get("role") == "assistant"
            and action == search_state["proposed_action"]
            and answer_messages[n + 1].get("role") == "tool"
            and project(tool.get("result")) == project(search_state.get("retrieved_after_search")))


def audit() -> dict:
    inventory_audit = audit_inventory()
    repair_ids = set(inventory_audit["v117_repair_candidate_record_ids"])
    raw_rows = read_jsonl(TRAIN)
    raw_by_id = {f"v117_train:{i:04d}": row for i, row in enumerate(raw_rows, 1)}
    inventory = {row["record_id"]: row for row in read_jsonl(INVENTORY)
                 if row.get("cohort") == "v117_train"}
    packet_groups = read_jsonl(REPLAY_INPUT)
    replayed_groups = read_jsonl(REPLAYED)
    report = read_json(REPORT)
    corpus_bytes = CORPUS.read_bytes()
    corpus_hash = sha(corpus_bytes)
    corpus_manifest = read_json(CORPUS_MANIFEST)
    if (corpus_hash != corpus_manifest.get("files", {}).get("training-corpus.jsonl")
            or corpus_hash != report["corpus_sha256"]):
        raise ValueError("repair_replay_corpus_hash_mismatch")
    corpus = [json.loads(line) for line in corpus_bytes.decode("utf-8-sig").splitlines()
              if line.strip()]
    index = _BM25Index(corpus)
    core = read_jsonl(CORE)
    supplement = read_jsonl(SUPPLEMENT)
    trained = core + supplement
    trained_questions = {norm(next(m["content"] for m in row["messages"]
                                    if m["role"] == "user")) for row in trained}
    trained_states = set()
    for row in trained:
        i = row["target_assistant_index"]
        trained_states.add(state_key(row["messages"][:i], action_at(row)))

    packet_by_id = {}
    for group in packet_groups:
        for state in group["states"]:
            if state["record_id"] in packet_by_id:
                raise ValueError("duplicate_repair_replay_input_record_id")
            packet_by_id[state["record_id"]] = (group, state)
    replay_by_id = {}
    group_to_replay = {}
    for group in replayed_groups:
        key = tuple(group["record_ids"])
        group_to_replay[key] = group
        for state in group["states"]:
            if state["record_id"] in replay_by_id:
                raise ValueError("duplicate_materialized_replay_record_id")
            replay_by_id[state["record_id"]] = (group, state)

    candidate_revised = []
    all_revised = []
    candidate_prompt_keys = defaultdict(list)
    query_change_actions = []
    bm25_replays = []
    group_change_count = 0
    all_group_rebuild_flags = True
    for packet_group in packet_groups:
        packet_id_states = {s["record_id"]: s for s in packet_group["states"]}
        materialized = group_to_replay.get(tuple(packet_group["record_ids"]))
        if materialized is None:
            raise ValueError("replay_input_group_has_no_materialized_replay")
        materialized_by_id = {s["record_id"]: s for s in materialized["states"]}
        if set(packet_id_states) != set(materialized_by_id):
            raise ValueError("replay_input_materialized_state_set_mismatch")

        edited_searches = [s for s in packet_group.get("search_replays", [])
                           if s.get("query_changed")]
        if edited_searches:
            group_change_count += 1
            all_group_rebuild_flags &= bool(
                packet_group.get("revised_search_requires_downstream_context_rebuild"))
            query_change_actions.extend(edited_searches)

        for record_id, packet_state in packet_id_states.items():
            raw = raw_by_id[record_id]
            inv = inventory[record_id]
            original_hash = digest(raw)
            if original_hash != inv["source_row_sha256"]:
                raise ValueError("inventory_to_frozen_source_hash_mismatch")
            revised = json.loads(json.dumps(raw, ensure_ascii=False))
            revised["messages"][1]["content"] = packet_group["question"]
            target_i = revised["target_assistant_index"]
            revised_action = packet_state["target_action"]
            revised["messages"][target_i]["content"] = json.dumps(
                revised_action, ensure_ascii=False, separators=(",", ":"))
            revised_hash = digest(revised)
            state = materialized_by_id[record_id]
            if (state["original_row_sha256"] != original_hash
                    or state["revised_row_sha256"] != revised_hash
                    or state["visible_messages"] != revised["messages"][:target_i]
                    or state["proposed_action"] != revised_action):
                raise ValueError("materialized_revision_hash_or_action_time_mismatch")
            if not state["within_token_limits"]:
                raise ValueError("materialized_revision_token_limit_failed")
            key = state_key(state["visible_messages"], state["proposed_action"])
            row_summary = {
                "record_id": record_id,
                "in_v117_repair_candidate_set": record_id in repair_ids,
                "source_row_sha256": original_hash,
                "revised_row_sha256": revised_hash,
                "revised_visible_prefix_sha256": digest(state["visible_messages"]),
                "action": state["proposed_action"]["action"],
                "action_target_sha256": sha(json.dumps(
                    state["proposed_action"], ensure_ascii=False,
                    sort_keys=True, separators=(",", ":")).encode("utf-8")),
                "prompt_tokens": state["prompt_tokens"],
                "target_tokens": state["target_tokens"],
                "sequence_tokens": state["total_tokens"],
                "within_limits": state["within_token_limits"],
                "exact_state_duplicate_in_revised_pool": False,
                "exact_state_duplicate_in_current_training_pool": key in trained_states,
                "normalized_question_overlaps_current_training_pool": norm(
                    packet_group["question"]) in trained_questions,
                "semantic_review": "not_run",
                "training_allowed": False,
            }
            all_revised.append((packet_group, state, row_summary, key))
            if record_id in repair_ids:
                candidate_revised.append((packet_group, state, row_summary, key))

        for search in packet_group.get("search_replays", []):
            query = search["query"]
            got = project(index.search(query, top_k=3))
            expected = project(search["results"])
            if got != expected:
                raise ValueError("repair_candidate_revised_query_bm25_mismatch")
            bm25_replays.append({
                "record_id": search["record_id"],
                "query_sha256": sha(query.encode("utf-8")),
                "query_changed_from_source": bool(search["query_changed"]),
                "top3_ids": [doc.get("id") for doc in got],
                "top3_matches_fixed_v114_training_corpus": True,
            })

    duplicate_count = Counter(key for *_, key in candidate_revised)
    for _, _, summary, key in candidate_revised:
        summary["exact_state_duplicate_in_revised_pool"] = duplicate_count[key] > 1
    revised_candidate_ids = {x[2]["record_id"] for x in candidate_revised}
    if len(revised_candidate_ids) != len(candidate_revised):
        raise ValueError("duplicate_revised_candidate_id")

    strict_links = []
    for packet_group in packet_groups:
        materialized = group_to_replay[tuple(packet_group["record_ids"])]
        searches = [s for s in materialized["states"]
                    if s["proposed_action"]["action"] == "search"]
        answers = [s for s in materialized["states"]
                   if s["proposed_action"]["action"] == "answer"]
        for search in searches:
            for answer in answers:
                if exact_prefix_link(search, answer):
                    strict_links.append({"search_record_id": search["record_id"],
                                         "answer_record_id": answer["record_id"],
                                         "both_repair_candidates": (search["record_id"] in repair_ids
                                                                     and answer["record_id"] in repair_ids)})

    q_overlaps = sum(x[2]["normalized_question_overlaps_current_training_pool"]
                     for x in candidate_revised)
    exact_train_duplicates = sum(x[2]["exact_state_duplicate_in_current_training_pool"]
                                 for x in candidate_revised)
    return {
        "version": "v128-revised-state-coverage-v1",
        "inputs_sha256": {
            "v117_train": sha(TRAIN.read_bytes()),
            "v117_inventory": sha(INVENTORY.read_bytes()),
            "v117_repair_inventory": sha(GOAL.joinpath("g2-repair-inventory-private.json").read_bytes()),
            "v114_training_corpus": corpus_hash,
            "v114_training_corpus_manifest": sha(CORPUS_MANIFEST.read_bytes()),
            "repair_replay_inputs": sha(REPLAY_INPUT.read_bytes()),
            "repair_replayed_states": sha(REPLAYED.read_bytes()),
            "repair_reconciliation_report": sha(REPORT.read_bytes()),
            "training_core": sha(CORE.read_bytes()),
            "v122_training_supplement": sha(SUPPLEMENT.read_bytes()),
        },
        "counts": {
            "v117_repair_candidates": len(repair_ids),
            "repair_candidates_with_materialized_revised_state": len(candidate_revised),
            "repair_candidates_still_without_materialized_revision": len(repair_ids - revised_candidate_ids),
            "other_disposition_states_in_repair_replay": len(all_revised) - len(candidate_revised),
            "revised_inputs_source_hash_action_time_and_hash_validated": len(all_revised),
            "all_materialized_states_within_token_limits": all(x[2]["within_limits"] for x in all_revised),
            "revised_pool_exact_duplicate_states": sum(v > 1 for v in duplicate_count.values()),
            "revised_candidate_states_exactly_duplicating_current_training_input": exact_train_duplicates,
            "revised_candidate_questions_overlapping_current_training_questions": q_overlaps,
            "search_replays_recomputed_against_v114_training_corpus": len(bm25_replays),
            "query_changed_actions": sum(x["query_changed_from_source"] for x in bm25_replays),
            "changed_query_question_groups_requiring_downstream_rebuild": group_change_count,
            "all_changed_query_groups_flagged_for_downstream_rebuild": all_group_rebuild_flags,
            "strict_action_result_prefix_links_in_35_groups": len(strict_links),
            "strict_links_both_states_in_v117_repair_candidate_set": sum(
                x["both_repair_candidates"] for x in strict_links),
            "semantic_double_approved_states_added": 0,
            "training_allowed": False,
        },
        "revised_candidate_states": [x[2] for x in sorted(candidate_revised,
                                                           key=lambda x: x[2]["record_id"])],
        "other_revised_states": [x[2] for x in sorted(all_revised,
                                                       key=lambda x: x[2]["record_id"])
                                 if not x[2]["in_v117_repair_candidate_set"]],
        "fixed_corpus_replays": bm25_replays,
        "strict_prefix_links": strict_links,
        "interpretation": [
            "Only 46 of the 107 flagged repair_candidate source states appear in the existing materialized revision set; the other 61 remain unrevised and must stay pending/excluded.",
            "The 46 revised states have exact raw-source and action-time bindings, distinct revised row hashes, and valid token replay. This does not establish semantic correctness.",
            "Query changes trigger downstream rebuild; same-question search and answer states are not promoted to strict pairs unless the exact prefix and retrieved result link is reproduced.",
            "BM25 retrieval equality, query novelty, and duplicate checks are mechanical evidence only.",
        ],
        "training_allowed": False,
        "data_ready_for_experiment": False,
    }


if __name__ == "__main__":
    result = audit()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "training_allowed": False},
                     ensure_ascii=False))
