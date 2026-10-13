"""CPU-only source, fixed-BM25, and tokenizer replay for v117 repair labels.

This replays only existing actions from the frozen v117 training source. It does
not author a query, create a revised trajectory, inspect gold answers, load a
model, call an API, or grant a training label.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoProcessor

from mini_agent_rl.tools.search import _BM25Index
from mini_agent_rl.training.sft import encode_assistant_only
from scripts.prepare_v116_review import PROCESSOR, render
from scripts.v123_cpu_core import digest, norm
from scripts.v128_audit_repairs import BASE, GOAL, audit as audit_inventory

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
INVENTORY = BASE / "inventory-private.jsonl"
CORPUS = ROOT / "data/evidence-sft-v117/candidate-corpus-private.json"
FREEZE = ROOT / "data/evidence-sft-v117/candidate-freeze.json"
REPLAY_STATES = BASE / "repair-draft-states-cpu-v2-private.jsonl"
BACKLOG = BASE / "v117-remaining-repair-backlog-private.json"
OUTPUT = GOAL / "g2-v117-repair-source-replay-private.json"
TOKENIZER_FILES = (
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json",
    "processor_config.json", "preprocessor_config.json", "chat_template.jinja",
    "vocab.json", "merges.txt", "tokenizer.model", "spiece.model",
)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def parse_action(row: dict) -> dict:
    return json.loads(row["messages"][row["target_assistant_index"]]["content"])


def tool_result_after_search(row: dict, query: str) -> list[dict] | None:
    messages = row["messages"]
    question_text = next((m["content"] for m in messages if m.get("role") == "user"), None)
    for i, message in enumerate(messages[:-1]):
        if message.get("role") != "assistant":
            continue
        try:
            action = json.loads(message.get("content", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        if action.get("action") != "search" or action.get("query") != query:
            continue
        if messages[i + 1].get("role") != "tool":
            continue
        try:
            payload = json.loads(messages[i + 1].get("content", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        result = payload.get("result")
        if isinstance(result, list):
            # Preserve only query-to-result linkage; the question check is done by caller.
            return result
    return None


def observation_signature(docs: list[dict] | None) -> str | None:
    if docs is None:
        return None
    normalized = [{k: doc.get(k) for k in ("id", "title", "text")}
                  for doc in docs if isinstance(doc, dict)]
    return digest(normalized)


def replay_token(row: dict, processor) -> dict:
    target_i = row["target_assistant_index"]
    prompt = row["messages"][:target_i]
    target_content = row["messages"][target_i]["content"]
    rendered = render(processor, [SimpleNamespace(**message) for message in prompt])
    target = {"role": "assistant", "content": target_content}
    encoded = encode_assistant_only(processor, prompt + [target], 2176,
                                    len(prompt), False)
    target_tokens = sum(token != -100 for token in encoded["labels"])
    return {
        "prompt_tokens": len(rendered),
        "target_tokens": target_tokens,
        "sequence_tokens": len(encoded["input_ids"]),
        "prompt_ids_sha256": digest(rendered),
        "target_mask_sha256": digest(encoded["labels"]),
        "within_limits": (len(rendered) <= 2048 and target_tokens <= 128
                          and len(encoded["input_ids"]) <= 2176),
    }


def audit() -> dict:
    inventory_result = audit_inventory()
    candidate_ids = set(inventory_result["v117_repair_candidate_record_ids"])
    train_rows = read_jsonl(TRAIN)
    inventory_rows = {row["record_id"]: row for row in read_jsonl(INVENTORY)
                      if row.get("cohort") == "v117_train"}
    corpus_bytes = CORPUS.read_bytes()
    corpus = json.loads(corpus_bytes.decode("utf-8-sig"))
    freeze = read_json(FREEZE)
    if sha_bytes(corpus_bytes) != freeze["corpus_sha256"]:
        raise ValueError("fixed_v117_corpus_hash_mismatch")
    if len(corpus) != freeze["documents"]:
        raise ValueError("fixed_v117_corpus_document_count_mismatch")

    expected_source_hashes: dict[str, set[str]] = defaultdict(set)
    backlog = read_json(BACKLOG)
    for group in backlog["groups"]:
        for state in group["flagged_states"]:
            if state["record_id"] in candidate_ids and state.get("effective_primary_disposition") == "repair_candidate":
                expected_source_hashes[state["record_id"]].add(state["source_row_sha256"])
    for row in read_jsonl(REPLAY_STATES):
        for state in row.get("states", []):
            record_id = state.get("record_id")
            if record_id in candidate_ids:
                expected_source_hashes[record_id].add(state["original_row_sha256"])
    if set(expected_source_hashes) != candidate_ids:
        raise ValueError("repair_candidate_source_hash_coverage_mismatch")
    if any(len(values) != 1 for values in expected_source_hashes.values()):
        raise ValueError("conflicting_source_hash_bindings")

    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    index = _BM25Index(corpus)
    by_question_query_results = {}
    for source_row in train_rows:
        question = next((m["content"] for m in source_row["messages"]
                         if m.get("role") == "user"), "")
        for i, message in enumerate(source_row["messages"][:-1]):
            if message.get("role") != "assistant":
                continue
            try:
                prior_action = json.loads(message.get("content", ""))
            except (TypeError, json.JSONDecodeError):
                continue
            if prior_action.get("action") == "search" and source_row["messages"][i + 1].get("role") == "tool":
                try:
                    payload = json.loads(source_row["messages"][i + 1].get("content", ""))
                except (TypeError, json.JSONDecodeError):
                    continue
                result = payload.get("result")
                if isinstance(result, list):
                    by_question_query_results.setdefault(
                        (norm(question), prior_action.get("query", "")), []).append(result)

    rows_out = []
    prompt_action_keys: dict[str, list[str]] = defaultdict(list)
    questions = Counter()
    actions = Counter()
    for record_id in sorted(candidate_ids):
        prefix, ordinal = record_id.rsplit(":", 1)
        if prefix != "v117_train":
            raise ValueError("unexpected_repair_candidate_cohort")
        row_i = int(ordinal) - 1
        if not 0 <= row_i < len(train_rows):
            raise ValueError("repair_record_id_outside_frozen_source")
        source_row = train_rows[row_i]
        source_hash = digest(source_row)
        if source_hash != next(iter(expected_source_hashes[record_id])):
            raise ValueError("repair_record_id_source_row_hash_mismatch")
        inventory = inventory_rows.get(record_id)
        if inventory is None or inventory.get("source_row_sha256") != source_hash:
            raise ValueError("inventory_source_row_binding_mismatch")
        target_i = source_row["target_assistant_index"]
        if (inventory.get("visible_messages") != source_row["messages"][:target_i]
                or inventory.get("proposed_action") != parse_action(source_row)):
            raise ValueError("inventory_action_time_prefix_mismatch")
        action = parse_action(source_row)
        action_type = action.get("action")
        if action_type not in {"search", "answer"}:
            raise ValueError("repair_action_schema_invalid")
        actions[action_type] += 1
        question = inventory["question"]
        questions[norm(question)] += 1
        target_value = action.get("query" if action_type == "search" else "answer", "")
        results = index.search(target_value, top_k=3) if action_type == "search" else []
        prior_ids = {doc.get("id") for doc in inventory.get("before_action", [])}
        result_ids = [doc.get("id") for doc in results]
        observed_candidates = (by_question_query_results.get((norm(question), target_value), [])
                               if action_type == "search" else [])
        replay_sig = observation_signature(results) if action_type == "search" else None
        observed_match = any(observation_signature(result) == replay_sig
                             for result in observed_candidates)
        token = replay_token(source_row, processor)
        visible_prefix_hash = digest(source_row["messages"][:target_i])
        target_state_hash = digest({"messages": source_row["messages"][:target_i + 1],
                                    "target_assistant_index": target_i,
                                    "thinking_enabled": source_row.get("thinking_enabled")})
        signature = digest({"visible_messages": inventory["visible_messages"],
                            "proposed_action": action})
        prompt_action_keys[signature].append(record_id)
        rows_out.append({
            "record_id": record_id,
            "source_row_sha256": source_hash,
            "inventory_record_sha256": inventory["record_sha256"],
            "target_state_sha256": target_state_hash,
            "action": action_type,
            "visible_prefix_sha256": visible_prefix_hash,
            "query_or_answer_sha256": sha_bytes(target_value.encode("utf-8")),
            "query_nonempty": bool(target_value.strip()),
            "prior_document_count": len(inventory.get("before_action", [])),
            "bm25_v2_replay": ({
                "top3_document_ids": result_ids,
                "top3_document_content_hashes": [
                    sha_bytes(json.dumps({k: d.get(k) for k in ("id", "title", "text")},
                                         ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8"))
                    for d in results],
                "new_top3_document_count": len(set(result_ids) - prior_ids),
                "overlap_with_before_action_count": len(set(result_ids) & prior_ids),
                "matching_observed_same_question_query_result": observed_match,
                "observed_same_question_query_result_count": len(observed_candidates),
                "interpretation": "CPU retrieval diagnostic only; not semantic relevance approval",
            } if action_type == "search" else None),
            "token_replay": token,
            "training_allowed": False,
        })

    duplicate_clusters = [sorted(ids) for ids in prompt_action_keys.values() if len(ids) > 1]
    inputs = {
        "v117_train": sha_bytes(TRAIN.read_bytes()),
        "v117_inventory": sha_bytes(INVENTORY.read_bytes()),
        "v117_fixed_corpus": sha_bytes(corpus_bytes),
        "v117_corpus_freeze": sha_bytes(FREEZE.read_bytes()),
        "v117_repair_backlog": sha_bytes(BACKLOG.read_bytes()),
        "v124_cpu_replayed_states": sha_bytes(REPLAY_STATES.read_bytes()),
        "repair_inventory_code": sha_bytes(Path(__file__).read_bytes()),
    }
    tokenizer_hashes = {name: sha_bytes((PROCESSOR / name).read_bytes())
                        for name in TOKENIZER_FILES if (PROCESSOR / name).is_file()}
    return {
        "version": "v128-v117-repair-source-replay-v1",
        "inputs_sha256": inputs,
        "processor": {"path": str(PROCESSOR), "local_files_only": True,
                      "model_loaded": False, "tokenizer_config_sha256": tokenizer_hashes},
        "retriever": {"version": "bm25-v2 top3", "fixed_corpus_sha256": sha_bytes(corpus_bytes),
                      "fixed_corpus_documents": len(corpus)},
        "counts": {
            "v117_repair_candidates": len(candidate_ids),
            "source_row_hashes_matched": len(rows_out),
            "inventory_prefix_and_target_bindings_matched": len(rows_out),
            "actions": dict(actions),
            "normalized_question_groups": len(questions),
            "questions_repeated_in_candidate_states": sum(n > 1 for n in questions.values()),
            "exact_duplicate_prompt_action_clusters": len(duplicate_clusters),
            "exact_duplicate_prompt_action_states": sum(len(g) for g in duplicate_clusters),
            "search_replays_with_matching_observed_tool_result": sum(
                row["bm25_v2_replay"]["matching_observed_same_question_query_result"]
                for row in rows_out if row["action"] == "search"),
            "search_states_with_any_same_question_query_tool_result": sum(
                row["bm25_v2_replay"]["observed_same_question_query_result_count"] > 0
                for row in rows_out if row["action"] == "search"),
            "all_tokens_within_limits": all(row["token_replay"]["within_limits"]
                                             for row in rows_out),
            "model_loaded": False,
            "api_called": False,
            "new_queries_authored": 0,
            "semantic_reviews_added": 0,
            "training_allowed": False,
        },
        "exact_duplicate_prompt_action_clusters": duplicate_clusters,
        "rows": rows_out,
        "limitations": [
            "BM25 result inclusion, new-document count, and observed-result equality do not establish query grounding or entailment.",
            "This is a replay of original search targets; it does not create revised labels or new action trajectories.",
            "Token limits only establish format feasibility; no semantic or training gate is passed by this replay.",
        ],
        "training_allowed": False,
        "data_ready_for_experiment": False,
    }


if __name__ == "__main__":
    result = audit()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "retriever": result["retriever"],
                      "training_allowed": False}, ensure_ascii=False))
