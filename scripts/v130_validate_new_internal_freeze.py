"""Independent mechanical replay of the one-time v130 internal freeze."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.prepare_v116_data import full_rows
from scripts.prepare_v116_review import PROCESSOR, canonical, render
from mini_agent_rl.domain import Message
from mini_agent_rl.training.sft import encode_assistant_only
from scripts.v123_cpu_core import ROOT, norm, read, readl, sha, write
from scripts.v130_build_new_internal_pool import (
    CORPUS_PATH, FROZEN_DIR, PREFLIGHT, SCREENED, SEED, _history,
)
from scripts.v130_diagnose_linked_title_policy import choose_next
from scripts.v129_review_protocol import contract_identity
from mini_agent_rl.tools.search import _BM25Index

BASE = ROOT / "data/evidence-sft-v124-forensic/goal-v128"
REPORT = BASE / "g4-frozen-replay-audit-v130-private.json"


def _hjson(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _tool_results(message: dict) -> list[dict]:
    if message.get("role") != "tool":
        raise ValueError("expected_tool_result_message")
    payload = json.loads(message["content"])
    if payload.get("ok") is not True or not isinstance(payload.get("result"), list):
        raise ValueError("tool_result_shape_invalid")
    return payload["result"]


def _replay_linked_queries(packet_row: dict, index, title_lookup: dict[str, str], corpus_lookup: dict[str, dict]) -> int:
    messages = packet_row["visible_messages"]
    question = packet_row["question"]
    used = {norm(question)}
    evidence_ids: set[str] = set()
    latest_docs: list[dict] = []
    search_count = 0
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        action = json.loads(message["content"])
        if action.get("action") != "search":
            continue
        query = action.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("visible_search_query_invalid")
        if search_count == 0:
            if norm(query) != norm(question):
                raise ValueError("initial_search_not_question_grounded")
        else:
            expected = choose_next(question, latest_docs, used, index, title_lookup,
                                   excluded_doc_ids=evidence_ids)
            if expected is None or norm(query) != norm(expected[0]):
                raise ValueError("visible_followup_query_policy_mismatch")
        if i + 1 >= len(messages):
            raise ValueError("visible_search_missing_tool_observation")
        observed = _tool_results(messages[i + 1])
        expected_results = index.search(query, top_k=3)
        if [doc.get("id") for doc in observed] != [doc.get("id") for doc in expected_results]:
            raise ValueError("visible_bm25_top3_id_order_mismatch")
        for doc, expected in zip(observed, expected_results):
            source = corpus_lookup.get(doc.get("id"), {})
            expected_ref = f"{doc.get('id')}:{hashlib.sha256(str(source.get('text', '')).encode()).hexdigest()[:12]}"
            if "ref" in doc:
                if (doc.get("ref") != expected_ref
                        or doc.get("id") not in evidence_ids):
                    raise ValueError("visible_reference_not_prior_document")
            elif (doc.get("title") != expected.get("title")
                  or not isinstance(doc.get("text"), str)
                  or not str(expected.get("text", "")).startswith(doc["text"])):
                raise ValueError("visible_bm25_top3_content_mismatch")
        latest_docs = expected_results
        evidence_ids.update(doc["id"] for doc in expected_results)
        used.add(norm(query))
        search_count += 1
    return search_count


def audit() -> dict:
    freeze_path = FROZEN_DIR / "candidate-freeze-private.json"
    packet_path = FROZEN_DIR / "review/internal-packets-private.jsonl"
    source_path = FROZEN_DIR / "review/source-map-private.jsonl"
    reuse_path = FROZEN_DIR / "review/old-vote-reuse-private.jsonl"
    freeze = read(freeze_path)
    packets = readl(packet_path)
    source_map = readl(source_path)
    reuse = readl(reuse_path)
    screen = read(PREFLIGHT)
    screened = readl(SCREENED)
    corpus = read(CORPUS_PATH)
    corpus_index = _BM25Index(corpus)
    title_lookup = {norm(doc["title"]): doc["title"] for doc in corpus if doc.get("title")}
    corpus_lookup = {str(doc["id"]): doc for doc in corpus}

    checks: dict[str, bool] = {}
    checks["screen_has_exactly_5000_unique_ids_and_hashes"] = (
        len(screened) == 5000 == len({r["id"] for r in screened}) == len({r["question_sha256"] for r in screened})
        and all(r.get("screen_order") == i and len(r.get("question_sha256", "")) == 64
                for i, r in enumerate(screened)))
    checks["screen_preflight_matches_manifest"] = (
        screen.get("source_questions_inspected") == 5000
        and screen.get("screened_manifest_sha256") == sha(SCREENED)
        and screen.get("corpus_sha256") == sha(CORPUS_PATH))
    eligible = sorted((r for r in screened if r["disposition"] == "eligible"),
                      key=lambda row: hashlib.sha256((SEED + "candidate:" + row["id"]).encode()).hexdigest())
    scope = [r for r in eligible if r.get("scope_hint_for_batch_order_only")]
    other = [r for r in eligible if not r.get("scope_hint_for_batch_order_only")]
    selected = scope[:60] + other[:40]
    if len(selected) < 100:
        selected_ids_temp = {r["id"] for r in selected}
        selected.extend(r for r in scope[60:] + other[40:] if r["id"] not in selected_ids_temp)
        selected = selected[:100]
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    if len(source_map) != len(packets):
        raise ValueError("source_map_packet_count_mismatch")
    for source, packet_row in zip(source_map, packets):
        grouped[source["source_id"]].append((source, packet_row))
    expected_ids = [r["id"] for r in selected]
    actual_ids = list(dict.fromkeys(row["source_id"] for row in source_map))
    checks["selection_replays_fixed_hash_and_scope_schedule"] = actual_ids == expected_ids and len(actual_ids) == 100
    checks["100_distinct_questions_200_states_and_strict_pairs"] = (
        len(actual_ids) == 100 and len(packets) == 200 and len({p["case_id"] for p in packets}) == 200
        and set(grouped) == set(actual_ids)
        and all(len(rows) == 2 and {p["proposed_action"]["action"] for _, p in rows} == {"search", "answer"}
                for rows in grouped.values()))
    checks["no_old_vote_reuse_or_training_eligibility"] = not reuse and freeze.get("training_eligible_states") == 0
    checks["review_protocol_identity_matches"] = freeze.get("review_protocol") == contract_identity()
    checks["packet_and_source_map_hashes_match"] = (
        freeze.get("packet_sha256") == sha(packet_path)
        and freeze.get("source_map_sha256") == sha(source_path)
        and freeze.get("old_vote_reuse_sha256") == sha(reuse_path))
    checks["frozen_corpus_and_source_screen_hashes_match"] = (
        freeze.get("corpus_sha256") == sha(CORPUS_PATH)
        and freeze.get("source_manifest_sha256") == sha(SCREENED)
        and freeze.get("screening_preflight_sha256") == sha(PREFLIGHT))

    history_ids, history_texts = _history()
    screened_text_hashes = {row["question_sha256"] for row in screened}
    history_text_hashes = {hashlib.sha256(text.encode()).hexdigest() for text in history_texts}
    selected_ids = set(actual_ids)
    selected_text_hashes = {source["question_sha256"] for source in source_map}
    checks["all_screened_ids_exclude_history"] = not ({row["id"] for row in screened} & history_ids)
    checks["all_screened_questions_exclude_history_text"] = not (screened_text_hashes & history_text_hashes)
    checks["selected_pairs_exclude_history_and_duplicate_text"] = (
        not (selected_ids & history_ids) and not (selected_text_hashes & history_text_hashes)
        and len(selected_text_hashes) == 100)

    source_rows = full_rows(actual_ids)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    token_failures = []
    query_replay_failures = []
    answer_mismatch = []
    for source_id in actual_ids:
        source_packets = [packet_row for _, packet_row in grouped[source_id]]
        source_packets.sort(key=lambda p: 0 if p["proposed_action"]["action"] == "search" else 1)
        question = source_rows[source_id]["question"]
        if any(packet_row["question"] != question for packet_row in source_packets):
            raise ValueError("paired_question_mismatch")
        for packet_row in source_packets:
            if (packet_row["prompt_tokens"] > 2048 or packet_row["target_tokens"] > 128
                    or packet_row["total_tokens"] > 2176):
                token_failures.append(packet_row["case_id"])
            try:
                messages = [Message(**message) for message in packet_row["visible_messages"]]
                prompt_ids = render(processor, messages)
                target = {"role": "assistant", "content": json.dumps(packet_row["proposed_action"],
                                                                        ensure_ascii=False, separators=(",", ":"))}
                encoded = encode_assistant_only(processor, packet_row["visible_messages"] + [target],
                                                2176, len(packet_row["visible_messages"]), False)
                target_tokens = sum(value != -100 for value in encoded["labels"])
                sequence_ids = encoded["input_ids"]
                if (len(prompt_ids) != packet_row["prompt_tokens"]
                        or target_tokens != packet_row["target_tokens"]
                        or len(sequence_ids) != packet_row["total_tokens"]
                        or hashlib.sha256(canonical(prompt_ids)).hexdigest() != packet_row["prompt_ids_sha256"]
                        or hashlib.sha256(canonical(sequence_ids)).hexdigest() != packet_row["sequence_ids_sha256"]
                        or hashlib.sha256(canonical(encoded["labels"])).hexdigest() != packet_row["target_mask_sha256"]):
                    token_failures.append(packet_row["case_id"])
            except Exception:
                token_failures.append(packet_row["case_id"])
            try:
                searches = _replay_linked_queries(packet_row, corpus_index, title_lookup, corpus_lookup)
                action = packet_row["proposed_action"]["action"]
                if action == "search":
                    action = packet_row["proposed_action"]
                    used = {norm(question)}
                    first_results = corpus_index.search(question, top_k=3)
                    expected = choose_next(question, first_results, used, corpus_index, title_lookup,
                                           excluded_doc_ids={d["id"] for d in first_results})
                    if expected is None or norm(action.get("query", "")) != norm(expected[0]):
                        raise ValueError("target_followup_query_mismatch")
                    expected_top3 = [{k: value for k, value in doc.items() if k != "score"}
                                     for doc in corpus_index.search(action["query"], top_k=3)]
                    if packet_row["retrieved_after_search"] != expected_top3:
                        raise ValueError("target_followup_retrieval_mismatch")
                    if searches != 1:
                        raise ValueError("search_packet_prior_search_count_invalid")
                elif action == "answer" and searches not in {2, 3}:
                    raise ValueError("answer_packet_search_depth_invalid")
            except Exception:
                query_replay_failures.append(packet_row["case_id"])
        answer_packet = next(p for p in source_packets if p["proposed_action"]["action"] == "answer")
        if answer_packet["proposed_action"].get("answer") != str(source_rows[source_id]["answer"]):
            answer_mismatch.append(source_id)
    checks["tokenization_and_budget_replay"] = not token_failures
    checks["visible_query_and_top3_replay"] = not query_replay_failures
    checks["official_answer_proposal_source_binding"] = not answer_mismatch
    checks["training_remains_forbidden"] = (freeze.get("training_allowed") is False
                                              and freeze.get("data_ready_for_experiment") is False
                                              and all(not s.get("training_eligible") for s in source_map))

    inputs = [freeze_path, packet_path, source_path, reuse_path, SCREENED, PREFLIGHT,
              CORPUS_PATH, Path(__file__), ROOT / "scripts/v130_build_new_internal_pool.py",
              ROOT / "scripts/v130_diagnose_linked_title_policy.py",
              ROOT / "scripts/v129_review_protocol.py"]
    result = {"version": "v130-new-internal-freeze-audit-v1", "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "passed": all(checks.values()),
        "question_count": len(actual_ids), "state_count": len(packets), "strict_pairs": sum(len(v) == 2 for v in grouped.values()),
        "screened_questions": len(screened), "eligible_questions": len(eligible),
        "token_replay_failures": token_failures, "query_replay_failures": query_replay_failures,
        "answer_binding_failures": answer_mismatch,
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        "training_allowed": False, "data_ready_for_experiment": False}
    write(REPORT, result)
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=True, indent=2))
