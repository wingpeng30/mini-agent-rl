"""Recount embedded searches in retained v117 answer trajectories and dedupe.

No new query is generated. Gold/source support labels are not read or used by
this audit. The input pool is the named, already-retained train candidate and
the already-reviewed v122 supplement only.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from scripts.v123_cpu_core import digest, norm

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v124-forensic"
GOAL = BASE / "goal-v128"
V117_TRAIN = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
CORE = BASE / "train-core-candidate-private.jsonl"
SUPPLEMENT = BASE / "double-reviewed-v122-supplement-candidate-private.jsonl"
EXTRACTED = BASE / "v126-retained-linked-search-state-candidates-private.jsonl"
EXTRACTION_REPORT = BASE / "v126-retained-linked-search-extraction-report-private.json"
REPAIR_INVENTORY = GOAL / "g2-repair-inventory-private.json"
BUILDER = ROOT / "scripts/build_v114_sft_candidates.py"
V117_PREPARER = ROOT / "scripts/prepare_v117.py"
V117_FINALIZER = ROOT / "scripts/finalize_v117.py"
OUTPUT = GOAL / "g2-embedded-search-reuse-audit-private.json"


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def action(row: dict) -> dict:
    return json.loads(row["messages"][row["target_assistant_index"]]["content"])


def state_key(messages: list[dict], proposed_action: dict) -> str:
    return digest({"visible_messages": messages, "proposed_action": proposed_action})


def user_question(row: dict) -> str:
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def project_docs(docs: list[dict]) -> list[dict]:
    return [{k: doc.get(k) for k in ("id", "title", "text")}
            for doc in docs if isinstance(doc, dict)]


def exact_answer_prefix_link(candidate: dict, answer_row: dict) -> bool:
    prefix = candidate["visible_messages"]
    answer_messages = answer_row["messages"][:answer_row["target_assistant_index"]]
    n = len(prefix)
    if len(answer_messages) < n + 2 or answer_messages[:n] != prefix:
        return False
    try:
        embedded_action = json.loads(answer_messages[n].get("content", ""))
        tool = json.loads(answer_messages[n + 1].get("content", ""))
    except (TypeError, json.JSONDecodeError):
        return False
    return (
        answer_messages[n].get("role") == "assistant"
        and embedded_action == candidate["proposed_action"]
        and answer_messages[n + 1].get("role") == "tool"
        and tool.get("ok") is True
        and project_docs(tool.get("result") or []) == project_docs(
            candidate.get("retrieved_after_search") or [])
        and action(answer_row).get("action") == "answer"
    )


def audit() -> dict:
    frozen_rows = read_jsonl(V117_TRAIN)
    core_rows = read_jsonl(CORE)
    supplement_rows = read_jsonl(SUPPLEMENT)
    repair_inventory = json.loads(REPAIR_INVENTORY.read_text(encoding="utf-8-sig"))
    repair_state_by_id = {r["record_id"]: r for r in repair_inventory["states"]}
    rows_by_id = {f"v117_train:{i:04d}": row
                  for i, row in enumerate(frozen_rows, 1)}
    frozen_ids_by_row_hash = defaultdict(list)
    for rid, row in rows_by_id.items():
        frozen_ids_by_row_hash[digest(row)].append(rid)

    frozen_search_targets = defaultdict(list)
    for rid, row in rows_by_id.items():
        target = action(row)
        if target.get("action") == "search":
            key = state_key(row["messages"][:row["target_assistant_index"]], target)
            frozen_search_targets[key].append(rid)

    retained_rows = core_rows + supplement_rows
    core_row_hashes = {digest(row) for row in core_rows}
    retained_exact_states = defaultdict(list)
    retained_questions = set()
    for idx, row in enumerate(retained_rows):
        target = action(row)
        key = state_key(row["messages"][:row["target_assistant_index"]], target)
        retained_exact_states[key].append(f"retained_candidate:{idx:04d}")
        retained_questions.add(norm(user_question(row)))
    retained_search_targets = defaultdict(list)
    for row in core_rows:
        target = action(row)
        if target.get("action") == "search":
            key = state_key(row["messages"][:row["target_assistant_index"]], target)
            matches = frozen_ids_by_row_hash.get(digest(row), [])
            retained_search_targets[key].extend(matches)

    embedded = {}
    embedded_occurrences = 0
    for row in core_rows:
        target_i = row["target_assistant_index"]
        if action(row).get("action") != "answer":
            continue
        messages = row["messages"]
        question = user_question(row)
        for i in range(2, target_i - 1):
            if (messages[i].get("role") != "assistant"
                    or messages[i + 1].get("role") != "tool"):
                continue
            try:
                proposed = json.loads(messages[i]["content"])
                observation = json.loads(messages[i + 1]["content"])
            except (TypeError, json.JSONDecodeError):
                continue
            if proposed.get("action") != "search" or observation.get("ok") is not True:
                continue
            embedded_occurrences += 1
            visible = messages[:i]
            if user_question({"messages": visible}) != question:
                raise ValueError("embedded_question_prefix_mismatch")
            key = state_key(visible, proposed)
            source_ids = frozen_ids_by_row_hash.get(digest(row), [])
            embedded.setdefault(key, {
                "state_sha256": key,
                "question_sha256": sha(norm(question).encode("utf-8")),
                "action_sha256": sha(json.dumps(proposed, ensure_ascii=False,
                                                sort_keys=True,
                                                separators=(",", ":")).encode("utf-8")),
                "answer_source_record_id": source_ids[0] if len(source_ids) == 1 else None,
                "answer_source_row_sha256": digest(row),
                "exact_frozen_search_target_record_ids": [],
                "exact_retained_search_target_record_ids": [],
                "exact_retained_candidate_state": False,
                "normalized_question_in_retained_train": norm(question) in retained_questions,
                "review_status": "no_review_for_this_embedded_prefix_unless_exact_target_overlap",
            })
            if key != state_key(visible, proposed):
                raise AssertionError("unreachable_state_key_mismatch")

    for key, item in embedded.items():
        item["exact_frozen_search_target_record_ids"] = sorted(frozen_search_targets.get(key, []))
        item["exact_retained_search_target_record_ids"] = sorted(retained_search_targets.get(key, []))
        item["exact_retained_candidate_state"] = key in retained_exact_states
        item["exact_frozen_target_statuses"] = [
            {"record_id": rid,
             "in_retained_core": digest(rows_by_id[rid]) in core_row_hashes,
             "current_v128_hold_dispositions": repair_state_by_id.get(rid, {}).get(
                 "current_dispositions", [])}
            for rid in item["exact_frozen_search_target_record_ids"]
        ]
        if item["exact_frozen_search_target_record_ids"]:
            item["review_status"] = "exact_frozen_search_target_overlap_check_source_disposition"

    extracted = read_jsonl(EXTRACTED)
    extraction_report = json.loads(EXTRACTION_REPORT.read_text(encoding="utf-8-sig"))
    extracted_rows = []
    seen_candidate_keys = set()
    for row in extracted:
        key = state_key(row["visible_messages"], row["proposed_action"])
        if key in seen_candidate_keys:
            raise ValueError("duplicate_extracted_candidate_state")
        seen_candidate_keys.add(key)
        if key not in embedded:
            raise ValueError("extracted_candidate_not_in_recounted_embedded_states")
        search_id = row["source_search_record_id_private"]
        answer_id = row["source_answer_record_id_private"]
        if search_id not in rows_by_id or answer_id not in rows_by_id:
            raise ValueError("extracted_source_id_missing")
        search_row, answer_row = rows_by_id[search_id], rows_by_id[answer_id]
        if digest(answer_row) != row["source_answer_row_sha256"]:
            raise ValueError("extracted_answer_source_hash_mismatch")
        source_action = action(search_row)
        if source_action != row["proposed_action"]:
            raise ValueError("extracted_action_not_equal_to_separately_reviewed_source_action")
        if norm(user_question(search_row)) != norm(user_question(answer_row)):
            raise ValueError("extracted_source_question_mismatch")
        exact_pair_link = exact_answer_prefix_link(row, answer_row)
        extracted_rows.append({
            "candidate_id_private": row["candidate_id_private"],
            "state_sha256": key,
            "source_search_record_id_private": search_id,
            "source_answer_record_id_private": answer_id,
            "source_answer_row_sha256": row["source_answer_row_sha256"],
            "source_search_row_is_in_retained_core": digest(search_row) in core_row_hashes,
            "source_answer_row_is_in_retained_core": digest(answer_row) in core_row_hashes,
            "exact_action_result_answer_prefix_link": exact_pair_link,
            "exact_retained_candidate_state": key in retained_exact_states,
            "exact_frozen_search_target_record_ids": sorted(frozen_search_targets.get(key, [])),
            "exact_frozen_target_statuses": [
                {"record_id": rid,
                 "in_retained_core": digest(rows_by_id[rid]) in core_row_hashes,
                 "current_v128_hold_dispositions": repair_state_by_id.get(rid, {}).get(
                     "current_dispositions", [])}
                for rid in sorted(frozen_search_targets.get(key, []))
            ],
            "normalized_question_in_retained_train": norm(user_question(answer_row)) in retained_questions,
            "source_action_and_prefix_citation_token_checks": "previously_mechanically_validated; re-bound to frozen source hash",
            "new_semantic_review": "not_run",
            "export_eligible": False,
            "training_allowed": False,
        })

    exact_frozen_overlap = sum(bool(x["exact_frozen_search_target_record_ids"])
                               for x in embedded.values())
    exact_retained_overlap = sum(bool(x["exact_retained_search_target_record_ids"])
                                 for x in embedded.values())
    held_nonretained_overlaps = [
        status for row in embedded.values()
        for status in row["exact_frozen_target_statuses"]
        if not status["in_retained_core"]
    ]
    if embedded_occurrences != 162 or len(embedded) != 162 or len(extracted_rows) != 10:
        raise ValueError(
            "embedded_search_recount_does_not_match_frozen_baseline:"
            f"occurrences={embedded_occurrences},unique={len(embedded)},"
            f"extracted={len(extracted_rows)}")
    if extraction_report.get("candidate_search_states") != len(extracted_rows):
        raise ValueError("extraction_report_candidate_count_mismatch")

    return {
        "version": "v128-embedded-search-reuse-audit-v1",
        "inputs_sha256": {
            "v117_frozen_train": sha(V117_TRAIN.read_bytes()),
            "retained_v117_core": sha(CORE.read_bytes()),
            "double_reviewed_v122_supplement": sha(SUPPLEMENT.read_bytes()),
            "v126_linked_search_candidates": sha(EXTRACTED.read_bytes()),
            "v126_extraction_report": sha(EXTRACTION_REPORT.read_bytes()),
            "g2_repair_inventory": sha(REPAIR_INVENTORY.read_bytes()),
            "v114_candidate_builder": sha(BUILDER.read_bytes()),
            "v117_candidate_preparer": sha(V117_PREPARER.read_bytes()),
            "v117_finalizer": sha(V117_FINALIZER.read_bytes()),
        },
        "counts": {
            "embedded_successful_search_tool_pairs": embedded_occurrences,
            "unique_question_prefix_action_states": len(embedded),
            "exact_overlap_frozen_search_targets": exact_frozen_overlap,
            "exact_overlap_retained_core_search_targets": exact_retained_overlap,
            "exact_overlap_frozen_search_targets_not_in_retained_core": exact_frozen_overlap - exact_retained_overlap,
            "exact_overlap_nonretained_records_marked_repair_candidate": sum(
                "repair_candidate" in r["current_v128_hold_dispositions"]
                for r in held_nonretained_overlaps),
            "different_prefix_states": len(embedded) - exact_retained_overlap,
            "exact_duplicate_occurrences_beyond_unique": embedded_occurrences - len(embedded),
            "previously_extracted_action_result_citation_candidates": len(extracted_rows),
            "extracted_candidates_exact_duplicate_current_retained_train_states": sum(
                r["exact_retained_candidate_state"] for r in extracted_rows),
            "extracted_candidates_exact_duplicate_existing_frozen_search_targets": sum(
                bool(r["exact_frozen_search_target_record_ids"]) for r in extracted_rows),
            "extracted_candidates_novel_to_frozen_search_target_states": sum(
                not r["exact_frozen_search_target_record_ids"] and
                not r["exact_retained_candidate_state"] for r in extracted_rows),
            "extracted_candidates_question_overlap_current_retained_train": sum(
                r["normalized_question_in_retained_train"] for r in extracted_rows),
            "extracted_candidates_new_semantic_reviews": 0,
            "extracted_candidates_both_source_rows_in_retained_core": sum(
                r["source_search_row_is_in_retained_core"] and
                r["source_answer_row_is_in_retained_core"] for r in extracted_rows),
            "extracted_candidates_exact_action_result_answer_prefix_links": sum(
                r["exact_action_result_answer_prefix_link"] for r in extracted_rows),
            "extracted_candidates_export_eligible": 0,
            "training_allowed": False,
        },
        "embedded_question_prefix_action_states": [embedded[k] for k in sorted(embedded)],
        "extracted_candidate_states": extracted_rows,
        "limitations": [
            "The historical builder composes query candidates from the question and already-visible text, but uses official supporting_facts as an oracle to score retrieved coverage, choose starting runs and select subsequent queries. This is oracle-assisted policy selection, not a model-generated action trace.",
            "Exact action-result-answer prefix, citation linkage and token validity are mechanical checks; they do not replace independent semantic review of the derived prefix state.",
            "A previously reviewed source action does not transfer semantic approval to a new embedded prefix state.",
            "No official supporting_facts values are used by this reuse audit; provenance is established from the named builder implementation and historical source manifest.",
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
