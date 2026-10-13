"""Build a bounded, non-model diagnostic pool without gold-directed search.

The output is for blind action/evidence review only. It is not a model trajectory
and never makes training or model-accuracy claims.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from scripts import v123_prepare_internal as old
from scripts.v123_cpu_core import ROOT, OUT as V123_OUT, norm, read, readl, sha, write, writel

BASE = ROOT / "data/evidence-sft-v124-forensic/goal-v128"
G4 = BASE / "g4-new-pool"
SOURCE = ROOT / "data/hotpot-full-train-v114"
CORPUS_PATH = V123_OUT / "internal/candidate-corpus-private.json"
SCREENED = G4 / "screened-source-private.jsonl"
PREFLIGHT = G4 / "screening-preflight-private.json"
FROZEN_DIR = G4 / "frozen-v130"
STAGING_DIR = G4 / ".frozen-v130-staging"
FREEZE = FROZEN_DIR / "candidate-freeze-private.json"
PACKETS = FROZEN_DIR / "review/internal-packets-private.jsonl"
SOURCE_MAP = FROZEN_DIR / "review/source-map-private.jsonl"
REUSE_MAP = FROZEN_DIR / "review/old-vote-reuse-private.jsonl"
MAX_SCAN = 5000
MAX_CANDIDATES = 100
SEED = "v130-one-new-internal-pool-v1:"
# The fixed linked-title rule is now backed by an exposed-only diagnostic and
# source-recorded below. It creates candidate actions, not model trajectories.
G4_METHOD_READY = True


def _require_method_ready() -> None:
    if not G4_METHOD_READY:
        raise RuntimeError("candidate_construction_method_not_approved; source budget remains untouched")


def _history() -> tuple[set[str], set[str]]:
    ids, texts = old.source_ids()
    # Every v123 source-screened prompt is exposed even if it was not selected.
    v123_screened_ids = {row["id"] for row in readl(V123_OUT / "internal/screened-source-private.jsonl")}
    ids.update(v123_screened_ids)
    for row in readl(V123_OUT / "internal/preselection-rejections-private.jsonl"):
        ids.add(row["id"])
    for qid in read(V123_OUT / "internal/candidate-freeze.json")["new_candidate_ids"]:
        ids.add(qid)
    # The frozen v123 source selection is counted as exposed from its authoritative map.
    for row in readl(V123_OUT / "internal/review/source-map-private.jsonl"):
        ids.add(row["source_id"])
    # Reconstruct normalized prompt exclusions for the already-exposed v123
    # screen IDs; these texts are historical, not charged to the new-pool cap.
    found = set()
    for path in _source_files():
        parquet = pq.ParquetFile(path)
        for group_index in range(parquet.metadata.num_row_groups):
            id_column = parquet.read_row_group(group_index, columns=["id"]).column("id")
            group_ids = [str(value.as_py()) for value in id_column]
            positions = [i for i, qid in enumerate(group_ids) if qid in v123_screened_ids]
            if not positions:
                continue
            question_column = parquet.read_row_group(group_index, columns=["question"]).column("question")
            for position in positions:
                qid = group_ids[position]
                texts.add(norm(question_column[position].as_py()))
                found.add(qid)
    if found != v123_screened_ids:
        raise ValueError("prior_v123_exposed_question_lookup_incomplete")
    return ids, texts


def _source_files() -> list[Path]:
    provenance = read(SOURCE / "download-provenance.json")
    result = []
    for item in provenance["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError("official_source_provenance_hash_mismatch")
        if path.parent.resolve() != SOURCE.resolve() or not path.name.startswith("train-"):
            raise ValueError("official_source_path_out_of_scope")
        result.append(path)
    if len(result) != 2:
        raise ValueError("expected_exactly_two_official_train_shards")
    return sorted(result)


def _metadata_order(exposed_ids: set[str]) -> list[dict]:
    # Scan only source IDs first. Then inspect exactly the first 5,000 unseen IDs
    # in a fixed hash order; duplicate/rejected questions still consume the cap.
    ids = []
    for path in _source_files():
        for batch in pq.ParquetFile(path).iter_batches(columns=["id"], batch_size=4096):
            for raw in batch.column("id").to_pylist():
                qid = str(raw)
                if qid not in exposed_ids:
                    ids.append(qid)
    ids = sorted(set(ids), key=lambda qid: hashlib.sha256((SEED + qid).encode()).hexdigest())[:MAX_SCAN]
    wanted = set(ids)
    rows = {}
    for path in _source_files():
        parquet = pq.ParquetFile(path)
        for group_index in range(parquet.metadata.num_row_groups):
            id_column = parquet.read_row_group(group_index, columns=["id"]).column("id")
            group_ids = [str(value.as_py()) for value in id_column]
            positions = [i for i, qid in enumerate(group_ids) if qid in wanted]
            if not positions:
                continue
            question_column = parquet.read_row_group(group_index, columns=["question"]).column("question")
            for position in positions:
                qid = group_ids[position]
                rows[qid] = {"id": qid, "question": question_column[position].as_py()}
    if set(rows) != wanted:
        raise ValueError("screening_question_lookup_incomplete")
    return [{**rows[qid], "screen_order": i} for i, qid in enumerate(ids)]


def _followups(question: str, first: list[dict], index, corpus: list[dict],
               max_additional_searches: int = 2) -> list[tuple[str, list[dict]]]:
    """Follow only exact title mentions in action-time visible result documents."""
    from scripts.v130_diagnose_linked_title_policy import choose_next

    title_lookup = {norm(doc["title"]): doc["title"] for doc in corpus if doc.get("title")}
    visible = list(first)
    all_docs = {doc["id"]: doc for doc in first}
    used = {norm(question)}
    followups = []
    for _ in range(max_additional_searches):
        step = choose_next(question, visible, used, index, title_lookup,
                           excluded_doc_ids=set(all_docs))
        if step is None:
            break
        query, results = step
        followups.append((query, results))
        used.add(norm(query))
        for doc in results:
            all_docs.setdefault(doc["id"], doc)
        visible = results
    return followups


def _scope_hint(question: str) -> bool:
    # Scheduling hint only; reviewer intersection supplies the actual risk label.
    return old.provisional_scope(question)


def screen() -> dict:
    _require_method_ready()
    if SCREENED.exists() or PREFLIGHT.exists() or FREEZE.exists():
        raise ValueError("one_time_pool_screen_already_started_refuse_rescan")
    from scripts.prepare_v117 import wording_issue
    from mini_agent_rl.tools.search import _BM25Index

    exposed_ids, exposed_texts = _history()
    metadata = _metadata_order(exposed_ids)
    if len(metadata) != MAX_SCAN:
        raise ValueError("fewer_than_5000_unexposed_source_questions")
    corpus = read(CORPUS_PATH)
    index = _BM25Index(corpus)
    rows, seen_questions = [], set(exposed_texts)
    rejects = Counter()
    for item in metadata:
        question = item["question"]
        key = norm(question)
        record = {"id": item["id"], "question_sha256": hashlib.sha256(key.encode()).hexdigest(),
                  "screen_order": item["screen_order"]}
        issue = wording_issue(question)
        if not key or key in seen_questions:
            issue = issue or "normalized_question_overlap"
        seen_questions.add(key)
        if not issue:
            first = index.search(question, top_k=3)
            followups = _followups(question, first, index, corpus) if first else []
            if not first or not followups:
                issue = "no_visible_evidence_grounded_second_search"
            else:
                record["first_results_sha256"] = hashlib.sha256(
                    json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                record["followup_query_sha256"] = [hashlib.sha256(norm(query).encode()).hexdigest()
                                                     for query, _ in followups]
                record["followup_results_sha256"] = [hashlib.sha256(
                    json.dumps(results, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    for _, results in followups]
                record["followup_count"] = len(followups)
                record["scope_hint_for_batch_order_only"] = _scope_hint(question)
        if issue:
            record["disposition"] = "rejected"
            record["reason"] = issue
            rejects[issue] += 1
        else:
            record["disposition"] = "eligible"
        rows.append(record)
    writel(SCREENED, rows)
    candidates = sorted((row for row in rows if row["disposition"] == "eligible"),
                        key=lambda row: hashlib.sha256((SEED + "candidate:" + row["id"]).encode()).hexdigest())
    report = {"version": "v130-new-pool-screen-v1", "status": "screened",
              "source": "provenance-verified HotpotQA distractor/train v114",
              "source_files_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in _source_files()},
              "historical_id_count_excluded_before_question_inspection": len(exposed_ids),
              "source_questions_inspected": len(rows), "budget_cap": MAX_SCAN,
              "candidate_cap": MAX_CANDIDATES, "eligible_question_count": len(candidates),
              "candidates_available": min(MAX_CANDIDATES, len(candidates)),
              "rejections": dict(rejects), "seed": SEED,
              "screened_manifest_sha256": sha(SCREENED), "corpus_sha256": sha(CORPUS_PATH),
              "selection_used_answer_or_support_facts": False,
              "query_policy": "question first; then exact corpus title mention from latest visible top3, absent from question/prior queries/all previous docs; query only if top3 adds an unseen document; up to three total searches",
              "training_allowed": False, "data_ready_for_experiment": False}
    write(PREFLIGHT, report)
    return report


def _construct(row: dict, first: list[dict], followups: list[tuple[str, list[dict]]],
               answer: str) -> tuple[dict, dict]:
    from mini_agent_rl.prompts import DIRECT_SYSTEM
    import json as json_module

    messages = [{"role": "system", "content": DIRECT_SYSTEM}, {"role": "user", "content": row["question"]}]
    messages.extend([
        {"role": "assistant", "content": json_module.dumps({"action": "search", "query": row["question"]}, ensure_ascii=False)},
        {"role": "tool", "content": json_module.dumps({"ok": True, "result": first}, ensure_ascii=False)},
    ])
    if not followups:
        raise ValueError("nonempty_visible_followup_required")
    follow_query, follow_results = followups[0]
    search_example = {"qid": row["id"], "question": row["question"], "messages": list(messages),
                      "action": {"action": "search", "query": follow_query}, "evidence": first,
                      "retrieval_top3": follow_results, "risk_tags": [],
                      "state_kind": "diagnostic_nonempty_evidence_search"}
    answer_messages = list(messages)
    answer_evidence = {doc["id"]: doc for doc in first}
    for query, results in followups:
        action = {"action": "search", "query": query}
        answer_messages.extend([
            {"role": "assistant", "content": json_module.dumps(action, ensure_ascii=False)},
            {"role": "tool", "content": json_module.dumps({"ok": True, "result": results}, ensure_ascii=False)},
        ])
        answer_evidence.update({doc["id"]: doc for doc in results})
    answer_example = {"qid": row["id"], "question": row["question"], "messages": answer_messages,
                      "action": {"action": "answer", "answer": answer},
                      "evidence": list(answer_evidence.values()),
                      "retrieval_top3": [], "risk_tags": [], "state_kind": "diagnostic_answer_proposal"}
    return search_example, answer_example


def freeze() -> dict:
    if FROZEN_DIR.exists() or STAGING_DIR.exists():
        raise ValueError("one_time_pool_freeze_already_exists_refuse_rebuild")
    if not PREFLIGHT.is_file() or not SCREENED.is_file():
        raise ValueError("screening_preflight_required")
    report = read(PREFLIGHT)
    if report.get("source_questions_inspected") != MAX_SCAN:
        raise ValueError("screening_budget_not_fully_accounted")
    eligible = sorted((r for r in readl(SCREENED) if r["disposition"] == "eligible"),
                      key=lambda row: hashlib.sha256((SEED + "candidate:" + row["id"]).encode()).hexdigest())
    scope_hints = [r for r in eligible if r.get("scope_hint_for_batch_order_only")]
    other = [r for r in eligible if not r.get("scope_hint_for_batch_order_only")]
    selected = scope_hints[:60] + other[:40]
    if len(selected) < MAX_CANDIDATES:
        selected_ids = {r["id"] for r in selected}
        selected.extend(r for r in scope_hints[60:] + other[40:] if r["id"] not in selected_ids)
        selected = selected[:MAX_CANDIDATES]
    if len(selected) < MAX_CANDIDATES:
        raise ValueError("fewer_than_100_candidate_questions_after_fixed_screen")

    from mini_agent_rl.tools.search import _BM25Index
    from scripts.prepare_v116_data import full_rows
    from scripts.prepare_v116_review import PROCESSOR, packet
    from scripts.build_v114_sft_candidates import public_example
    from scripts.v129_review_protocol import contract_identity
    from transformers import AutoProcessor

    source_rows = full_rows([row["id"] for row in selected])
    corpus = read(CORPUS_PATH)
    index = _BM25Index(corpus)
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    packet_rows, source_map = [], []
    selected_ids = [candidate["id"] for candidate in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected_source_ids_not_unique")
    history_ids, history_texts = _history()
    if set(selected_ids) & history_ids:
        raise ValueError("selected_source_id_history_overlap")
    question_keys = [norm(source_rows[qid]["question"]) for qid in selected_ids]
    if len(question_keys) != len(set(question_keys)) or set(question_keys) & history_texts:
        raise ValueError("selected_source_question_history_overlap")
    for candidate in selected:
        row = source_rows[candidate["id"]]
        if hashlib.sha256(norm(row["question"]).encode()).hexdigest() != candidate["question_sha256"]:
            raise ValueError("screened_question_hash_mismatch")
        first = index.search(row["question"], top_k=3)
        followups = _followups(row["question"], first, index, corpus)
        if not followups:
            raise ValueError("frozen_candidate_replay_changed")
        if ([hashlib.sha256(norm(query).encode()).hexdigest() for query, _ in followups]
                != candidate.get("followup_query_sha256")):
            raise ValueError("frozen_followup_query_hash_changed")
        first_hash = hashlib.sha256(json.dumps(first, ensure_ascii=False, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
        follow_hashes = [hashlib.sha256(json.dumps(results, ensure_ascii=False, sort_keys=True,
                                                    separators=(",", ":")).encode()).hexdigest()
                         for _, results in followups]
        if first_hash != candidate.get("first_results_sha256"):
            raise ValueError("frozen_first_results_hash_changed")
        if follow_hashes != candidate.get("followup_results_sha256"):
            raise ValueError("frozen_followup_results_hash_changed")
        pair = _construct(row, first, followups, str(row["answer"]))
        for example in pair:
            source = public_example(example, 0)
            packed = packet({"split": "internal", "source": "programmatic_diagnostic_scaffold_official_answer_proposal",
                             "risk_tags_provisional": [], "packet": source}, processor)
            packet_rows.append(packed)
            source_map.append({"case_id": packed["case_id"], "source_id": row["id"],
                               "question_sha256": hashlib.sha256(norm(row["question"]).encode()).hexdigest(),
                               "answer_proposal_source": "official_train_answer_visible_as_candidate_action",
                               "model_trajectory": False, "training_eligible": False})
    if len(packet_rows) != 200 or len(source_map) != 200:
        raise ValueError("freeze_must_contain_exactly_200_states")
    if len({p["case_id"] for p in packet_rows}) != 200:
        raise ValueError("freeze_case_ids_not_unique")
    grouped: dict[str, list[dict]] = {}
    for mapping, packet_row in zip(source_map, packet_rows):
        grouped.setdefault(mapping["source_id"], []).append(packet_row)
    if set(grouped) != set(selected_ids) or any(
            len(group) != 2 or {p["proposed_action"]["action"] for p in group} != {"search", "answer"}
            for group in grouped.values()):
        raise ValueError("freeze_source_pair_integrity_failed")

    G4.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir()
    staging_packets = STAGING_DIR / "review/internal-packets-private.jsonl"
    staging_source_map = STAGING_DIR / "review/source-map-private.jsonl"
    staging_reuse = STAGING_DIR / "review/old-vote-reuse-private.jsonl"
    writel(staging_packets, packet_rows)
    writel(staging_source_map, source_map)
    writel(staging_reuse, [])
    freeze_report = {"version": "v130-new-internal-freeze-v1", "status": "frozen_for_blind_review",
        "question_count": len(selected), "state_count": len(packet_rows), "strict_pairs": len(selected),
        "source_questions_inspected": report["source_questions_inspected"],
        "source_manifest_sha256": sha(SCREENED), "screening_preflight_sha256": sha(PREFLIGHT),
        "screen_execution_incident_path": "data/evidence-sft-v124-forensic/goal-v128/g4-screen-execution-incident-v130-private.md",
        "screen_execution_incident_sha256": sha(BASE / "g4-screen-execution-incident-v130-private.md"),
        "corpus_path": CORPUS_PATH.relative_to(ROOT).as_posix(), "corpus_sha256": sha(CORPUS_PATH),
        "packet_sha256": sha(staging_packets), "source_map_sha256": sha(staging_source_map),
        "old_vote_reuse_sha256": sha(staging_reuse),
        "retriever": "pinned v123 BM25 index, top3", "context_limit": 2048,
        "review_protocol": contract_identity(),
        "query_construction_uses_only_question_and_visible_results": True,
        "uses_gold_or_support_facts_for_selection_or_query": False,
        "answers_are_official_source_proposals_not_model_outputs": True,
        "not_model_accuracy_evidence": True, "training_eligible_states": 0,
        "training_allowed": False, "data_ready_for_experiment": False}
    write(STAGING_DIR / "candidate-freeze-private.json", freeze_report)
    STAGING_DIR.rename(FROZEN_DIR)
    return freeze_report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("screen", "freeze"))
    args = parser.parse_args()
    result = screen() if args.mode == "screen" else freeze()
    print(json.dumps(result, ensure_ascii=False, indent=2))
