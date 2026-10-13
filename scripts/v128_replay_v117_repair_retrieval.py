"""Correct BM25 replay for the 107 v117 repair candidates.

The first G2 replay accidentally used v117's 1,986-document internal corpus.
Training traces must be checked against the manifest-bound v114 training
corpus. This script corrects only retrieval provenance and reuses the already
verified source/action-time and tokenizer results; it does not repeat tokenizing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mini_agent_rl.tools.search import _BM25Index
from scripts.v123_cpu_core import digest
from scripts.v128_audit_repairs import BASE, GOAL, audit as audit_inventory

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
INVENTORY = BASE / "inventory-private.jsonl"
TRAIN_CORPUS = ROOT / "data/evidence-sft-v114-clean/training-corpus.jsonl"
TRAIN_CORPUS_MANIFEST = ROOT / "data/evidence-sft-v114-clean/manifest.json"
PRIOR_REPLAY = GOAL / "g2-v117-repair-source-replay-private.json"
LINEAGE = BASE / "lineage-private.jsonl"
OUTPUT = GOAL / "g2-v117-repair-retrieval-replay-v2-private.json"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def action(row: dict) -> dict:
    return json.loads(row["messages"][row["target_assistant_index"]]["content"])


def audit() -> dict:
    candidates = set(audit_inventory()["v117_repair_candidate_record_ids"])
    train_bytes = TRAIN.read_bytes()
    inventory_bytes = INVENTORY.read_bytes()
    prior_bytes = PRIOR_REPLAY.read_bytes()
    train_rows = [json.loads(line) for line in train_bytes.decode("utf-8-sig").splitlines()
                  if line.strip()]
    inventory_rows = {row["record_id"]: row for row in read_jsonl(INVENTORY)
                      if row.get("cohort") == "v117_train"}
    lineage_bytes = LINEAGE.read_bytes()
    lineage = {row["record_id"]: row for row in read_jsonl(LINEAGE)}
    prior = json.loads(prior_bytes.decode("utf-8"))
    prior_rows = {row["record_id"]: row for row in prior["rows"]}
    prior_sha = sha_bytes(prior_bytes)
    if set(prior_rows) != candidates or len(prior_rows) != 107:
        raise ValueError("prior_source_replay_candidate_set_mismatch")
    if prior["inputs_sha256"].get("v117_train") != sha_bytes(train_bytes):
        raise ValueError("prior_source_replay_train_hash_mismatch")
    if prior["inputs_sha256"].get("v117_inventory") != sha_bytes(inventory_bytes):
        raise ValueError("prior_source_replay_inventory_hash_mismatch")
    if not all(prior_rows[rid]["token_replay"]["within_limits"] for rid in candidates):
        raise ValueError("prior_token_replay_not_all_within_limits")

    corpus_bytes = TRAIN_CORPUS.read_bytes()
    corpus_hash = sha_bytes(corpus_bytes)
    manifest = read_json(TRAIN_CORPUS_MANIFEST)
    expected_hash = manifest.get("files", {}).get("training-corpus.jsonl")
    if corpus_hash != expected_hash or corpus_hash != "17bee3d1f6f29b52c07f7ea5fd674a508974290043b5b6a44e146305e11fc90c":
        raise ValueError("manifest_bound_v114_training_corpus_hash_mismatch")
    corpus = [json.loads(line) for line in corpus_bytes.decode("utf-8-sig").splitlines()
              if line.strip()]
    if len(corpus) != manifest.get("train_corpus_documents") or len(corpus) != 2171:
        raise ValueError("v114_training_corpus_document_count_mismatch")
    index = _BM25Index(corpus)

    rows_out = []
    matches = 0
    lineage_matches = 0
    for record_id in sorted(candidates):
        ordinal = int(record_id.rsplit(":", 1)[1])
        row = train_rows[ordinal - 1]
        inv = inventory_rows.get(record_id)
        prior_item = prior_rows[record_id]
        source_hash = digest(row)
        if (source_hash != prior_item["source_row_sha256"]
                or source_hash != inv.get("source_row_sha256")):
            raise ValueError("frozen_source_hash_changed_during_retrieval_replay")
        target = action(row)
        if target.get("action") != "search":
            continue
        query = target.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("empty_original_search_query")
        result = index.search(query, top_k=3)
        prior_ids = {d.get("id") for d in inv.get("before_action", [])}
        prior_matches = lineage.get(record_id, {}).get("matches", [])
        historical_result = (prior_matches[0].get("retrieved_after_search")
                             if prior_matches else None)
        projected_result = [{k: d.get(k) for k in ("id", "title", "text")}
                            for d in result]
        exact_match = historical_result == projected_result
        matches += bool(exact_match)
        lineage_matches += bool(historical_result is not None)
        rows_out.append({
            "record_id": record_id,
            "source_row_sha256": source_hash,
            "target_state_sha256": prior_item["target_state_sha256"],
            "query_sha256": sha_bytes(query.encode("utf-8")),
            "bm25_v2_training_top3_ids": [d.get("id") for d in result],
            "bm25_v2_training_top3_document_hashes": [
                digest({k: d.get(k) for k in ("id", "title", "text")}) for d in result],
            "new_document_count_vs_before_action": len(
                {d.get("id") for d in result} - prior_ids),
            "prior_document_overlap_count": len(
                {d.get("id") for d in result} & prior_ids),
            "lineage_retrieved_after_search_available": historical_result is not None,
            "historical_lineage_result_exact_match": bool(exact_match),
            "interpretation": "fixed-corpus CPU replay only; semantic query grounding is unreviewed",
            "training_allowed": False,
        })

    if len(rows_out) != prior["counts"]["actions"]["search"]:
        raise ValueError("search_action_replay_coverage_mismatch")
    tokenizer_hashes = prior["processor"].get("tokenizer_config_sha256", {})
    return {
        "version": "v128-v117-repair-retrieval-replay-v2",
        "supersedes": ["g2-v117-repair-source-replay-private.json retrieval metrics only",
                       "g2-v117-repair-retrieval-replay-private.json result-link metric only"],
        "supersedes_reason": "The first file used the v117 internal corpus (1,986 docs); the intermediate retrieval file used an unreliable same-question trace matcher. This version uses the v114 manifest-bound training corpus and each record's hash-bound lineage observation. Source bindings and tokenizer results from the first file remain separately reusable.",
        "prior_replay_sha256": prior_sha,
        "inputs_sha256": {
            "v117_train": sha_bytes(train_bytes),
            "v117_inventory": sha_bytes(inventory_bytes),
            "v114_training_corpus": corpus_hash,
            "v114_training_manifest": sha_bytes(TRAIN_CORPUS_MANIFEST.read_bytes()),
            "v117_lineage": sha_bytes(lineage_bytes),
            "prior_source_and_token_replay": prior_sha,
        },
        "reused_tokenizer_config_sha256": tokenizer_hashes,
        "retriever": {"name": "BM25-v2 top3", "corpus_documents": len(corpus),
                      "corpus_sha256": corpus_hash},
        "counts": {
            "v117_repair_candidate_states": len(candidates),
            "search_states_replayed": len(rows_out),
            "search_states_with_lineage_result": lineage_matches,
            "historical_results_exactly_matching_correct_corpus": matches,
            "new_queries_authored": 0,
            "tokenizer_replay_reused": prior["counts"]["v117_repair_candidates"],
            "token_limits_passed": prior["counts"]["all_tokens_within_limits"],
            "semantic_reviews_added": 0,
            "model_loaded": False,
            "api_called": False,
            "training_allowed": False,
        },
        "rows": rows_out,
        "limitations": [
            "Top-three retrieval coverage is mechanical and does not prove a query was grounded, necessary, or semantically relevant.",
            "Retrieved-result equality is checked against the source-ID-bound lineage record, not inferred from a similar question or query.",
            "No query was edited and no repair candidate was promoted by this replay.",
        ],
        "training_allowed": False,
        "data_ready_for_experiment": False,
    }


if __name__ == "__main__":
    result = audit()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "retriever": result["retriever"]},
                     ensure_ascii=False))
