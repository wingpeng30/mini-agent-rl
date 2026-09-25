"""Freeze a new, performance-blind internal candidate pool from unused train questions.

This is a candidate construction step only. None of its states are eligible for
internal evaluation until both independent semantic reviews have approved them.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from scripts import prepare_v117 as prior
from scripts.build_v114_sft_candidates import build_index, norm, public_example, support_facts
from scripts.prepare_v116_data import ROOT, SOURCE, full_rows, paired_examples, sha, write, writel

OUT = ROOT / "data/evidence-sft-v122-internal-candidates"


def source_order(history_ids: set[str], history_questions: set[str]) -> list[dict]:
    provenance = json.loads((SOURCE / "download-provenance.json").read_text(encoding="utf-8"))
    for item in provenance["files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("source_shard_hash_mismatch")
    seen: set[str] = set()
    metadata: list[dict] = []
    for file in sorted(SOURCE.glob("train-*-of-*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(columns=["id", "question"], batch_size=4096):
            for item in batch.to_pylist():
                key = norm(item["question"])
                if item["id"] in history_ids or key in history_questions or key in seen:
                    continue
                seen.add(key)
                metadata.append(item)
    metadata.sort(key=lambda row: hashlib.sha256(("v122-internal:" + row["id"]).encode()).hexdigest())
    return metadata[:2000]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # The v117 history scanner includes all v117/v118/v121/v122 JSON/JSONL
    # artifacts and excludes sealed directories without reading their contents.
    prior.OUT = OUT
    ids, questions, inventory = prior.history()
    if not inventory["ready"]:
        raise RuntimeError("historical_exposure_scan_incomplete")
    metadata = source_order(ids, questions)
    rows = full_rows([row["id"] for row in metadata])
    chosen: list[dict] = []
    rejects: list[dict] = []
    for item in metadata:
        row = rows[item["id"]]
        issue = prior.wording_issue(row["question"])
        if issue:
            rejects.append({"source_id": row["id"], "reason": issue})
            continue
        if len(support_facts(row)) < 2:
            rejects.append({"source_id": row["id"], "reason": "insufficient_annotated_facts"})
            continue
        _, solo = build_index([row])
        if paired_examples(row, solo) is None:
            rejects.append({"source_id": row["id"], "reason": "no_structural_pair_in_source"})
            continue
        chosen.append(row)
        if len(chosen) == 200:
            break
    write(OUT / "preselection-rejections.json", {
        "screened": len(chosen) + len(rejects), "rejections": rejects,
        "counts": dict(Counter(r["reason"] for r in rejects)),
    })
    if len(chosen) < 200:
        write(OUT / "candidate-gap.json", {"needed": 200, "found": len(chosen), "scanned_limit": 2000})
        raise RuntimeError("insufficient_200_candidates")
    corpus, index = build_index(chosen)
    corpus_path = OUT / "candidate-corpus-private.json"
    write(corpus_path, corpus)
    write(OUT / "candidate-freeze.json", {
        "version": "v122-new-internal-candidates-v1", "ids": [r["id"] for r in chosen],
        "source_count": len(chosen), "documents": len(corpus), "corpus_sha256": sha(corpus_path),
        "retriever": "BM25", "top_k": 3, "source": "official_hotpotqa_distractor_train",
        "selection": "SHA256(v122-internal:id), first 200 wording-and-structure-screened within 2000",
        "model_scores_used": False, "double_review_pending": True,
    })
    candidates: list[dict] = []
    no_pair: list[str] = []
    for row in chosen:
        pair = paired_examples(row, index)
        if pair is None:
            no_pair.append(row["id"])
            continue
        for state in pair:
            candidates.append({
                "source_id": row["id"], "split": "internal", "source": "programmatically_constructed_official_train",
                "state_kind": state["state_kind"], "risk_tags_provisional": prior.provisional_risks(row["question"]),
                "packet": public_example(state, 0),
            })
    writel(OUT / "candidate-private.jsonl", candidates)
    write(OUT / "candidate-construction.json", {
        "candidate_questions": 200, "pairable_questions": len(candidates) // 2,
        "unpairable_ids": no_pair, "corpus_sha256": sha(corpus_path), "pre_review_only": True,
    })
    print(json.dumps({"history_ids": len(ids), "history_questions": len(questions),
                      "preselection_rejected": len(rejects), "fixed_corpus_docs": len(corpus),
                      "pairable_questions": len(candidates) // 2}), flush=True)


if __name__ == "__main__":
    main()
