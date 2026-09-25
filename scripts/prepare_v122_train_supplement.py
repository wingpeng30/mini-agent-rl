"""Freeze generic bridge-relation SFT candidate pairs from unused train questions.

Selection is hash-ordered and independent of v122 development answers. This
module only constructs candidates; a two-reviewer semantic gate is required
before any pair can be added to the 345-state v117 training export.
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

OUT = ROOT / "data/evidence-sft-v122-train-candidates"


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
                if "relation" in prior.provisional_risks(item["question"]):
                    metadata.append(item)
    metadata.sort(key=lambda row: hashlib.sha256(("v122-train-relation:" + row["id"]).encode()).hexdigest())
    return metadata[:2400]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior.OUT = OUT
    ids, questions, inventory = prior.history()
    if not inventory["ready"]:
        raise RuntimeError("historical_exposure_scan_incomplete")
    internal = json.loads((ROOT / "data/evidence-sft-v122-internal-candidates/candidate-freeze.json").read_text(encoding="utf-8"))
    ids.update(internal["ids"])
    metadata = source_order(ids, questions)
    rows = full_rows([item["id"] for item in metadata])
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
        if len(chosen) == 120:
            break
    write(OUT / "preselection-rejections.json", {
        "screened": len(chosen) + len(rejects), "rejections": rejects,
        "counts": dict(Counter(r["reason"] for r in rejects)),
    })
    if len(chosen) < 120:
        write(OUT / "candidate-gap.json", {"needed": 120, "found": len(chosen), "scanned_limit": 2400})
        raise RuntimeError("insufficient_relation_candidates")
    corpus, index = build_index(chosen)
    path = OUT / "candidate-corpus-private.json"
    write(path, corpus)
    write(OUT / "candidate-freeze.json", {
        "version": "v122-targeted-train-candidates-v1", "ids": [r["id"] for r in chosen],
        "source_count": len(chosen), "documents": len(corpus), "corpus_sha256": sha(path),
        "retriever": "BM25", "top_k": 3, "source": "official_hotpotqa_distractor_train",
        "selection": "SHA256(v122-train-relation:id), first 120 structurally valid within 2400 relation-screened",
        "development_model_scores_used": False, "double_review_pending": True,
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
                "source_id": row["id"], "split": "train", "source": "programmatically_constructed_official_train",
                "state_kind": state["state_kind"], "risk_tags_provisional": prior.provisional_risks(row["question"]),
                "packet": public_example(state, 0),
            })
    writel(OUT / "candidate-private.jsonl", candidates)
    write(OUT / "candidate-construction.json", {
        "candidate_questions": 120, "pairable_questions": len(candidates) // 2,
        "unpairable_ids": no_pair, "corpus_sha256": sha(path), "pre_review_only": True,
    })
    print(json.dumps({"history_ids": len(ids), "history_questions": len(questions),
                      "preselection_rejected": len(rejects), "fixed_corpus_docs": len(corpus),
                      "pairable_questions": len(candidates) // 2}), flush=True)


if __name__ == "__main__":
    main()
