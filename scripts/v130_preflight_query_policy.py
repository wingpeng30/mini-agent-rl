"""Performance-blind replay of the v130 follow-up query rule on exposed v123 IDs."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from scripts.v123_cpu_core import OUT as V123_OUT, ROOT, read, sha
from scripts.v130_build_new_internal_pool import _followup
from mini_agent_rl.tools.search import _BM25Index

REPORT = ROOT / "data/evidence-sft-v124-forensic/goal-v128/g4-query-policy-pilot-private.json"


def run() -> dict:
    freeze = V123_OUT / "internal/candidate-freeze.json"
    corpus_path = V123_OUT / "internal/candidate-corpus-private.json"
    ids = set(read(freeze)["new_candidate_ids"][:100])
    questions = []
    files = sorted((ROOT / "data/hotpot-full-train-v114").glob("train-*.parquet"))
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(columns=["id", "question"], batch_size=4096):
            questions.extend(row["question"] for row in batch.to_pylist() if row["id"] in ids)
    if len(questions) != len(ids):
        raise ValueError("exposed_pilot_question_lookup_incomplete")
    index = _BM25Index(read(corpus_path))
    counts = Counter()
    examples = []
    for question in questions:
        first = index.search(question, top_k=3)
        follow = _followup(question, first, index) if first else None
        counts["first_search_nonempty"] += bool(first)
        counts["visible_title_followup_adds_new_document"] += bool(follow)
        counts["no_followup_adds_new_document"] += follow is None
        if follow and len(examples) < 5:
            examples.append({"question": question, "first_titles": [d["title"] for d in first],
                             "followup_query": follow[0],
                             "followup_titles": [d["title"] for d in follow[1]]})
    return {"version": "v130-exposed-query-policy-pilot-v1", "pilot_source": "v123 exposed first 100 selected IDs only",
            "questions": len(questions), "counts": dict(counts), "examples": examples,
            "input_sha256": {freeze.relative_to(ROOT).as_posix(): sha(freeze),
                             corpus_path.relative_to(ROOT).as_posix(): sha(corpus_path)},
            "answer_or_support_facts_read": False, "candidate_pool_budget_consumed": 0,
            "accuracy_claim": False, "training_allowed": False}


if __name__ == "__main__":
    result = run()
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"questions": result["questions"], "counts": result["counts"],
                      "report": REPORT.relative_to(ROOT).as_posix()}, ensure_ascii=False))
