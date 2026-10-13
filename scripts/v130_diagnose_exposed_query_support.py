"""Post-hoc exact support retrieval diagnostic on already exposed v123 IDs only."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.build_v114_sft_candidates import support_facts
from scripts.prepare_v116_data import full_rows
from scripts.v123_cpu_core import OUT as V123_OUT, ROOT, read, sha, write
from scripts.v130_build_new_internal_pool import _followup
from mini_agent_rl.tools.search import _BM25Index

REPORT = ROOT / "data/evidence-sft-v124-forensic/goal-v128/g4-query-policy-exposed-support-diagnostic-private.json"


def run() -> dict:
    freeze = V123_OUT / "internal/candidate-freeze.json"
    corpus_path = V123_OUT / "internal/candidate-corpus-private.json"
    ids = read(freeze)["new_candidate_ids"][:100]
    rows = full_rows(ids)  # Exposed development records only; never used to choose queries.
    index = _BM25Index(read(corpus_path))
    counts = Counter()
    for qid in ids:
        row = rows[qid]
        facts = support_facts(row)
        first = index.search(row["question"], top_k=3)
        follow = _followup(row["question"], first, index) if first else None

        def exact_hits(documents: list[dict]) -> set[tuple[str, str]]:
            return {(title, quote) for title, quote in facts
                    if any(doc["title"] == title and quote in doc["text"] for doc in documents)}

        first_hits = exact_hits(first)
        follow_hits = exact_hits(follow[1]) if follow else set()
        counts["questions"] += 1
        counts["first_has_any_exact_support_fact"] += bool(first_hits)
        counts["visible_title_followup_returns_new_document"] += bool(follow)
        counts["followup_adds_missing_exact_support_fact"] += bool(follow_hits - first_hits)
        counts["first_or_followup_contains_all_exact_support_facts"] += bool(set(facts) <= first_hits | follow_hits)
    inputs = [freeze, corpus_path, *sorted((ROOT / "data/hotpot-full-train-v114").glob("train-*.parquet")), Path(__file__)]
    return {"version": "v130-exposed-query-support-diagnostic-v1",
            "scope": "Post-hoc diagnostic on the already exposed v123 new-candidate first 100 IDs only.",
            "query_generation_or_selection_used_support_facts": False,
            "support_facts_used_only_after_replay_for_diagnostic": True,
            "counts": dict(counts),
            "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
            "accuracy_claim": False, "new_pool_questions_inspected": 0,
            "training_eligible_states": 0, "training_allowed": False,
            "data_ready_for_experiment": False}


if __name__ == "__main__":
    report = run()
    write(REPORT, report)
    print(json.dumps({"counts": report["counts"], "report": REPORT.relative_to(ROOT).as_posix()}, ensure_ascii=True))
