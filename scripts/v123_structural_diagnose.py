"""Explain v123 pair losses on the already frozen shared BM25 corpus."""
from __future__ import annotations

import json
from collections import Counter

from scripts.v123_cpu_core import OUT, read, readl, sha, write, writel
from scripts.v123_review import INTERNAL


def analyze() -> dict:
    from mini_agent_rl.tools.search import _BM25Index
    from scripts.build_v114_sft_candidates import build_examples, support_facts
    from scripts.prepare_v116_data import full_rows

    failures = readl(INTERNAL / "structural-rejections-private.jsonl")
    source = full_rows([row["id"] for row in failures])
    corpus = read(INTERNAL / "candidate-corpus-private.json")
    index = _BM25Index(corpus)
    details = []
    reasons = Counter()
    for failure in failures:
        row = source[failure["id"]]
        examples = build_examples([row], index)
        kinds = Counter(item["state_kind"] for item in examples)
        if failure["reason"] == "packet_invalid:token_limit":
            reason = "paired_but_packet_exceeds_token_limit"
        elif not kinds["nonempty_evidence_search"]:
            reason = "no_nonempty_search_state_on_shared_corpus"
        elif not kinds["supported_answer"]:
            reason = "no_supported_answer_path_on_shared_corpus"
        else:
            reason = "search_answer_prefix_not_pairable"
        reasons[reason] += 1
        details.append({"source_id": failure["id"], "failure": failure["reason"],
                        "diagnosis": reason, "state_kinds": dict(kinds),
                        "support_fact_count": len(support_facts(row))})
    report = {"version": "v123-structural-loss-analysis-v1",
              "source_rejections_sha256": sha(INTERNAL / "structural-rejections-private.jsonl"),
              "frozen_corpus_sha256": sha(INTERNAL / "candidate-corpus-private.json"),
              "questions": len(failures), "diagnoses": dict(reasons), "training_allowed": False}
    write(OUT / "diagnosis/v123-structural-loss-report.json", report)
    writel(OUT / "diagnosis/v123-structural-loss-private.jsonl", details)
    return report


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
