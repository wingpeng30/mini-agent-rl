"""Post-hoc annotated-fact coverage for old exposed development traces.

Official supporting facts are read only after all historical queries exist;
they are never used to compose or select a new query or training target.
"""
from __future__ import annotations

from collections import Counter

from scripts.v123_cpu_core import ROOT, OUT, read, readl, write, writel


def analyze() -> dict:
    from scripts.build_v114_sft_candidates import facts_visible, support_facts
    from scripts.prepare_v116_data import full_rows

    fixed = read(ROOT / "runs/v122-download/v122-unified/fixed-C.json")["rows"]
    source = full_rows([r["task_id"] for r in fixed])
    corpus = read(ROOT / "data/evidence-sft-v117/frozen/internal-corpus.json")
    rows = []
    counts = Counter()
    for trace in fixed:
        qid = trace["task_id"]
        facts = support_facts(source[qid])
        retrieved = [doc for step in trace["retrieved_docs"] for doc in step["documents"]]
        first = trace["retrieved_docs"][0]["documents"] if trace["retrieved_docs"] else []
        calls = trace["calls"]
        final_visible = calls[-1].get("visible_evidence", []) if calls else []
        corpus_hits = sum(any(doc["title"] == title and quote in doc["text"] for doc in corpus)
                          for title, quote in facts)
        first_hits = sum(any(doc["title"] == title and quote in doc["text"] for doc in first)
                         for title, quote in facts)
        retrieved_hits = sum(any(doc["title"] == title and quote in doc["text"] for doc in retrieved)
                             for title, quote in facts)
        visible_hits = sum(any(doc["title"] == title and quote in doc["text"] for doc in final_visible)
                           for title, quote in facts)
        row = {"task_id": qid, "em": trace["candidate_em"], "searches": len(trace["search_queries"]),
               "annotated_facts": len(facts), "in_frozen_corpus": corpus_hits,
               "in_first_top3": first_hits, "in_all_retrieved_top3": retrieved_hits,
               "visible_after_token_processing": visible_hits,
               "all_facts_in_corpus": bool(facts) and corpus_hits == len(facts),
               "all_facts_retrieved": bool(facts) and retrieved_hits == len(facts),
               "all_facts_visible_at_last_action": bool(facts) and visible_hits == len(facts),
               "coverage_is_posthoc_annotation_proxy": True}
        rows.append(row)
        counts["questions"] += 1
        for key in ("all_facts_in_corpus", "all_facts_retrieved", "all_facts_visible_at_last_action"):
            counts[key] += bool(row[key])
        if row["all_facts_retrieved"] and not row["all_facts_visible_at_last_action"]:
            counts["retrieved_then_missing_after_token_processing"] += 1
    writel(OUT / "diagnosis/annotated-coverage-private.jsonl", rows)
    result = {"version": "v123-annotated-coverage-v1", "status": "passed", "counts": dict(counts),
              "query_construction_used_official_support_facts": False,
              "coverage_is_semantic_sufficiency": False, "training_allowed": False}
    write(OUT / "diagnosis/annotated-coverage-report.json", result)
    return result
