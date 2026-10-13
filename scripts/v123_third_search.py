"""CPU analysis of already observed search opportunities; never generates queries."""
from __future__ import annotations

from collections import Counter

from scripts.v123_cpu_core import ROOT, OUT, read, readl, write, writel


def analyze() -> dict:
    from mini_agent_rl.tools.search import _BM25Index
    from scripts.audit_v121_arms_api import digest
    from scripts.audit_v122_api import packets as audit_packets
    from scripts.merge_v122_audit import review_for

    questions = {r["id"]: r["question"] for r in readl(ROOT / "runs/v122-download/v122-audit-questions.jsonl")}
    reports = {arm: {r["task_id"]: r for r in read(ROOT / f"runs/v122-download/v122-unified/fixed-{arm}.json")["rows"]}
               for arm in "ABCD"}
    api = {r["case_id"]: r for r in readl(ROOT / "runs/v122-download/deepseek-audit.jsonl")}
    luna = {r["case_id"]: r for r in readl(ROOT / "runs/v122-download/luna-blind-c-audit-v2.jsonl")}
    packets = {r["case_id"]: r for r in readl(ROOT / "runs/v122-download/luna-blind-c-packets.jsonl")}
    all_packets = {p["case_id"]: p for p in audit_packets(ROOT / "runs/v122-download/v122-unified",
                   {qid: {"question": text} for qid, text in questions.items()})}
    index = _BM25Index(read(ROOT / "data/evidence-sft-v117/frozen/internal-corpus.json"))
    rows = []
    counts = Counter()
    for task, c in reports["C"].items():
        case = digest("v122:C:" + questions[task])[:24]
        left, le = review_for(packets[case], api.get(case))
        right, re = review_for(packets[case], luna.get(case))
        if le or re:
            status = "review_invalid"
        elif left["final_support"] != right["final_support"] or left["final_support"] == "uncertain":
            status = "review_disputed"
        elif c["candidate_em"] == 1:
            status = "already_exact"
        elif left["final_support"] == "yes":
            status = "supported_but_strictly_wrong"
        elif len(c["search_queries"]) < 2:
            status = "fewer_than_two_searches"
        elif left["final_support"] == "no":
            status = "two_searches_unsupported_answer"
        else:
            status = "no_answer_or_unknown"
        alternatives = []
        if len(c["search_queries"]) == 2 and status in {"two_searches_unsupported_answer", "review_disputed"}:
            for arm in "BD":
                other = reports[arm][task]
                if len(other["search_queries"]) < 2 or other["search_queries"][0] != c["search_queries"][0]:
                    continue
                query = other["search_queries"][1]
                if query.casefold().strip() == c["search_queries"][1].casefold().strip():
                    continue
                alt_case = digest("v122:" + arm + ":" + questions[task])[:24]
                alt_review, alt_errors = review_for(all_packets[alt_case], api.get(alt_case))
                if alt_errors or len(alt_review["searches"]) < 2:
                    continue
                second = alt_review["searches"][1]
                if second.get("query_grounded") != "yes" or second.get("query_targets_gap") != "yes":
                    continue
                # The original arm trace already contains the alternative retrieval.
                # Use only that recorded result; do not invent a teacher query.
                recorded = other["retrieved_docs"][1]["documents"]
                fresh = index.search(query, top_k=3)
                exact = [{k: d[k] for k in ("id", "title", "text")} for d in fresh] == [
                    {k: d[k] for k in ("id", "title", "text")} for d in recorded]
                if not exact:
                    raise ValueError("alternative_historical_top3_mismatch")
                alternatives.append({"arm": arm, "query": query, "top3_ids": [d["id"] for d in fresh],
                                     "same_pre_second_search": True, "bm25_replay_exact": exact,
                                     "single_api_query_grounded": second["query_grounded"],
                                     "single_api_query_targets_gap": second["query_targets_gap"],
                                     "single_api_relation_visible_after": second.get("relation_visible_after"),
                                     "independent_semantic_grounding": "not_run"})
        rows.append({"task_id": task, "status": status, "em": c["candidate_em"],
                     "searches": len(c["search_queries"]), "existing_alternative_second_queries": alternatives,
                     "third_search_model_action": "not_run", "third_search_accuracy": "not_run"})
        counts[status] += 1
    writel(OUT / "diagnosis/third-search-opportunities-private.jsonl", rows)
    result = {"version": "v123-third-search-cpu-v1", "status": "passed", "counts": dict(counts),
              "alternative_second_queries_replayed": sum(len(r["existing_alternative_second_queries"]) for r in rows),
              "third_search_model_action": "not_run", "third_search_accuracy": "not_run",
              "training_allowed": False}
    write(OUT / "diagnosis/third-search-report.json", result)
    return result
