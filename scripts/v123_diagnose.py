"""Recompute historical scores and retrieval facts without model inference."""
from __future__ import annotations

import hashlib
from collections import Counter

from scripts.v123_cpu_core import ROOT, OUT, read, readl, sha, write, writel


def diagnose() -> dict:
    from mini_agent_rl.tools.search import _BM25Index
    from scripts.audit_v121_arms_api import digest
    from scripts.merge_v122_audit import review_for

    report_dir = ROOT / "runs/v122-download"
    fixed = report_dir / "v122-unified"
    questions = {r["id"]: r["question"] for r in readl(report_dir / "v122-audit-questions.jsonl")}
    api = {r["case_id"]: r for r in readl(report_dir / "deepseek-audit.jsonl")}
    luna = {r["case_id"]: r for r in readl(report_dir / "luna-blind-c-audit-v2.jsonl")}
    c_packets = {r["case_id"]: r for r in readl(report_dir / "luna-blind-c-packets.jsonl")}
    baseline = {}
    v118_rows = []
    for model in ("v101", "v112", "v118"):
        source = ROOT / f"backups/v118/extracted/runs/v118/eval/{model}-tasks.json"
        data = read(source)
        rows = data["rows"]
        baseline[model] = {"questions": len(rows), "em": sum(r["normalized_exact_match"] for r in rows),
                           "f1": sum(r["token_f1"] for r in rows) / len(rows), "source_sha256": sha(source)}
        for row in rows:
            v118_rows.append({"model": model, "task_id": row["task_id"],
                              "em": row["normalized_exact_match"], "f1": row["token_f1"],
                              "termination": row["termination"], "protocol_valid": row["protocol_valid"],
                              "first_observable_fault": "unresolved_historical" if not row["normalized_exact_match"] else None})

    corpus_path = ROOT / "data/evidence-sft-v117/frozen/internal-corpus.json"
    corpus = read(corpus_path)
    index = _BM25Index(corpus)
    rows_out = []
    arms = {}
    audit_counts = Counter()
    q_regressions = Counter()
    repair_audit = Counter()
    for arm in "ABCD":
        data = read(fixed / f"fixed-{arm}.json")
        if len(data["rows"]) != 50 or len({r["task_id"] for r in data["rows"]}) != 50:
            raise ValueError("fixed_arm_incomplete")
        em = sum(r["candidate_em"] for r in data["rows"])
        f1 = sum(r["candidate_f1"] for r in data["rows"]) / 50
        searches = sum(len(r["search_queries"]) for r in data["rows"])
        if (em != data["metrics"]["correct"] or abs(f1 - data["metrics"]["mean_f1"]) > 1e-12 or
                searches != data["metrics"]["searches"]):
            raise ValueError("fixed_arm_summary_mismatch")
        arms[arm] = {"em": em, "f1": f1, "searches": searches,
                     "completed_answers": sum(r["termination"] == "answer" for r in data["rows"]),
                     "settings_sha256": data["settings_sha256"]}
        for row in data["rows"]:
            case_id = digest("v122:" + arm + ":" + questions[row["task_id"]])[:24]
            if case_id not in api:
                raise ValueError("api_case_missing")
            calls = row["calls"]
            for call_index, call in enumerate(calls):
                if call.get("role") != "repair":
                    continue
                prior = calls[call_index - 1] if call_index else {}
                repair_audit[(arm, prior.get("blocked_reason"), call.get("action"),
                              prior.get("visible_evidence") == call.get("visible_evidence"))] += 1
            blocks = [c.get("blocked_reason") for c in calls if c.get("blocked_reason")]
            if any(c.get("parse_error") or c.get("output_truncated") for c in calls):
                protocol = "invalid_or_truncated"
            else:
                protocol = "valid_raw_actions"
            retrieved = []
            for search in row["retrieved_docs"]:
                exact = [{key: d.get(key) for key in ("id", "title", "text")}
                         for d in index.search(search["query"], top_k=3)] == [
                             {key: d.get(key) for key in ("id", "title", "text")}
                             for d in search["documents"]]
                retrieved.append({"query_sha256": hashlib.sha256(search["query"].encode()).hexdigest(),
                                  "top3_exact": exact})
            if not all(r["top3_exact"] for r in retrieved):
                raise ValueError("historical_bm25_replay_mismatch")
            review_status = "single_review_only"
            support = None
            query_targets = []
            new_relation = []
            if arm == "C":
                packet = c_packets.get(case_id)
                if not packet:
                    raise ValueError("c_packet_missing")
                left, le = review_for(packet, api[case_id])
                right, re = review_for(packet, luna.get(case_id))
                if le or re:
                    review_status = "invalid_review"
                elif left["final_support"] != right["final_support"] or left["final_support"] == "uncertain":
                    review_status = "disputed_or_uncertain"
                else:
                    review_status = "double_review_agree"
                    support = left["final_support"]
                for review in (left, right):
                    if review:
                        query_targets.append([s.get("query_targets_gap") for s in review["searches"]])
                        new_relation.append([s.get("relation_visible_after") for s in review["searches"]])
            tags = []
            if protocol != "valid_raw_actions": tags.append("protocol_or_truncation")
            if "unsupported_query_anchor" in blocks: tags.append("surface_gate_blocked")
            if "repeat" in blocks: tags.append("repeat_repair")
            if arm == "C" and len(new_relation) == 2 and len(new_relation[0]) > 1 and all(
                    review[1] == "no" for review in new_relation):
                tags.append("second_retrieval_no_missing_relation")
            if row["termination"] == "search_limit_insufficient": tags.append("search_limit")
            if not row["candidate_answer"]: tags.append("no_final_answer")
            if arm == "C" and support == "no" and row["candidate_answer"]: tags.append("answer_unsupported")
            if arm == "C" and row["candidate_em"] == 0 and support == "yes": tags.append("strict_score_or_answer_conflict")
            if arm in "BD" and "surface_gate_blocked" in tags: q_regressions[arm] += 1
            fault = next(iter(tags), None)
            audit_counts[(arm, review_status)] += 1
            rows_out.append({"arm": arm, "task_id": row["task_id"], "case_id": case_id,
                             "em": row["candidate_em"], "f1": row["candidate_f1"],
                             "termination": row["termination"], "protocol": protocol,
                             "searches": len(row["search_queries"]), "bm25_replay": retrieved,
                             "review_status": review_status, "final_support": support,
                             "query_targets_gap_reviews": query_targets,
                             "relation_visible_after_reviews": new_relation,
                             "fault_tags": tags, "first_observable_fault": fault,
                             "unknown": fault is None and row["candidate_em"] == 0})
    writel(OUT / "diagnosis/v118-tasks.jsonl", v118_rows)
    writel(OUT / "diagnosis/v122-fixed.jsonl", rows_out)
    report = {"version": "v123-cpu-diagnosis-v1", "status": "passed", "v118": baseline,
              "v122": arms, "v122_audit_counts": {f"{a}:{s}": v for (a, s), v in sorted(audit_counts.items())},
              "q_surface_blocked_cases": dict(q_regressions), "bm25_replayed_searches": sum(
                  len(r["bm25_replay"]) for r in rows_out),
              "repair_calls_by_prior_block_action_and_same_visible_evidence": {
                  f"{arm}:{block}:{action}:{same}": count for (arm, block, action, same), count in repair_audit.items()},
              "historical_em_unchanged": True, "third_search_model_accuracy": "not_run",
              "third_search_observation": "No stored third-search model action; alternative second queries cannot estimate third-search behavior.",
              "next_inference_test": {"baseline": "same first two searches and visible evidence",
                                      "intervention": "allow one evidence-grounded third search only when a relation remains missing",
                                      "thresholds": {"net_em_gain_min": 3, "f1_no_decrease": True,
                                                     "correct_retention_min": .95, "protocol_min": .98,
                                                     "unresolved_support_disputes": 0}},
              "training_allowed": False}
    write(OUT / "diagnosis/report.json", report)
    return report
