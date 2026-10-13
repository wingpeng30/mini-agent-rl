"""Read-only failure decomposition of the one frozen v123 candidate pool."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from scripts.v123_cpu_core import OUT, readl, sha, write, writel
from scripts.v123_prepare_internal import provisional_scope
from scripts.v123_review import INTERNAL, REVIEW, all_votes, packets_and_source, valid_review


def analyze() -> dict:
    freeze, packets, source = packets_and_source()
    api = all_votes("api", packets)
    luna = all_votes("luna", packets)
    old_ids, new_ids = set(freeze["old_candidate_ids"]), set(freeze["new_candidate_ids"])
    structural = readl(INTERNAL / "structural-rejections-private.jsonl")
    by_question = defaultdict(list)
    action_counts = Counter()
    cohort_counts = Counter()
    flag_counts = Counter()
    invalid_counts = Counter()
    approved_missing_fact = Counter()
    risk_counts = Counter()
    query_forms = Counter()
    details = []
    for packet in packets:
        case = packet["case_id"]
        qid = source[case]
        action = packet["proposed_action"]["action"]
        cohort = "old" if qid in old_ids else "new"
        row = api[case]
        valid, approved, labels = valid_review(packet, row)
        vote = row.get("decision") or {}
        outcome = "approved" if approved else "rejected" if valid else "invalid"
        action_counts[(action, outcome)] += 1
        cohort_counts[(cohort, outcome)] += 1
        if not valid:
            attempts = row.get("attempts") or []
            invalid_counts[attempts[-1].get("message", "invalid_old_vote") if attempts else "invalid_old_vote"] += 1
        if valid:
            for label in labels:
                risk_counts[(cohort, action, outcome, label)] += 1
            if action == "search":
                for key in ("search_needed", "query_grounded", "retrieval_relevant"):
                    flag_counts[(cohort, outcome, key, str(vote.get(key)))] += 1
                if approved:
                    value = str(vote.get("missing_fact", "")).strip().casefold()
                    if value in {"none", "n/a", "not applicable", "unknown", "missing"}:
                        approved_missing_fact[(cohort, value)] += 1
        if action == "search":
            query = packet["proposed_action"]["query"]
            titles = {doc.get("title", "") for doc in packet["before_action"]}
            texts = [doc.get("text", "") for doc in packet["before_action"]]
            if query in titles:
                form = "visible_title"
            elif query.startswith(packet["question"] + " "):
                form = "question_plus_visible_seed"
            elif any(query in body for body in texts):
                form = "visible_text_span"
            else:
                form = "other"
            query_forms[(cohort, outcome, form)] += 1
        lv, la, ll = valid_review(packet, luna.get(case))
        by_question[qid].append({"case_id": case, "action": action, "api": outcome,
                                 "api_labels": labels if valid else [],
                                 "luna": "approved" if la else "rejected" if lv else "unreviewed",
                                 "luna_labels": ll if lv else [],
                                 "search_needed": vote.get("search_needed") if valid else None,
                                 "query_grounded": vote.get("query_grounded") if valid else None,
                                 "retrieval_relevant": vote.get("retrieval_relevant") if valid else None,
                                 "missing_fact": vote.get("missing_fact") if valid else None,
                                 "reason": vote.get("reason") if valid else None})
    pair_counts = Counter()
    hint_counts = Counter()
    risk_pair_counts = Counter()
    for qid in freeze["old_candidate_ids"] + freeze["new_candidate_ids"]:
        states = by_question.get(qid, [])
        cohort = "old" if qid in old_ids else "new"
        if not states:
            pair_status = "not_pairable"
        elif len(states) != 2:
            pair_status = "incomplete_packet"
        elif all(s["api"] == "approved" for s in states):
            pair_status = "api_approved_pair"
        elif any(s["api"] == "invalid" for s in states):
            pair_status = "api_invalid_at_least_one"
        else:
            pair_status = "api_rejected_at_least_one"
        pair_counts[(cohort, pair_status)] += 1
        if states:
            hint = "scope_hint" if provisional_scope(next(p["question"] for p in packets if p["case_id"] == states[0]["case_id"])) else "no_scope_hint"
            hint_counts[(cohort, hint, pair_status)] += 1
            if pair_status == "api_approved_pair":
                for s in states:
                    for label in s["api_labels"]:
                        risk_pair_counts[(cohort, label)] += 1
        details.append({"source_id": qid, "cohort": cohort, "pair_status": pair_status,
                        "states": states})
    report = {
        "version": "v123-candidate-failure-analysis-v1",
        "source_freeze_sha256": sha(INTERNAL / "candidate-freeze.json"),
        "api_vote_sha256": sha(REVIEW / "api-v123.jsonl"),
        "luna_vote_sha256": sha(REVIEW / "luna-v123.jsonl"),
        "structural_rejections": {"total": len(structural), "by_reason": dict(Counter(x["reason"] for x in structural)),
                                  "old": sum(x["old_approved"] for x in structural),
                                  "new": sum(not x["old_approved"] for x in structural)},
        "state_outcomes_by_action": {f"{a}:{o}": n for (a, o), n in sorted(action_counts.items())},
        "state_outcomes_by_cohort": {f"{c}:{o}": n for (c, o), n in sorted(cohort_counts.items())},
        "pair_outcomes_by_cohort": {f"{c}:{s}": n for (c, s), n in sorted(pair_counts.items())},
        "scope_hint_pair_outcomes": {f"{c}:{h}:{s}": n for (c, h, s), n in sorted(hint_counts.items())},
        "api_approved_pair_risk_states": {f"{c}:{l}": n for (c, l), n in sorted(risk_pair_counts.items())},
        "state_risk_labels": {f"{c}:{a}:{o}:{l}": n for (c, a, o, l), n in sorted(risk_counts.items())},
        "search_review_flags": {f"{c}:{o}:{k}:{v}": n for (c, o, k, v), n in sorted(flag_counts.items())},
        "search_query_forms": {f"{c}:{o}:{form}": n for (c, o, form), n in sorted(query_forms.items())},
        "invalid_api_errors": dict(invalid_counts),
        "approved_search_nonspecific_missing_fact": {f"{c}:{v}": n for (c, v), n in sorted(approved_missing_fact.items())},
        "training_allowed": False,
    }
    out = OUT / "diagnosis"
    write(out / "v123-candidate-failures.json", report)
    writel(out / "v123-candidate-failures-private.jsonl", details)
    return report


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
