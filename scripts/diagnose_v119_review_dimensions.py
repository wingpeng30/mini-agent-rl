"""Separate answer support from search necessity/query/benefit in the v118 blind reviews.

This is a development diagnostic. It never changes the frozen v118 scores or
turns a disputed/invalid reviewer judgment into a training label.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from scripts.merge_v118_semantic_reviews import index_reviews, readl, sha, validate_candidates


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def vote(a: str, b: str, bad_a: bool = False, bad_b: bool = False) -> str:
    if bad_a or bad_b:
        return "invalid_citation"
    if a == b and a in {"yes", "no", "supported", "unsupported"}:
        return a
    if a == b and a in {"uncertain", "not_evaluable"}:
        return a
    return "disputed"


def run(args):
    directory = Path(args.packet_dir)
    manifest = read(directory / "manifest.json")
    states_file, tasks_file = directory / "states-blind.jsonl", directory / "changed-tasks-blind.jsonl"
    if sha(states_file) != manifest["state_packet_sha256"] or sha(tasks_file) != manifest["changed_task_packet_sha256"]:
        raise ValueError("blind_packet_hash_mismatch")
    packets = {section: {row["audit_id"]: row for row in readl(path)}
               for section, path in (("states", states_file), ("changed_tasks", tasks_file))}
    if len(packets["states"]) != 100 or len(packets["changed_tasks"]) != 26:
        raise ValueError("review_coverage_mismatch")
    reviews = [read(Path(args.reviewer_a)), read(Path(args.reviewer_b))]
    if [r["reviewer"] for r in reviews] != ["luna_agent", "deepseek_api"] or any(
            r.get("packet_manifest_sha256") != sha(directory / "manifest.json") for r in reviews):
        raise ValueError("review_identity_or_packet_binding_mismatch")
    private = read(Path(args.private_mapping))
    states = {r["state_id"]: r for r in readl(Path(args.private_states))}
    if len(states) != 100:
        raise ValueError("private_state_coverage_mismatch")
    result_rows = []
    counters = collections.Counter()
    for section in ("states", "changed_tasks"):
        rows = [index_reviews(review, section) for review in reviews]
        judgments = [validate_candidates(row, packets[section], section, reviewer)
                     for row, reviewer in zip(rows, ("luna_agent", "deepseek_api"))]
        bad = []
        for (_, problems) in judgments:
            bad.append({(p["audit_id"], p["alias"], p["reason"]) for p in problems})
        for audit_id, packet in packets[section].items():
            mapping = private["states" if section == "states" else "tasks"][audit_id]
            if section == "states" and mapping["state_id"] not in states:
                raise ValueError("unmapped_private_state")
            target = states[mapping["state_id"]]["target"]["action"] if section == "states" else None
            for alias, candidate in packet["anonymous_outputs"].items():
                model = mapping["anonymous_to_model"][alias]
                left, right = (judgments[0][0][audit_id][alias], judgments[1][0][audit_id][alias])
                action = (candidate.get("action") or {}).get("action") if section == "states" else "answer"
                prefix = (audit_id, alias)
                bad_pre = [any((prefix[0], prefix[1], reason) in failures for reason in
                               ("quote_not_in_action_time_input", "support_without_quote")) for failures in bad]
                bad_result = [any((prefix[0], prefix[1], reason) in failures for reason in
                                  ("search_quote_not_in_retrieval_results", "search_benefit_without_quote")) for failures in bad]
                entry = {"section": section, "audit_id": audit_id, "model": model,
                         "target_action": target, "actual_action": action,
                         "invalid_pre_action_citation": any(bad_pre),
                         "invalid_search_result_citation": any(bad_result)}
                if action == "search":
                    for dimension in ("search_necessity", "query_justified", "relevant_new_evidence"):
                        problems = bad_result if dimension == "relevant_new_evidence" else bad_pre
                        entry[dimension] = vote(left[dimension], right[dimension], *problems)
                        counters[(section, target, model, action, dimension, entry[dimension])] += 1
                else:
                    entry["answer_support"] = vote(left["evidence_verdict"], right["evidence_verdict"], *bad_pre)
                    counters[(section, target, model, action, "answer_support", entry["answer_support"])] += 1
                result_rows.append(entry)
    result = {"version": "v119-review-dimensions-v1", "diagnostic_only": True,
              "source_packet_manifest_sha256": sha(directory / "manifest.json"),
              "reviewer_sha256": [sha(Path(args.reviewer_a)), sha(Path(args.reviewer_b))],
              "private_states_sha256": sha(Path(args.private_states)),
              "state_cases": 100, "changed_task_cases": 26,
              "counters": [{"section": sec, "target_action": target, "model": model,
                            "actual_action": action, "dimension": dim, "result": status, "count": count}
                           for (sec, target, model, action, dim, status), count in sorted(counters.items(),
                               key=lambda pair: tuple(str(value) for value in pair[0]))],
              "rows": result_rows,
              "independent100_opened": False, "final_test_accessed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in ("packet-dir", "reviewer-a", "reviewer-b", "private-mapping", "private-states", "output"):
        parser.add_argument("--" + arg, required=True)
    report = run(parser.parse_args())
    print(json.dumps({key: report[key] for key in ("version", "state_cases", "changed_task_cases", "diagnostic_only")},
                     ensure_ascii=False))
