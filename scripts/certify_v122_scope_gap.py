"""Prove whether the frozen v122 internal pool can still meet scope coverage."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from scripts.finalize_v117 import valid_v117_vote
from scripts.finalize_v122_internal import OUT, REVIEW, digest, loadl
from scripts.prepare_v116_data import ROOT, sha, write


def run() -> dict:
    manifest = json.loads((REVIEW / "manifest.json").read_text(encoding="utf-8"))
    packets = {row["case_id"]: row for row in loadl(REVIEW / "internal-packets.jsonl")}
    source = {case: row["source_id"] for row in loadl(REVIEW / "eligible-private.jsonl") for case in row["case_ids"]}
    if len(packets) != manifest["states"] or set(source) != set(packets):
        raise ValueError("candidate_packet_or_source_coverage_invalid")
    api: dict[str, dict] = {}
    for name, spec in manifest["batches"].items():
        batch = REVIEW / name
        review = REVIEW / f"api-{name}"
        if sha(batch) != spec["sha256"] or not review.is_file():
            raise ValueError("missing_or_changed_api_batch")
        expected = {row["case_id"] for row in loadl(batch)}
        rows = loadl(review)
        if len(rows) != spec["states"] or {row.get("case_id") for row in rows} != expected:
            raise ValueError("incomplete_api_review_batch")
        for row in rows:
            case = row["case_id"]
            if case in api or row.get("input_file_sha256") != spec["sha256"] or row.get("packet_sha256") != digest(packets[case]):
                raise ValueError("api_review_binding_invalid")
            api[case] = row
    if set(api) != set(packets):
        raise ValueError("api_review_not_complete")
    api_approved: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for case, row in api.items():
        vote = row.get("decision")
        valid, _ = valid_v117_vote(packets[case], vote)
        if valid and vote["approved"]:
            api_approved[source[case]].append((case, vote))
    complete = {qid: records for qid, records in api_approved.items() if len(records) == 2}
    details = {row["case_id"]: row for row in loadl(REVIEW / "merge-details.jsonl")}
    if set(details) != set(packets):
        raise ValueError("merge_details_stale_or_incomplete")
    luna_seen = {row["case_id"] for path in REVIEW.glob("luna-internal-batch-*.jsonl") for row in loadl(path)}
    possible_unreviewed_scope = []
    actual_scope = []
    for qid, records in complete.items():
        pair_details = [details[case] for case, _ in records]
        if any(case in luna_seen and not details[case]["accepted"] for case, _ in records):
            continue
        for case, vote in records:
            if "scope" not in vote["risk_labels"]:
                continue
            detail = details[case]
            if case not in luna_seen:
                possible_unreviewed_scope.append(case)
            elif all(row["accepted"] for row in pair_details) and "scope" in detail["risk_labels"]:
                actual_scope.append(case)
    report = {"version": "v122-scope-feasibility-certificate-v1",
              "candidate_packet_sha256": sha(REVIEW / "internal-packets.jsonl"),
              "api_reviewed": len(api), "api_approved_complete_pairs": len(complete),
              "current_double_approved_scope_states": len(actual_scope),
              "potential_unreviewed_scope_states": len(possible_unreviewed_scope),
              "maximum_scope_states_under_frozen_pool": len(actual_scope) + len(possible_unreviewed_scope),
              "required_scope_states": 5,
              "scope_gate_mathematically_impossible": len(actual_scope) + len(possible_unreviewed_scope) < 5,
              "training_permitted": False,
              "independent100_accessed": False, "final_test_accessed": False}
    write(ROOT / "reports/v122-internal-scope-impossibility.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
