"""Recompute the frozen internal-pool impossibility certificate from saved votes."""
from __future__ import annotations

import json
from collections import Counter

from scripts.v123_cpu_core import readl, sha, write
from scripts.v123_review import INTERNAL, REVIEW, packets_and_source, status


def certify() -> dict:
    freeze, packets, _ = packets_and_source()
    report = status()
    api = readl(REVIEW / "api-v123.jsonl")
    luna = readl(REVIEW / "luna-v123.jsonl")
    journal = readl(REVIEW / "api-request-journal.jsonl")
    started = Counter(row["case_id"] for row in journal if row.get("event") == "started")
    finished = Counter(row["case_id"] for row in journal if row.get("event") == "finished")
    if len(packets) != report["candidate_states"] or report["counts"]["api_records"] != len(packets):
        raise ValueError("api_coverage_not_complete")
    if len(api) != len(started) or started != finished or any(n != 1 for n in started.values()):
        raise ValueError("api_journal_has_duplicate_or_unfinished_request")
    pair_upper = report["possible_complete_pairs_upper"]
    risk_upper = report["possible_risk_state_upper"]
    if pair_upper >= 50 and all(risk_upper[label] >= 5 for label in ("relation", "time", "scope", "measurement")):
        raise ValueError("pool_not_proven_insufficient")
    result = {
        "version": "v123-internal-insufficiency-v1",
        "conclusion": "frozen_pool_cannot_meet_original_internal_gate",
        "interpretation": "Upper bound under the saved audit votes and fixed no-retry rule; it does not prove that every semantic vote was correct.",
        "candidate_freeze_sha256": sha(INTERNAL / "candidate-freeze.json"),
        "candidate_packet_sha256": freeze["packet_sha256"],
        "token_replay_sha256": sha(INTERNAL / "token-replay.json"),
        "api_votes_sha256": sha(REVIEW / "api-v123.jsonl"),
        "api_journal_sha256": sha(REVIEW / "api-request-journal.jsonl"),
        "luna_votes_sha256": sha(REVIEW / "luna-v123.jsonl"),
        "source_questions": report["candidate_questions"],
        "candidate_states": len(packets),
        "new_api_requests": len(api),
        "new_luna_reviews": len(luna),
        "counts": report["counts"],
        "current_complete_double_approved_pairs": report["complete_double_approved_pairs"],
        "current_pair_risks": report["complete_pair_risks"],
        "required_complete_pairs": 50,
        "possible_complete_pairs_upper": pair_upper,
        "complete_pair_deficit_even_if_remaining_luna_approves": max(0, 50 - pair_upper),
        "required_risk_states_per_label": 5,
        "possible_risk_state_upper": risk_upper,
        "risk_state_deficits_even_if_remaining_luna_approves": {
            label: max(0, 5 - risk_upper[label]) for label in ("relation", "time", "scope", "measurement")},
        "unreviewed_luna_states": len(packets) - report["counts"]["luna_records"],
        "stop_reason": "pair_scope_and_measurement_upper_bounds_below_prespecified_thresholds",
        "training_allowed": False,
    }
    write(INTERNAL / "insufficiency-certificate.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(certify(), ensure_ascii=False, indent=2))
