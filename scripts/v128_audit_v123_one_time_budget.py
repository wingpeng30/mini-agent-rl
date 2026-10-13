"""Recompute the consumed one-time v123 internal-pool budget and its gate bound."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "data/evidence-sft-v123/internal/candidate-freeze.json"
CERT = ROOT / "data/evidence-sft-v123/internal/insufficiency-certificate.json"
OUTPUT = ROOT / "data/evidence-sft-v124-forensic/goal-v128/g4-one-time-budget-audit-private.json"


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def audit() -> dict:
    freeze_raw, cert_raw = FREEZE.read_bytes(), CERT.read_bytes()
    freeze = json.loads(freeze_raw.decode("utf-8-sig"))
    cert = json.loads(cert_raw.decode("utf-8-sig"))
    old_ids, new_ids = freeze["old_candidate_ids"], freeze["new_candidate_ids"]
    if sha(freeze_raw) != cert["candidate_freeze_sha256"]:
        raise ValueError("candidate_freeze_hash_mismatch")
    if len(old_ids) != 45 or len(set(old_ids)) != 45:
        raise ValueError("old_candidate_inventory_mismatch")
    if len(new_ids) != 100 or len(set(new_ids)) != 100:
        raise ValueError("one_time_new_candidate_budget_not_exactly_consumed")
    if set(old_ids) & set(new_ids):
        raise ValueError("old_and_new_candidate_id_overlap")
    if (freeze.get("new_questions") != 100 or freeze.get("screened") != 5000
            or freeze.get("all_questions") != 145 or cert.get("source_questions") != 145
            or cert.get("candidate_states") != 246):
        raise ValueError("candidate_freeze_and_certificate_count_mismatch")
    max_pairs = cert["possible_complete_pairs_upper"]
    risk_upper = cert["possible_risk_state_upper"]
    thresholds = {"complete_pairs": 50, "risk_states_per_label": 5}
    if max_pairs >= thresholds["complete_pairs"]:
        raise ValueError("saved_pair_upper_does_not_prove_frozen_pool_shortfall")
    if any(risk_upper[k] >= thresholds["risk_states_per_label"]
           for k in ("scope", "measurement")):
        raise ValueError("saved_scope_or_measurement_upper_no_longer_below_gate")
    return {
        "version": "v128-v123-one-time-budget-audit-v1",
        "inputs_sha256": {"candidate_freeze": sha(freeze_raw), "insufficiency_certificate": sha(cert_raw)},
        "candidate_pool": {
            "screened_source_questions": freeze["screened"],
            "new_questions": len(new_ids), "previously_approved_questions": len(old_ids),
            "combined_questions": len(old_ids) + len(new_ids),
            "new_ids_unique": len(set(new_ids)) == len(new_ids),
            "old_new_id_overlap": 0,
        },
        "saved_vote_upper_bounds": {
            "possible_complete_pairs": max_pairs,
            "possible_scope_states": risk_upper["scope"],
            "possible_measurement_states": risk_upper["measurement"],
            "required_complete_pairs": thresholds["complete_pairs"],
            "required_scope_states": thresholds["risk_states_per_label"],
            "required_measurement_states": thresholds["risk_states_per_label"],
        },
        "second_pool_or_reseed_authorized": False,
        "new_pool_started": False,
        "semantic_unusability_proven": False,
        "data_ready_for_experiment": False,
        "training_allowed": False,
        "conclusion": (
            "The one authorized v123 candidate pool consumed the 5,000-source/100-new-question budget. "
            "Under its saved audit votes and no-retry rule, its frozen upper bounds remain 49 pairs, "
            "2 scope states and 4 measurement states, below the fixed gate. This is a process-bound "
            "insufficiency result, not proof that the source distribution lacks valid questions. "
            "A new pool or reseed requires explicit new-scope authorization."
        ),
    }


if __name__ == "__main__":
    result = audit()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pool": result["candidate_pool"],
                      "upper_bounds": result["saved_vote_upper_bounds"],
                      "new_pool_started": False, "training_allowed": False}, ensure_ascii=False))
