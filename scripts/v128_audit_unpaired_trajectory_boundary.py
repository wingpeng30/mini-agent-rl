"""Reconcile the 35 unpaired diagnostics without exposing gold fields or creating traces."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v124-forensic"
DIAGNOSTICS = BASE / "remaining-unpaired-diagnostic-private.jsonl"
BOUNDARY = BASE / "unpaired-training-boundary-private.json"
ORACLE_AUDIT = BASE / "v126-unpaired-oracle-construction-audit-private.json"
BUILDER = ROOT / "scripts/build_v114_sft_candidates.py"
OUT = BASE / "goal-v128/g2-unpaired-trajectory-boundary-audit-private.json"


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def audit() -> dict:
    diag_raw, boundary_raw, oracle_raw, builder_raw = (
        DIAGNOSTICS.read_bytes(), BOUNDARY.read_bytes(), ORACLE_AUDIT.read_bytes(), BUILDER.read_bytes())
    # Deliberately retain only non-gold structural fields from each row.
    diag = []
    for line in diag_raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        diag.append({k: row.get(k) for k in (
            "source_id", "status", "frozen_rejection_reason", "reason",
            "generated_empty_searches", "generated_nonempty_searches", "generated_supported_answers",
            "source_list_sha256", "corpus_sha256", "training_allowed")})
    boundary = json.loads(boundary_raw.decode("utf-8-sig"))
    oracle = json.loads(oracle_raw.decode("utf-8-sig"))
    records = boundary.get("records", [])
    by_id = {r["source_id"]: r for r in records}
    if len(diag) != 35 or len(by_id) != len(records) or len(by_id) != 35:
        raise ValueError("unpaired_inventory_count_or_duplicate_mismatch")
    if {r["source_id"] for r in diag} != set(by_id):
        raise ValueError("unpaired_boundary_source_coverage_mismatch")
    if any(r["training_allowed"] is not False for r in diag) or boundary.get("training_allowed") is not False:
        raise ValueError("unpaired_training_flag_invalid")
    if boundary.get("complete_pairs_created") != 0 or boundary.get("independent_reviews_added") != 0:
        raise ValueError("unpaired_boundary_claims_new_pair_or_review")
    audit_inputs = oracle.get("inputs", {})
    expected = {
        "data\\evidence-sft-v124-forensic\\remaining-unpaired-diagnostic-private.jsonl": sha(diag_raw),
        "data\\evidence-sft-v124-forensic\\v122-train-unpaired-semantic-triage-private.jsonl": None,
        "data\\evidence-sft-v124-forensic\\v123-unpaired-semantic-triage-private.jsonl": None,
        "scripts\\build_v114_sft_candidates.py": sha(builder_raw),
    }
    for rel, expected_hash in expected.items():
        if expected_hash is None:
            path = ROOT / Path(rel.replace("\\", "/"))
            expected_hash = sha(path.read_bytes())
        if audit_inputs.get(rel) != expected_hash:
            raise ValueError("oracle_construction_audit_input_hash_mismatch")
    statuses = Counter(r["frozen_rejection_reason"] for r in diag)
    reasons = Counter(r["reason"] for r in diag)
    dispositions = Counter(r["triage_disposition"] for r in records)
    if dict(dispositions) != {"exclude_original": 5, "replay_candidate": 18, "rewrite_question": 12}:
        raise ValueError("unpaired_semantic_triage_count_mismatch")
    if dict(statuses) != {"unpairable_in_frozen_construction": 13,
                          "no_pair_in_new_shared_corpus": 21,
                          "packet_invalid:token_limit": 1}:
        raise ValueError("unpaired_structural_loss_count_mismatch")
    # Validate that the prior builder audit records oracle use; don't use its proposed actions.
    findings = oracle.get("code_path_findings", [])
    if not any("supporting-fact annotations" in x for x in findings):
        raise ValueError("oracle_builder_dependency_not_documented")
    if oracle.get("diagnostic_questions") != 35 or oracle.get("training_allowed") is not False:
        raise ValueError("oracle_audit_scope_or_training_flag_invalid")
    return {
        "version": "v128-unpaired-trajectory-boundary-audit-v1",
        "inputs_sha256": {"diagnostics": sha(diag_raw), "training_boundary": sha(boundary_raw),
                          "oracle_construction_audit": sha(oracle_raw), "builder_source": sha(builder_raw)},
        "question_count": len(diag),
        "triage_dispositions": dict(dispositions),
        "frozen_structural_rejection_reasons": dict(statuses),
        "generated_diagnostic_counts_only": {
            "nonempty_search_states": sum(r["generated_nonempty_searches"] or 0 for r in diag),
            "empty_search_states": sum(r["generated_empty_searches"] or 0 for r in diag),
            "supported_answer_states": sum(r["generated_supported_answers"] or 0 for r in diag),
        },
        "persisted_model_generated_trajectories": 0,
        "new_complete_pairs_created": 0,
        "independent_reviews_added_in_source_ledger": 0,
        "gold_or_supporting_facts_used_to_author_review_packet": False,
        "independent_semantic_pair_review_performed": False,
        "training_eligible_states": 0,
        "training_allowed": False,
        "interpretation": [
            "The 18 replay_candidate and 12 rewrite_question dispositions are old primary triage suggestions, not saved actions, retrieval observations, answers, or independent approvals.",
            "The diagnostic builder selects queries and answer states using official supporting-fact annotations; its counts are an oracle-conditioned structural upper bound, not model behavior.",
            "No Luna pair review was requested for absent trajectories: a reviewer cannot certify a model-generated search/result/answer chain that was never persisted. All 35 remain non-exportable for this repair path.",
            "For future authorized source scope, review only exact materialized tokenized contexts and their action-time evidence; if a real linked pair does not exist, record the gap rather than synthesize it from gold.",
        ],
    }


if __name__ == "__main__":
    report = audit()
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "question_count", "triage_dispositions", "frozen_structural_rejection_reasons",
        "persisted_model_generated_trajectories", "new_complete_pairs_created",
        "independent_semantic_pair_review_performed", "training_eligible_states", "training_allowed",
    )}, ensure_ascii=False))
