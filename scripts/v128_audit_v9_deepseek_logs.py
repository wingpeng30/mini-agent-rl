"""Audit the named v9 repair-review journals without calling any review service."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.v124_deepseek_review_v9_batch import validate as validate_v9

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v124-forensic/pending-dual-review-v9"
GOAL = ROOT / "data/evidence-sft-v124-forensic/goal-v128"
OUTPUT = GOAL / "g3-v9-request-audit-private.json"
BATCHES = ("v9-batch-01", "v9-batch-02", "v9-batch-03")


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def json_file(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), raw


def jsonl_file(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def audit() -> dict:
    index, index_bytes = json_file(BASE / "index-private.json")
    if index.get("total_groups") != 33 or index.get("total_states") != 55:
        raise ValueError("v9_index_count_mismatch")
    g2_inventory_path = GOAL / "g2-repair-inventory-private.json"
    g2_inventory, g2_inventory_bytes = json_file(g2_inventory_path)
    repair_ids = set(g2_inventory["v117_repair_candidate_record_ids"])
    coverage_path = GOAL / "g2-revised-state-coverage-private.json"
    coverage, coverage_bytes = json_file(coverage_path)
    other_revised = {row["record_id"]: row for row in coverage["other_revised_states"]}
    replay_path = ROOT / "data/evidence-sft-v124-forensic/repair-draft-states-cpu-v2-private.jsonl"
    if index.get("source_replay_sha256") != sha(replay_path.read_bytes()):
        raise ValueError("v9_source_replay_not_bound_to_named_g2_materialization")
    batch_summaries, all_source_ids, all_packet_hashes = [], [], []
    decisions_by_source: dict[str, str] = {}
    total_decisions = Counter()
    for batch in BATCHES:
        packet_path = BASE / f"{batch}-packet-private.json"
        manifest_path = BASE / f"{batch}-manifest.json"
        binding_path = BASE / f"{batch}-binding-private.json"
        journal_path = BASE / "deepseek-journals" / f"{batch}-journal-private.jsonl"
        raw_path = BASE / "deepseek-responses" / batch / "attempt-01-raw.json"
        validated_path = BASE / "deepseek-responses" / batch / "validated-review-private.json"

        packet_with_hash, packet_bytes = json_file(packet_path)
        supplied_hash = packet_with_hash.pop("packet_sha256")
        canonical = json.dumps(packet_with_hash, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode("utf-8")
        payload_hash = sha(canonical)
        manifest, manifest_bytes = json_file(manifest_path)
        if (payload_hash != supplied_hash or payload_hash != manifest.get("packet_payload_sha256")
                or sha(packet_bytes) != manifest.get("packet_file_sha256")
                or manifest.get("batch_id") != batch):
            raise ValueError(f"{batch}:packet_manifest_binding_mismatch")
        packet_states = [s for group in packet_with_hash["groups"] for s in group["states"]]
        if len(packet_states) != manifest.get("states"):
            raise ValueError(f"{batch}:manifest_state_count_mismatch")

        binding, binding_bytes = json_file(binding_path)
        if binding.get("packet_sha256") != payload_hash:
            raise ValueError(f"{batch}:binding_packet_mismatch")
        sources = [r["source_record_id"] for r in binding["source_mapping"]]
        anon_ids = {r["state_id"] for r in binding["source_mapping"]}
        if len(sources) != len(set(sources)) or anon_ids != {s["state_id"] for s in packet_states}:
            raise ValueError(f"{batch}:binding_source_or_state_mismatch")
        all_source_ids.extend(sources)
        all_packet_hashes.append(payload_hash)

        journal = jsonl_file(journal_path)
        expected_statuses = ["batch_started", "sending", "response_received", "valid_vote_saved"]
        if [r.get("status") for r in journal] != expected_statuses:
            raise ValueError(f"{batch}:unexpected_or_incomplete_journal_sequence")
        request_row, response_row, saved_row = journal[1], journal[2], journal[3]
        raw, raw_bytes = json_file(raw_path)
        validated, validated_bytes = json_file(validated_path)
        if (request_row.get("packet_sha256") != payload_hash
                or response_row.get("packet_sha256") != payload_hash
                or saved_row.get("packet_sha256") != payload_hash
                or response_row.get("response_sha256") != sha(raw_bytes)
                or saved_row.get("validated_review_sha256") != sha(validated_bytes)
                or validated.get("packet_sha256") != payload_hash
                or validated.get("model") != "deepseek-flash"):
            raise ValueError(f"{batch}:journal_response_vote_hash_mismatch")
        try:
            validate_v9(packet_with_hash, validated)
            replay_validation = {"passed": True, "error": None}
        except (KeyError, TypeError, ValueError) as exc:
            replay_validation = {"passed": False,
                                 "error": type(exc).__name__ + ":" + str(exc)}
        decisions = Counter(r["decision"] for r in validated["reviews"])
        total_decisions.update(decisions)
        vote_by_anon = {r["state_id"]: r["decision"] for r in validated["reviews"]}
        decisions_by_source.update({source_row["source_record_id"]: vote_by_anon[source_row["state_id"]]
                                    for source_row in binding["source_mapping"]})
        batch_summaries.append({
            "batch_id": batch, "packet_payload_sha256": payload_hash,
            "packet_file_sha256": sha(packet_bytes), "manifest_sha256": sha(manifest_bytes),
            "binding_sha256": sha(binding_bytes), "journal_sha256": sha(journal_path.read_bytes()),
            "request_sha256": request_row.get("request_sha256"),
            "raw_response_sha256": sha(raw_bytes), "validated_review_sha256": sha(validated_bytes),
            "raw_response_reported_model": raw.get("model"),
            "validated_state_count": len(validated["reviews"]),
            "decisions": dict(sorted(decisions.items())),
            "manifest_review_attempted_flag_stale": manifest.get("review_attempted") is False,
            "journal_confirms_response_and_vote_saved": True,
            "current_source_validator_replay": replay_validation,
            "validator_source_hash_recorded_at_send": False,
        })

    if len(set(all_packet_hashes)) != len(BATCHES):
        raise ValueError("duplicate_v9_packet_hash_across_batches")
    if len(all_source_ids) != len(set(all_source_ids)):
        raise ValueError("duplicate_source_record_across_v9_batches")
    if sum(x["validated_state_count"] for x in batch_summaries) != index["total_states"]:
        raise ValueError("v9_index_journal_review_state_count_mismatch")

    dispute_path = ROOT / "data/evidence-sft-v124-forensic/v9-batch02-primary-disagreement-private.json"
    disputes, disputes_bytes = json_file(dispute_path)
    batch02_path = BASE / "v9-batch-02-packet-private.json"
    batch02_packet, _ = json_file(batch02_path)
    batch02_hash = batch02_packet.pop("packet_sha256")
    batch02_binding, _ = json_file(BASE / "v9-batch-02-binding-private.json")
    batch02_review, batch02_review_bytes = json_file(
        BASE / "deepseek-responses/v9-batch-02/validated-review-private.json")
    source_by_anon = {row["state_id"]: row for row in batch02_binding["source_mapping"]}
    vote_by_anon = {row["state_id"]: row for row in batch02_review["reviews"]}
    findings = disputes.get("findings", [])
    for finding in findings:
        anon = finding["state_id_in_blind_packet"]
        source = source_by_anon.get(anon)
        vote = vote_by_anon.get(anon)
        if (source is None or vote is None
                or source["source_record_id"] != finding["source_record_id"]
                or source["original_row_sha256"] != finding["source_row_sha256"]
                or vote["decision"] != finding["deepseek_decision"]
                or not finding.get("training_use", "").startswith("hold")):
            raise ValueError("v9_batch02_disagreement_not_bound_to_source_and_vote")
    if (disputes.get("packet_sha256") != batch02_hash
            or disputes.get("initial_review_sha256") != sha(batch02_review_bytes)
            or disputes.get("state_count") != len(batch02_review["reviews"])):
        raise ValueError("v9_batch02_disagreement_report_binding_mismatch")
    inventory_by_id = {row["record_id"]: row for row in g2_inventory["states"]}
    dispute_dispositions = Counter()
    for finding in findings:
        source_id = finding["source_record_id"]
        if source_id in repair_ids:
            disposition = "v117_repair_candidate"
        else:
            inv = inventory_by_id[source_id]
            disposition = "+".join(inv.get("current_dispositions", [])) or "no_current_disposition"
        dispute_dispositions[disposition] += 1

    correction_batch = "v9-batch-01"
    correction_journal_path = BASE / "deepseek-journals" / f"{correction_batch}-citation-correction-private.jsonl"
    correction_raw_path = BASE / "deepseek-responses" / f"{correction_batch}-citation-correction" / "citation-correction-raw.json"
    correction_path = BASE / "deepseek-responses" / f"{correction_batch}-citation-correction" / "citation-correction-validated-private.json"
    original_path = BASE / "deepseek-responses" / correction_batch / "validated-review-private.json"
    packet_path = BASE / f"{correction_batch}-packet-private.json"
    packet_with_hash, _ = json_file(packet_path)
    correction_packet_hash = packet_with_hash.pop("packet_sha256")
    correction_journal = jsonl_file(correction_journal_path)
    correction_raw, correction_raw_bytes = json_file(correction_raw_path)
    correction, correction_bytes = json_file(correction_path)
    original, original_bytes = json_file(original_path)
    if ([x.get("status") for x in correction_journal] != [
            "citation_only_correction_sending", "response_received", "citation_correction_saved"]
            or correction_journal[0].get("packet_sha256") != correction_packet_hash
            or correction_journal[1].get("response_sha256") != sha(correction_raw_bytes)
            or correction_journal[2].get("correction_sha256") != sha(correction_bytes)
            or correction.get("packet_sha256") != correction_packet_hash
            or correction.get("original_review_sha256") != sha(original_bytes)
            or correction.get("correction_type") != "citations_only"
            or correction.get("decisions_and_rationales_unchanged") is not True):
        raise ValueError("v9_citation_correction_journal_or_binding_mismatch")
    schema_fields = ("decision", "issues", "rationale", "query_grounded",
                     "relevant_new_evidence", "answer_supported")
    old = {x["state_id"]: x for x in original["reviews"]}
    new = {x["state_id"]: x for x in correction["reviews"]}
    if set(old) != set(new) or any(any(old[k].get(f) != new[k].get(f) for f in schema_fields)
                                   for k in old):
        raise ValueError("v9_citation_correction_changed_semantic_fields")
    validate_v9(packet_with_hash, correction)
    batch_summaries[0]["citation_only_correction_current_validator_passed"] = True
    batch_summaries[0]["effective_citation_record_sha256"] = sha(correction_bytes)
    manifest_stale_count = sum(x["manifest_review_attempted_flag_stale"] for x in batch_summaries)
    source_class_counts = Counter()
    source_class_decisions: dict[str, Counter] = {}
    for source_id, decision in decisions_by_source.items():
        if source_id in repair_ids:
            category = "v117_repair_candidate"
        else:
            inventory_row = inventory_by_id[source_id]
            dispositions = inventory_row.get("current_dispositions", [])
            category = "+".join(dispositions) if dispositions else "no_current_disposition"
        source_class_counts[category] += 1
        source_class_decisions.setdefault(category, Counter())[decision] += 1
    return {
        "version": "v128-v9-deepseek-request-audit-v1",
        "inputs_sha256": {"index": sha(index_bytes)},
        "batch_count": len(batch_summaries),
        "reviewed_states": sum(x["validated_state_count"] for x in batch_summaries),
        "single_reviewer_decisions": dict(sorted(total_decisions.items())),
        "source_states_intersecting_v117_repair_candidates": len(set(all_source_ids) & repair_ids),
        "source_states_outside_v117_repair_candidates": len(set(all_source_ids) - repair_ids),
        "source_states_by_g2_disposition": dict(sorted(source_class_counts.items())),
        "single_reviewer_decisions_by_g2_disposition": {
            k: dict(sorted(v.items())) for k, v in sorted(source_class_decisions.items())},
        "g2_repair_inventory_sha256": sha(g2_inventory_bytes),
        "g2_materialized_repair_source_sha256": sha(replay_path.read_bytes()),
        "batch02_primary_disagreement_report_sha256": sha(disputes_bytes),
        "batch02_primary_disagreement_states_bound_to_votes": len(findings),
        "batch02_primary_disagreement_states_still_held": sum(
            str(x.get("training_use", "")).startswith("hold") for x in findings),
        "batch02_disagreement_states_by_g2_disposition": dict(sorted(dispute_dispositions.items())),
        "g2_revised_state_coverage_sha256": sha(coverage_bytes),
        "nonrepair_v9_states_already_exactly_duplicating_current_training_input": sum(
            bool(other_revised.get(source_id, {}).get("exact_state_duplicate_in_current_training_pool"))
            for source_id in set(all_source_ids) - repair_ids),
        "nonrepair_v9_states_overlapping_current_training_question": sum(
            bool(other_revised.get(source_id, {}).get("normalized_question_overlaps_current_training_pool"))
            for source_id in set(all_source_ids) - repair_ids),
        "historically_journaled_vote_saves": sum(
            bool(x["journal_confirms_response_and_vote_saved"]) for x in batch_summaries),
        "initial_outputs_passing_current_validator": sum(
            bool(x["current_source_validator_replay"]["passed"]) for x in batch_summaries),
        "citation_corrected_batch01_passes_current_validator": True,
        "dual_review_approvals": 0,
        "training_eligible_from_this_audit": 0,
        "training_allowed": False,
        "batch_results": batch_summaries,
        "citation_correction": {
            "same_packet_as_v9_batch_01": True,
            "request_sha256": correction_journal[0].get("request_sha256"),
            "raw_response_sha256": sha(correction_raw_bytes),
            "validated_correction_sha256": sha(correction_bytes),
            "response_reported_model": correction_raw.get("model"),
            "citation_only": True,
            "decisions_and_rationales_unchanged": True,
            "reviewed_states": len(correction["reviews"]),
            "not_a_second_semantic_vote": True,
        },
        "manifest_attempt_flag_stale_count": manifest_stale_count,
        "endpoint_host_logged_in_v9_request_journal": all(
            "api_base_host" in jsonl_file(BASE / "deepseek-journals" / f"{batch}-journal-private.jsonl")[0]
            for batch in BATCHES),
        "interpretation": [
            "All three named v9 batches have hash-bound HTTP 200 responses and journaled vote-save artifacts; these are one reviewer only, not semantic dual approval.",
            "The current validator re-run on the initial batch-01 output fails approved_search_missing_retrieved_evidence_citation; the one allowed citation-only correction is hash-bound, preserves semantic fields and passes the current validator. The send-time validator hash was not journaled, so this establishes output drift, not which exact validator version ran then.",
            "The v9 batch manifests still say review_attempted=false despite journals proving requests and saved results; treat journals and saved files as authoritative and flag the manifest status as stale.",
            "The separate batch-01 citation-only correction preserves semantic decisions and is not an additional semantic vote.",
            "The 55 anonymized batch states are joined to the named G2 repair inventory; the separate batch-02 primary-disagreement report is hash-bound back to seven API votes and all seven remain held.",
            "The journal stores the requested model label but not the API base host; endpoint identity cannot be established from this evidence.",
            "This audit does not inspect response rationale semantics or clear the seven v9 batch-02 primary-disagreement holds.",
        ],
        "data_ready_for_experiment": False,
    }


if __name__ == "__main__":
    result = audit()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "batch_count", "reviewed_states", "single_reviewer_decisions",
        "historically_journaled_vote_saves", "initial_outputs_passing_current_validator",
        "citation_corrected_batch01_passes_current_validator", "dual_review_approvals", "manifest_attempt_flag_stale_count",
        "endpoint_host_logged_in_v9_request_journal", "training_allowed")}, ensure_ascii=False))
