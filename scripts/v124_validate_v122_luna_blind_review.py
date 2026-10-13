"""Validate and bind the independent Luna votes for the v122 train candidates."""
from __future__ import annotations

import json
from collections import Counter

from scripts.finalize_v117 import valid_v117_vote
from scripts.finalize_v122_train_supplement import choose_relation_pairs
from scripts.v123_cpu_core import digest, readl, write
from scripts.v124_forensic import OUT

BASE = OUT / "pending-luna-v122-train"
TRAIN = OUT.parents[1] / "data/evidence-sft-v122-train-candidates"
REPORT = OUT / "v122-train-luna-review-validation-private.json"


def main() -> None:
    manifest = json.loads((BASE / "manifest-private.json").read_text(encoding="utf-8"))
    bindings = readl(BASE / "blind-binding-private.jsonl")
    binding_by_anon = {r["case_id"]: r for r in bindings}
    packets = [p for name in manifest["batches"] for p in readl(BASE / name)]
    packet_by_anon = {p["case_id"]: p for p in packets}
    if (len(bindings) != 56 or len(packet_by_anon) != 56 or
            set(binding_by_anon) != set(packet_by_anon) or
            digest(packets) != manifest["packet_payload_sha256"] or
            digest(bindings) != manifest["binding_sha256"]):
        raise RuntimeError("blind_packet_binding_or_hash_mismatch")

    source_packets = {p["case_id"]: p for p in readl(
        TRAIN / "review/internal-packets.jsonl")}
    pair_audit = json.loads((OUT / "v122-train-pair-integrity-audit-private.json").read_text(
        encoding="utf-8"))
    pair_audit_by_source = {r["source_id"]: r for r in pair_audit["pairs"]}
    inv_by_id = {r["record_id"]: r for r in readl(OUT / "inventory-private.jsonl")}
    primary = {r["record_id"]: r for f in sorted(OUT.glob("primary-batch-*.jsonl"))
               for r in readl(f)}
    for path in sorted(OUT.glob("primary-amendments*-private.jsonl")):
        for c in readl(path):
            row = primary[c["record_id"]]
            primary[c["record_id"]] = {**row,
                "disposition": c["effective_disposition"]}
    for c in readl(OUT / "primary-adjudication-overrides-private.jsonl"):
        row = primary[c["record_id"]]
        if row["disposition"] != c["supersedes_effective_disposition"]:
            raise RuntimeError("primary_override_chain_mismatch")
        primary[c["record_id"]] = {**row,
            "disposition": c["effective_disposition"]}

    api = {}
    for path in sorted((TRAIN / "review").glob("api-internal-batch-*.jsonl")):
        for r in readl(path):
            if r["case_id"] in api:
                raise RuntimeError("duplicate_deepseek_vote")
            api[r["case_id"]] = r
    luna_rows = readl(BASE / "luna-decisions-private.jsonl")
    luna_by_anon = {r.get("case_id"): r for r in luna_rows}
    if len(luna_by_anon) != len(luna_rows) or set(luna_by_anon) != set(packet_by_anon):
        raise RuntimeError("luna_vote_case_coverage_mismatch")

    errors = []
    decisions = Counter()
    valid_luna = {}
    for anon_id, packet in packet_by_anon.items():
        vote = luna_by_anon[anon_id]
        valid, reason = valid_v117_vote(packet, vote)
        if not valid:
            errors.append({"case_id": anon_id, "reason": reason})
            continue
        valid_luna[anon_id] = vote
        decisions["approve" if vote["approved"] else "reject"] += 1

    source_to_blind = {}
    for anon_id, binding in binding_by_anon.items():
        source_to_blind.setdefault(binding["source_id"], []).append(anon_id)
    source_to_blind = {source: sorted(ids, key=lambda x: int(x[1:]))
                       for source, ids in source_to_blind.items()}
    pair_rows = []
    candidate_pairs = []
    deepseek_decisions = Counter()
    deepseek_technical_error_records = 0
    for path in sorted((TRAIN / "review").glob("api-internal-batch-*.jsonl")):
        for r in readl(path):
            if r.get("errors"):
                deepseek_technical_error_records += 1
            vote = r.get("decision")
            if isinstance(vote, dict):
                deepseek_decisions["approve" if vote.get("approved") is True else
                                   "reject" if vote.get("approved") is False else "invalid"] += 1
            else:
                deepseek_decisions["invalid"] += 1

    for source_id, anon_ids in source_to_blind.items():
        if len(anon_ids) != 2:
            raise RuntimeError("blind_group_state_count_mismatch")
        member_results = []
        for anon_id in anon_ids:
            binding = binding_by_anon[anon_id]
            packet = packet_by_anon[anon_id]
            original_packet = source_packets[binding["original_case_id"]]
            if binding["original_packet_sha256"] != original_packet["packet_sha256"]:
                raise RuntimeError("original_packet_binding_mismatch")
            anonymous_content = {k: v for k, v in packet.items() if k != "packet_sha256"}
            expected_content = {"case_id": anon_id, "group_id": binding["group_id"],
                "question": original_packet["question"],
                "proposed_action": original_packet["proposed_action"],
                "before_action": original_packet["before_action"],
                "retrieved_after_search": original_packet["retrieved_after_search"]}
            if (packet.get("packet_sha256") != digest(anonymous_content) or
                    anonymous_content != expected_content or
                    not pair_audit_by_source[source_id]["eligible_for_independent_review"]):
                raise RuntimeError("blind_packet_content_or_selection_binding_mismatch")
            source = inv_by_id[binding["record_id"]]
            review = primary[binding["record_id"]]
            api_row = api[binding["original_case_id"]]
            api_vote = api_row.get("decision")
            api_valid, api_reason = valid_v117_vote(original_packet, api_vote)
            if api_row.get("packet_sha256") != digest(original_packet):
                api_valid, api_reason = False, "api_packet_hash_mismatch"
            luna_vote = valid_luna.get(anon_id)
            member_results.append({"case_id": anon_id, "record_id": binding["record_id"],
                "primary_disposition": review.get("disposition"),
                "api_valid": api_valid,
                "api_approved": bool(api_valid and api_vote and api_vote.get("approved") is True),
                "api_reason": api_reason,
                "luna_valid": luna_vote is not None,
                "luna_approved": bool(luna_vote and luna_vote["approved"]),
                "api_risk_labels": sorted(set(api_vote.get("risk_labels", [])) if api_vote else set()),
                "luna_risk_labels": sorted(set(luna_vote.get("risk_labels", [])) if luna_vote else set()),
                "risk_intersection": sorted(set(api_vote.get("risk_labels", [])) &
                    set(luna_vote.get("risk_labels", []))) if api_vote and luna_vote else [],
                "training_allowed": False})
        complete = (all(r["primary_disposition"] == "retain_candidate" and
                        r["api_valid"] and r["api_approved"] and r["luna_valid"] and
                        r["luna_approved"] for r in member_results))
        pair_rows.append({"source_id": source_id, "states": member_results,
            "independent_double_approved_pair": complete, "training_allowed": False})
        if complete:
            candidate_pairs.append((source_id, [
                (None, set(r["risk_intersection"])) for r in member_results]))

    selected = choose_relation_pairs(candidate_pairs, count=25, relation_min=20)
    selected_ids = [source_id for source_id, _ in selected] if selected else []
    selected_relation_states = sum(
        1 for _, records in selected or [] for _, labels in records if "relation" in labels)
    report = {
        "version": "v124-v122-train-luna-validation-v1",
        "anonymous_packet_sha256": manifest["packet_payload_sha256"],
        "luna_output_sha256": digest(luna_rows),
        "luna_reviews_received": len(luna_rows),
        "luna_valid_votes": len(valid_luna),
        "luna_decisions": dict(decisions),
        "luna_validation_errors": errors,
        "deepseek": {"states": len(api), "valid_approves": deepseek_decisions["approve"],
            "valid_rejects": deepseek_decisions["reject"],
            "invalid": deepseek_decisions["invalid"],
            "records_with_historical_technical_errors": deepseek_technical_error_records,
            "technical_error_interpretation":
                "Attempt metadata only; final validity is recomputed from the final vote and packet binding."},
        "independent_double_approved_pairs": sum(
            r["independent_double_approved_pair"] for r in pair_rows),
        "pairs_with_both_reviewers_valid": sum(all(s["api_valid"] and s["luna_valid"]
            for s in r["states"]) for r in pair_rows),
        "selected_25_pair_gate": bool(selected),
        "selected_relation_risk_states": selected_relation_states,
        "selected_source_ids_private": selected_ids,
        "pair_rows": pair_rows,
        "reviewed_pool_gate_passed": bool(selected and selected_relation_states >= 20),
        "training_allowed": False,
    }
    write(REPORT, report)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"pair_rows", "selected_source_ids_private",
                                   "luna_validation_errors"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
