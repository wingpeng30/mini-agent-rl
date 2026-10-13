"""Recheck a fixed set of exposed v123 reviewer-error sentinels without changing votes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.v123_cpu_core import ROOT, readl, write
from scripts.v125_review_protocol import valid_vote

REVIEW = ROOT / "data/evidence-sft-v123/internal/review"
OUT = ROOT / "data/evidence-sft-v124-forensic"
CASE_IDS = (
    "a6cd61cef547e6293793", "1fa491912b01a051fd8c",
    "c1656960c8b6359c439a", "e529d32f46a2002c9250",
    "3a13592e3b1e37bf9a1b", "c8d256260a9adbe58f68",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    packet_path = REVIEW / "internal-packets.jsonl"
    api_path = REVIEW / "api-v123.jsonl"
    luna_path = REVIEW / "luna-v123.jsonl"
    packets = {r["case_id"]: r for r in readl(packet_path)}
    api = {r["case_id"]: r for r in readl(api_path)}
    luna = {r["case_id"]: r for r in readl(luna_path)}
    if not set(CASE_IDS) <= set(packets) or not set(CASE_IDS) <= set(api):
        raise ValueError("sentinel_case_missing")

    findings = []
    for case_id in CASE_IDS:
        packet = packets[case_id]
        api_vote = api[case_id]["decision"]
        api_valid, api_status = valid_vote(packet, api_vote)
        luna_vote = luna.get(case_id, {}).get("decision")
        luna_valid, luna_status = (valid_vote(packet, luna_vote) if luna_vote is not None
                                   else (False, "no_luna_ticket_for_sentinel"))
        action = packet["proposed_action"]["action"]
        cited_before = [c for c in api_vote.get("citations", [])
                        if c.get("stage") == "before_action"]
        quoted_doc_ids = {c.get("document_id") for c in cited_before}
        is_east_neighbor_sentinel = (
            case_id == "3a13592e3b1e37bf9a1b" and
            any("Utah is bordered by Colorado to the east" in doc["text"]
                for doc in packet["before_action"])
        )
        explicit_single_doc_denial = (
            case_id in {"a6cd61cef547e6293793", "1fa491912b01a051fd8c"} and
            "not permitted" in api_vote.get("reason", "").casefold()
        )
        api_redundant_search_approval = (
            action == "search" and api_vote.get("approved") is True and
            case_id in {"c1656960c8b6359c439a", "e529d32f46a2002c9250"}
        )
        findings.append({
            "case_id": case_id,
            "action": action,
            "api_vote": api_vote.get("approved"),
            "api_vote_mechanically_valid_under_v125": api_valid,
            "api_validation_status": api_status,
            "luna_vote": luna_vote.get("approved") if isinstance(luna_vote, dict) else None,
            "luna_vote_mechanically_valid_under_v125": luna_valid,
            "luna_validation_status": luna_status,
            "distinct_before_action_documents_cited_by_api": len(quoted_doc_ids),
            "api_reason_explicitly_demands_single_document_for_multihop": explicit_single_doc_denial,
            "before_action_directly_contains_utah_eastern_neighbor": is_east_neighbor_sentinel,
            "api_approved_redundant_search_sentinel": api_redundant_search_approval,
            "disposition": "historical_votes_unchanged_fresh_independent_reaudit_required",
            "training_allowed": False,
        })

    report = {
        "version": "v125-exposed-v123-review-sentinel-audit-v1",
        "scope": "six named exposed v123 development states only; no sealed evaluation material",
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                   for p in (packet_path, api_path, luna_path)},
        "multi_document_false_denial_candidates": sum(
            r["api_reason_explicitly_demands_single_document_for_multihop"] for r in findings),
        "direct_evidence_false_denial_candidate": sum(
            r["before_action_directly_contains_utah_eastern_neighbor"] and r["luna_vote"] is False
            for r in findings),
        "redundant_search_false_approval_candidates": sum(
            r["api_approved_redundant_search_sentinel"] for r in findings),
        "votes_changed": False,
        "training_or_strategy_tuning_use": False,
        "findings": findings,
        "training_allowed": False,
    }
    write(OUT / "v125-exposed-v123-review-sentinel-audit-private.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
