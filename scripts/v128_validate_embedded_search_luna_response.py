"""Mechanically validate the one-shot blind Luna diagnostic; never promotes a vote."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.v128_review_protocol import valid_vote

ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / "data/evidence-sft-v124-forensic/goal-v128"
PACKET = GOAL / "g2-embedded-search-luna-packet-private.json"
MANIFEST = GOAL / "g2-embedded-search-luna-packet-manifest-private.json"
RESPONSE = GOAL / "g2-embedded-search-luna-response-private.json"
OUTPUT = GOAL / "g2-embedded-search-luna-validation-private.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_response() -> dict:
    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = json.loads(RESPONSE.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("response_must_contain_exactly_six_votes")
    cases = packet.get("cases", [])
    if [x.get("case_id") for x in rows] != [x.get("case_id") for x in cases]:
        raise ValueError("response_case_order_or_identity_mismatch")
    if sha(PACKET) != manifest.get("packet_sha256"):
        raise ValueError("packet_manifest_hash_mismatch")
    results = []
    for case, vote in zip(cases, rows):
        ok, reason = valid_vote(case, vote)
        prior_citations = [c for c in vote.get("citations", [])
                           if isinstance(c, dict) and c.get("stage") == "before_action"]
        prompt_compliant = (bool(prior_citations) if case.get("before_action") else
                            any(isinstance(c, dict) and c.get("stage") == "question"
                                for c in vote.get("citations", [])))
        results.append({"case_id": case["case_id"], "mechanically_valid": ok,
                        "validator_result": reason,
                        "approved_single_vote": vote.get("approved") if ok else None,
                        "review_prompt_prior_context_cited": prompt_compliant})
    valid = [r for r in results if r["mechanically_valid"]]
    approved = [r for r in valid if r["approved_single_vote"] is True]
    rejected = [r for r in valid if r["approved_single_vote"] is False]
    case04 = next(x for x in cases if x["case_id"] == "case-04")
    visible = next((d["text"] for d in case04["before_action"]
                    if d["id"] == "doc-02"), "")
    exact_excerpt = ("is a popular song with music and lyrics by Harold Rome, the title tune "
                     "from his 1952 show, Wish You Were Here (musical).  It was introduced in "
                     "the show and on the RCA Victor cast album by Jack Cassidy.")
    contradiction = exact_excerpt in visible
    case04_result = next(x for x in results if x["case_id"] == "case-04")
    report = {
        "version": "v128-embedded-search-luna-validation-v1",
        "packet_sha256": sha(PACKET), "manifest_sha256": sha(MANIFEST),
        "response_sha256": sha(RESPONSE), "protocol_version": packet["protocol"]["version"],
        "reviewer_count": 1, "case_count": len(rows),
        "mechanically_valid_votes": len(valid), "single_vote_approvals": len(approved),
        "single_vote_rejections": len(rejected), "double_review_approvals": 0,
        "review_prompt_prior_context_citation_compliant_votes": sum(
            bool(x["review_prompt_prior_context_cited"]) for x in results),
        "training_eligible_states": 0, "training_allowed": False,
        "case_results": results,
        "manual_discrepancy": {
            "case_id": "case-04", "reviewer_vote_preserved": True,
            "reviewer_decision": "reject", "discrepancy_found": contradiction,
            "evidence_document_id": "doc-02",
            "evidence_excerpt": exact_excerpt,
            "reviewer_rationale_conflicts_with_visible_evidence": contradiction,
            "review_prompt_citation_requirement_met": case04_result[
                "review_prompt_prior_context_cited"],
            "frozen_validator_missed_rejection_citation_requirement": not case04_result[
                "review_prompt_prior_context_cited"],
            "interpretation": "The action-before evidence explicitly grounds the query's Harold Rome/song/introduction relation; retrieved evidence adds Jack Cassidy's actor/singer/director role. This primary audit is not an independent second review and does not reverse the vote.",
            "re_review_or_retry_requested": False,
            "candidate_status": "excluded_pending_disagreement; no training use"
        },
        "limitations": [
            "Mechanical validity proves schema and exact citation binding only, not semantic correctness.",
            "The frozen validator returns early for rejections and therefore does not enforce the protocol's prior-context citation requirement for case-04.",
            "Five approvals are single-review diagnostics, not independent dual approvals.",
            "The case-04 manual conflict is an exposed primary audit finding, not a second vote."
        ]
    }
    return report


if __name__ == "__main__":
    report = validate_response()
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mechanically_valid_votes": report["mechanically_valid_votes"],
                      "single_vote_approvals": report["single_vote_approvals"],
                      "single_vote_rejections": report["single_vote_rejections"],
                      "double_review_approvals": 0, "training_allowed": False,
                      "case04_discrepancy": report["manual_discrepancy"]["discrepancy_found"]},
                     ensure_ascii=False))
