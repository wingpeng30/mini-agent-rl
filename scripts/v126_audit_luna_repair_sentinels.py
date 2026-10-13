"""Record exposed primary diagnostics for the one Luna v117 repair packet.

This is not a second independent review and never changes historical votes.
"""
from __future__ import annotations

import hashlib
import json

from scripts.v123_cpu_core import ROOT, readl, write
from scripts.v124_forensic import OUT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    work = OUT / "pending-luna-v125-v117-repair"
    packet_path = work / "work-luna-01.jsonl"
    binding_path = work / "binding-private.jsonl"
    vote_path = work / "luna-decisions-private.jsonl"
    validation_path = OUT / "v125-v117-repair-luna-validation-v2-private.json"
    paths = [packet_path, binding_path, vote_path, validation_path,
             ROOT / "scripts/v126_review_protocol.py"]
    packets, bindings, votes = readl(packet_path), readl(binding_path), readl(vote_path)
    if not (len(packets) == len(bindings) == len(votes) == 6):
        raise ValueError("review_packet_count_changed")
    by_case = {p["case_id"]: (p, b, v) for p, b, v in zip(packets, bindings, votes)}
    checks = json.loads(validation_path.read_text(encoding="utf-8"))["case_checks_private"]
    validity = {row["case_id"]: row for row in checks}

    p4, b4, v4 = by_case["R004"]
    answer = p4["proposed_action"].get("answer")
    answer_year_visible = any(answer in d.get("text", "") for d in p4.get("before_action", []))
    answer_cited = any(answer in c.get("quote", "") for c in v4.get("citations", []))
    r4_reason = v4.get("reason", "")

    p3, b3, v3 = by_case["R003"]
    r3_question = p3.get("question", "")
    r3_query = p3.get("proposed_action", {}).get("query", "")
    r3_retrieval = " ".join(d.get("text", "") for d in p3.get("retrieved_after_search", []))

    p1, b1, v1 = by_case["R001"]
    r1_query = p1.get("proposed_action", {}).get("query", "")
    r1_before = " ".join(d.get("text", "") for d in p1.get("before_action", []))
    r1_retrieval = " ".join(d.get("text", "") for d in p1.get("retrieved_after_search", []))

    report = {
        "version": "v126-v117-repair-luna-sentinel-audit-v1",
        "inputs": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "scope": "one exposed primary audit of six anonymized Luna decisions; not independent adjudication",
        "findings_private": [
            {
                "case_id": "R004", "record_id_private": b4["record_id"],
                "mechanically_valid": validity["R004"]["valid"],
                "luna_decision": v4["approved"],
                "answer": answer,
                "answer_value_visible_in_before_action": answer_year_visible,
                "answer_value_in_luna_citations": answer_cited,
                "luna_reason_claims_answer_absent": "none of the before_action documents states" in r4_reason,
                "diagnosis": "strong_false_denial_candidate: the rejected answer value is visible in an uncited before_action document; primary audit only, re-audit independently before any vote correction",
                "training_allowed": False,
            },
            {
                "case_id": "R003", "record_id_private": b3["record_id"],
                "mechanically_valid": validity["R003"]["valid"],
                "luna_decision": v3["approved"],
                "before_action_document_count": len(p3.get("before_action", [])),
                "question_contains_query_entity": r3_query.casefold() in r3_question.casefold(),
                "retrieval_contains_program_list_marker": "syndicated programming" in r3_retrieval.casefold(),
                "retrieval_contains_answer_candidate": "judge judy" in r3_retrieval.casefold(),
                "diagnosis": "protocol-limited review: v125 required a prior-document citation although none existed; v126 permits an exact question citation as search-gap context, but this vote is not retroactively repaired or approved",
                "training_allowed": False,
            },
            {
                "case_id": "R001", "record_id_private": b1["record_id"],
                "mechanically_valid": validity["R001"]["valid"],
                "luna_decision": v1["approved"],
                "query_is_verbatim_before_action_substring": r1_query in r1_before,
                "retrieval_contains_missing_city": "Matewan, West Virginia" in r1_retrieval,
                "diagnosis": "query is grounded and retrieval supplies the gap, but it largely repeats the clue; specificity/efficiency remains a reviewer judgment, not proven by mechanical validation",
                "training_allowed": False,
            },
        ],
        "mechanical_luna_complete_pairs": 1,
        "independent_double_approved_pairs": 0,
        "historical_votes_changed": False,
        "training_allowed": False,
    }
    out = OUT / "v126-v117-repair-luna-sentinel-audit-private.json"
    if out.exists():
        raise FileExistsError(out)
    write(out, report)
    print(json.dumps({
        "version": report["version"],
        "r004_answer_visible": answer_year_visible,
        "r004_luna_vote_mechanically_valid": validity["R004"]["valid"],
        "r003_has_prior_docs": bool(p3.get("before_action")),
        "r003_question_covers_query_entity": r3_query.casefold() in r3_question.casefold(),
        "r003_retrieval_contains_list_and_answer": "syndicated programming" in r3_retrieval.casefold() and "judge judy" in r3_retrieval.casefold(),
        "double_approved_pairs": 0,
        "training_allowed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
