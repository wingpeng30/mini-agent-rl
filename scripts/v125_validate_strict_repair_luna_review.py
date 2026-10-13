"""Validate the isolated Luna output for the three strict v117 repair pairs."""
from __future__ import annotations

import hashlib
import json

from scripts.v123_cpu_core import ROOT, readl, write
from scripts.v124_forensic import OUT
from scripts.v125_review_protocol import valid_vote

WORK = OUT / "pending-luna-v125-v117-repair"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = WORK / "manifest-private.json"
    packet_path = WORK / "work-luna-01.jsonl"
    rubric_path = WORK / "rubric.txt"
    binding_path = WORK / "binding-private.jsonl"
    vote_path = WORK / "luna-decisions-private.jsonl"
    validation_path = OUT / "v125-v117-repair-luna-validation-v2-private.json"
    if validation_path.exists():
        raise FileExistsError("refuse_to_overwrite_luna_repair_validation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (sha(packet_path) != manifest["packet_sha256"] or
            sha(binding_path) != manifest["binding_sha256"] or
            sha(rubric_path) != manifest["rubric_sha256"]):
        raise ValueError("review_input_hash_mismatch")
    packets = readl(packet_path)
    bindings = readl(binding_path)
    votes = readl(vote_path)
    if len(packets) != 6 or len(bindings) != 6 or len(votes) != 6:
        raise ValueError("review_count_mismatch")
    if [p["case_id"] for p in packets] != [v.get("case_id") for v in votes]:
        raise ValueError("review_order_or_case_id_mismatch")
    if len({v.get("case_id") for v in votes}) != len(votes):
        raise ValueError("duplicate_luna_case_id")

    checks = []
    for packet, binding, vote in zip(packets, bindings, votes):
        if binding["case_id"] != packet["case_id"]:
            raise ValueError("blind_binding_case_mismatch")
        if binding["review_packet_sha256"] != hashlib.sha256(
                json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
            # digest() is the canonical binding implementation used at package time.
            from scripts.v123_cpu_core import digest
            if binding["review_packet_sha256"] != digest(packet):
                raise ValueError("blind_binding_packet_mismatch")
        valid, status = valid_vote(packet, vote)
        checks.append({"case_id": packet["case_id"], "record_id_private": binding["record_id"],
                       "valid": valid, "status": status,
                       "approved": vote.get("approved") if isinstance(vote, dict) else None})
    link_report = json.loads((OUT / "v125-v117-repair-trajectory-link-audit-private.json").read_text(
        encoding="utf-8"))
    decisions = {row["case_id"]: row["approved"] if row["valid"] else None
                 for row in checks}
    record_to_case = {row["record_id"]: row["case_id"] for row in bindings}
    pair_rows = []
    for group in link_report["findings"]:
        for link in group["strict_action_result_citation_links"]:
            if (link["search_record_id"] in record_to_case and
                    link["answer_record_id"] in record_to_case):
                pair_cases = [record_to_case[link["search_record_id"]],
                              record_to_case[link["answer_record_id"]]]
                pair_votes = [decisions[c] for c in pair_cases]
                pair_rows.append({"state_case_ids_private": pair_cases,
                    "luna_valid_states": sum(v is not None for v in pair_votes),
                    "luna_approved_states": sum(v is True for v in pair_votes),
                    "luna_approved_pair": all(v is True for v in pair_votes),
                    "pair_mechanically_complete": all(v is not None for v in pair_votes),
                    "deepseek_review": "not_run",
                    "training_allowed": False})
    report = {
        "version": "v125-v117-repair-luna-validation-v2",
        "supersedes": "v125-v117-repair-luna-validation-private.json",
        "supersedes_reason": "v1 report's reviewed_states count was overwritten by a pair-local variable; raw Luna output and per-case checks were unaffected.",
        "manifest_sha256": sha(manifest_path),
        "packet_sha256": sha(packet_path),
        "rubric_sha256": sha(rubric_path),
        "luna_output_sha256": sha(vote_path),
        "reviewed_states": len(votes), "valid_states": sum(r["valid"] for r in checks),
        "invalid_states": sum(not r["valid"] for r in checks),
        "invalid_reason_counts": {
            reason: sum(r["status"] == reason for r in checks)
            for reason in sorted({r["status"] for r in checks if not r["valid"]})
        },
        "decisions": {"approve": sum(r["approved"] is True for r in checks),
                      "reject": sum(r["approved"] is False for r in checks)},
        "independent_double_review": False,
        "case_checks_private": checks,
        "pair_checks_private": pair_rows,
        "training_allowed": False,
    }
    write(validation_path, report)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"case_checks_private", "pair_checks_private"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
