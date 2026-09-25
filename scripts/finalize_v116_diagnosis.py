"""Merge corrected, independent v115 blind reviews without forcing disagreements."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.review_v116_api import validate_citations

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v116/diagnosis-blind/rereview"


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []


def hashrow(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def run():
    packets = rows(BASE / "selected.jsonl")
    index = {p["audit_id"]: p for p in packets}
    if len(index) != len(packets):
        raise ValueError("duplicate_packet")
    luna = {}
    for path in sorted(BASE.glob("luna-batch-*.jsonl")):
        for row in rows(path):
            key = row["audit_id"]
            if key in luna:
                raise ValueError("duplicate_luna_review")
            luna[key] = row
    corrected = set()
    for row in rows(BASE / "luna-corrections.jsonl"):
        key = row.get("audit_id")
        if key not in luna or key in corrected or not row.get("supersedes_batch") or not row.get("correction_reason"):
            raise ValueError("invalid_or_duplicate_luna_correction")
        corrected.add(key)
        luna[key] = row
    bindings = {r["audit_id"]: r for r in rows(BASE / "luna-diagnosis-hash-bindings.jsonl")}
    api = {r["audit_id"]: r for r in rows(BASE / "api-reviews.jsonl")}
    if len(api) != len(rows(BASE / "api-reviews.jsonl")):
        raise ValueError("duplicate_api_review")
    retried = set()
    for row in rows(BASE / "api-retry-reviews.jsonl"):
        key = row.get("audit_id")
        if key not in api or api[key].get("decision") is not None or key in retried:
            raise ValueError("invalid_or_duplicate_api_retry")
        retried.add(key)
        if row.get("decision") is not None:
            api[key] = row
    out, counts = [], Counter()
    for key, packet in index.items():
        candidate_results = {}
        for alias in ("A", "B", "C"):
            left = luna.get(key, {}).get("candidates", {}).get(alias)
            right_record = api.get(key)
            right = (right_record.get("decision") or {}).get("candidates", {}).get(alias) if right_record else None
            statuses = []
            for reviewer, vote in (("luna", left), ("api", right)):
                valid = bool(vote and vote.get("verdict") in {"supported", "unsupported", "uncertain"}
                             and vote.get("reason"))
                if valid:
                    try:
                        validate_citations(packet, vote, alias)
                    except (ValueError, KeyError, TypeError):
                        valid = False
                if reviewer == "luna" and valid:
                    bind = bindings.get(key)
                    valid = bool(bind and bind.get("canonical_packet_sha256") == hashrow(packet))
                if reviewer == "api" and valid:
                    valid = right_record.get("packet_sha256") == hashrow(packet)
                statuses.append({"reviewer": reviewer, "valid": valid,
                                 "verdict": vote.get("verdict") if valid else None})
            resolved = (statuses[0]["verdict"] if all(s["valid"] for s in statuses)
                        and statuses[0]["verdict"] == statuses[1]["verdict"]
                        and statuses[0]["verdict"] != "uncertain" else None)
            state = "agreed" if resolved else "unresolved"
            counts[state] += 1
            cause = (None if resolved else "invalid_review" if not all(s["valid"] for s in statuses)
                     else "disagreement" if statuses[0]["verdict"] != statuses[1]["verdict"]
                     else "both_uncertain")
            if cause:
                counts[cause] += 1
            candidate_results[alias] = {"status": state, "evidence_support": resolved,
                                        "unresolved_reason": cause, "reviews": statuses}
        out.append({"audit_id": key, "kind": packet["kind"], "candidates": candidate_results})
    report = {"version": "v116-v115-blind-merge-v1", "cases": len(out), "counts": dict(counts),
              "requires_two_valid_independent_reviews": True, "cases_detail": out}
    target = ROOT / "reports/v116-v115-semantic-merge.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases_detail"}, ensure_ascii=False))


if __name__ == "__main__":
    run()
