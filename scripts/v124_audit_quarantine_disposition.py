"""Recompute the quarantine inventory without promoting or rewriting labels."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from scripts.v123_cpu_core import digest, readl, write
from scripts.v124_forensic import OUT
from scripts.v124_verify_primary import verify as verify_primary

REPORT = OUT / "quarantine-processing-audit-private.json"

ISSUE_FAMILIES = {
    "time_or_version_anchor": ("unanchored", "without_as_of", "current_", "present_tense",
                               "recent", "date_anchor", "snapshot"),
    "ambiguous_or_nonunique_target": ("ambiguous", "non_unique", "same_name", "not_unique",
                                      "state_not_unique", "collision"),
    "false_or_unsupported_premise": ("false_premise", "unsupported", "no_", "wrong_",
                                     "not_proven", "no_named", "did_not"),
    "query_or_action_grounding": ("query_", "search_", "answer_lookahead", "answer_visible"),
    "source_or_citation_integrity": ("corrupt", "citation", "source_", "provenance",
                                     "cross_park", "evidence_splice"),
    "exposure_or_contamination": ("exposed", "leak", "contaminat"),
}


def main() -> None:
    verified = verify_primary()
    inventory = {r["record_id"]: r for r in readl(OUT / "inventory-private.jsonl")}
    original = {r["record_id"]: r for p in sorted(OUT.glob("primary-batch-*.jsonl"))
                for r in readl(p)}
    effective = dict(original)
    for p in sorted(OUT.glob("primary-amendments*-private.jsonl")):
        for change in readl(p):
            row = original[change["record_id"]]
            if digest(row) != change["superseded_primary_review_sha256"]:
                raise RuntimeError("primary_amendment_hash_mismatch")
            effective[change["record_id"]] = {**row,
                "disposition": change["effective_disposition"]}
    for change in readl(OUT / "primary-adjudication-overrides-private.jsonl"):
        row = effective[change["record_id"]]
        if row["disposition"] != change["supersedes_effective_disposition"]:
            raise RuntimeError("primary_override_chain_mismatch")
        effective[change["record_id"]] = {**row,
            "disposition": change["effective_disposition"]}

    source_audit = json.loads((OUT / "quarantine-source-review-private.json").read_text(
        encoding="utf-8"))
    confirmed = {r["record_id"]: r for r in source_audit["records"]}
    quarantine = []
    family_counts = Counter()
    by_cohort = Counter()
    for record_id, review in sorted(effective.items()):
        if review["disposition"] != "quarantine_pending_repair":
            continue
        source = inventory[record_id]
        issues = review.get("issues", [])
        families = [name for name, needles in ISSUE_FAMILIES.items()
                    if any(any(n in issue.casefold() for n in needles)
                           for issue in issues)]
        if not families:
            families = ["other_or_unclassified"]
        family_counts.update(families)
        by_cohort[source["cohort"]] += 1
        source_result = confirmed.get(record_id)
        quarantine.append({
            "record_id": record_id,
            "record_sha256": source["record_sha256"],
            "cohort": source["cohort"],
            "effective_primary_disposition": review["disposition"],
            "issues": issues,
            "repair_proposal": review.get("repair_proposal"),
            "confirmed_source_review": bool(source_result),
            "source_review_diagnosis": source_result.get("diagnosis") if source_result else None,
            "processing_status": ("source_rechecked_hold" if source_result
                                   else "needs_source_or_semantic_resolution"),
            "approved_training_label": False,
            "training_allowed": False,
        })
    expected = verified["effective_dispositions"].get("quarantine_pending_repair", 0)
    if len(quarantine) != expected or len(quarantine) != 120:
        raise RuntimeError("effective_quarantine_count_mismatch")
    report = {
        "version": "v124-quarantine-processing-audit-v1",
        "primary_ledger_verified": verified,
        "quarantine_states": len(quarantine),
        "quarantine_by_cohort": dict(sorted(by_cohort.items())),
        "multi_label_issue_families": dict(sorted(family_counts.items())),
        "source_rechecked_states": len(confirmed),
        "source_rechecked_still_held": sum(
            row["processing_status"] == "source_rechecked_hold" for row in quarantine),
        "records": quarantine,
        "interpretation": (
            "This report accounts for every effective quarantine state and its current blocker. "
            "Issue families may overlap; no item is semantically cleared or promoted by this audit."),
        "training_allowed": False,
    }
    write(REPORT, report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
