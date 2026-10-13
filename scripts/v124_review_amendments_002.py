"""Clarify that the Navy article's 2017 scope is not securely tied to personnel."""
from scripts.v123_cpu_core import digest, readl, writel
from scripts.v124_forensic import OUT


def main():
    original = next(r for r in readl(OUT / "primary-batch-012.jsonl")
                    if r["record_id"] == "v117_train:0119")
    assert original["disposition"] == "repair_candidate"
    rows = [{"record_id": original["record_id"], "record_sha256": original["record_sha256"],
             "superseded_primary_review_sha256": digest(original),
             "effective_disposition": "quarantine_pending_repair",
             "new_issue": "personnel_as_of_date_not_explicitly_bound",
             "reason": "The source states the personnel count, then gives vessel and aircraft figures followed by 'as of 2017'. Its grammar does not unambiguously date the preceding personnel count. The suggested 2017-anchored question would add an unproved time relation. Keep the count as a source value while withholding a dated repair until source provenance independently establishes its date.",
             "proposed_question": None,
             "status": "needs_dated_source_provenance_before_repair",
             "independent_review": False, "approved_training_label": False,
             "training_allowed": False}]
    path = OUT / "primary-amendments-002-private.jsonl"
    if path.exists() and readl(path) != rows:
        raise ValueError("existing_primary_amendments_002_differ")
    if not path.exists():
        writel(path, rows)
    print("Saved source-scope correction without modifying original vote.")


if __name__ == "__main__":
    main()
