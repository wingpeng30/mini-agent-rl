"""Bind paired prompt repairs discovered after the original search-state reviews."""
from scripts.v123_cpu_core import digest, readl, writel
from scripts.v124_forensic import OUT


CHANGES = [
    (21, "region_scale_undefined_for_answer_pair",
     "Search is grounded and retrieves the missing range, but answer row 108 confirms that 'same region' lacks a defined geographic scale. The search state's prompt must change together with its answer state before the pair can be retained."),
    (51, "presidential_ordinal_needs_term_anchor",
     "Search retrieves the historical 45th-president descriptor, but answer row 176 shows the question never limits the office term. Reword the question around the presidency beginning in 2017 for both states; do not treat an unqualified person ordinal as timeless."),
]


def main():
    rows = []
    for number, issue, reason in CHANGES:
        record_id = f"v117_train:{number:04}"
        batch = f"primary-batch-{(number - 1) // 10 + 1:03}.jsonl"
        original = next(r for r in readl(OUT / batch) if r["record_id"] == record_id)
        assert original["disposition"] == "retain_candidate"
        rows.append({"record_id": record_id, "record_sha256": original["record_sha256"],
                     "superseded_primary_review_sha256": digest(original),
                     "effective_disposition": "repair_candidate", "new_issue": issue,
                     "reason": reason,
                     "status": "needs_paired_question_context_token_replay_and_independent_double_review",
                     "independent_review": False, "approved_training_label": False,
                     "training_allowed": False})
    path = OUT / "primary-amendments-003-private.jsonl"
    if path.exists() and readl(path) != rows:
        raise ValueError("existing_primary_amendments_003_differ")
    if not path.exists():
        writel(path, rows)
    print("Saved two append-only paired-state corrections.")


if __name__ == "__main__":
    main()
