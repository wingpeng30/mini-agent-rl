"""Append-only primary-review correction; original review remains preserved."""
from scripts.v123_cpu_core import digest, readl, writel
from scripts.v124_forensic import OUT


def main():
    inventory = readl(OUT / "inventory-private.jsonl")
    target = next(r for r in inventory if r["record_id"] == "v117_train:0083")
    original = next(r for r in readl(OUT / "primary-batch-009.jsonl") if r["record_id"] == target["record_id"])
    assert original["record_sha256"] == target["record_sha256"]
    assert original["disposition"] == "retain_candidate"
    affected = [r["record_id"] for r in inventory if r["question"] == target["question"]]
    assert affected == ["v117_train:0083", "v117_train:0112"]
    rows = [{"record_id": target["record_id"], "record_sha256": target["record_sha256"],
             "superseded_primary_review_sha256": digest(original),
             "effective_disposition": "repair_candidate",
             "new_issue": "created_by_figer_not_established_by_ownership",
             "reason": "The Figer source says 'His organisation' and calls him co-owner with Wagner Ribeiro. Neither statement establishes that Figer created the company. The retrieved Zé Roberto page verifies the Palmeiras role but cannot repair the creation premise.",
             "proposed_question": "What is the name of Juan Figer's organisation that represents the Palmeiras left wing back and midfielder Zé Roberto?",
             "same_question_record_ids": affected,
             "status": "needs_same_question_context_token_replay_and_independent_double_review",
             "independent_review": False, "approved_training_label": False,
             "training_allowed": False}]
    path = OUT / "primary-amendments-private.jsonl"
    if path.exists() and readl(path) != rows:
        raise ValueError("existing_primary_amendments_differ")
    if not path.exists():
        writel(path, rows)
    print("Saved one superseding primary-review correction; original opinion retained.")


if __name__ == "__main__":
    main()
