"""Preserve, then correct the effective primary view of Southbridge answer row 4."""
from scripts.v123_cpu_core import digest, readl, writel
from scripts.v124_forensic import OUT


def main():
    source = next(r for r in readl(OUT / "inventory-private.jsonl") if r["record_id"] == "v117_train:0004")
    original = next(r for r in readl(OUT / "primary-batch-001.jsonl") if r["record_id"] == source["record_id"])
    assert original["disposition"] == "retain_candidate"
    change = {
        "record_id": source["record_id"],
        "record_sha256": source["record_sha256"],
        "superseded_primary_review_sha256": digest(original),
        "effective_disposition": "repair_candidate",
        "new_issue": "town_mislabeled_city_in_paired_question",
        "reason": "Search row 343 exposes that this same question calls the Town of Southbridge 'a city'. Although the event answer is correct and visible here, the answer cannot be retained with the inaccurate question text. The pair requires one wording repair and exact replay before independent review.",
        "status": "needs_paired_question_context_token_replay_and_independent_double_review",
        "independent_review": False,
        "approved_training_label": False,
        "training_allowed": False,
    }
    path = OUT / "primary-amendments-004-private.jsonl"
    if path.exists():
        assert readl(path) == [change], "existing_amendment_differs"
    else:
        writel(path, [change])
    print("Saved immutable primary amendment for v117_train:0004")


if __name__ == "__main__":
    main()
