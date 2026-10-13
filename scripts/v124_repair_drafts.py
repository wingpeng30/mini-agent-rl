"""Materialize reviewed repair proposals as non-trainable drafts only."""
from copy import deepcopy

from scripts.v123_cpu_core import ROOT, digest, readl, writel
from scripts.v124_forensic import OUT


def main():
    original = readl(ROOT / "data/evidence-sft-v117/frozen/train.jsonl")[11]
    review = next(r for r in readl(OUT / "primary-batch-002.jsonl") if r["record_id"] == "v117_train:0012")
    inventory = next(r for r in readl(OUT / "inventory-private.jsonl") if r["record_id"] == review["record_id"])
    assert digest(original) == inventory["source_row_sha256"]
    revised = deepcopy(original)
    user = next(m for m in revised["messages"] if m["role"] == "user")
    question = user["content"]
    user["content"] = "What year was Mike Cavan's successor as head football coach at Southern Methodist University born?"
    rows = [{"record_id": review["record_id"], "original_row_sha256": digest(original),
             "repair_type": "question_role_disambiguation", "original_question": question,
             "revised_row": revised, "primary_review_sha256": digest(review),
             "status": "needs_context_token_replay_and_independent_double_review", "training_allowed": False}]
    path = OUT / "repair-drafts-private.jsonl"
    if path.exists() and readl(path) != rows:
        raise ValueError("existing_repair_drafts_differ")
    if not path.exists():
        writel(path, rows)
    print("One concrete repair draft saved; original dataset unchanged.")


if __name__ == "__main__":
    main()
