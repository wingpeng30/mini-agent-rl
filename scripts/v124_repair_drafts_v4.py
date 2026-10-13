"""Write non-trainable prompt repairs for newly found question defects."""
from copy import deepcopy

from scripts.v123_cpu_core import ROOT, digest, readl, writel
from scripts.v124_forensic import OUT


QUESTIONS = {
    51: "According to the cited presidency beginning January 20, 2017, what numbered U.S. president was the person who feuded with Megyn Kelly in Settle for More?",
    141: "Which ballpark has been home to the St. Louis Cardinals, the team Moose Baxter played for, since 2006?",
    165: "According to the supplied WLAX syndicated-programming list, which long-running American arbitration-based reality court show was included?",
    174: "What is the name of Michael Hayden's best-known commission at the airport on Chicago's Far Northwest Side?",
    176: "According to the cited presidency beginning January 20, 2017, what numbered U.S. president was the person who feuded with Megyn Kelly in Settle for More?",
}


def main():
    inventory = readl(OUT / "inventory-private.jsonl")
    by_id = {row["record_id"]: row for row in inventory}
    frozen = readl(ROOT / "data/evidence-sft-v117/frozen/train.jsonl")
    reviews = {row["record_id"]: row for path in sorted(OUT.glob("primary-batch-*.jsonl"))
               for row in readl(path)}
    rows = []
    for number, new_question in QUESTIONS.items():
        record_id = f"v117_train:{number:04}"
        source, review = by_id[record_id], reviews[record_id]
        original = frozen[number - 1]
        assert digest(original) == source["source_row_sha256"]
        revised = deepcopy(original)
        user = next(m for m in revised["messages"] if m["role"] == "user")
        assert user["content"] == source["question"]
        user["content"] = new_question
        assert digest(revised) != digest(original)
        peers = [row["record_id"] for row in inventory if row["question"] == source["question"]]
        rows.append({"record_id": record_id, "original_row_sha256": digest(original),
                     "primary_review_sha256": digest(review),
                     "same_question_record_ids": peers,
                     "same_question_all_states_reviewed": all(peer in reviews for peer in peers),
                     "revised_row": revised,
                     "status": "needs_paired_context_token_replay_and_independent_double_review",
                     "approved_training_label": False, "training_allowed": False})
    path = OUT / "repair-drafts-v4-private.jsonl"
    if path.exists() and readl(path) != rows:
        raise ValueError("existing_repair_drafts_v4_differ")
    if not path.exists():
        writel(path, rows)
    print(f"Saved {len(rows)} non-trainable question drafts.")


if __name__ == "__main__":
    main()
