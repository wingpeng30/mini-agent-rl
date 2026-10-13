"""Create immutable, non-trainable repair drafts with same-question impact lists."""
from copy import deepcopy

from scripts.v123_cpu_core import ROOT, digest, readl, writel
from scripts.v124_forensic import OUT


SPECS = {
    37: ("question_false_inspiration_premise",
         "What is the real name of the singer known as Lana Del Rey, whose music Entertainment Weekly used in describing Julie Mintz's sound?", None),
    48: ("question_novel_disambiguation",
         "What is the subtitle of the first novel written by the author of The Life of Charlotte Bronte?", None),
    61: ("answer_preserve_lower_bound", None, "over 35,000"),
    65: ("answer_crater_entity", None, "Serpent Mound crater"),
    68: ("question_attended_not_graduated",
         "Which two universities did the World War II hero for whom USS Elrod was named attend?", None),
}


def main():
    inventory = readl(OUT / "inventory-private.jsonl")
    by_id = {row["record_id"]: row for row in inventory}
    originals = readl(ROOT / "data/evidence-sft-v117/frozen/train.jsonl")
    reviews = {row["record_id"]: row for path in sorted(OUT.glob("primary-batch-*.jsonl"))
               for row in readl(path)}
    assert len(reviews) == sum(len(readl(path)) for path in OUT.glob("primary-batch-*.jsonl"))
    rows = []
    for number, (kind, question_replacement, answer_replacement) in SPECS.items():
        record_id = f"v117_train:{number:04}"
        source = by_id[record_id]
        review = reviews[record_id]
        assert review["disposition"] == "repair_candidate"
        original = originals[number - 1]
        assert digest(original) == source["source_row_sha256"]
        affected = [r["record_id"] for r in inventory if r["question"] == source["question"]]
        revised = deepcopy(original)
        if question_replacement is not None:
            user = next(m for m in revised["messages"] if m["role"] == "user")
            assert user["content"] == source["question"]
            user["content"] = question_replacement
        if answer_replacement is not None:
            target = revised["target_assistant_index"]
            action = revised["messages"][target]
            import json
            parsed = json.loads(action["content"])
            assert parsed == source["proposed_action"] and parsed["action"] == "answer"
            parsed["answer"] = answer_replacement
            action["content"] = json.dumps(parsed, ensure_ascii=False)
        assert digest(revised) != digest(original)
        rows.append({"record_id": record_id, "original_row_sha256": digest(original),
                     "repair_type": kind, "original_question": source["question"],
                     "same_question_record_ids": affected,
                     "same_question_all_states_reviewed": all(r in reviews for r in affected),
                     "revised_row": revised, "primary_review_sha256": digest(review),
                     "status": "needs_same_question_context_token_replay_and_independent_double_review",
                     "approved_training_label": False, "training_allowed": False})
    path = OUT / "repair-drafts-v2-private.jsonl"
    if path.exists() and readl(path) != rows:
        raise ValueError("existing_repair_drafts_v2_differ")
    if not path.exists():
        writel(path, rows)
    print(f"Saved {len(rows)} immutable non-trainable drafts with same-question impact IDs.")


if __name__ == "__main__":
    main()
