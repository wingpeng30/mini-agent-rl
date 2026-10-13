"""Draft paired question/query repairs without changing frozen source records."""
from copy import deepcopy
import json

from scripts.v123_cpu_core import ROOT, digest, readl, writel
from scripts.v124_forensic import OUT


# A null replacement leaves that component unchanged. Each query change needs a
# fresh retrieval run, and every revised prompt needs tokenizer/context replay.
SPECS = {
    21: ("Are Euptelea and Richea native to the same continent?", None,
         "paired_question_repair_from_answer_row_108"),
    83: ("What is the name of Juan Figer's organisation that represents the Palmeiras left wing back and midfielder Zé Roberto?", None,
         "unsupported_creation_premise"),
    107: ("What not-for-profit teaching hospital is located in the Maryland place ranked fourth most populous by the cited 2013 Census Bureau estimates?", None,
          "rank_snapshot_anchor"),
    108: ("Are Euptelea and Richea native to the same continent?", None,
          "region_scale_disambiguation"),
    109: (None, "Murlough Nature Reserve County Down",
          "query_targets_missing_reserve_location"),
    112: ("What is the name of Juan Figer's organisation that represents the Palmeiras left wing back and midfielder Zé Roberto?", None,
          "unsupported_creation_premise"),
    117: (None, "Percy Jones birth date",
          "query_targets_missing_person_birth_date"),
    125: ("Which storm, nicknamed the No-Name Storm, was described as more severe than the December 1992 nor'easter along the Maryland coast?", None,
          "comparison_scope_repair"),
    136: ("Of which city was Sid Hatfield police chief before Charles Everett Lively assassinated him?", None,
          "missing_verb_and_city_role"),
    139: ("Of which city was Sid Hatfield police chief before Charles Everett Lively assassinated him?", None,
          "missing_verb_and_city_role"),
}


def main():
    inventory = readl(OUT / "inventory-private.jsonl")
    by_id = {row["record_id"]: row for row in inventory}
    frozen = readl(ROOT / "data/evidence-sft-v117/frozen/train.jsonl")
    reviews = {row["record_id"]: row for path in sorted(OUT.glob("primary-batch-*.jsonl"))
               for row in readl(path)}
    rows = []
    for number, (revised_question, revised_query, reason) in SPECS.items():
        record_id = f"v117_train:{number:04}"
        source, review = by_id[record_id], reviews[record_id]
        original = frozen[number - 1]
        assert digest(original) == source["source_row_sha256"]
        revised = deepcopy(original)
        if revised_question is not None:
            user = next(m for m in revised["messages"] if m["role"] == "user")
            assert user["content"] == source["question"]
            user["content"] = revised_question
        if revised_query is not None:
            target = revised["target_assistant_index"]
            action = revised["messages"][target]
            parsed = json.loads(action["content"])
            assert parsed == source["proposed_action"] and parsed["action"] == "search"
            parsed["query"] = revised_query
            action["content"] = json.dumps(parsed, ensure_ascii=False)
        assert digest(revised) != digest(original)
        peers = [row["record_id"] for row in inventory if row["question"] == source["question"]]
        rows.append({"record_id": record_id, "original_row_sha256": digest(original),
                     "primary_review_sha256": digest(review), "repair_reason": reason,
                     "same_question_record_ids": peers,
                     "same_question_all_states_reviewed": all(peer in reviews for peer in peers),
                     "revised_row": revised,
                     "status": "needs_retrieval_context_token_replay_and_independent_double_review",
                     "approved_training_label": False, "training_allowed": False})
    path = OUT / "repair-drafts-v3-private.jsonl"
    if path.exists() and readl(path) != rows:
        raise ValueError("existing_repair_drafts_v3_differ")
    if not path.exists():
        writel(path, rows)
    print(f"Saved {len(rows)} non-trainable drafts; frozen records unchanged.")


if __name__ == "__main__":
    main()
