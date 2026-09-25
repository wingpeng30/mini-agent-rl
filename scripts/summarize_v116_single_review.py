"""Report one-reviewer evidence-quality gaps without certifying SFT data."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from scripts.finalize_v116 import ROOT, WORK, REVIEW, agent_reviews, loadl, valid_vote


def main():
    result = {"version": "v116-single-review-gap-v1", "provisional_only": True,
              "can_train": False, "second_independent_api_review_complete": False}
    for split in ("train", "internal"):
        packets = {r["case_id"]: r for r in loadl(REVIEW / f"{split}-packets.jsonl")}
        votes = agent_reviews(split)
        invalid = {key: valid_vote(packets[key], vote)[1] for key, vote in votes.items()
                   if key in packets and valid_vote(packets[key], vote)[0] is False}
        approved = {key: vote for key, vote in votes.items() if key in packets and vote["approved"]
                    and key not in invalid}
        summary = {"candidate_states": len(packets), "reviewed_states": len(votes),
                   "single_approved_states": len(approved), "single_rejected_states": len(votes) - len(approved),
                   "invalid_reviews": invalid}
        if split == "internal":
            ids = {r["packet"]["case_id"]: r["source_id"] for r in loadl(WORK / "internal-candidate-private.jsonl")}
            by_source = defaultdict(list)
            for key, packet in packets.items():
                by_source[ids[key]].append((packet, votes.get(key)))
            complete = {sid: entries for sid, entries in by_source.items()
                        if len(entries) == 2 and all(vote and vote["approved"] and packet["case_id"] in approved
                                                     for packet, vote in entries)}
            summary["complete_single_approved_questions"] = len(complete)
            summary["complete_single_approved_states"] = sum(map(len, complete.values()))
            summary["question_gap_before_second_review"] = 50 - len(complete)
            summary["risk_counts_on_complete_pairs"] = dict(Counter(tag for entries in complete.values()
                                                                      for _, vote in entries
                                                                      for tag in vote["risk_labels"]))
            summary["rejected_questions"] = [{"source_id": sid,
                                             "question": entries[0][0]["question"],
                                             "reasons": [vote["reason"] for _, vote in entries
                                                         if vote and not vote["approved"]]}
                                            for sid, entries in by_source.items()
                                            if any(vote and not vote["approved"] for _, vote in entries)]
        result[split] = summary
    target = ROOT / "reports/v116-single-review-gap.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": {k: v for k, v in result["train"].items() if k != "invalid_reviews"},
                      "internal": {k: v for k, v in result["internal"].items()
                                   if k not in {"invalid_reviews", "rejected_questions"}}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
