"""Supplementary CPU exposure audit for v118's inherited v041 lineage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_v133_goal_g_cpu import OUT, ROOT, norm, qhash, readl, sha, write


def main():
    candidates = list(readl(OUT / "review-candidates.jsonl"))
    if len(candidates) != 19:
        raise ValueError("frozen_candidate_count_changed")
    inputs = [ROOT / "data/hotpot-agent-v041/train.jsonl", ROOT / "data/hotpot-agent-v041/validation.jsonl"]
    known_ids, known_questions, known_titles, question_texts = set(), set(), set(), []
    for path in inputs:
        for row in readl(path):
            meta = row["metadata"]
            task_id = meta["task_id"]
            if task_id.startswith("hotpot-"):
                known_ids.add(task_id[7:])
            for title in meta.get("support_titles", []):
                known_titles.add(norm(title))
            for message in row["messages"]:
                if message["role"] == "user":
                    known_questions.add(qhash(message["content"]))
                    question_texts.append(norm(message["content"]))
    rows = []
    for candidate in candidates:
        titles = [norm(t) for t in candidate["evidence_titles"]]
        reasons = []
        if candidate["source_id"] in known_ids:
            reasons.append("source_id_overlap")
        if candidate["question_sha256"] in known_questions:
            reasons.append("question_overlap")
        if any(t in known_titles for t in titles):
            reasons.append("evidence_title_overlap")
        if any(t in question for t in titles for question in question_texts):
            reasons.append("named_entity_in_ancestor_question")
        rows.append({"source_id": candidate["source_id"], "reasons": reasons,
                     "ancestor_source_id_overlap": "source_id_overlap" in reasons,
                     "ancestor_question_overlap": "question_overlap" in reasons})
    write(OUT / "ancestry-audit.json", {"version": "v133-goal-g-v041-ancestry-v1",
          "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
          "known_source_ids": len(known_ids), "known_questions": len(known_questions),
          "known_support_titles": len(known_titles), "rows": rows,
          "overlap_count": sum(bool(x["reasons"]) for x in rows)})
    print(json.dumps({"candidates": len(rows), "overlap_count": sum(bool(x["reasons"]) for x in rows)}))


if __name__ == "__main__":
    main()
