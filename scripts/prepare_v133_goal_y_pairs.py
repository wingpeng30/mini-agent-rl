"""Verify Goal Y's audited real-context preference pairs and freeze raw tokens."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
X = ROOT / "runs/v133-goal-x-audit-and-transfer"
OUT = ROOT / "runs/v133-goal-y-three-class-transfer"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(s) for s in stream if s.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare() -> dict:
    proposals_path = OUT / "pair-proposals.jsonl"
    traces_path = X / "train-real-search.jsonl"
    output = OUT / "pairs.json"
    audit_output = OUT / "pair-main-review.jsonl"
    if output.exists() or audit_output.exists():
        raise FileExistsError("Goal Y pairs already frozen")
    proposals = rows(proposals_path)
    traces = {v["case_id"]: v for v in rows(traces_path)}
    if len(proposals) != 25 or len({v["case_id"] for v in proposals}) != 25:
        raise RuntimeError("Expected 25 distinct audited pair proposals")
    if Counter(v["category"] for v in proposals) != Counter({"occupation": 12, "entity_type": 8, "origin": 5}):
        raise RuntimeError("Audited category counts changed")
    frozen, review = [], []
    for pair in proposals:
        case_id = pair["case_id"]
        trace = traces[case_id]
        if not trace["both_support_full_text_visible"] or trace["reviewed_category"] != pair["category"]:
            raise RuntimeError(f"Not an approved real context: {case_id}")
        if pair["question"] != trace["question"] or pair["official_answer"] != trace["official_answer"]:
            raise RuntimeError(f"Question or label mismatch: {case_id}")
        if pair["chosen"] != trace["official_answer"] or pair["chosen"] == pair["rejected"]:
            raise RuntimeError(f"Chosen/rejected mismatch: {case_id}")
        if not pair["reason"] or not pair["defect_type"].endswith("one_sided_attribute_claim"):
            raise RuntimeError(f"No specific evidence-defect review: {case_id}")
        supports = pair["evidence"]["chosen_both_sides"]
        if {v["doc_id"] for v in supports} != set(trace["support_ids"]):
            raise RuntimeError(f"Support ID mismatch: {case_id}")
        actual = {doc["id"]: doc for step in trace["trace"]
                  for doc in step.get("observation", {}).get("result", [])}
        if any(actual[v["doc_id"]]["text"] != v["text"] for v in supports):
            raise RuntimeError(f"Audit evidence does not equal real tool output: {case_id}")
        rejected_side = pair["evidence"]["rejected_one_side"]["doc_id"]
        counterpart = pair["evidence"]["counterpart_visible"]["doc_id"]
        if rejected_side == counterpart or {rejected_side, counterpart} != set(trace["support_ids"]):
            raise RuntimeError(f"One-sided defect mapping invalid: {case_id}")
        ids = trace["answer_prompt_token_ids"]
        mask = trace["answer_prompt_attention_mask"]
        if not ids or len(ids) != len(mask) or len(ids) > 2048:
            raise RuntimeError(f"Invalid actual prompt tokens: {case_id}")
        frozen.append({"case_id": case_id, "category": pair["category"], "question": pair["question"],
                       "chosen": pair["chosen"], "rejected": pair["rejected"],
                       "input_ids": ids, "input_mask": mask,
                       "input_sha256": trace["answer_prompt_sha256"],
                       "support_ids": trace["support_ids"], "defect_type": pair["defect_type"]})
        review.append({"case_id": case_id, "decision": "accepted", "reason": pair["reason"],
                       "chosen": pair["chosen"], "rejected": pair["rejected"],
                       "support_quotes": [{"doc_id": v["doc_id"], "text": v["text"]} for v in supports]})
    with audit_output.open("w", encoding="utf-8", newline="\n") as stream:
        for item in review:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    output.write_text(json.dumps({"pairs": frozen, "source_hashes": {
        "proposals": sha(proposals_path), "real_search": sha(traces_path),
        "main_review": sha(audit_output)}}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"pairs": len(frozen), "by_category": dict(Counter(v["category"] for v in frozen)),
            "pairs_sha256": sha(output), "main_review_sha256": sha(audit_output)}


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, sort_keys=True))
