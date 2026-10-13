"""Make model-blind, deduplicated Goal Z development evidence packets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import run_v133_goal_z_multihop_sft as z


OUT = z.OUT
LABELS = ("incumbent", "42-answer", "42-trajectory", "43-answer", "43-trajectory")
SPLITS = ("dev", "old", "y")


def evidence(row: dict) -> list[dict]:
    seen = set()
    docs = []
    for step in row["trace"]:
        for document in step.get("observation", {}).get("result", []):
            if document["id"] not in seen:
                seen.add(document["id"])
                docs.append({key: document[key] for key in ("id", "title", "text")})
    return docs


def fingerprint(question: str, answer: str | None, docs: list[dict]) -> str:
    payload = json.dumps([question, answer, docs], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    if (OUT / "blind-review-map.json").exists():
        raise FileExistsError("Blind review already prepared")
    gold = {"dev": {x["id"]: x for x in z.cases("dev")},
            "old": {x["case_id"]: x for x in z.q.dev_cases()},
            "y": {x["case_id"]: x for x in z.y.dev_cases()}}
    baseline_review = {x["id"]: x for x in z.read(OUT / "baseline-dev-reviewed.json")}
    old_review = {(x["case_id"], x["answer"]): x for x in
                  z.read(z.t.OUT / "semantic-review.json")["blind_rows"]}
    y_review = {x["case_id"]: x for x in z.rows(z.y.OUT / "baseline-semantic-rows.jsonl")}
    base = {split: {x["case_id"]: x for x in z.rows(OUT / f"eval-{split}-incumbent.jsonl")}
            for split in SPLITS}
    candidates = {}
    mapping = []
    inherited = []
    for split in SPLITS:
        for label in LABELS:
            path = OUT / f"eval-{split}-{label}.jsonl"
            values = z.rows(path)
            if len(values) != {"dev": 64, "old": 32, "y": 48}[split]:
                raise RuntimeError(f"Evaluation incomplete: {path}")
            for row in values:
                key = row["case_id"]
                case = gold[split][key]
                question = case["question"] if split == "dev" else (
                    case["question"] if "question" in case else case["messages"][1]["content"])
                answer = row["answer"]
                docs = evidence(row)
                token = fingerprint(question, answer, docs)
                entry = {"split": split, "model": label, "case_id": key, "review_id": token}
                mapping.append(entry)
                baseline_row = base[split][key]
                if answer == baseline_row["answer"] and docs == evidence(baseline_row):
                    if split == "dev":
                        category = baseline_review[key]["category"]
                        cited = baseline_review[key]["evidence_ids"]
                        reason = baseline_review[key]["reason"]
                    elif split == "old":
                        item = old_review[(key, answer)]
                        category = {"sufficient": "correct_sufficient",
                                    "missing_qualifier": "missing_required_information",
                                    "unsupported_or_wrong": "factual_error",
                                    "undetermined": "indeterminate"}[item["classification"]]
                        cited = [d["id"] for d in item["visible_evidence"]]
                        reason = item["reason"]
                    else:
                        item = y_review[key]
                        category = {"sufficient": "correct_sufficient",
                                    "factual_or_unsupported": "factual_error",
                                    "undetermined": "indeterminate"}[item["category"]]
                        cited = item["cited_doc_ids"]
                        reason = item["reason"]
                    inherited.append({"review_id": token, "category": category,
                                      "evidence_ids": cited, "reason": reason,
                                      "inherited_from": f"{split}-reviewed-baseline"})
                    continue
                candidates[token] = {"review_id": token, "split": split,
                    "question": question, "official_answer": case["answer"] if split == "dev" else case["official_answer"],
                    "answer": answer, "reason": row["reason"], "docs": docs,
                    "rubric_sha256": z.sha(OUT / "evidence-rubric.json")}
    inherited_by_id = {x["review_id"]: x for x in inherited}
    for token in inherited_by_id:
        candidates.pop(token, None)
    z.write(OUT / "blind-review-map.json", mapping)
    z.write(OUT / "blind-review-inherited.json", list(inherited_by_id.values()))
    for shard in (0, 1):
        packet = [value for token, value in sorted(candidates.items()) if int(token[0], 16) % 2 == shard]
        path = OUT / f"blind-review-packet-{shard}.jsonl"
        if path.exists():
            raise FileExistsError(path)
        for value in packet:
            z.append(path, value)
    print(json.dumps({"output_rows": len(mapping), "inherited_unique": len(inherited_by_id),
                      "new_unique": len(candidates)}))


if __name__ == "__main__":
    main()
