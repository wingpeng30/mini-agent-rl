"""Freeze Goal Z incumbent evidence scores before any new model is trained."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import run_v133_goal_z_multihop_sft as z


OUT = z.OUT
CATEGORIES = {"correct_sufficient", "missing_required_information", "factual_error",
              "unsupported_by_visible_evidence", "indeterminate"}


def main() -> None:
    target = OUT / "baseline-evidence-review.json"
    if target.exists():
        raise FileExistsError(target)
    dev = {v["case_id"]: v for v in z.rows(OUT / "eval-dev-incumbent.jsonl")}
    if len(dev) != 64:
        raise RuntimeError("Incomplete development baseline")
    reviews = z.rows(OUT / "baseline-dev-musique-review.jsonl") + \
              z.rows(OUT / "baseline-dev-2wiki-review.jsonl")
    if len(reviews) != 64 or {v["id"] for v in reviews} != set(dev):
        raise RuntimeError("Baseline evidence review IDs incomplete")
    correction = {v["id"]: v for v in z.rows(OUT / "baseline-dev-main-corrections.jsonl")}
    applied = []
    for row in reviews:
        if row["category"] not in CATEGORIES:
            raise RuntimeError("Unknown semantic category")
        cited = correction.get(row["id"], {}).get("evidence_ids_override", row["evidence_ids"])
        visible = {d["id"] for step in dev[row["id"]]["trace"]
                   for d in step.get("observation", {}).get("result", [])}
        if not set(cited).issubset(visible):
            raise RuntimeError("Review cited unseen evidence")
        category = row["category"]
        if row["id"] in correction:
            value = correction[row["id"]]
            if category != value["from"] or value["to"] not in CATEGORIES:
                raise RuntimeError("Main adjudication mismatch")
            category = value["to"]
        if category == "correct_sufficient" and (dev[row["id"]]["reason"] != "answer" or
                                                  not cited):
            raise RuntimeError("Success lacks visible evidence or answer")
        applied.append({"id": row["id"], "category": category, "evidence_ids": cited,
                        "reason": correction[row["id"]]["reason"] if row["id"] in correction else row["reason"]})
    old = z.rows(OUT / "eval-old-incumbent.jsonl")
    goal_t = {v["case_id"]: v for v in z.rows(z.t.OUT / "eval-42-32.jsonl")}
    if len(old) != 32 or any((v["answer"], v["reason"], v["searches"]) !=
                             (goal_t[v["case_id"]]["answer"], goal_t[v["case_id"]]["reason"],
                              goal_t[v["case_id"]]["searches"]) for v in old):
        raise RuntimeError("Old baseline differs from Goal T semantic review source")
    historical = z.read(z.t.OUT / "semantic-review.json")
    known = {(v["case_id"], v["answer"]): v["classification"] for v in historical["blind_rows"]}
    old_semantic = {split: sum(known[(v["case_id"], v["answer"])] == "sufficient" for v in old
                               if v["split"] == split) for split in ("new_eval", "goal_G_regression")}
    goal_y = z.rows(OUT / "eval-y-incumbent.jsonl")
    previous_y = {v["case_id"]: v for v in z.rows(z.y.OUT / "dev-incumbent.jsonl")}
    if len(goal_y) != 48 or any((v["answer"], v["reason"], v["searches"]) !=
                                (previous_y[v["case_id"]]["answer"], previous_y[v["case_id"]]["reason"],
                                 previous_y[v["case_id"]]["searches"]) for v in goal_y):
        raise RuntimeError("Goal Y baseline differs from prior reviewed outputs")
    if z.read(z.y.OUT / "baseline-semantic-review.json")["sufficient"] != 40:
        raise RuntimeError("Goal Y reviewed baseline changed")
    categories = dict(Counter(v["category"] for v in applied))
    output = {"version": "goal-z-baseline-review-v1", "cases": {"dev": 64, "goal_y": 48},
              "dev_output_sha256": z.sha(OUT / "eval-dev-incumbent.jsonl"),
              "old_output_sha256": z.sha(OUT / "eval-old-incumbent.jsonl"),
              "goal_y_output_sha256": z.sha(OUT / "eval-y-incumbent.jsonl"),
              "review_source_sha256": {"musique": z.sha(OUT / "baseline-dev-musique-review.jsonl"),
                                       "2wiki": z.sha(OUT / "baseline-dev-2wiki-review.jsonl"),
                                       "main_corrections": z.sha(OUT / "baseline-dev-main-corrections.jsonl")},
              "rubric_sha256": z.sha(OUT / "evidence-rubric.json"),
              "dev_sufficient": categories.get("correct_sufficient", 0),
              "dev_categories": categories, "old_semantic_sufficient": old_semantic,
              "goal_y_semantic_sufficient": 40,
              "review": "Blind agent-assisted review with saved main adjudication; not independent human gold."}
    z.write(OUT / "baseline-dev-reviewed.json", applied)
    z.write(target, output)
    print(json.dumps({"dev_sufficient": output["dev_sufficient"],
                      "dev_categories": categories, "old": old_semantic, "goal_y": 40}))


if __name__ == "__main__":
    main()
