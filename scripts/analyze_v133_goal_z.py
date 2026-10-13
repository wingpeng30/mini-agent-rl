"""Recompute Goal Z development comparisons without loading a model."""
from __future__ import annotations

import json
from collections import Counter

import run_v133_goal_z_multihop_sft as z


LABELS = ("incumbent", "42-answer", "43-answer", "42-trajectory", "43-trajectory")
SUCCESS = "correct_sufficient"


def review_index() -> dict[str, dict]:
    out = z.OUT
    votes = (z.read(out / "blind-review-inherited.json") +
             z.rows(out / "blind-review-votes-0.jsonl") +
             z.rows(out / "blind-review-votes-1.jsonl"))
    indexed = {x["review_id"]: x for x in votes}
    for correction in z.rows(out / "blind-review-main-corrections.jsonl"):
        row = indexed[correction["review_id"]]
        assert row["category"] == correction["from"]
        indexed[correction["review_id"]] = row | {"category": correction["to"],
            "reason": correction["reason"]}
    return indexed


def bucket(row: dict, category: str, support_titles: list[str]) -> str:
    if category == SUCCESS:
        return "success"
    if row["reason"] != "answer":
        return "invalid_or_no_answer"
    searches = [step for step in row["trace"] if step["action"] == "search"]
    if not searches:
        return "no_first_search"
    first_titles = {doc["title"].casefold() for doc in
                    searches[0].get("observation", {}).get("result", [])}
    if support_titles[0].casefold() not in first_titles:
        return "first_support_not_retrieved"
    if len(searches) < 2:
        return "stopped_after_first_support"
    if not row["support_retrieved"]:
        return "continued_without_both_supports"
    return "both_supports_visible_answer_failed"


def main() -> None:
    out = z.OUT
    decision = z.read(out / "development-decision.json")
    mapping = z.read(out / "blind-review-map.json")
    reviews = review_index()
    categories = {(x["split"], x["model"], x["case_id"]): reviews[x["review_id"]]["category"]
                  for x in mapping}
    dev_cases = {x["id"]: x for x in z.cases("dev")}
    evals = {label: {x["case_id"]: x for x in z.rows(out / f"eval-dev-{label}.jsonl")}
             for label in LABELS}
    fixed = {label: {x["case_id"]: x for x in z.rows(out / f"fixed-dev-{label}.jsonl")}
             for label in LABELS}
    if any(len(evals[label]) != 64 or len(fixed[label]) != 64 for label in LABELS):
        raise RuntimeError("Incomplete development or fixed context output")
    transitions = {}
    buckets = {}
    for label in LABELS:
        baseline_success = {key for key in dev_cases
                            if categories[("dev", "incumbent", key)] == SUCCESS}
        model_success = {key for key in dev_cases if categories[("dev", label, key)] == SUCCESS}
        transitions[label] = {"gained": sorted(model_success - baseline_success),
                              "lost": sorted(baseline_success - model_success),
                              "retained": sorted(model_success & baseline_success)}
        buckets[label] = dict(Counter(bucket(evals[label][key], categories[("dev", label, key)],
                                             dev_cases[key]["support_titles"])
                                      for key in dev_cases))
    rows = []
    for key, case in dev_cases.items():
        rows.append({"case_id": key, "source": case["source"], "question": case["question"],
                     "official_answer": case["answer"], "support_titles": case["support_titles"],
                     "models": {label: {"answer": evals[label][key]["answer"],
                                        "category": categories[("dev", label, key)],
                                        "em": evals[label][key]["em"],
                                        "fixed_em": fixed[label][key].get("em", 0),
                                        "searches": evals[label][key]["searches"],
                                        "support_retrieved": evals[label][key]["support_retrieved"],
                                        "failure_bucket": bucket(evals[label][key],
                                            categories[("dev", label, key)], case["support_titles"])}
                                for label in LABELS}})
    target = out / "development-per-case.jsonl"
    if target.exists():
        raise FileExistsError(target)
    for row in rows:
        z.append(target, row)
    summary = {"status": decision["status"], "selected_route": decision["selected_route"],
               "counts": decision["counts"], "transitions": transitions,
               "failure_buckets": buckets,
               "fixed_em": {label: sum(x.get("em", 0) for x in fixed[label].values())
                            for label in LABELS},
               "confirmed": False, "product_adapter": "Goal T seed 42 update 32"}
    z.write(out / "analysis.json", summary)
    print(json.dumps({"transitions": {label: {key: len(value) for key, value in items.items()}
                                      for label, items in transitions.items()},
                      "failure_buckets": buckets}, ensure_ascii=False))


if __name__ == "__main__":
    main()
