"""CPU-only Goal Y semantic gate, paired differences, and report."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-y-three-class-transfer"
MODELS = ("incumbent", "42-a", "42-b", "43-a", "43-b")


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def classify():
    baseline = rows(OUT / "baseline-semantic-rows.jsonl")
    new = rows(OUT / "dev-semantic-delta-rows.jsonl")
    mapping = read(OUT / "dev-semantic-delta-map.json")
    baseline_summary = read(OUT / "baseline-semantic-review.json")
    if baseline_summary["row_review_sha256"] != sha(OUT / "baseline-semantic-rows.jsonl"):
        raise RuntimeError("Baseline semantic review changed")
    if len(baseline) != 48 or len({x["case_id"] for x in baseline}) != 48:
        raise RuntimeError("Baseline semantic rows incomplete")
    if len(new) != 11 or len({x["review_id"] for x in new}) != 11:
        raise RuntimeError("Delta semantic rows incomplete")
    by_review = {x["review_id"]: x for x in new}
    packet_case_ids = {x["review_id"]: x["case_id"]
                       for x in rows(OUT / "dev-semantic-delta-packet.jsonl")}
    for correction in rows(OUT / "dev-semantic-main-corrections.jsonl"):
        original = by_review[correction["review_id"]]
        if (packet_case_ids[correction["review_id"]] != correction["case_id"] or
                original["category"] != correction["original_category"]):
            raise RuntimeError("Main adjudication does not match auxiliary review")
        by_review[correction["review_id"]] = {**original,
            "category": correction["corrected_category"],
            "one_sided_error": correction["one_sided_error"],
            "reason": correction["reason"]}
    base = {x["case_id"]: x for x in baseline}
    result = {"incumbent": base}
    for model in MODELS[1:]:
        result[model] = dict(base)
        for row in mapping:
            if row["model"] == model:
                result[model][row["case_id"]] = by_review[row["review_id"]]
        if set(result[model]) != set(base): raise RuntimeError("Model semantic IDs changed")
    return result


def main():
    status = read(OUT / "state.json")
    if status["status"] != "awaiting_semantic_review":
        raise RuntimeError("Wrong run state")
    if any((OUT / f"confirm-{name}.jsonl").exists() for name in MODELS):
        raise RuntimeError("Confirmation set unexpectedly opened")
    dev = {name: {x["case_id"]: x for x in rows(OUT / f"dev-{name}.jsonl")} for name in MODELS}
    old = {name: rows(OUT / f"old-{name}.jsonl") for name in MODELS}
    fixed = {name: {x["case_id"]: x for x in rows(OUT / f"fixed-{name}.jsonl")} for name in MODELS}
    sem = classify()
    cases = {x["case_id"]: x for x in rows(OUT / "dev-cases.jsonl")}
    if any(len(dev[n]) != 48 or len(old[n]) != 32 or len(fixed[n]) != 48 for n in MODELS):
        raise RuntimeError("Incomplete evaluation")
    if any(set(dev[n]) != set(cases) or set(fixed[n]) != set(cases) for n in MODELS):
        raise RuntimeError("New dev case IDs changed")
    if any(fixed["incumbent"][k]["answer"] != dev["incumbent"][k]["answer"] for k in cases):
        raise RuntimeError("Incumbent fixed context replay changed answer")
    base_ids = {k for k, v in sem["incumbent"].items() if v["category"] == "sufficient"}
    stats, changes = {}, []
    for name in MODELS:
        data = dev[name]
        reviews = sem[name]
        success_ids = {k for k, v in reviews.items() if v["category"] == "sufficient"}
        stats[name] = {"success": len(success_ids), "strict_em": sum(v["em"] for v in data.values()),
            "by_category_success": {cat: sum(k in success_ids for k, case in cases.items()
                                         if case["attribute_category"] == cat)
                                    for cat in ("occupation", "entity_type", "source")},
            "factual_or_unsupported": sum(v["category"] == "factual_or_unsupported" for v in reviews.values()),
            "one_sided_error": sum(v["one_sided_error"] for v in reviews.values()),
            "missing_qualifier": sum(v["category"] == "missing_qualifier" for v in reviews.values()),
            "undetermined": sum(v["category"] == "undetermined" for v in reviews.values()),
            "invalid": sum(v["reason"] == "invalid_action" for v in data.values()),
            "both_support_retrieved": sum(v["both_support_retrieved"] for v in data.values()),
            "searches": sum(v["searches"] for v in data.values()),
            "old_em": sum(v["em"] for v in old[name]),
            "old_regression_em": sum(v["em"] for v in old[name] if v["split"] == "goal_G_regression"),
            "fixed_em": sum(v["em"] for v in fixed[name].values()),
            "gained_ids": sorted(success_ids - base_ids), "lost_ids": sorted(base_ids - success_ids)}
        if name == "incumbent": continue
        for case_id in sorted(cases):
            before = dev["incumbent"][case_id]
            after = data[case_id]
            if (before["answer"] != after["answer"] or
                    sem["incumbent"][case_id]["category"] != reviews[case_id]["category"]):
                baseline_docs = [(doc["id"], doc["text"]) for step in before["trace"]
                                 for doc in step.get("observation", {}).get("result", [])]
                model_docs = [(doc["id"], doc["text"]) for step in after["trace"]
                              for doc in step.get("observation", {}).get("result", [])]
                changes.append({"model": name, "case_id": case_id,
                    "category": cases[case_id]["attribute_category"], "question": cases[case_id]["question"],
                    "baseline_answer": before["answer"], "model_answer": after["answer"],
                    "baseline_review": sem["incumbent"][case_id]["category"],
                    "model_review": reviews[case_id]["category"],
                    "baseline_reason": sem["incumbent"][case_id]["reason"],
                    "model_reason": reviews[case_id]["reason"],
                    "retrieved_context_changed": baseline_docs != model_docs,
                    "fixed_context_baseline": fixed["incumbent"][case_id]["answer"],
                    "fixed_context_model": fixed[name][case_id]["answer"]})
    base = stats["incumbent"]
    arms = {}
    for arm in ("a", "b"):
        members = [stats[f"{seed}-{arm}"] for seed in (42, 43)]
        arms[arm] = {"qualified": all(v["success"] >= base["success"] + 3
            and v["factual_or_unsupported"] <= base["factual_or_unsupported"]
            and v["one_sided_error"] <= base["one_sided_error"]
            and v["invalid"] <= base["invalid"] for v in members),
            "dev_success": [v["success"] for v in members], "required_each": base["success"] + 3}
    if any(value["qualified"] for value in arms.values()):
        raise RuntimeError("A route qualified; old semantic gate must be completed before decision")
    decision = {"status": "complete_development_gate_failed", "selected": "goal-t-incumbent",
        "source_adapter": str(read(OUT / "config.json")["source"]), "confirmation_opened": False,
        "reason": "Neither route gained at least 3/48 semantic successes in both seeds",
        "original_48_pair_plan": "stopped_for_data_shortfall_12_8_5",
        "small_experiment_pairs": 25, "development": stats, "arm_gates": arms,
        "old_semantic_review": "not_required_after_development_gate_failure",
        "checkpoint_release": False,
        "source_hashes": {name: sha(OUT / name) for name in
            ("pairs.json", "dev-freeze.json", "baseline-semantic-rows.jsonl",
            "dev-semantic-delta-rows.jsonl", "dev-semantic-delta-map.json",
            "dev-semantic-main-corrections.jsonl")}}
    save(OUT / "decision.json", decision)
    with (OUT / "dev-changes.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for item in changes: stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    status["status"] = decision["status"]
    status["decision_sha256"] = sha(OUT / "decision.json")
    (OUT / "state.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"arm_gates": arms, "summary": {n: {k: v for k, v in d.items()
            if k in ("success", "strict_em", "old_em", "old_regression_em", "fixed_em", "one_sided_error")}
            for n, d in stats.items()}, "changes": len(changes)}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, sort_keys=True))
