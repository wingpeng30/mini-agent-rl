"""Score blinded Goal Z development reviews and enforce the frozen gate."""
from __future__ import annotations

import json
from collections import Counter

import run_v133_goal_z_multihop_sft as z


OUT = z.OUT
LABELS = ("incumbent", "42-answer", "42-trajectory", "43-answer", "43-trajectory")
CATEGORIES = {"correct_sufficient", "missing_required_information", "factual_error",
              "unsupported_by_visible_evidence", "indeterminate"}


def main() -> None:
    decision_path = OUT / "development-decision.json"
    if decision_path.exists():
        raise FileExistsError(decision_path)
    state = z.read(z.STATE)
    if state["status"] != "awaiting_dev_evidence_review":
        raise RuntimeError("Development evaluation not ready")
    mapping = z.read(OUT / "blind-review-map.json")
    inherited = z.read(OUT / "blind-review-inherited.json")
    reviewed = z.rows(OUT / "blind-review-votes-0.jsonl") + z.rows(OUT / "blind-review-votes-1.jsonl")
    corrections = z.rows(OUT / "blind-review-main-corrections.jsonl")
    by_id = {}
    for row in inherited + reviewed:
        if row["review_id"] in by_id and by_id[row["review_id"]]["category"] != row["category"]:
            raise RuntimeError("Conflicting blind review")
        by_id[row["review_id"]] = row
    for row in corrections:
        if row["review_id"] not in by_id or by_id[row["review_id"]]["category"] != row["from"]:
            raise RuntimeError("Main correction mismatch")
        by_id[row["review_id"]] = by_id[row["review_id"]] | {
            "category": row["to"], "reason": row["reason"],
            "evidence_ids": row.get("evidence_ids_override", by_id[row["review_id"]]["evidence_ids"])}
    packet = z.rows(OUT / "blind-review-packet-0.jsonl") + z.rows(OUT / "blind-review-packet-1.jsonl")
    expected = {x["review_id"] for x in packet} | {x["review_id"] for x in inherited}
    if set(by_id) != expected or len(reviewed) != len({x["review_id"] for x in reviewed}):
        raise RuntimeError("Blind review incomplete or duplicated")
    packet_by_id = {x["review_id"]: x for x in packet}
    for review_id, row in by_id.items():
        if row["category"] not in CATEGORIES:
            raise RuntimeError("Unknown category")
        if review_id in packet_by_id:
            visible = {d["id"] for d in packet_by_id[review_id]["docs"]}
            if not set(row["evidence_ids"]).issubset(visible):
                raise RuntimeError("Review cites unseen evidence")
            if row["category"] == "correct_sufficient" and (
                    not row["evidence_ids"] or packet_by_id[review_id]["reason"] != "answer"):
                raise RuntimeError("Success without evidence or answer")
    case_ids = {split: set(v["case_id"] for v in mapping if v["split"] == split and v["model"] == "incumbent")
                for split in ("dev", "old", "y")}
    if {k: len(v) for k, v in case_ids.items()} != {"dev": 64, "old": 32, "y": 48}:
        raise RuntimeError("Review mapping case count mismatch")
    counts = {}
    for split in ("dev", "old", "y"):
        for label in LABELS:
            entries = [x for x in mapping if x["split"] == split and x["model"] == label]
            if len(entries) != len(case_ids[split]) or {x["case_id"] for x in entries} != case_ids[split]:
                raise RuntimeError("Model review rows missing")
            categories = Counter(by_id[x["review_id"]]["category"] for x in entries)
            if split == "old":
                old_cases = {x["case_id"]: x["split"] for x in z.q.dev_cases()}
                categories["main_sufficient"] = sum(by_id[x["review_id"]]["category"] == "correct_sufficient"
                                                     for x in entries if old_cases[x["case_id"]] == "new_eval")
                categories["regression_sufficient"] = sum(by_id[x["review_id"]]["category"] == "correct_sufficient"
                                                           for x in entries if old_cases[x["case_id"]] == "goal_G_regression")
            output = z.rows(OUT / f"eval-{split}-{label}.jsonl")
            categories["invalid"] = sum(v["reason"] == "invalid_action" for v in output)
            categories["em"] = sum(v["em"] for v in output)
            categories["searches"] = sum(v["searches"] for v in output)
            counts[f"{split}-{label}"] = dict(categories)
    baseline = counts["dev-incumbent"]
    if baseline.get("correct_sufficient", 0) != z.read(OUT / "baseline-evidence-review.json")["dev_sufficient"]:
        raise RuntimeError("Baseline review changed")
    qualified = {}
    for arm in ("answer", "trajectory"):
        seed_results = []
        for seed in (42, 43):
            label = f"{seed}-{arm}"
            dev = counts[f"dev-{label}"]
            old = counts[f"old-{label}"]
            yy = counts[f"y-{label}"]
            good = (dev.get("correct_sufficient", 0) >= baseline.get("correct_sufficient", 0) + 4 and
                    all(dev.get(key, 0) <= baseline.get(key, 0) for key in
                        ("factual_error", "unsupported_by_visible_evidence", "invalid")) and
                    old["main_sufficient"] >= counts["old-incumbent"]["main_sufficient"] and
                    old["regression_sufficient"] >= counts["old-incumbent"]["regression_sufficient"] and
                    yy.get("correct_sufficient", 0) >= counts["y-incumbent"].get("correct_sufficient", 0) and
                    z.read(OUT / f"seed{label}/summary.json")["updates"] == 128)
            seed_results.append({"seed": seed, "qualified": good,
                                 "dev_success": dev.get("correct_sufficient", 0),
                                 "dev_factual": dev.get("factual_error", 0),
                                 "dev_unsupported": dev.get("unsupported_by_visible_evidence", 0),
                                 "old_main": old["main_sufficient"],
                                 "old_regression": old["regression_sufficient"],
                                 "goal_y": yy.get("correct_sufficient", 0)})
        qualified[arm] = seed_results
    eligible = [arm for arm, values in qualified.items() if all(v["qualified"] for v in values)]
    if eligible:
        eligible.sort(key=lambda arm: (-min(v["dev_success"] for v in qualified[arm]),
                                       -sum(v["dev_success"] for v in qualified[arm]),
                                       sum(v["dev_factual"] for v in qualified[arm]),
                                       0 if arm == "answer" else 1))
        selected = eligible[0]
        state["status"] = "confirmation_authorized"
    else:
        selected = None
        state["status"] = "complete_development_gate_failed"
    result = {"status": state["status"], "selected_route": selected, "eligible_routes": eligible,
              "seed_gates": qualified, "counts": counts,
              "review_sha256": {"map": z.sha(OUT / "blind-review-map.json"),
                                "votes_0": z.sha(OUT / "blind-review-votes-0.jsonl"),
                                "votes_1": z.sha(OUT / "blind-review-votes-1.jsonl"),
                                "main_corrections": z.sha(OUT / "blind-review-main-corrections.jsonl")}}
    z.write(decision_path, result)
    z.STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": state["status"], "selected_route": selected,
                      "seed_gates": qualified}))


if __name__ == "__main__":
    main()
