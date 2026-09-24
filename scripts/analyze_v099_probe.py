"""Clustered paired analysis for the v099 120-case frozen probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analyze_grpo_effect_v095 import SEEDS, bootstrap


METRICS = ("answered", "strict_exact_match", "token_f1", "search", "invalid_action", "repeated_query")


def load(root: Path, label: str) -> tuple[dict[str, dict[str, dict]], dict]:
    folder = root / f"v099-probe120-{label}"
    decisions = [json.loads(line) for line in (folder / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, dict[str, dict]] = {}
    for row in decisions:
        grouped.setdefault(row["task_id"], {})[row["condition"]] = row
    if len(grouped) != 30 or any(len(rows) != 4 for rows in grouped.values()):
        raise RuntimeError(f"incomplete probe: {label}")
    cases = [json.loads(line) for line in (folder / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    non_system = {
        row["case_id"]: [message for message in row["messages"] if message["role"] != "system"]
        for row in cases
    }
    return grouped, {
        "case_count": len(cases),
        "non_system_messages_sha256": hashlib.sha256(json.dumps(non_system, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
    }


def value(row: dict, metric: str) -> float:
    return float(row[metric])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reports")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    labels = ["sft", *(f"v096-seed{s}" for s in SEEDS), *(f"answer-seed{s}" for s in SEEDS)]
    loaded = {label: load(root, label) for label in labels}
    runs = {label: item[0] for label, item in loaded.items()}
    audits = {label: item[1] for label, item in loaded.items()}
    input_identical = len({item["non_system_messages_sha256"] for item in audits.values()}) == 1
    conditions = sorted(next(iter(next(iter(runs.values())).values())).keys())

    per_seed = {}
    for seed in SEEDS:
        left, right = runs[f"v096-seed{seed}"], runs[f"answer-seed{seed}"]
        by_condition = {}
        for condition in conditions:
            by_condition[condition] = {
                metric: bootstrap([value(right[task][condition], metric) - value(left[task][condition], metric) for task in sorted(left)])
                for metric in METRICS
            }
        per_seed[str(seed)] = by_condition

    cross = {}
    for condition in conditions:
        cross[condition] = {}
        for metric in METRICS:
            task_deltas = []
            for task in sorted(runs["sft"]):
                task_deltas.append(sum(
                    value(runs[f"answer-seed{seed}"][task][condition], metric)
                    - value(runs[f"v096-seed{seed}"][task][condition], metric)
                    for seed in SEEDS
                ) / len(SEEDS))
            cross[condition][metric] = bootstrap(task_deltas)

    overall = {}
    for metric in METRICS:
        task_deltas = []
        for task in sorted(runs["sft"]):
            values = []
            for seed in SEEDS:
                for condition in conditions:
                    values.append(value(runs[f"answer-seed{seed}"][task][condition], metric)
                                  - value(runs[f"v096-seed{seed}"][task][condition], metric))
            task_deltas.append(sum(values) / len(values))
        overall[metric] = bootstrap(task_deltas)

    payload = {
        "version": "v0.9.9",
        "cluster_unit": "30 original questions",
        "bootstrap_seed": 42,
        "bootstrap_repeats": 10_000,
        "input_audit": {"identical_non_system_messages": input_identical, "runs": audits},
        "per_seed_by_condition": per_seed,
        "cross_seed_by_condition": cross,
        "cross_seed_overall": overall,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
