"""Clustered paired probe analysis for v100 30-task versus v099 10-task."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analyze_grpo_effect_v095 import SEEDS, bootstrap


METRICS = ("answered", "strict_exact_match", "token_f1", "search", "invalid_action", "repeated_query")


def load(folder: Path) -> tuple[dict[str, dict[str, dict]], str]:
    decisions = [json.loads(line) for line in (folder / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, dict[str, dict]] = {}
    for row in decisions:
        grouped.setdefault(row["task_id"], {})[row["condition"]] = row
    if len(grouped) != 30 or any(len(rows) != 4 for rows in grouped.values()):
        raise RuntimeError(f"incomplete probe: {folder}")
    cases = [json.loads(line) for line in (folder / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    non_system = {row["case_id"]: [m for m in row["messages"] if m["role"] != "system"] for row in cases}
    digest = hashlib.sha256(json.dumps(non_system, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return grouped, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reports")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    small, large, hashes = {}, {}, []
    for seed in SEEDS:
        small[seed], digest_small = load(root / f"v099-probe120-answer-seed{seed}")
        large[seed], digest_large = load(root / f"v100-probe120-seed{seed}")
        hashes.extend((digest_small, digest_large))
    conditions = sorted(next(iter(next(iter(small.values())).values())).keys())

    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {
            condition: {
                metric: bootstrap([
                    float(large[seed][task][condition][metric]) - float(small[seed][task][condition][metric])
                    for task in sorted(small[seed])
                ]) for metric in METRICS
            } for condition in conditions
        }
    cross = {}
    for condition in conditions:
        cross[condition] = {}
        for metric in METRICS:
            cross[condition][metric] = bootstrap([
                sum(float(large[seed][task][condition][metric]) - float(small[seed][task][condition][metric]) for seed in SEEDS) / len(SEEDS)
                for task in sorted(small[SEEDS[0]])
            ])
    overall = {}
    for metric in METRICS:
        overall[metric] = bootstrap([
            sum(float(large[seed][task][condition][metric]) - float(small[seed][task][condition][metric])
                for seed in SEEDS for condition in conditions) / (len(SEEDS) * len(conditions))
            for task in sorted(small[SEEDS[0]])
        ])
    payload = {
        "version": "v0.10.0",
        "cluster_unit": "30 original questions",
        "input_non_system_messages_identical": len(set(hashes)) == 1,
        "per_seed_by_condition": per_seed,
        "cross_seed_by_condition": cross,
        "cross_seed_overall": overall,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
