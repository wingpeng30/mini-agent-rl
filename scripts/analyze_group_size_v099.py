"""Estimate whether increasing GRPO group size can rescue reward variance.

This is a read-only diagnostic over independent frozen rollouts.  It does not
claim that pooled empirical probabilities equal a future sampling run; it is a
pre-registered feasibility check used before spending GPU time.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from audit_answer_reward_v098 import read


def variance_probability(values: list[float], group_size: int) -> float:
    counts = Counter(values)
    total = len(values)
    return 1.0 - sum((count / total) ** group_size for count in counts.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", action="append", type=Path, required=True)
    parser.add_argument("--group-size", action="append", type=int, default=[])
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = args.group_size or [4, 8]

    runs = [read(path) for path in args.db]
    by_task: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for row in run["rows"]:
            by_task[row["task_id"]].append(row["candidate_reward"])

    rng = random.Random(args.seed)
    estimates = {}
    for size in sizes:
        probabilities = {
            task_id: variance_probability(values, size)
            for task_id, values in sorted(by_task.items())
        }
        counts = []
        for _ in range(args.simulations):
            varied = 0
            for values in by_task.values():
                sampled = [rng.choice(values) for _ in range(size)]
                varied += len(set(sampled)) > 1
            counts.append(varied)
        counts.sort()
        estimates[str(size)] = {
            "expected_nonzero_groups": sum(probabilities.values()),
            "probability_at_least_5_of_10": sum(value >= 5 for value in counts) / len(counts),
            "simulated_p05": counts[int(0.05 * len(counts))],
            "simulated_median": counts[len(counts) // 2],
            "simulated_p95": counts[int(0.95 * len(counts))],
            "per_task_variance_probability": probabilities,
        }

    payload = {
        "version": "v0.9.9-preflight",
        "method": "pooled-three-seed empirical distribution with replacement",
        "limitations": "forecast only; future frozen sampling remains the decision evidence",
        "source_hashes_unchanged": all(run["sha256_before"] == run["sha256_after"] for run in runs),
        "task_count": len(by_task),
        "rollouts_per_task": sorted({len(values) for values in by_task.values()}),
        "estimates": estimates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
