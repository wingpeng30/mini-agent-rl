"""Classify a frozen rollout export without changing rewards or selecting tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def analyze(path: Path) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups[row["rollout"]["task_id"]].append(row["rollout"])
    classifications = defaultdict(int)
    protocol_failures = defaultdict(int)
    total_variance_groups = 0
    legal_variance_groups = 0
    details = []
    for task_id, rows in groups.items():
        rewards = [float(row["reward"]) for row in rows]
        legal = [row.get("termination_reason") == "answer" for row in rows]
        correct = [float(row.get("reward_components", {}).get("exact_match", 0)) > 0 for row in rows]
        legal_rewards = [reward for reward, ok in zip(rewards, legal) if ok]
        total_variance = len(set(rewards)) > 1
        legal_variance = len(set(legal_rewards)) > 1
        total_variance_groups += int(total_variance)
        legal_variance_groups += int(legal_variance)
        terminations = [str(row.get("termination_reason") or "missing") for row in rows]
        for reason in terminations:
            if reason != "answer": protocol_failures[reason] += 1
        if all(correct): category = "all_correct"
        elif not any(correct) and total_variance: category = "all_wrong_reward_variance"
        elif not any(correct): category = "all_wrong"
        elif legal_variance: category = "legal_reward_variance"
        elif total_variance and not all(legal): category = "variance_from_protocol_failure"
        else: category = "other_mixed"
        classifications[category] += 1
        details.append({"task_id": task_id, "category": category, "rewards": rewards,
                        "legal_count": sum(legal), "correct_count": sum(correct),
                        "total_reward_variance": total_variance, "legal_reward_variance": legal_variance,
                        "termination_reasons": terminations})
    return {
        "version": "v0.9.4",
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "task_count": len(groups),
        "rollout_count": sum(len(rows) for rows in groups.values()),
        "classifications": dict(classifications),
        "total_variance_groups": total_variance_groups,
        "legal_variance_groups": legal_variance_groups,
        "protocol_failure_counts": dict(protocol_failures),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
