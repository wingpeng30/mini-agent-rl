"""Read-only diagnosis of zero-variance GRPO groups.

The decision rule is fixed in this file: low action diversity with acceptable
protocol reliability selects group_size=8 as the sole pilot change; diverse
actions with flat reward selects no sampling change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from mini_agent_rl.storage.sqlite import SQLiteStore
from mini_agent_rl.domain import RolloutGroup


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_signature(transitions) -> str:
    actions = []
    for transition in transitions:
        call = transition.response.tool_call
        if call:
            actions.append({"action": call.name, "arguments": call.arguments})
        else:
            actions.append({"action": "answer", "content": transition.response.content})
    return json.dumps(actions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def query_signature(transitions) -> tuple[str, ...]:
    return tuple(
        str(t.response.tool_call.arguments.get("query", "")).strip().casefold()
        for t in transitions if t.response.tool_call and t.response.tool_call.name == "search"
    )


def diagnose(path: Path) -> dict:
    store = SQLiteStore(path)
    groups = []
    invalid_rollouts = total_rollouts = 0
    groups_in_db = [RolloutGroup.model_validate_json(row[0]) for row in store.conn.execute("SELECT data FROM rollout_groups")]
    for group in groups_in_db:
        rows = []
        for rollout_id in group.rollout_ids:
            rollout = store.get_rollout(rollout_id)
            if rollout is None:
                raise RuntimeError(f"missing rollout {rollout_id}")
            transitions = store.get_transitions(rollout_id)
            total_rollouts += 1
            invalid_rollouts += int(rollout.termination_reason == "invalid_action")
            rows.append({
                "reward": float(rollout.reward), "termination": rollout.termination_reason,
                "action": action_signature(transitions), "queries": query_signature(transitions),
                "answer": (rollout.final_answer or "").strip().casefold(),
                "components": rollout.reward_components,
            })
        rewards = [row["reward"] for row in rows]
        groups.append({
            "task_id": group.task_id, "zero_variance": len(set(rewards)) == 1,
            "unique_actions": len({row["action"] for row in rows}),
            "unique_query_sequences": len({row["queries"] for row in rows}),
            "unique_answers": len({row["answer"] for row in rows}),
            "termination_counts": dict(Counter(row["termination"] for row in rows)),
            "correct_count": sum(float(row["components"].get("exact_match", 0.0)) > 0 for row in rows),
            "support_coverage_values": sorted({float(row["components"].get("support_coverage", 0.0)) for row in rows}),
            "evidence_supported_values": sorted({float(row["components"].get("evidence_supported", 0.0)) for row in rows}),
            "rewards": rewards,
        })
    store.close()
    zero = [row for row in groups if row["zero_variance"]]
    return {
        "db": str(path), "sha256": sha(path), "group_count": len(groups),
        "zero_variance_groups": len(zero), "invalid_rollouts": invalid_rollouts,
        "rollouts": total_rollouts,
        "zero_variance_unique_actions": [row["unique_actions"] for row in zero],
        "zero_variance_unique_queries": [row["unique_query_sequences"] for row in zero],
        "zero_variance_unique_answers": [row["unique_answers"] for row in zero],
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [diagnose(path) for path in args.db]
    unique = [value for report in reports for value in report["zero_variance_unique_actions"]]
    invalid = sum(report["invalid_rollouts"] for report in reports)
    rollouts = sum(report["rollouts"] for report in reports)
    median_unique = statistics.median(unique) if unique else None
    invalid_rate = invalid / rollouts if rollouts else 1.0
    # Frozen decision: do not increase temperature.  Expand group size only
    # when the observed zero-variance groups have low action coverage and the
    # protocol is already reliable enough to make extra samples useful.
    use_group8 = bool(unique) and median_unique <= 2 and invalid_rate <= 0.10
    decision = "group_size_8" if use_group8 else "no_sampling_change"
    zero_rows = [row for report in reports for row in report["groups"] if row["zero_variance"]]
    cause_counts = Counter()
    for row in zero_rows:
        if row["unique_actions"] <= 2:
            cause_counts["low_action_diversity"] += 1
        elif row["correct_count"] == 0 and len(row["support_coverage_values"]) == 1:
            cause_counts["diverse_actions_all_wrong_same_support_reward"] += 1
        elif row["correct_count"] == len(row["rewards"]):
            cause_counts["diverse_actions_all_correct"] += 1
        else:
            cause_counts["diverse_actions_flat_composite_reward"] += 1
    result = {
        "version": "v0.9.7", "kind": "sampling_diversity_diagnostic",
        "pre_registered_rule": {"low_diversity_median_unique_actions_lte": 2,
                                "maximum_invalid_rate": 0.10,
                                "temperature_change_allowed": False},
        "aggregate": {"zero_variance_groups": len(unique), "median_unique_actions": median_unique,
                      "invalid_rate": invalid_rate, "cause_counts": dict(cause_counts)},
        "decision": decision, "runs": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
