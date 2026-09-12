"""离线基准评测与指标汇总，不依赖本地 LLM 或训练后端。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .domain import RolloutStatus, Task
from .runtime import AgentRunner


async def evaluate_tasks(runner: AgentRunner, tasks: list[Task], group_size: int) -> dict[str, Any]:
    """运行任务集并输出可比较的基线指标。"""
    groups = [await runner.run_group(task, group_size) for task in tasks]
    rollouts = [runner.store.get_rollout(rid) for group in groups for rid in group.rollout_ids]
    valid = [item for item in rollouts if item is not None]
    n = len(valid) or 1
    exact = sum(item.reward_components.get("exact_match", 0.0) for item in valid) / n
    searches = sum(-item.reward_components.get("search_cost", 0.0) / 0.03 for item in valid) / n
    return {
        "task_count": len(tasks), "group_size": group_size, "rollout_count": len(valid),
        "execution_success_rate": sum(item.status == RolloutStatus.SUCCEEDED for item in valid) / n,
        "exact_match_rate": exact, "mean_reward": sum(item.reward for item in valid) / n,
        "mean_search_count": searches,
        "failed_rollout_count": sum(item.status == RolloutStatus.FAILED for item in valid),
    }


def load_tasks(path: str | Path) -> list[Task]:
    return [Task.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
