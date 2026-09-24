"""SFT JSONL 的只读质量审计。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from .hotpot import _file_sha256, _jsonl
import math
import statistics
from ..domain import Task, Transition
from ..reward.functions import (CompositeReward, ExactMatchReward, EvidenceReward, SupportCoverageReward,
                                SearchCostPenalty, RepeatedSearchPenalty, InvalidActionPenalty, MaxStepsPenalty,
                                UngroundedAnswerPenalty, OverSearchAfterEvidencePenalty, EvidenceStopReward)
from ..reward.functions import IncompleteEvidenceAnswerPenalty, EmptyEvidenceAnswerPenalty, AnswerTokenF1Reward


def _reward_from_provenance(provenance: dict[str, Any]) -> CompositeReward | None:
    """仅重建已落库的公开 reward 参数；未知版本宁可返回 None 也不猜测。"""
    known = {
        "ExactMatchReward": lambda values: ExactMatchReward(values.get("weight", 1.0)),
        "EvidenceReward": lambda values: EvidenceReward(), "SupportCoverageReward": lambda values: SupportCoverageReward(),
        "SearchCostPenalty": lambda values: SearchCostPenalty(),
        "RepeatedSearchPenalty": lambda values: RepeatedSearchPenalty(values.get("penalty", -0.5)),
        "InvalidActionPenalty": lambda values: InvalidActionPenalty(), "MaxStepsPenalty": lambda values: MaxStepsPenalty(),
        "UngroundedAnswerPenalty": lambda values: UngroundedAnswerPenalty(),
        "OverSearchAfterEvidencePenalty": lambda values: OverSearchAfterEvidencePenalty(values.get("penalty", -0.2)),
        "EvidenceStopReward": lambda values: EvidenceStopReward(values.get("weight", 0.3)),
        "IncompleteEvidenceAnswerPenalty": lambda values: IncompleteEvidenceAnswerPenalty(values.get("penalty", -0.15)),
        "EmptyEvidenceAnswerPenalty": lambda values: EmptyEvidenceAnswerPenalty(values.get("penalty", -0.10)),
        "AnswerTokenF1Reward": lambda values: AnswerTokenF1Reward(values.get("weight", 0.20)),
    }
    meta = provenance.get("_meta", {})
    payload = {name: values for name, values in provenance.items() if name != "_meta"}
    version = meta.get("reward_version", "v096") if isinstance(meta, dict) else "v096"
    if version not in {"v096", "v098-answer-v1"} or not payload or any(name not in known or not isinstance(values, dict) for name, values in payload.items()):
        return None
    return CompositeReward([known[name](values) for name, values in payload.items()], version=version)

def audit_rl_signal(db_path: str | Path) -> dict[str, Any]:
    """只读审计 SQLite 中的 RL 轨迹，避免把损坏信号送入训练。"""
    import sqlite3
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"审计数据库不存在：{path}")
    # mode=ro 防止拼错路径时创建空数据库，也保证审计不会改写实验产物。
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True); conn.row_factory = sqlite3.Row
    rows = [json.loads(row[0]) for row in conn.execute("SELECT data FROM rollouts").fetchall()]
    groups = [json.loads(row[0]) for row in conn.execute("SELECT data FROM rollout_groups").fetchall()]
    transitions = [json.loads(row[0]) for row in conn.execute("SELECT data FROM transitions").fetchall()]
    tasks = {json.loads(row[0]).get("id"): json.loads(row[0]) for row in conn.execute("SELECT data FROM tasks").fetchall()}
    report = {"db": str(db_path), "rollouts": len(rows), "groups": len(groups),
              "checks": {"finite": True, "length_consistent": True, "mask_observation_clean": True,
                         "reward_component_sum_consistent": True, "reward_recompute_consistent": True, "group_statistics_consistent": True,
                         "advantage_mean_zero": True, "immediate_reward_consistent": True,
                         "infrastructure_failures": 0},
              "counts": {"zero_variance_groups": 0, "nonzero_variance_groups": 0, "eligible": 0},
              "reward_component_nonzero_counts": {}, "termination_counts": {},
              "correlations": {}, "unverifiable": {"missing_prompt_tokens": 0, "missing_reward_provenance": 0}}
    by_group: dict[str, list[dict[str, Any]]] = {}
    for item in rows:
        status = item.get("status")
        if status == "FAILED": report["checks"]["infrastructure_failures"] += 1
        if item.get("eligible_for_rl"): report["counts"]["eligible"] += 1
        adv = item.get("advantage")
        if adv is not None and not math.isfinite(float(adv)): report["checks"]["finite"] = False
        if abs(sum(float(value) for value in item.get("reward_components", {}).values()) - float(item.get("reward", 0.0))) > 1e-8:
            report["checks"]["reward_component_sum_consistent"] = False
        reward = _reward_from_provenance(item.get("reward_config", {}))
        task = tasks.get(item.get("task_id"))
        if reward is None or task is None:
            report["unverifiable"]["missing_reward_provenance"] += 1
        else:
            # transition_by_rollout 尚未构建，延后到下方完成独立计算。
            item["_audit_reward"] = reward
        for name, value in item.get("reward_components", {}).items():
            if float(value) != 0.0:
                report["reward_component_nonzero_counts"][name] = report["reward_component_nonzero_counts"].get(name, 0) + 1
        reason = item.get("termination_reason") or "unknown"
        report["termination_counts"][reason] = report["termination_counts"].get(reason, 0) + 1
        by_group.setdefault(item.get("group_id", ""), []).append(item)
    transition_by_rollout: dict[str, list[dict[str, Any]]] = {}
    for transition in transitions:
        transition_by_rollout.setdefault(transition.get("rollout_id", ""), []).append(transition)
        response = transition.get("response", {})
        ids, logs, mask = response.get("action_token_ids"), response.get("old_logprobs"), response.get("action_mask")
        prompt_ids = response.get("prompt_token_ids")
        if not prompt_ids:
            report["unverifiable"]["missing_prompt_tokens"] += 1
        if ids is None or logs is None or mask is None or not (len(ids) == len(logs) == len(mask)):
            report["checks"]["length_consistent"] = False
        if any(not math.isfinite(float(value)) for value in (logs or [])):
            report["checks"]["finite"] = False
        if mask is not None and any(value not in (0, 1) for value in mask):
            report["checks"]["mask_observation_clean"] = False
        if transition.get("observation") is not None and mask is not None and len(mask) != len(ids or []):
            report["checks"]["mask_observation_clean"] = False
    for item in rows:
        own = transition_by_rollout.get(item.get("id", ""), [])
        over_count = sum("over_search_after_evidence" in t.get("reward_tags", []) for t in own)
        over_component = float(item.get("reward_components", {}).get("over_search_after_evidence", 0.0))
        if abs(over_component - (-0.2 * over_count)) > 1e-8:
            report["checks"]["immediate_reward_consistent"] = False
        reward = item.get("_audit_reward")
        task_data = tasks.get(item.get("task_id"))
        if reward is not None and task_data is not None:
            recomputed = reward.evaluate(Task.model_validate(task_data), [Transition.model_validate(value) for value in own], item.get("final_answer"))
            expected = {key: float(value) for key, value in item.get("reward_components", {}).items()}
            if recomputed != expected or abs(sum(recomputed.values()) - float(item.get("reward", 0.0))) > 1e-8:
                report["checks"]["reward_recompute_consistent"] = False
    for group in groups:
        std = float(group.get("std_reward", 0.0) or 0.0)
        report["counts"]["zero_variance_groups" if abs(std) < 1e-8 else "nonzero_variance_groups"] += 1
        items = [item for item in by_group.get(group.get("id", ""), []) if item.get("eligible_for_rl")]
        rewards = [float(item.get("reward", 0.0)) for item in items]
        mean = statistics.mean(rewards) if rewards else 0.0
        expected_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        if abs(mean - float(group.get("mean_reward", 0.0))) > 1e-8 or abs(expected_std - std) > 1e-8:
            report["checks"]["group_statistics_consistent"] = False
        advantages = [float(item["advantage"]) for item in items if item.get("advantage") is not None]
        if advantages and abs(sum(advantages) / len(advantages)) > 1e-6:
            report["checks"]["advantage_mean_zero"] = False
    lengths, advantages, rewards, searches = [], [], [], []
    for item in rows:
        own = transition_by_rollout.get(item.get("id", ""), [])
        lengths.append(sum(len(t.get("response", {}).get("action_token_ids") or []) for t in own))
        advantages.append(float(item.get("advantage") or 0.0)); rewards.append(float(item.get("reward", 0.0)))
        searches.append(sum(bool(t.get("response", {}).get("tool_calls")) for t in own))
    def corr(left, right):
        if len(left) < 2: return None
        ml, mr = statistics.mean(left), statistics.mean(right)
        denominator = math.sqrt(sum((x-ml)**2 for x in left) * sum((y-mr)**2 for y in right))
        return sum((x-ml)*(y-mr) for x,y in zip(left,right)) / denominator if denominator else None
    report["correlations"] = {"advantage_action_length": corr(advantages, lengths), "advantage_search_count": corr(advantages, searches), "advantage_reward": corr(advantages, rewards)}
    # 没有可复现的 reward 配置时只报告 unknown，不能把分项求和误称为独立重算。
    report["checks"]["reward_recompute_consistent"] = (report["unverifiable"]["missing_reward_provenance"] == 0
                                                          and report["checks"]["reward_recompute_consistent"]
                                                          and report["checks"]["reward_component_sum_consistent"])
    checks_ok = all(value is True for key, value in report["checks"].items() if key != "infrastructure_failures") and report["checks"]["infrastructure_failures"] == 0
    required_nonzero = math.ceil(len(groups) / 2) if groups else 1
    report["gates"] = {"required_nonzero_variance_groups": required_nonzero,
                       "nonzero_variance_rate": report["counts"]["nonzero_variance_groups"] / len(groups) if groups else 0.0}
    report["passed"] = checks_ok and report["counts"]["nonzero_variance_groups"] >= required_nonzero
    conn.close(); return report


def audit_sft_directory(data_dir: str | Path) -> dict[str, Any]:
    """统计样本、搜索步数、重复问题及 supporting-document 跨 split 泄漏。"""
    root = Path(data_dir)
    summaries: dict[str, Any] = {}
    titles_by_split: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        questions = [item["messages"][1]["content"] for item in records]
        search_counts = [item.get("metadata", {}).get("search_count", 0) for item in records]
        titles = {title for item in records for title in item.get("metadata", {}).get("support_titles", [])}
        titles_by_split[split] = titles
        summaries[split] = {"samples": len(records), "unique_questions": len(set(questions)), "duplicate_questions": len(questions) - len(set(questions)), "search_count_distribution": dict(Counter(search_counts)), "supporting_documents": len(titles), "mean_messages": round(sum(len(item["messages"]) for item in records) / max(len(records), 1), 2)}
    overlaps = {"train_validation": len(titles_by_split["train"] & titles_by_split["validation"]), "train_test": len(titles_by_split["train"] & titles_by_split["test"]), "validation_test": len(titles_by_split["validation"] & titles_by_split["test"])}
    return {"data_dir": str(root), "splits": summaries, "document_overlap": overlaps, "passed_document_isolation": not any(overlaps.values())}


def audit_final_test(source_dir: str | Path, final_dir: str | Path) -> dict[str, Any]:
    """验证最终测试集与已用开发数据的 task/document 隔离及 manifest 哈希。"""
    source, final = Path(source_dir), Path(final_dir)
    source_records = [item for name in ("train", "validation", "test") for item in _jsonl(source / f"{name}.jsonl")]
    final_path = final / "final-test.jsonl"
    final_records = _jsonl(final_path)
    source_ids = {item["metadata"]["task_id"] for item in source_records}
    final_ids = [item["metadata"]["task_id"] for item in final_records]
    source_titles = {title for item in source_records for title in item["metadata"].get("support_titles", [])}
    final_titles = {title for item in final_records for title in item["metadata"].get("support_titles", [])}
    manifest_path = final / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    actual_hash = _file_sha256(final_path)
    return {"source_dir": str(source), "final_dir": str(final), "final_count": len(final_records),
            "duplicate_final_task_ids": len(final_ids) - len(set(final_ids)), "task_id_overlap": len(source_ids & set(final_ids)),
            "supporting_document_overlap": len(source_titles & final_titles), "final_sha256": actual_hash,
            "manifest_sha256_matches": manifest.get("final_test", {}).get("sha256") == actual_hash,
            "passed": not (len(final_ids) - len(set(final_ids))) and not (source_ids & set(final_ids)) and not (source_titles & final_titles) and manifest.get("final_test", {}).get("sha256") == actual_hash}
