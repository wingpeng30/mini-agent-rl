"""SFT JSONL 的只读质量审计。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from .hotpot import _file_sha256, _jsonl
import math
import statistics

def audit_rl_signal(db_path: str | Path) -> dict[str, Any]:
    """只读审计 SQLite 中的 RL 轨迹，避免把损坏信号送入训练。"""
    import sqlite3
    conn = sqlite3.connect(str(db_path)); conn.row_factory = sqlite3.Row
    rows = [json.loads(row[0]) for row in conn.execute("SELECT data FROM rollouts").fetchall()]
    groups = [json.loads(row[0]) for row in conn.execute("SELECT data FROM rollout_groups").fetchall()]
    transitions = [json.loads(row[0]) for row in conn.execute("SELECT data FROM transitions").fetchall()]
    report = {"db": str(db_path), "rollouts": len(rows), "groups": len(groups),
              "checks": {"finite": True, "length_consistent": True, "mask_observation_clean": True,
                         "reward_recompute_consistent": True, "group_statistics_consistent": True,
                         "advantage_mean_zero": True, "immediate_reward_consistent": True,
                         "infrastructure_failures": 0},
              "counts": {"zero_variance_groups": 0, "nonzero_variance_groups": 0, "eligible": 0},
              "reward_component_nonzero_counts": {}, "termination_counts": {},
              "correlations": {}}
    by_group: dict[str, list[dict[str, Any]]] = {}
    for item in rows:
        status = item.get("status")
        if status == "FAILED": report["checks"]["infrastructure_failures"] += 1
        if item.get("eligible_for_rl"): report["counts"]["eligible"] += 1
        adv = item.get("advantage")
        if adv is not None and not math.isfinite(float(adv)): report["checks"]["finite"] = False
        if abs(sum(float(value) for value in item.get("reward_components", {}).values()) - float(item.get("reward", 0.0))) > 1e-8:
            report["checks"]["reward_recompute_consistent"] = False
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
        if ids is not None and not (logs is not None and mask is not None and len(ids) == len(logs) == len(mask)):
            report["checks"]["length_consistent"] = False
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
