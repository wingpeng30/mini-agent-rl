"""离线基准评测与指标汇总，不依赖本地 LLM 或训练后端。"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .domain import RolloutStatus, Task
from .runtime import AgentRunner
from .answer_scoring import normalize_answer, token_f1


def answer_metrics(prediction: str | None, reference: str | None) -> dict[str, float]:
    """同时返回历史严格匹配和新增诊断指标。"""
    strict_pred = re.sub(r"\s+", " ", (prediction or "").strip().lower())
    strict_gold = re.sub(r"\s+", " ", (reference or "").strip().lower())
    normalized_pred, normalized_gold = normalize_answer(prediction), normalize_answer(reference)
    return {
        "strict_exact_match": float(strict_pred == strict_gold),
        "normalized_exact_match": float(normalized_pred == normalized_gold),
        "token_f1": token_f1(prediction, reference),
    }


def file_sha256(path: str | Path) -> str:
    """流式计算数据文件指纹，避免把整份评测集载入哈希缓冲区。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentEnvironment:
    """把任务切片与检索语料绑定，避免 task_limit 偷换环境。"""

    tasks: list[Task]
    corpus: list[dict[str, Any]]
    task_ids: list[str]
    task_order_sha256: str
    corpus_sha256: str
    corpus_document_count: int
    source_data_sha256: str
    corpus_limit: int

    def fingerprint(self, retriever_version: str, observation_encoding: str = "legacy") -> dict[str, Any]:
        return {
            "source_data_sha256": self.source_data_sha256,
            "task_ids": self.task_ids,
            "task_order_sha256": self.task_order_sha256,
            "corpus_sha256": self.corpus_sha256,
            "corpus_document_count": self.corpus_document_count,
            "corpus_limit": self.corpus_limit,
            "retriever_version": retriever_version,
            "observation_encoding": observation_encoding,
        }


async def evaluate_tasks(runner: AgentRunner, tasks: list[Task], group_size: int) -> dict[str, Any]:
    """运行任务集并输出可比较的基线指标。"""
    groups = [await runner.run_group(task, group_size) for task in tasks]
    rollouts = [runner.store.get_rollout(rid) for group in groups for rid in group.rollout_ids]
    valid = [item for item in rollouts if item is not None]
    n = len(valid) or 1
    exact = sum(item.reward_components.get("exact_match", 0.0) for item in valid) / n
    metric_rows = [answer_metrics(item.final_answer, next(task.answer for task in tasks if task.id == item.task_id)) for item in valid if item.task_id in {task.id for task in tasks}]
    searches = sum(-item.reward_components.get("search_cost", 0.0) / 0.03 for item in valid) / n
    transitions = [t for item in valid for t in runner.store.get_transitions(item.id)]
    # 非法 JSON 会在 parser 阶段失败，因而没有 Transition；不能只统计已落库的
    # transition，否则会把格式失败错误地排除在 invalid action rate 之外。
    invalid_markers = ("模型输出不是合法 JSON", "模型动作必须", "动作必须是包含 query")
    invalid = sum(
        item.reward_components.get("invalid_action", 0.0) < 0
        or any(marker in (item.error or "") for marker in invalid_markers)
        for item in valid
    )
    repeated = sum(
        item.reward_components.get("repeated_search", 0.0) < 0 or "重复工具动作" in (item.error or "")
        for item in valid
    )
    search_transitions = [transition for transition in transitions if transition.response.tool_call]
    responses = [transition.response for transition in transitions]
    output_truncated = sum(bool(response.usage.get("output_truncated")) for response in responses)
    empty_direct = 0
    for rollout in valid:
        saw_empty_result = False
        for transition in runner.store.get_transitions(rollout.id):
            if transition.response.tool_call and not (transition.observation or {}).get("result"):
                saw_empty_result = True
            elif saw_empty_result and transition.response.content and not transition.response.tool_call:
                empty_direct += 1
    local_client = runner.model
    report = {
        "task_count": len(tasks), "group_size": group_size, "rollout_count": len(valid),
        "execution_success_rate": sum(item.status == RolloutStatus.SUCCEEDED for item in valid) / n,
        "exact_match_rate": exact, "strict_exact_match_rate": sum(row["strict_exact_match"] for row in metric_rows) / (len(metric_rows) or 1),
        "normalized_exact_match_rate": sum(row["normalized_exact_match"] for row in metric_rows) / (len(metric_rows) or 1),
        "token_f1": sum(row["token_f1"] for row in metric_rows) / (len(metric_rows) or 1),
        "mean_reward": sum(item.reward for item in valid) / n,
        "mean_search_count": searches,
        "failed_rollout_count": sum(item.status == RolloutStatus.FAILED for item in valid),
        "valid_action_rate": (n - invalid) / n,
        "search_action_success_rate": sum((transition.observation or {}).get("ok") is True for transition in search_transitions) / (len(search_transitions) or 1),
        "invalid_action_rate": invalid / n,
        "repeated_search_rate": repeated / n,
        "average_steps": len(transitions) / n,
        "output_budget_exhaustion_rate": output_truncated / (len(responses) or 1),
        "mean_reasoning_tokens": sum(response.reasoning_tokens for response in responses) / (len(responses) or 1),
        "mean_json_action_tokens": sum(response.json_action_tokens for response in responses) / (len(responses) or 1),
        "generated_message_end_rate": sum(bool(response.usage.get("generated_message_end")) for response in responses) / (len(responses) or 1),
        "generated_eos_rate": sum(bool(response.usage.get("generated_eos")) for response in responses) / (len(responses) or 1),
        "empty_result_direct_answer_count": empty_direct,
        "metric_denominators": {"rollouts": n, "answer_metrics": len(metric_rows), "search_actions": len(search_transitions)},
        "model_load_seconds": getattr(local_client, "load_seconds", None),
        "peak_gpu_memory_gb": local_client.gpu_memory_peak_gb() if hasattr(local_client, "gpu_memory_peak_gb") else None,
    }
    if hasattr(local_client, "gpu_memory_stats"):
        report["gpu_memory"] = local_client.gpu_memory_stats()
    if hasattr(local_client, "context_usage_summary"):
        report["context_usage"] = local_client.context_usage_summary()
    return report


def _diagnose_rollout(rollout, transitions: list[Any], task: Task) -> dict[str, Any]:
    """从已落库轨迹生成只读失败分类，不重新计算 reward。"""
    metrics = answer_metrics(rollout.final_answer, task.answer)
    searches = [t for t in transitions if t.response.tool_call and t.response.tool_call.name == "search"]
    observations = [t.observation or {} for t in searches]
    found_titles = {str(doc.get("title", "")).casefold() for obs in observations for doc in (obs.get("result") or []) if isinstance(doc, dict)}
    wanted_titles = {str(title).casefold() for title in task.metadata.get("support_titles", [])}
    usage = [t.response.usage for t in transitions]
    parse_error = next((t.response.parse_error for t in transitions if t.response.parse_error), None)
    answered = rollout.termination_reason == "answer"
    if parse_error and "不是合法 JSON" in parse_error:
        parse_error_category = "invalid_json"
    elif parse_error and "JSON 对象" in parse_error:
        parse_error_category = "non_object_json"
    elif parse_error:
        parse_error_category = "invalid_action_schema"
    else:
        parse_error_category = None
    return {
        "rollout_id": rollout.id, "task_id": rollout.task_id, "termination_reason": rollout.termination_reason,
        "reward": rollout.reward, "reward_components": rollout.reward_components, "final_answer": rollout.final_answer,
        **metrics, "answered": answered, "searched": bool(searches),
        "empty_search_result": any(not (obs.get("result") or []) for obs in observations),
        "supporting_document_coverage": len(found_titles & wanted_titles) / len(wanted_titles) if wanted_titles else None,
        "prompt_truncated": any(bool(item.get("prompt_truncated")) for item in usage),
        "parse_error": parse_error, "parse_error_category": parse_error_category,
        "repeated_action": rollout.termination_reason == "repeated_action",
        "invalid_action": rollout.termination_reason == "invalid_action",
        "not_strict_match": metrics["strict_exact_match"] == 0.0,
        "wrong_answer": answered and metrics["strict_exact_match"] == 0.0,
        "ungrounded_wrong_answer": (answered and not searches and metrics["strict_exact_match"] == 0.0),
        "evidence_supported": rollout.reward_components.get("evidence_supported", 0.0) > 0,
        "steps": len(transitions),
    }


def diagnose_store(store, data_path: str | Path, output: str | Path) -> dict[str, Any]:
    """分析 SQLite rollout，并写入汇总与逐条 JSONL 明细。"""
    records = [json.loads(line) for line in Path(data_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = {item["metadata"]["task_id"]: Task(id=item["metadata"]["task_id"], question=next(m["content"] for m in item["messages"] if m["role"] == "user"), answer=item["metadata"]["gold_answer"], metadata=item["metadata"]) for item in records}
    rollouts = [store.get_rollout(row[0]) for row in store.conn.execute("SELECT id FROM rollouts")]
    rows = [_diagnose_rollout(r, store.get_transitions(r.id), tasks[r.task_id]) for r in rollouts if r and r.task_id in tasks]
    def aggregate(items):
        count = len(items) or 1
        return {"count": len(items), "strict_exact_match_rate": sum(x["strict_exact_match"] for x in items) / count, "normalized_exact_match_rate": sum(x["normalized_exact_match"] for x in items) / count, "token_f1": sum(x["token_f1"] for x in items) / count, "answer_completion_rate": sum(x["answered"] for x in items) / count, "repeated_rate": sum(x["repeated_action"] for x in items) / count, "invalid_rate": sum(x["invalid_action"] for x in items) / count, "truncation_rate": sum(x["prompt_truncated"] for x in items) / count}
    by_reason = {key: aggregate([x for x in rows if x["termination_reason"] == key]) for key in sorted({x["termination_reason"] for x in rows})}
    by_difficulty = {key: aggregate([x for x in rows if tasks[x["task_id"]].metadata.get("difficulty", "unknown") == key]) for key in sorted({tasks[x["task_id"]].metadata.get("difficulty", "unknown") for x in rows})}
    detail_path = Path(output).with_suffix(".jsonl")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {"rollout_count": len(rows), "data_sha256": file_sha256(data_path), "details_file": str(detail_path), "overall": aggregate(rows), "by_termination_reason": by_reason, "by_difficulty": by_difficulty, "failure_counts": {"empty_search_result": sum(x["empty_search_result"] for x in rows), "ungrounded_wrong_answer": sum(x["ungrounded_wrong_answer"] for x in rows), "wrong_answer": sum(x["wrong_answer"] for x in rows), "not_strict_match": sum(x["not_strict_match"] for x in rows), "evidence_unsupported_answer": sum(x["answered"] and not x["evidence_supported"] for x in rows)}, "details": rows}
    write_report(report, output)
    return report


def load_tasks(path: str | Path) -> list[Task]:
    return [Task.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_hotpot_environment(path: str | Path, *, task_limit: int | None = None, corpus_limit: int | None = None, task_ids: list[str] | None = None) -> ExperimentEnvironment:
    """独立构建任务和 corpus；corpus_limit 不随 task_limit 隐式变化。"""
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = records if task_ids is None else [record for record in records if record["metadata"]["task_id"] in set(task_ids)]
    tasks_records = selected[:task_limit] if task_limit is not None else selected
    corpus_records = selected[:corpus_limit] if corpus_limit is not None else selected
    if not tasks_records:
        raise ValueError("评测数据为空")
    if corpus_limit is not None and task_limit is not None and corpus_limit < task_limit:
        raise ValueError("corpus_limit 必须大于等于 task_limit，避免任务不在固定语料范围内")
    tasks: list[Task] = []
    corpus: dict[str, dict[str, Any]] = {}
    for record in tasks_records:
        messages, metadata = record["messages"], record["metadata"]
        question = next(message["content"] for message in messages if message["role"] == "user")
        tasks.append(Task(id=metadata["task_id"], question=question, answer=metadata["gold_answer"], metadata={"source": metadata["source"], "support_titles": metadata.get("support_titles", [])}))
    for record in corpus_records:
        for message in record["messages"]:
            if message["role"] != "tool":
                continue
            observation = json.loads(message["content"])
            for document in observation.get("result", []):
                corpus[document["id"]] = {key: document[key] for key in ("id", "title", "text")}
    corpus_list = [corpus[key] for key in sorted(corpus)]
    corpus_bytes = json.dumps(corpus_list, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    task_ids_value = [task.id for task in tasks]
    task_bytes = json.dumps(task_ids_value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return ExperimentEnvironment(tasks, corpus_list, task_ids_value, hashlib.sha256(task_bytes).hexdigest(), hashlib.sha256(corpus_bytes).hexdigest(), len(corpus_list), file_sha256(path), corpus_limit if corpus_limit is not None else len(corpus_records))


def hotpot_sft_to_tasks_and_corpus(path: str | Path, limit: int | None = None, corpus_limit: int | None = None) -> tuple[list[Task], list[dict[str, Any]]]:
    """从本项目生成的 Hotpot SFT 测试集恢复评测任务与离线检索环境。

    gold answer 仅放入 ``Task.answer`` 供 reward 计算；模型消息和 SearchTool 语料中
    都不会出现该字段。tool observation 中的 supporting documents 是环境状态，
    会被按文档 ID 去重后提供给 SearchTool。
    """
    environment = load_hotpot_environment(path, task_limit=limit, corpus_limit=corpus_limit or limit)
    return environment.tasks, environment.corpus


def write_report(report: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def rl_rollout_metrics(store, rollout_ids: list[str]) -> dict[str, Any]:
    """汇总 RL 采集质量；策略终止和基础设施失败必须分开报告。"""
    rollouts = [store.get_rollout(rollout_id) for rollout_id in rollout_ids]
    rollouts = [rollout for rollout in rollouts if rollout]
    count = len(rollouts) or 1
    transitions = [item for rollout in rollouts for item in store.get_transitions(rollout.id)]
    actions = [item.response for item in transitions if item.response.action_token_ids is not None]
    complete = [response for response in actions if response.old_logprobs and response.action_mask and len(response.action_token_ids) == len(response.old_logprobs) == len(response.action_mask)]
    reasons = {reason: sum(rollout.termination_reason == reason for rollout in rollouts) / count for reason in ("answer", "invalid_action", "repeated_action", "max_steps")}
    advantages = [rollout.advantage for rollout in rollouts if rollout.advantage is not None]
    return {
        "rollout_count": len(rollouts), "answer_completion_rate": reasons["answer"],
        "infrastructure_failure_rate": sum(not rollout.eligible_for_rl for rollout in rollouts) / count,
        "invalid_action_rate": reasons["invalid_action"], "repeated_action_rate": reasons["repeated_action"],
        "max_steps_rate": reasons["max_steps"], "logprob_completeness": len(complete) / (len(actions) or 1),
        "average_action_tokens": sum(len(response.action_token_ids or []) for response in actions) / (len(actions) or 1),
        "advantage_mean": sum(advantages) / (len(advantages) or 1),
    }
