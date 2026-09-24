"""固定历史后的停止决策探针；前置动作不是策略动作，也不会进入训练。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from .domain import Message, Task
from .evaluation import answer_metrics, file_sha256, load_hotpot_environment, normalize_answer
from .model import LocalModelConfig, TransformersModelClient, local_environment_report
from .training.gates import evaluate_probe_gate, resolve_memory_limit_gb
from .model.context import ContextBudgetError, budget_messages
from .runtime.runner import AgentRunner, _query_key
from .prompts import DIRECT_SYSTEM, THINK_SYSTEM, PROMPT_VERSION
from .tools import SearchTool, ToolRegistry

CONDITIONS = ("one_hop", "two_hop", "empty_result", "repeated_history")


def load_frozen_cases(path: str | Path, data_path: str | Path = "data/hotpot-agent-v041/validation.jsonl") -> list[dict[str, Any]]:
    """读取 v0.8.5 冻结 case；不重新选题，也不覆盖源文件。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到冻结 case 文件：{source}")
    cases = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != 120 or {case.get("condition") for case in cases} != set(CONDITIONS):
        raise ValueError("冻结 case 不符合 30 题 × 4 条件的 v0.8.5 协议")
    if len({case.get("case_id") for case in cases}) != 120:
        raise ValueError("冻结 case_id 不唯一")
    # v0.8.5 case 本身不含 support_titles；仅在离线审计阶段按 task ID 补充，
    # 此字段不会进入 messages、检索排序或预算选择。
    supports = {str(row["metadata"]["task_id"]): row.get("metadata", {}).get("support_titles", [])
                for row in _records(data_path, 100)}
    for case in cases:
        case.setdefault("audit", {})["support_titles"] = supports.get(str(case["task_id"]), [])
    return cases


def _reject_final(path: str | Path) -> None:
    """探针与训练准备均禁止读取封存 final-test。"""
    value = str(Path(path).resolve()).lower()
    if "final-test" in value or "hotpot-agent-v072" in value:
        raise ValueError("禁止使用封存最终集")


def _records(path: str | Path, limit: int) -> list[dict[str, Any]]:
    _reject_final(path)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < limit:
        raise ValueError(f"数据不足：需要 {limit} 条，实际只有 {len(rows)} 条")
    return rows[:limit]


def check_split_overlap(train_path: str | Path, validation_path: str | Path, limit: int = 100) -> dict[str, Any]:
    """按 ID 与规范化问题审计分割；有重叠即停，不临时替换题目。"""
    train, validation = _records(train_path, limit), _records(validation_path, limit)
    def identity(row: dict[str, Any]) -> tuple[str, str]:
        question = next(message["content"] for message in row["messages"] if message["role"] == "user")
        return str(row["metadata"]["task_id"]), normalize_answer(question)
    train_pairs, validation_pairs = [identity(row) for row in train], [identity(row) for row in validation]
    train_ids, train_questions = zip(*train_pairs); validation_ids, validation_questions = zip(*validation_pairs)
    overlap = {"task_ids": sorted(set(train_ids) & set(validation_ids)),
               "normalized_questions": sorted(set(train_questions) & set(validation_questions))}
    return {"train_prefix_sha256": hashlib.sha256(json.dumps(train_ids).encode()).hexdigest(),
            "validation_prefix_sha256": hashlib.sha256(json.dumps(validation_ids).encode()).hexdigest(),
            "train_data_sha256": file_sha256(train_path), "validation_data_sha256": file_sha256(validation_path),
            "overlap": overlap, "passed": not any(overlap.values())}


def _queries(record: dict[str, Any]) -> list[str]:
    """从既有 SFT 动作获取固定历史，绝不依据待测模型表现筛题。"""
    result: list[str] = []
    for message in record["messages"]:
        if message["role"] != "assistant":
            continue
        try:
            action = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        if action.get("action") == "search" and str(action.get("query", "")).strip():
            result.append(str(action["query"]))
    if len(result) < 2:
        raise ValueError(f"缺少双跳历史: {record['metadata']['task_id']}")
    return result


def _coverage(docs: list[dict[str, Any]], titles: list[str]) -> dict[str, Any]:
    """support title 只在生成后离线评分，不能进入模型消息、排序或裁剪。"""
    wanted = {str(title).casefold() for title in titles}
    found = {str(doc.get("title", "")).casefold() for doc in docs}
    matched = sorted(found & wanted)
    return {"support_count": len(wanted), "matched_titles": matched,
            "support_coverage": len(matched) / len(wanted) if wanted else None,
            "coverage_bucket": "complete" if wanted and wanted <= found else ("partial" if matched else "none")}


def build_cases(data: str | Path, limit: int = 30, corpus_limit: int = 100) -> list[dict[str, Any]]:
    """建立四种固定条件，审计字段不传给模型。"""
    _reject_final(data)
    environment = load_hotpot_environment(data, task_limit=limit, corpus_limit=corpus_limit)
    search = SearchTool(environment.corpus, "bm25-v2")
    shell = object.__new__(AgentRunner); shell.anti_repeat_prompt = True
    cases: list[dict[str, Any]] = []
    for task, record in zip(environment.tasks, _records(data, limit), strict=True):
        queries = _queries(record)
        histories = {"one_hop": queries[:1], "two_hop": queries[:2], "empty_result": queries[:1],
                     "repeated_history": [queries[0], queries[0]]}
        for condition in CONDITIONS:
            seen: list[str] = []; docs_all: list[dict[str, Any]] = []; history = []
            messages = shell._messages(Task(id=task.id, question=task.question, answer=""))
            for query in histories[condition]:
                seen.append(_query_key(query))
                docs = [] if condition == "empty_result" else search._search(query)
                docs_all.extend(docs)
                observation = {"ok": True, "result": docs, "searched_queries": sorted(set(seen))}
                history.append({"query": query, "raw_observation": observation})
                messages.extend([Message(role="assistant", content=json.dumps({"action": "search", "query": query}, ensure_ascii=False)),
                                 Message(role="tool", content=json.dumps(observation, ensure_ascii=False))])
            cases.append({"case_id": f"{task.id}:{condition}", "task_id": task.id, "condition": condition,
                          "messages": [message.model_dump(mode="json") for message in messages], "history_queries": seen,
                          "gold_answer": task.answer, "audit": {"history": history, "support_titles": task.metadata.get("support_titles", []),
                          **_coverage(docs_all, task.metadata.get("support_titles", []))},
                          "environment": environment.fingerprint("bm25-v2")})
    return cases


def classify(case: dict[str, Any], response: Any) -> dict[str, Any]:
    """只计算新生成动作；人工前置重复不能冒充模型重复。"""
    call = response.tool_call
    answered = not response.parse_error and call is None
    repeated = bool(call and _query_key(str(call.arguments.get("query", ""))) in case["history_queries"])
    signature = "answer" if answered else (f"search:{_query_key(str(call.arguments.get('query', '')))}" if call else "invalid")
    visible_docs = response.usage.get("retained_tool_documents", [])
    visible_titles = {str(item.get("title", "")).casefold() for item in visible_docs if isinstance(item, dict)}
    wanted = {str(item).casefold() for item in case.get("audit", {}).get("support_titles", [])}
    visible_coverage = len(wanted & visible_titles) / len(wanted) if wanted else None
    return {"case_id": case["case_id"], "task_id": case["task_id"], "condition": case["condition"], "answered": answered,
            "repeated_query": repeated, "invalid_action": bool(response.parse_error), "search": bool(call), "action_signature": signature,
            **answer_metrics(response.content if answered else "", case["gold_answer"]),
            # 保持轻量 unit test 与旧手工 case 兼容；真实构造路径始终写入 audit。
            "audit": {**case.get("audit", {"coverage_bucket": "none", "support_coverage": None, "history": []}),
                      "visible_support_coverage": visible_coverage,
                      "document_changes": response.usage.get("document_changes", [])},
            "response": response.model_dump(mode="json")}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按条件与真实文档覆盖汇总，不以覆盖标签指导生成。"""
    count = len(rows)
    if not count:
        return {"count": 0}
    mean = lambda key: sum(float(row[key]) for row in rows) / count
    buckets = {key: [row for row in rows if row["audit"]["coverage_bucket"] == key] for key in ("none", "partial", "complete")}
    return {"count": count, "answered_rate": mean("answered"), "new_search_rate": mean("search"),
            "repeated_query_rate": mean("repeated_query"), "invalid_action_rate": mean("invalid_action"),
            "strict_exact_match_rate": mean("strict_exact_match"), "token_f1": mean("token_f1"),
            "empty_result_direct_answer_rate": mean("answered") if rows[0]["condition"] == "empty_result" else None,
            "complete_coverage_answer_rate": sum(row["answered"] for row in buckets["complete"]) / len(buckets["complete"]) if buckets["complete"] else None,
            "complete_coverage_correct_rate": sum(row["strict_exact_match"] for row in buckets["complete"]) / len(buckets["complete"]) if buckets["complete"] else None,
            "partial_coverage_continue_search_rate": sum(row["search"] for row in buckets["partial"]) / len(buckets["partial"]) if buckets["partial"] else None,
            "coverage_buckets": {key: len(value) for key, value in buckets.items()}}


async def evaluate(cases: list[dict[str, Any]], output: Path, adapter: str, max_input_tokens: int = 768,
                   observation_encoding: str = "legacy", memory_limit_gb: float | None = None,
                   model_path: str = r"D:\qwen_08b", enable_thinking: bool = False,
                   max_new_tokens: int = 128, generation_timeout: float = 600.0) -> dict[str, Any]:
    """冻结 SFT 的单步探针；每条落盘，显存或异常即停止。"""
    output.mkdir(parents=True, exist_ok=False)
    # Preserve every frozen history/action/tool message; only the system protocol varies.
    cases = [{**case, "messages": [{**message, "content": THINK_SYSTEM if enable_thinking else DIRECT_SYSTEM} if message["role"] == "system" else message for message in case["messages"]]} for case in cases]
    client = TransformersModelClient(LocalModelConfig(model_id=model_path, adapter_path=adapter, load_in_4bit=False, max_input_tokens=max_input_tokens,
                                                       observation_encoding=observation_encoding, enable_thinking=enable_thinking,
                                                       max_new_tokens=max_new_tokens, timeout=generation_timeout, gradient_checkpointing=False))
    memory_limit_gb = resolve_memory_limit_gb(memory_limit_gb, local_environment_report().get("gpu_memory_gb"))
    registry = ToolRegistry(); registry.register(SearchTool([], "bm25-v2"))
    manifest = {"version": "v0.8.8", "model_path": model_path, "adapter": adapter, "seed": 42, "temperature": 0.0, "context_budget": max_input_tokens,
                "observation_encoding": observation_encoding, "enable_thinking": enable_thinking, "prompt_version": PROMPT_VERSION,
                "max_new_tokens": max_new_tokens, "generation_timeout": generation_timeout,
                "case_count": len(cases), "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "semantics": "teacher-history one-step probe; no weight updates; history actions excluded from policy metrics",
                "gates": {"lossy_trim_max": 0.05, "reserved_gib_max": memory_limit_gb}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "cases.jsonl").write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    rows: list[dict[str, Any]] = []; reason = None; attempted = 0; failed = 0
    try:
        for index, case in enumerate(cases, 1):
            attempted += 1
            try:
                response = await client.generate([Message.model_validate(message) for message in case["messages"]], registry.definitions(), temperature=0.0, seed=42)
                row = classify(case, response)
            except ContextBudgetError as exc:
                # 失败 case 也必须落盘，不能从分母消失。
                row = {"case_id": case["case_id"], "task_id": case["task_id"], "condition": case["condition"],
                       "answered": False, "search": False, "repeated_query": False, "invalid_action": True,
                       "action_signature": "context_budget_error", "strict_exact_match": 0.0, "token_f1": 0.0,
                       "audit": {**case.get("audit", {}), "visible_support_coverage": None, "document_changes": []},
                       "response": {"usage": {"context_budget_error": True}, "error": f"ContextBudgetError: {exc}"}}
                failed += 1
            except Exception as exc:
                row = {"case_id": case["case_id"], "task_id": case["task_id"], "condition": case["condition"],
                       "answered": False, "search": False, "repeated_query": False, "invalid_action": True,
                       "action_signature": "infrastructure_error", "strict_exact_match": 0.0, "token_f1": 0.0,
                       "audit": {**case.get("audit", {}), "visible_support_coverage": None, "document_changes": []},
                       "response": {"usage": {}, "error": f"{type(exc).__name__}: {exc}"}}
                rows.append(row)
                with (output / "decisions.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                failed += 1; reason = f"infrastructure_error: {type(exc).__name__}"
                break
            rows.append(row)
            with (output / "decisions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{index}/{len(cases)} {case['condition']} action={row['action_signature']}", flush=True)
            reserved = client.gpu_memory_stats().get("peak_reserved_gb")
            if reserved is None:
                reason = "UNVERIFIABLE: missing peak_reserved_gb"; break
            if reserved > memory_limit_gb:
                reason = "memory_gate"; break
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    by_condition = {condition: _aggregate([row for row in rows if row["condition"] == condition]) for condition in CONDITIONS}
    total = max(len(rows), 1)
    context = {"prompt_truncation_rate": sum(bool(row["response"]["usage"].get("prompt_truncated")) for row in rows) / total,
               "observation_compaction_rate": sum(bool(row["response"]["usage"].get("observation_compacted")) for row in rows) / total,
               "history_drop_rate": sum(bool(row["response"]["usage"].get("history_dropped")) for row in rows) / total,
               "context_budget_errors": sum(bool(row["response"]["usage"].get("context_budget_error")) for row in rows),
               "lossless_expression_change_rate": sum(bool(row["response"]["usage"].get("lossless_expression_changed")) for row in rows) / total,
               "lossy_body_trim_rate": sum(bool(row["response"]["usage"].get("lossy_body_trimmed")) for row in rows) / total,
               "documents_deleted": sum(int(row["response"]["usage"].get("documents_deleted", 0)) for row in rows)}
    visible_coverages = [row["audit"].get("visible_support_coverage") for row in rows if row["audit"].get("visible_support_coverage") is not None]
    context["visible_support_coverage"] = sum(visible_coverages) / len(visible_coverages) if visible_coverages else None
    retentions = [row["audit"]["visible_support_coverage"] / row["audit"]["support_coverage"] for row in rows
                  if row["audit"].get("support_coverage") not in (None, 0) and row["audit"].get("visible_support_coverage") is not None]
    context["visible_support_retention"] = sum(retentions) / len(retentions) if retentions else None
    changes = [change for row in rows for change in row["audit"].get("document_changes", [])]
    original_chars = sum(int(change.get("original_text_chars", 0)) for change in changes if not change.get("reused_reference"))
    visible_chars = sum(int(change.get("visible_text_chars", 0)) for change in changes if not change.get("reused_reference"))
    context["body_text_retention"] = visible_chars / original_chars if original_chars else None
    memory = client.gpu_memory_stats()
    gate = evaluate_probe_gate(expected=len(cases), attempted=attempted, completed=len(rows), failed=failed,
                               context=context, memory=memory, memory_limit_gb=memory_limit_gb)
    status = "DIAGNOSTIC_COMPLETE" if gate["status"] == "PASS" else gate["status"]
    report = {"status": status, "reason": reason or (None if gate["status"] == "PASS" else "probe_gate"), "gate": gate, "training_allowed": False,
              "training_pending": "需独立完成训练输入重放、奖励重算、mask/reference/LoRA-only 与训练候选 reward 方差验收",
              "by_condition": by_condition, "context": context, "expected": len(cases), "attempted": attempted,
              "completed": len(rows), "failed": failed, "not_run": max(0, len(cases)-attempted),
              "policy_fingerprint": client.policy_fingerprint(), "memory": memory}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def compare_probe_reports(left: Path, right: Path, output: Path) -> dict[str, Any]:
    """仅允许预算/编码不同的逐题比较，并按题目而非四条件扩大样本。"""
    def rows(path: Path) -> dict[str, dict[str, Any]]:
        return {row["case_id"]: row for row in (json.loads(line) for line in (path / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if line)}
    a, b = rows(left), rows(right)
    if set(a) != set(b):
        raise ValueError("两个探针 case_id 不一致，拒绝比较")
    left_manifest = json.loads((left / "manifest.json").read_text(encoding="utf-8"))
    right_manifest = json.loads((right / "manifest.json").read_text(encoding="utf-8"))
    for key in ("adapter", "seed", "temperature", "cases_sha256", "case_count"):
        if left_manifest.get(key) != right_manifest.get(key):
            raise ValueError(f"探针配置 {key} 不一致，拒绝比较")
    changed = [key for key in a if a[key]["action_signature"] != b[key]["action_signature"]]
    answers_changed = [key for key in a if a[key]["response"].get("content") != b[key]["response"].get("content")]
    queries_changed = [key for key in a if a[key]["action_signature"].startswith("search:")
                       and a[key]["action_signature"] != b[key]["action_signature"]]
    by_condition = {}
    for condition in CONDITIONS:
        keys = [key for key, row in a.items() if row["condition"] == condition]
        by_condition[condition] = {"question_count": len({a[key]["task_id"] for key in keys}),
            "strict_em_delta": sum(b[key]["strict_exact_match"]-a[key]["strict_exact_match"] for key in keys)/max(len(keys), 1),
            "token_f1_delta": sum(b[key]["token_f1"]-a[key]["token_f1"] for key in keys)/max(len(keys), 1)}
    result = {"left": str(left), "right": str(right), "case_count": len(a),
              "question_count": len({row["task_id"] for row in a.values()}), "changed_action_count": len(changed),
              "action_change_rate": len(changed) / max(len(a), 1), "changed_case_ids": changed,
              "query_change_count": len(queries_changed),
              "answer_text_change_count": len(answers_changed), "answer_text_change_rate": len(answers_changed) / max(len(a), 1),
              "strict_em_delta": sum(row["strict_exact_match"] for row in b.values()) / len(b) - sum(row["strict_exact_match"] for row in a.values()) / len(a),
              "token_f1_delta": sum(row["token_f1"] for row in b.values()) / len(b) - sum(row["token_f1"] for row in a.values()) / len(a),
              "by_condition": by_condition, "left_manifest": left_manifest, "right_manifest": right_manifest,
              "left_report": json.loads((left / "report.json").read_text(encoding="utf-8")),
              "right_report": json.loads((right / "report.json").read_text(encoding="utf-8"))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def audit_probe_budget(cases: list[dict[str, Any]], output: Path, model_path: str, max_input_tokens: int,
                       observation_encoding: str) -> dict[str, Any]:
    """只加载本地 tokenizer，预先审计 120 个 case 的最终 prompt，不运行生成。"""
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError("预算审计需要本地 transformers 环境") from exc
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    rows = []
    for case in cases:
        messages = [Message.model_validate(message) for message in case["messages"]]
        def render(items: list[Message]) -> int:
            chat = [{"role": item.role, "content": item.content} for item in items]
            return int(processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True,
                       return_dict=True, return_tensors="pt")["input_ids"].shape[1])
        required_tokens = render(messages)
        try:
            budgeted = budget_messages(messages, max_input_tokens, render,
                                       lambda text: len(tokenizer.encode(text, add_special_tokens=False)), observation_encoding)
        except Exception as exc:
            # 预算失败是可诊断结果，不能让后续 case 消失。
            rows.append({"case_id": case["case_id"], "condition": case["condition"], "error": str(exc),
                         "full_input_tokens": required_tokens,
                         "usage": {"context_budget_error": True, "prompt_truncated": False,
                                   "lossy_body_trimmed": False, "documents_deleted": 0, "history_dropped": False}})
            continue
        chat = [{"role": item.role, "content": item.content} for item in budgeted.messages]
        final = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        wanted = {str(item).casefold() for item in case["audit"].get("support_titles", [])}
        visible = {str(item.get("title", "")).casefold() for item in budgeted.retained_tool_documents}
        raw = float(case["audit"].get("support_coverage") or 0.0)
        actual = len(wanted & visible) / len(wanted) if wanted else None
        rows.append({"case_id": case["case_id"], "condition": case["condition"],
                     "full_input_tokens": required_tokens,
                     "raw_support_coverage": raw, "visible_support_coverage": actual,
                     "final_prompt_token_ids": final["input_ids"][0].tolist(),
                     "final_attention_mask": final.get("attention_mask")[0].tolist() if final.get("attention_mask") is not None else None,
                     "visible_messages": [item.model_dump(mode="json") for item in budgeted.messages],
                     "document_changes": budgeted.document_changes, "usage": budgeted.usage})
    count = len(rows) or 1
    retention = [row["visible_support_coverage"] / row["raw_support_coverage"] for row in rows
                 if row.get("raw_support_coverage") and row.get("visible_support_coverage") is not None]
    full_lengths = sorted(row["full_input_tokens"] for row in rows)
    percentile = lambda q: full_lengths[min(len(full_lengths)-1, int((len(full_lengths)-1)*q))] if full_lengths else None
    report = {"version": "v0.8.7", "mode": "tokenizer-only", "case_count": len(rows), "model_path": model_path,
              "context_budget": max_input_tokens, "observation_encoding": observation_encoding,
              "metrics": {"legacy_prompt_truncation_rate": sum(bool(row["usage"].get("prompt_truncated")) for row in rows) / count,
                          "lossy_body_trim_rate": sum(bool(row["usage"].get("lossy_body_trimmed")) for row in rows) / count,
                          "documents_deleted": sum(int(row["usage"].get("documents_deleted", 0)) for row in rows),
                          "history_drop_rate": sum(bool(row["usage"].get("history_dropped")) for row in rows) / count,
                          "budget_errors": sum(bool(row["usage"].get("context_budget_error")) for row in rows),
                          "visible_support_retention": sum(retention) / len(retention) if retention else None,
                          "full_input_tokens": {"min": full_lengths[0] if full_lengths else None, "p50": percentile(.5),
                                                "p95": percentile(.95), "max": full_lengths[-1] if full_lengths else None,
                                                "over_budget": sum(value > max_input_tokens for value in full_lengths)}}, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise FileExistsError(f"审计输出已存在，拒绝覆盖：{output}")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="困难停止决策诊断；不更新模型权重")
    parser.add_argument("--data", default="data/hotpot-agent-v041/validation.jsonl")
    parser.add_argument("--train-data", default="data/hotpot-agent-v041/train.jsonl")
    parser.add_argument("--limit", type=int, default=30); parser.add_argument("--corpus-limit", type=int, default=100)
    parser.add_argument("--max-input-tokens", type=int, default=768); parser.add_argument("--output", default="reports/v085-stop-probe")
    parser.add_argument("--observation-encoding", choices=("legacy", "compact-v1", "compact-v2", "compact-v3"), default="legacy")
    parser.add_argument("--frozen-cases", default="reports/v085-stop-768/cases.jsonl")
    parser.add_argument("--adapter", default="checkpoints/qwen35-08b-sft-v052")
    parser.add_argument("--compare-left"); parser.add_argument("--compare-right")
    parser.add_argument("--audit-only", action="store_true"); parser.add_argument("--audit-output")
    args = parser.parse_args()
    if args.compare_left or args.compare_right:
        if not args.compare_left or not args.compare_right: parser.error("比较时必须同时给出 --compare-left 与 --compare-right")
        compare_probe_reports(Path(args.compare_left), Path(args.compare_right), Path(args.output)); return
    split = check_split_overlap(args.train_data, args.data, 100)
    if not split["passed"]: raise SystemExit("训练/诊断分割存在 ID 或问题重叠，已停止，不替换题目")
    cases = load_frozen_cases(args.frozen_cases, args.data)
    if args.audit_only:
        audit_probe_budget(cases, Path(args.audit_output or args.output), r"D:\qwen_08b", args.max_input_tokens, args.observation_encoding); return
    report = asyncio.run(evaluate(cases, Path(args.output), args.adapter, args.max_input_tokens, args.observation_encoding))
    (Path(args.output) / "split-audit.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"停止探针状态={report['status']}；原因={report['reason']}；目录={args.output}")


if __name__ == "__main__":
    main()
