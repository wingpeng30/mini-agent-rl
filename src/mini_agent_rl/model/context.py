"""结构化上下文预算：保护动作协议与问题，只压缩环境历史。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from ..domain import Message


class ContextBudgetError(RuntimeError):
    """最小 system + user + 最近环境状态也无法装入上下文预算。"""


@dataclass
class ContextBudgetResult:
    messages: list[Message]
    usage: dict[str, int | str | bool]


def _compact_tool(message: Message, max_docs: int, max_text_chars: int) -> Message:
    """按原始相关度顺序压缩工具 JSON，不读取 gold 或 reward 元数据。"""
    try:
        payload = json.loads(message.content)
    except (json.JSONDecodeError, TypeError):
        # 非结构化工具输出只能按字符截断，并明确保留为字符串。
        content = message.content[:max_text_chars] if max_text_chars else ""
        return message.model_copy(update={"content": content})
    if not isinstance(payload, dict):
        return message.model_copy(update={"content": json.dumps(payload, ensure_ascii=False)[:max_text_chars]})
    result = []
    for document in (payload.get("result") or [])[:max_docs]:
        if not isinstance(document, dict):
            continue
        kept = {key: document[key] for key in ("id", "title", "score") if key in document}
        if "text" in document and max_text_chars:
            kept["text"] = str(document["text"])[:max_text_chars]
        result.append(kept)
    compact = {key: payload[key] for key in ("ok", "latency_ms", "searched_queries") if key in payload}
    compact["result"] = result
    return message.model_copy(update={"content": json.dumps(compact, ensure_ascii=False, separators=(",", ":"))})


def _history_groups(messages: list[Message]) -> list[list[Message]]:
    """把 assistant/tool 视为不可拆的历史组，避免产生孤立 tool 消息。"""
    groups: list[list[Message]] = []
    current: list[Message] = []
    for message in messages:
        if message.role == "assistant" and current:
            groups.append(current); current = []
        current.append(message)
        if message.role == "tool":
            groups.append(current); current = []
    if current:
        groups.append(current)
    return groups


def budget_messages(
    messages: list[Message],
    max_tokens: int,
    render_length: Callable[[list[Message]], int],
    content_length: Callable[[str], int],
) -> ContextBudgetResult:
    """返回可渲染的预算化消息和可审计 token 统计。"""
    original_length = render_length(messages)
    system = next((message for message in messages if message.role == "system"), None)
    question = next((message for message in messages if message.role == "user"), None)
    if system is None or question is None:
        raise ContextBudgetError("上下文必须包含 system 动作协议和 user 问题")
    mandatory = [system, question]
    if render_length(mandatory) > max_tokens:
        raise ContextBudgetError("context_budget_error：system 动作协议与 user 问题超过上下文预算")

    def finish(candidate: list[Message], stage: str) -> ContextBudgetResult:
        actual = render_length(candidate)
        original_observation = sum(content_length(m.content) for m in messages if m.role == "tool")
        actual_observation = sum(content_length(m.content) for m in candidate if m.role == "tool")
        usage = {
            "context_budget": max_tokens,
            "original_prompt_tokens": original_length,
            "actual_prompt_tokens": actual,
            "system_tokens": content_length(system.content),
            "question_tokens": content_length(question.content),
            "history_tokens": sum(content_length(m.content) for m in candidate if m.role == "assistant"),
            "observation_tokens": actual_observation,
            "dropped_observation_tokens": max(0, original_observation - actual_observation),
            "prompt_truncated": actual < original_length,
            "truncation_stage": stage,
        }
        return ContextBudgetResult(candidate, usage)

    if original_length <= max_tokens:
        return finish(messages, "none")

    mandatory_ids = {id(system), id(question)}
    history = [message for message in messages if id(message) not in mandatory_ids]
    groups = _history_groups(history)
    # 先保留完整历史，只逐步压缩 observation；文档顺序保持 SearchTool 的相关度顺序。
    strategies = ((3, 160), (3, 96), (2, 64), (1, 48), (1, 0))
    for max_docs, max_chars in strategies:
        compacted = [
            _compact_tool(message, max_docs, max_chars) if message.role == "tool" else message
            for message in history
        ]
        candidate = mandatory + compacted
        if render_length(candidate) <= max_tokens:
            return finish(candidate, "observation_compacted")

    # 仍超限时从最旧历史组开始删除；最新 action/observation 组始终保留。
    minimal_groups = [[_compact_tool(m, 1, 0) if m.role == "tool" else m for m in group] for group in groups]
    for start in range(1, len(minimal_groups)):
        candidate = mandatory + [m for group in minimal_groups[start:] for m in group]
        if render_length(candidate) <= max_tokens:
            return finish(candidate, "history_dropped")
    candidate = mandatory + (minimal_groups[-1] if minimal_groups else [])
    if render_length(candidate) <= max_tokens:
        return finish(candidate, "observation_minimal")
    raise ContextBudgetError("context_budget_error：保留 system、user 和最近环境状态后仍超过上下文预算")
