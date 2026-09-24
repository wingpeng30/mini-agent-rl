"""结构化上下文预算：协议、问题与证据变化均可审计。"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from ..domain import Message

ObservationEncoding = Literal["legacy", "compact-v1", "compact-v2", "compact-v3"]


class ContextBudgetError(RuntimeError):
    """最小输入或完整可引用证据无法放进给定预算。"""


@dataclass
class ContextBudgetResult:
    messages: list[Message]
    usage: dict[str, Any]
    retained_tool_documents: list[dict[str, Any]]
    document_changes: list[dict[str, Any]]


def _sentence_prefix(text: str, limit: int) -> tuple[str, tuple[int, int] | None]:
    """只在完整句子边界缩短正文，并返回可从原文重建的偏移。"""
    if len(text) <= limit:
        return text, (0, len(text))
    matches = list(re.finditer(r"[.!?。！？](?:\s|$)", text[:limit]))
    if not matches:
        return "", None
    end = matches[-1].end()
    return text[:end].strip(), (0, end)


def _doc_key(document: dict[str, Any], index: int) -> tuple[tuple[str, str, str] | None, str, str, str]:
    """v2 引用同时确认 ID、标题和正文；缺失 ID 绝不自动合并。"""
    raw_id = document.get("id")
    doc_id = str(raw_id).strip() if raw_id is not None else ""
    title, text = str(document.get("title", "")), str(document.get("text", ""))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ((doc_id, title, digest) if doc_id else None), doc_id or f"anonymous-{index}", title, text


def _payload(message: Message) -> dict[str, Any] | None:
    try:
        value = json.loads(message.content)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _encode_tool(message: Message, *, encoding: ObservationEncoding, text_cap: int | None,
                 seen: dict[tuple[str, str, str], str], rank_caps: tuple[int | None, ...] | None = None,
                 tool_index: int = 0) -> tuple[Message, list[dict[str, Any]], list[dict[str, Any]]]:
    """编码工具结果；compact-v2 不改变排名，且不使用 gold 信息。"""
    payload = _payload(message)
    if payload is None:
        return message.model_copy(update={"content": json.dumps({"ok": False, "error": "invalid_tool_json"}, ensure_ascii=False)}), [], []
    result: list[dict[str, Any]] = []
    visible: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for rank, raw in enumerate(payload.get("result") or [], 1):
        if not isinstance(raw, dict):
            continue
        key, doc_id, title, text = _doc_key(raw, rank)
        change: dict[str, Any] = {"document_id": doc_id, "rank": rank, "title": title, "original_text": text,
            "original_text_chars": len(text), "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "visible": False, "reused_reference": False, "body_trimmed": False, "deleted": False,
            "content_conflict": False, "visible_segments": [], "reference_target": None}
        if encoding == "compact-v1" and doc_id and (doc_id, title, hashlib.sha256(text.encode("utf-8")).hexdigest()) in seen:
            encoded = {"id": doc_id, "ref": True}
            change.update(visible=True, reused_reference=True, visible_text_chars=0)
        elif encoding in ("compact-v2", "compact-v3") and key is not None and key in seen:
            target = seen[key]; encoded = {"id": doc_id, "ref": target}
            change.update(visible=True, reused_reference=True, reference_target=target, visible_text_chars=0)
        else:
            if encoding == "compact-v1" and doc_id:
                seen[(doc_id, title, hashlib.sha256(text.encode("utf-8")).hexdigest())] = doc_id
            if encoding in ("compact-v2", "compact-v3") and key is not None:
                change["content_conflict"] = any(old[0] == key[0] and old != key for old in seen)
            cap = rank_caps[rank - 1] if rank_caps is not None and rank <= len(rank_caps) else text_cap
            body, segment = _sentence_prefix(text, cap) if cap is not None else (text, (0, len(text)))
            if text and not body:
                change.update(deleted=True, budget_unsatisfied=True); changes.append(change); continue
            change.update(visible=True, visible_text_chars=len(body), visible_segments=[list(segment)] if segment else [],
                          body_trimmed=bool(text and len(body) < len(text)))
            encoded = {"id": doc_id, "title": title, "text": body}
            if encoding == "legacy" and "score" in raw:
                encoded["score"] = raw["score"]
            if encoding in ("compact-v2", "compact-v3") and key is not None:
                target = f"{doc_id}:{key[2][:12]}"; seen[key] = target
                encoded["source"] = target; change["reference_target"] = target
        change["tool_index"] = tool_index
        result.append(encoded); visible.append({"id": doc_id, "title": title,
            "original_text_chars": len(text), "visible_text_chars": int(change.get("visible_text_chars", 0))}); changes.append(change)
    compact = {name: payload[name] for name in ("ok", "searched_queries") if name in payload}
    compact["result"] = result
    return message.model_copy(update={"content": json.dumps(compact, ensure_ascii=False, separators=(",", ":"))}), visible, changes


def budget_messages(messages: list[Message], max_tokens: int, render_length: Callable[[list[Message]], int],
                    content_length: Callable[[str], int], observation_encoding: ObservationEncoding = "legacy") -> ContextBudgetResult:
    """预算输入。v2 不删除动作历史或文档，无法装入时明确失败。"""
    original_length = render_length(messages)
    system = next((m for m in messages if m.role == "system"), None)
    question = next((m for m in messages if m.role == "user"), None)
    if system is None or question is None:
        raise ContextBudgetError("上下文必须包含 system 与 user 问题")
    mandatory = [system, question]
    if render_length(mandatory) > max_tokens:
        raise ContextBudgetError("context_budget_error：system 与 user 超过预算")
    required_ids = {id(system), id(question)}
    history = [m for m in messages if id(m) not in required_ids]

    def finish(candidate: list[Message], visible: list[dict[str, Any]], changes: list[dict[str, Any]], stage: str) -> ContextBudgetResult:
        actual = render_length(candidate); raw_obs = sum(content_length(m.content) for m in messages if m.role == "tool")
        seen_obs = sum(content_length(m.content) for m in candidate if m.role == "tool")
        trimmed = sum(bool(c.get("body_trimmed")) for c in changes); deleted = sum(bool(c.get("deleted")) for c in changes)
        usage = {"context_budget": max_tokens, "original_prompt_tokens": original_length, "actual_prompt_tokens": actual,
            "system_tokens": content_length(system.content), "question_tokens": content_length(question.content),
            "history_tokens": sum(content_length(m.content) for m in candidate if m.role == "assistant"), "observation_tokens": seen_obs,
            "dropped_observation_tokens": max(0, raw_obs-seen_obs), "prompt_truncated": actual < original_length,
            "truncation_stage": stage, "observation_encoding": observation_encoding,
            "lossless_expression_changed": observation_encoding.startswith("compact") and actual < original_length and not trimmed and not deleted,
            "lossy_body_trimmed": trimmed > 0, "documents_deleted": deleted,
            "reused_document_references": sum(bool(c.get("reused_reference")) for c in changes),
            "observation_compacted": stage == "observation_compacted", "history_dropped": False, "context_budget_error": False}
        return ContextBudgetResult(candidate, usage, visible, changes)

    # v0.8.5 legacy：能容纳时绝不改写原消息。
    if observation_encoding == "legacy" and original_length <= max_tokens:
        return finish(messages, [], [], "none")

    def encoded(cap: int | None, max_docs: int | None = None,
                rank_caps: tuple[int | None, ...] | None = None):
        seen: dict[tuple[str, str, str], str] = {}; out: list[Message] = []; visible: list[dict[str, Any]] = []; changes: list[dict[str, Any]] = []
        tool_index = 0
        for message in history:
            if message.role == "tool":
                value, docs, audit = _encode_tool(message, encoding=observation_encoding, text_cap=cap, seen=seen,
                                                  rank_caps=rank_caps, tool_index=tool_index)
                tool_index += 1
                if max_docs is not None:
                    # 仅为历史 legacy 回退保留：旧版会删除低排名文档；v2 永不使用。
                    tool = _payload(value) or {}; kept = (tool.get("result") or [])[:max_docs]
                    for change in audit[max_docs:]: change.update(visible=False, deleted=True)
                    tool["result"] = kept
                    value = value.model_copy(update={"content": json.dumps(tool, ensure_ascii=False, separators=(",", ":"))})
                    docs = docs[:max_docs]
                out.append(value); visible += docs; changes += audit
            else:
                out.append(message)
        return out, visible, changes

    if observation_encoding == "compact-v3":
        # 每个阶段先缩短 rank-3，再缩短 rank-2，最后才触及 rank-1；每次均用
        # chat template 重算真实 token 长度，避免用字符数推断是否已经满足预算。
        strategies = ((None, None, None), (None, None, 320), (None, None, 192),
                      (None, 320, 192), (None, 192, 128), (320, 192, 128),
                      (192, 128, 96), (128, 96, 64), (96, 64, 48))
        for rank_caps in strategies:
            transformed, visible, changes = encoded(None, rank_caps=rank_caps)
            if any(c.get("budget_unsatisfied") for c in changes):
                continue
            candidate = mandatory + transformed
            if render_length(candidate) <= max_tokens:
                return finish(candidate, visible, changes,
                              "none" if rank_caps == (None, None, None) else "observation_compacted")
        raise ContextBudgetError("context_budget_error：低排名优先裁剪后仍无法保留完整动作历史与每篇至少一句证据")

    caps = (None, 320, 192, 128, 96, 64, 48) if observation_encoding == "compact-v2" else (None, 256, 160, 96, 64, 48)
    for cap in caps:
        transformed, visible, changes = encoded(cap)
        if any(c.get("budget_unsatisfied") for c in changes):
            continue
        candidate = mandatory + transformed
        if render_length(candidate) <= max_tokens:
            return finish(candidate, visible, changes, "none" if cap is None else "observation_compacted")
    if observation_encoding == "legacy":
        for docs, cap in ((2, 64), (1, 48), (1, 0)):
            transformed, visible, changes = encoded(cap, docs)
            if render_length(mandatory + transformed) <= max_tokens:
                return finish(mandatory + transformed, visible, changes, "observation_compacted")
    raise ContextBudgetError("context_budget_error：完整动作历史与每篇至少一句证据仍无法放入预算")
