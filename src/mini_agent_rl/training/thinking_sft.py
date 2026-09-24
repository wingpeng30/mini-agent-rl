"""Build evidence-bounded teacher data for optional thinking-mode SFT."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx

from ..tools import SearchTool
from ..prompts import DIRECT_SYSTEM, THINK_SYSTEM

TEACHER_SYSTEM = '''Return JSON only: {"reasoning":"one or two short sentences based only on visible evidence","action":{"action":"search","query":"..."} | {"action":"answer","answer":"..."},"evidence_ids":["visible-document-id"]}. Choose the next offline retrieval action. Do not use outside knowledge, future evidence, support labels, or gold answers. Do not repeat a query.'''
# Official peak prices on 2026-09-17, CNY per token.  The ceiling is deliberately
# calculated with the more expensive window so a run remains under budget.
INPUT_CNY_PER_TOKEN, OUTPUT_CNY_PER_TOKEN = 2.0 / 1_000_000, 8.0 / 1_000_000


def _env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1); values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_corpus(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for record in records:
        for message in record["messages"]:
            if message["role"] == "tool":
                for doc in json.loads(message["content"]).get("result", []):
                    docs[doc["id"]] = {key: doc[key] for key in ("id", "title", "text")}
    return [docs[key] for key in sorted(docs)]


def _normal(text: str) -> str: return " ".join(text.casefold().split())
def _split(task_id: str) -> str: return "validation" if int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 10 == 0 else "train"


def _teacher(client: httpx.Client, key: str, model: str, question: str, history: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    # The teacher emits a short structured label.  Disable its hidden reasoning so
    # max_tokens is reserved for the JSON object rather than an invisible trace.
    payload = {"model": model, "temperature": 0, "max_tokens": 260, "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": json.dumps({"question": question, "history": history}, ensure_ascii=False)}]}
    response = client.post("https://api.deepseek.com/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload)
    response.raise_for_status(); raw = response.json()
    return json.loads(raw["choices"][0]["message"].get("content") or "{}"), raw.get("usage", {})


def _row(question: str, history: list[dict[str, Any]], target: str, thinking: bool, metadata: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, str]] = [{"role": "system", "content": THINK_SYSTEM if thinking else DIRECT_SYSTEM}, {"role": "user", "content": question}]
    for item in history:
        messages.extend(({"role": "assistant", "content": json.dumps(item["action"], ensure_ascii=False)}, {"role": "tool", "content": json.dumps({"ok": True, "result": item["result"]}, ensure_ascii=False)}))
    messages.append({"role": "assistant", "content": target})
    return {"messages": messages, "target_assistant_index": len(messages) - 1, "thinking_enabled": thinking, "metadata": metadata}


def generate_thinking_sft(source_data: str | Path, output_dir: str | Path, env_file: str | Path, model: str = "deepseek-flash", task_limit: int = 500, max_steps: int = 3, api_budget_cny: float = 30.0, pilot_only: bool = False) -> dict[str, Any]:
    """Generate only auditable, real-BM25-v2 teacher state/action samples."""
    source, output = Path(source_data), Path(output_dir)
    if output.exists(): raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}")
    key = _env(Path(env_file)).get("DEEPSEEK_API_KEY")
    if not key: raise RuntimeError("未找到 DEEPSEEK_API_KEY")
    records = _load(source); selected = records[:min(task_limit, 20 if pilot_only else task_limit)]
    tool = SearchTool(_source_corpus(records), "bm25-v2"); output.mkdir(parents=True)
    counts = {"train": 0, "validation": 0, "search": 0, "answer": 0, "rejected": 0, "teacher_input_tokens": 0, "teacher_output_tokens": 0}
    with httpx.Client(timeout=90) as client, (output / "teacher-audit.jsonl").open("w", encoding="utf-8") as audit, (output / "train.jsonl").open("w", encoding="utf-8") as train, (output / "validation.jsonl").open("w", encoding="utf-8") as validation:
        for record in selected:
            metadata = record["metadata"]; task_id = metadata["task_id"]; question = next(m["content"] for m in record["messages"] if m["role"] == "user")
            history: list[dict[str, Any]] = []; split = _split(task_id)
            for step in range(max_steps):
                estimated_next = 3_000 * INPUT_CNY_PER_TOKEN + 260 * OUTPUT_CNY_PER_TOKEN
                observed_cost = counts["teacher_input_tokens"] * INPUT_CNY_PER_TOKEN + counts["teacher_output_tokens"] * OUTPUT_CNY_PER_TOKEN
                if observed_cost + estimated_next > api_budget_cny:
                    audit.write(json.dumps({"task_id": task_id, "step": step, "accepted": False, "reason": "api_budget_reached"}) + "\n")
                    break
                try: teacher, usage = _teacher(client, key, model, question, history)
                except Exception as exc:
                    audit.write(json.dumps({"task_id": task_id, "step": step, "accepted": False, "reason": type(exc).__name__}) + "\n"); counts["rejected"] += 1; break
                counts["teacher_input_tokens"] += int(usage.get("prompt_tokens") or 0); counts["teacher_output_tokens"] += int(usage.get("completion_tokens") or 0)
                reasoning, action, ids = teacher.get("reasoning"), teacher.get("action"), teacher.get("evidence_ids")
                visible = {doc["id"] for item in history for doc in item["result"]}; accepted, reason = isinstance(reasoning, str) and 20 <= len(reasoning) <= 600, "reasoning_length"
                if not isinstance(action, dict) or action.get("action") not in {"search", "answer"}: accepted, reason = False, "invalid_action"
                if not isinstance(ids, list) or any(value not in visible for value in ids): accepted, reason = False, "invalid_evidence_ids"
                result: list[dict[str, Any]] = []
                if accepted and action["action"] == "search":
                    query = action.get("query", ""); old = {_normal(item["action"].get("query", "")) for item in history}; result = tool._search(query) if isinstance(query, str) else []
                    if not isinstance(query, str) or not query.strip() or _normal(query) in old or not result: accepted, reason = False, "invalid_repeated_or_empty_search"
                if accepted and action["action"] == "answer":
                    if not isinstance(action.get("answer"), str) or _normal(action["answer"]) != _normal(metadata["gold_answer"]) or not ids: accepted, reason = False, "unsupported_or_incorrect_answer"
                audit.write(json.dumps({"task_id": task_id, "step": step, "teacher": teacher, "usage": usage, "accepted": accepted, "reason": reason, "history_sha256": hashlib.sha256(json.dumps(history, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}, ensure_ascii=False) + "\n")
                if not accepted: counts["rejected"] += 1; break
                action_json = json.dumps(action, ensure_ascii=False, separators=(",", ":")); common = {"task_id": task_id, "step": step, "teacher_model": model, "action": action["action"]}
                rows = (_row(question, history, f"<think>\n{reasoning.strip()}\n</think>\n\n{action_json}", True, common), _row(question, history, action_json, False, common))
                for row in rows: (validation if split == "validation" else train).write(json.dumps(row, ensure_ascii=False) + "\n"); counts[split] += 1
                counts[action["action"]] += 1
                if action["action"] == "answer": break
                history.append({"action": action, "result": result})
    observed_cost = counts["teacher_input_tokens"] * INPUT_CNY_PER_TOKEN + counts["teacher_output_tokens"] * OUTPUT_CNY_PER_TOKEN
    manifest = {"version": "thinking-sft-v090", "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "retriever_version": "bm25-v2", "teacher_model_requested": model, "task_limit": len(selected), "max_steps": max_steps, "api_budget_cny": api_budget_cny, "price_assumption": {"input_cny_per_million": 2.0, "output_cny_per_million": 8.0, "window": "peak"}, "observed_cost_ceiling_cny": observed_cost, "counts": counts, "protocol": "Teacher does not receive gold answers or support-title labels; answer matching occurs only after generation."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
