"""Generation qualification for every saved thinking-SFT checkpoint.

This is intentionally a standalone, append-only evaluator: it never changes an
adapter and writes raw token boundaries beside every decision.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from mini_agent_rl.domain import Message
from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
from mini_agent_rl.model.local import ActionParseError, parse_agent_action


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_states(path: Path) -> dict[tuple[str, int], dict[bool, dict[str, Any]]]:
    states: dict[tuple[str, int], dict[bool, dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        meta = row["metadata"]
        key = (meta["task_id"], int(meta["step"]))
        states.setdefault(key, {})[bool(row["thinking_enabled"])] = row
    bad = [key for key, pair in states.items() if set(pair) != {False, True}]
    if bad:
        raise ValueError(f"validation pairs are incomplete: {bad[:3]}")
    return states


def expected(row: dict[str, Any]) -> dict[str, Any]:
    text = row["messages"][row["target_assistant_index"]]["content"]
    if row["thinking_enabled"]:
        text = text.split("</think>", 1)[1].strip()
    content, tool = parse_agent_action(text)
    return {"kind": "search" if tool else "answer", "value": tool.arguments["query"] if tool else content}


async def evaluate(adapter: Path, states: dict[tuple[str, int], dict[bool, dict[str, Any]]],
                   output: Path, model_path: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for thinking in (False, True):
        client = TransformersModelClient(LocalModelConfig(
            model_id=model_path, adapter_path=str(adapter), dtype="bfloat16", load_in_4bit=False,
            gradient_checkpointing=False, max_input_tokens=2048, max_new_tokens=512,
            timeout=600, enable_thinking=thinking, deterministic=True, observation_encoding="compact-v3",
        ))
        for ordinal, key in enumerate(sorted(states)):
            row = states[key][thinking]
            messages = [Message.model_validate(item) for item in row["messages"][:-1]]
            started = time.perf_counter()
            try:
                response = await client.generate(messages, [], temperature=0.0, seed=42)
                observed = {"kind": "invalid", "value": None}
                if not response.parse_error:
                    observed = {"kind": "search" if response.tool_calls else "answer",
                                "value": response.tool_calls[0].arguments["query"] if response.tool_calls else response.content}
                direct_json = bool(thinking and response.reasoning_text is None and response.raw_text.lstrip().startswith("{"))
                item = {"task_id": key[0], "step": key[1], "ordinal": ordinal, "thinking": thinking,
                        "expected": expected(row), "observed": observed, "valid": response.parse_error is None,
                        "parse_error": response.parse_error, "raw_text": response.raw_text,
                        "action_token_ids": response.action_token_ids, "prompt_token_ids": response.prompt_token_ids,
                        "prompt_attention_mask": response.prompt_attention_mask,
                        "action_mask": response.action_mask, "old_logprobs": response.old_logprobs,
                        "usage": response.usage, "direct_json_degradation": direct_json,
                        "elapsed_seconds": round(time.perf_counter() - started, 4)}
            except Exception as exc:  # retain infrastructure failures as data
                item = {"task_id": key[0], "step": key[1], "ordinal": ordinal, "thinking": thinking,
                        "expected": expected(row), "observed": {"kind": "error", "value": None}, "valid": False,
                        "parse_error": f"infrastructure: {type(exc).__name__}: {exc}", "raw_text": "",
                        "action_token_ids": [], "prompt_token_ids": [], "prompt_attention_mask": [],
                        "action_mask": [], "old_logprobs": [], "usage": {}, "direct_json_degradation": False,
                        "elapsed_seconds": round(time.perf_counter() - started, 4)}
            records.append(item)
    detail = output.with_suffix(".jsonl")
    detail.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    modes: dict[str, Any] = {}
    for thinking in (False, True):
        rows = [item for item in records if item["thinking"] == thinking]
        answer = [item for item in rows if item["expected"]["kind"] == "answer"]
        modes["thinking" if thinking else "direct"] = {
            "count": len(rows), "legal_rate": sum(item["valid"] for item in rows) / len(rows),
            "budget_exhaustion_rate": sum(bool(item["usage"].get("output_truncated")) for item in rows) / len(rows),
            "answer_em": (sum(item["valid"] and item["observed"] == item["expected"] for item in answer) / len(answer)),
            "valid_search": sum(item["valid"] and item["observed"]["kind"] == "search" for item in rows),
            "valid_answer": sum(item["valid"] and item["observed"]["kind"] == "answer" for item in rows),
            "direct_json_degradation": sum(item["direct_json_degradation"] for item in rows),
            "infrastructure_errors": sum(str(item["parse_error"]).startswith("infrastructure:") for item in rows),
            "sequence_audit_errors": sum(
                len(item["prompt_token_ids"]) != len(item["prompt_attention_mask"])
                or len(item["action_token_ids"]) != len(item["action_mask"])
                or len(item["action_token_ids"]) != len(item["old_logprobs"])
                or any(not math.isfinite(value) for value in item["old_logprobs"])
                for item in rows
            ),
            "mean_latency_seconds": sum(item["elapsed_seconds"] for item in rows) / len(rows),
        }
    report = {"adapter": str(adapter), "adapter_sha256": sha256(next(adapter.glob("adapter_model.*"))),
              "detail": str(detail), "detail_sha256": sha256(detail), "modes": modes}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def checkpoint_eval_loss(path: Path) -> float:
    state = path / "trainer_state.json"
    if not state.exists():
        return float("inf")
    history = json.loads(state.read_text(encoding="utf-8")).get("log_history", [])
    values = [item["eval_loss"] for item in history if "eval_loss" in item]
    return values[-1] if values else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/thinking-sft-v090/validation.jsonl")
    parser.add_argument("--adapter-root", default="checkpoints/qwen35-08b-thinking-sft-v090")
    parser.add_argument("--model", default="/models/qwen_08b")
    parser.add_argument("--output-dir", default="reports/thinking-sft-v090-qualification")
    args = parser.parse_args()
    data = Path(args.data); root = Path(args.adapter_root); out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    states = read_states(data)
    candidates = sorted([path for path in root.glob("checkpoint-*") if (path / "adapter_config.json").exists()],
                        key=lambda path: int(path.name.split("-")[-1]))
    candidates.append(root)
    reports = []
    for candidate in candidates:
        report = asyncio.run(evaluate(candidate, states, out / f"{candidate.name}.json", args.model))
        report["validation_eval_loss"] = checkpoint_eval_loss(candidate)
        reports.append(report)
    ranked = sorted(reports, key=lambda item: (-item["modes"]["thinking"]["legal_rate"],
                                                -item["modes"]["thinking"]["answer_em"],
                                                item["validation_eval_loss"], int(Path(item["adapter"]).name.split("-")[-1]) if "checkpoint-" in item["adapter"] else 10**9))
    summary = {"data": str(data), "data_sha256": sha256(data), "state_count": len(states),
               "selection_rule": "thinking legal_rate desc, answer_em desc, validation eval_loss asc, earlier checkpoint",
               "candidates": reports, "selected": ranked[0]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
