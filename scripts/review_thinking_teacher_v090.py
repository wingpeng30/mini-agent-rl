"""Fixed, independent evidence-support audit for the thinking-SFT teacher set.

The reviewer only sees the state visible to the policy and the teacher target;
it never receives gold answers, support labels, or the previous acceptance flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx

from mini_agent_rl.training.thinking_sft import _env

REVIEW_SYSTEM = """You independently audit one retrieval-agent target. Return JSON only:
{"supported":true|false,"reason":"short reason"}. Judge whether its stated reasoning and final JSON action are justified solely by the visible messages. A search may be supported when the visible evidence has a real unresolved gap. An answer is supported only when visible tool evidence supports it. Do not use outside knowledge or infer missing documents."""


def records(data: Path) -> list[dict[str, Any]]:
    values = []
    for name in ("train.jsonl", "validation.jsonl"):
        for line in (data / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["thinking_enabled"]:
                values.append(row)
    return values


def fixed_sample(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    # Stable stratification by action, split, and state hash; no result-driven choice.
    # Source split can be obtained deterministically from task id, matching data builder.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        meta = row["metadata"]
        split = "validation" if int(hashlib.sha256(meta["task_id"].encode()).hexdigest(), 16) % 10 == 0 else "train"
        groups.setdefault((split, meta["action"]), []).append(row)
    selected: list[dict[str, Any]] = []
    ordered_groups = sorted(groups)
    base, remainder = divmod(count, len(ordered_groups))
    for group_index, key in enumerate(ordered_groups):
        ordered = sorted(groups[key], key=lambda row: hashlib.sha256((row["metadata"]["task_id"] + ":" + str(row["metadata"]["step"])).encode()).hexdigest())
        quota = base + (1 if group_index < remainder else 0)
        selected.extend(ordered[:quota])
    return selected[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/thinking-sft-v090")
    parser.add_argument("--env-file", default="/root/Tracefix/.env")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", default="reports/thinking-sft-v090-teacher-review-50.jsonl")
    args = parser.parse_args()
    data = Path(args.data); selected = fixed_sample(records(data), args.count)
    audit = {(item["task_id"], int(item["step"])): item for item in (json.loads(line) for line in (data / "teacher-audit.jsonl").read_text(encoding="utf-8").splitlines()) if item.get("accepted")}
    env = _env(Path(args.env_file)); key = env.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing from env file")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with httpx.Client(timeout=90) as client:
        for row in selected:
            meta = row["metadata"]; state = (meta["task_id"], int(meta["step"]))
            teacher = audit[state]["teacher"]
            visible = row["messages"][:-1]
            payload = {"model": "deepseek-flash", "thinking": {"type": "disabled"}, "temperature": 0,
                       "max_tokens": 180, "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": json.dumps({"visible_messages": visible, "teacher_target": teacher}, ensure_ascii=False)}]}
            response = client.post("https://api.deepseek.com/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=payload)
            response.raise_for_status()
            review = json.loads(response.json()["choices"][0]["message"]["content"])
            item = {"task_id": state[0], "step": state[1], "action": meta["action"], "review": review,
                    "teacher_audit_hash": hashlib.sha256(json.dumps(audit[state], ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                    "usage": response.json().get("usage", {})}
            results.append(item)
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results), encoding="utf-8")
    unsupported = sum(not item["review"].get("supported", False) for item in results)
    summary = {"count": len(results), "unsupported": unsupported, "unsupported_rate": unsupported / len(results),
               "expand_to_full": unsupported / len(results) > 0.05, "detail": str(output)}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
