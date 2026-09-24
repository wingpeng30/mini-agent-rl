"""Build evidence-bounded verifier SFT data from frozen v101 trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx

from mini_agent_rl.answer_verifier import VERIFIER_SYSTEM, parse_and_apply, verifier_messages, visible_documents
from mini_agent_rl.training.thinking_sft import _env

BUILD_SYSTEM = """Create verifier supervision using only the question, visible evidence, and candidate answer. Return JSON with actual_action and one hard_negative. actual_action and hard_negative.action each have decision, answer, evidence_refs. decision is keep, replace, or insufficient. keep and insufficient must copy the candidate exactly. replace must give the corrected answer and at least one {id,quote} whose quote occurs verbatim in that document. hard_negative.candidate must be a plausible wrong entity, relation, date, or answer boundary drawn from the visible evidence, and its action should correct it. Never use outside knowledge."""
REVIEW_SYSTEM = """Audit one answer-verifier training record using only the supplied question, evidence, candidate and proposed action. Return JSON {"accepted":true|false,"reason":"..."}. A replacement is accepted only if the question and verbatim cited visible evidence directly support it; keep is accepted only if the candidate is supported; insufficient must preserve the candidate. Never use outside knowledge."""


def table(path: Path, name: str) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        return [json.loads(row[0]) for row in conn.execute(f"SELECT data FROM {name}")]
    finally:
        conn.close()


def split(task_id: str) -> str:
    return "validation" if int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 5 == 0 else "train"


def call(client: httpx.Client, key: str, system: str, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    request = {"model": "deepseek-flash", "thinking": {"type": "disabled"}, "temperature": 0,
               "max_tokens": 700, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
    response = client.post("https://api.deepseek.com/chat/completions",
                           headers={"Authorization": f"Bearer {key}"}, json=request)
    response.raise_for_status()
    body = response.json()
    return json.loads(body["choices"][0]["message"]["content"]), body.get("usage", {})


def sft_row(task_id: str, question: str, docs: list[dict[str, str]], candidate: str,
            action: dict[str, Any], origin: str) -> dict[str, Any]:
    messages = [message.model_dump(mode="json") for message in verifier_messages(question, docs, candidate)]
    messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))})
    return {"messages": messages, "target_assistant_index": len(messages) - 1, "thinking_enabled": False,
            "metadata": {"task_id": task_id, "origin": origin, "decision": action.get("decision"),
                         "evidence_sha256": hashlib.sha256(json.dumps(docs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-cny", type=float, default=10.0)
    args = parser.parse_args()
    before = hashlib.sha256(args.db.read_bytes()).hexdigest()
    rollouts = {row["task_id"]: row for row in table(args.db, "rollouts")}
    tasks = {row["id"]: row for row in table(args.db, "tasks")}
    transitions: dict[str, list[dict[str, Any]]] = {}
    for row in table(args.db, "transitions"):
        transitions.setdefault(row["rollout_id"], []).append(row)
    key = _env(args.env_file).get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    total_input = total_output = 0
    with httpx.Client(timeout=90) as client:
        for index, task_id in enumerate(sorted(rollouts), 1):
            rollout = rollouts[task_id]
            steps = sorted(transitions.get(rollout["id"], []), key=lambda row: row["step"])
            if not steps or rollout.get("termination_reason") != "answer" or not rollout.get("final_answer"):
                rejected.append({"task_id": task_id, "reason": "no_final_answer"}); continue
            question = tasks[task_id]["question"]
            docs = visible_documents(steps[-1].get("messages", []))
            candidate = str(rollout["final_answer"])
            if not docs:
                rejected.append({"task_id": task_id, "reason": "no_visible_evidence"}); continue
            teacher, usage = call(client, key, BUILD_SYSTEM, {"question": question, "evidence": docs, "candidate_answer": candidate})
            total_input += int(usage.get("prompt_tokens", 0)); total_output += int(usage.get("completion_tokens", 0))
            estimated = total_input * 2 / 1_000_000 + total_output * 8 / 1_000_000
            if estimated > args.budget_cny:
                raise RuntimeError(f"DeepSeek budget exceeded: {estimated:.4f} CNY")
            candidates = [(candidate, teacher.get("actual_action"), "actual")]
            hard = teacher.get("hard_negative") or {}
            candidates.append((str(hard.get("candidate", "")), hard.get("action"), "hard_negative"))
            # A supported gold candidate provides a conservative keep example;
            # gold is never included in verifier inputs for any other record.
            gold = str(tasks[task_id].get("answer", ""))
            candidates.append((gold, {"decision": "keep", "answer": gold, "evidence_refs": []}, "gold_keep"))
            for candidate_value, action, origin in candidates:
                if not candidate_value or not isinstance(action, dict):
                    rejected.append({"task_id": task_id, "origin": origin, "reason": "missing_teacher_action"}); continue
                applied = parse_and_apply(json.dumps(action, ensure_ascii=False), candidate_value, docs)
                if applied.action is None or applied.status.startswith("fallback_"):
                    rejected.append({"task_id": task_id, "origin": origin, "reason": applied.status}); continue
                if origin == "hard_negative" and applied.status != "replace":
                    rejected.append({"task_id": task_id, "origin": origin, "reason": "hard_negative_not_replace"}); continue
                review, review_usage = call(client, key, REVIEW_SYSTEM, {"question": question, "evidence": docs,
                    "candidate_answer": candidate_value, "proposed_action": action})
                total_input += int(review_usage.get("prompt_tokens", 0)); total_output += int(review_usage.get("completion_tokens", 0))
                estimated = total_input * 2 / 1_000_000 + total_output * 8 / 1_000_000
                audit.append({"task_id": task_id, "origin": origin, "review": review})
                if estimated > args.budget_cny:
                    raise RuntimeError(f"DeepSeek budget exceeded: {estimated:.4f} CNY")
                if not bool(review.get("accepted")):
                    rejected.append({"task_id": task_id, "origin": origin, "reason": "review_rejected", "detail": review.get("reason")}); continue
                accepted.append(sft_row(task_id, question, docs, candidate_value, action, origin))
            print(f"{index}/{len(rollouts)} accepted={len(accepted)} rejected={len(rejected)}", flush=True)
    by_split = {"train": [], "validation": []}
    for row in accepted:
        by_split[split(row["metadata"]["task_id"])].append(row)
    for name, rows in by_split.items():
        (args.output_dir / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.output_dir / "teacher-audit.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit), encoding="utf-8")
    (args.output_dir / "rejected.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected), encoding="utf-8")
    counts = {name: {decision: sum(row["metadata"]["decision"] == decision for row in rows) for decision in ("keep", "replace", "insufficient")} for name, rows in by_split.items()}
    manifest = {"version": "v107", "source_db_sha256": before, "source_unchanged": before == hashlib.sha256(args.db.read_bytes()).hexdigest(),
                "records": {name: len(rows) for name, rows in by_split.items()}, "decisions": counts,
                "unique_tasks": {name: len({row["metadata"]["task_id"] for row in rows}) for name, rows in by_split.items()},
                "rejected": len(rejected), "reviewed": len(audit), "usage": {"input_tokens": total_input, "output_tokens": total_output,
                "estimated_cny": total_input * 2 / 1_000_000 + total_output * 8 / 1_000_000},
                "system_sha256": hashlib.sha256(VERIFIER_SYSTEM.encode()).hexdigest()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
