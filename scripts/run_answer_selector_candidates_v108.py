"""Generate frozen evidence-span candidates and report attainable coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from mini_agent_rl.answer_scoring import normalize_answer
from mini_agent_rl.answer_selector import (candidate_messages, correct_option_labels,
    evidence_sha256, ordered_options, parse_candidates)
from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.model.local import (LocalModelConfig, TransformersModelClient,
    generation_terminator_ids, trim_generation_terminators)


def rows(path: Path, table: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        return [json.loads(row[0]) for row in connection.execute(f"SELECT data FROM {table}")]
    finally:
        connection.close()


def split(task_id: str) -> str:
    return "validation" if int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % 5 == 0 else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import torch

    source_hash = hashlib.sha256(args.db.read_bytes()).hexdigest()
    task_by_id = {row["id"]: row for row in rows(args.db, "tasks")}
    rollout_by_task = {row["task_id"]: row for row in rows(args.db, "rollouts")}
    steps_by_rollout: dict[str, list[dict[str, Any]]] = {}
    for row in rows(args.db, "transitions"):
        steps_by_rollout.setdefault(row["rollout_id"], []).append(row)

    client = TransformersModelClient(LocalModelConfig(model_id=args.model_path,
        adapter_path=args.adapter, max_input_tokens=2048, max_new_tokens=128,
        observation_encoding="compact-v3", enable_thinking=False,
        gradient_checkpointing=False, recompute_old_logprobs=False))
    client._load()
    tokenizer = getattr(client._processor, "tokenizer", client._processor)
    terminators = generation_terminator_ids(tokenizer)
    output_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, task_id in enumerate(sorted(rollout_by_task), 1):
        task, rollout = task_by_id[task_id], rollout_by_task[task_id]
        steps = sorted(steps_by_rollout.get(rollout["id"], []), key=lambda item: item["step"])
        row: dict[str, Any] = {"task_id": task_id, "split": split(task_id),
            "question": task["question"], "gold": task["answer"],
            "original": rollout.get("final_answer"), "termination_reason": rollout.get("termination_reason")}
        if not steps or not rollout.get("final_answer") or rollout.get("termination_reason") != "answer":
            row["status"] = "unverifiable_no_final_answer"; output_rows.append(row); continue
        final = steps[-1]
        documents = visible_documents(final.get("messages", []))
        response = final.get("response") or {}
        audit_ok = bool(response.get("prompt_token_ids")) and (
            len(response.get("prompt_token_ids", [])) == len(response.get("prompt_attention_mask", [])))
        row.update({"evidence": documents, "evidence_sha256": evidence_sha256(documents),
                    "source_prompt_audit_ok": audit_ok})
        if not documents or not audit_ok:
            row["status"] = "unverifiable_input"; output_rows.append(row); continue
        try:
            messages = candidate_messages(task["question"], documents, str(rollout["final_answer"]))
            inputs = client._inputs(messages, "v108_candidate_generation")
            client._model.config.use_cache = True
            with torch.inference_mode():
                output = client._model.generate(**inputs, max_new_tokens=128, do_sample=False,
                    eos_token_id=terminators or None)
            token_ids = output[0, inputs["input_ids"].shape[1]:].tolist()
            parse_ids, terminal = trim_generation_terminators(token_ids, tokenizer)
            raw = client._processor.decode(parse_ids, skip_special_tokens=False)
            alternatives, error = parse_candidates(raw, str(rollout["final_answer"]), documents)
            options = ordered_options(task_id, str(rollout["final_answer"]), alternatives)
            labels = correct_option_labels(options, str(task["answer"]))
            original_correct = normalize_answer(str(rollout["final_answer"])) == normalize_answer(str(task["answer"]))
            attainable = labels != ["NONE"]
            row.update({"status": "ok", "raw": raw, "candidate_error": error,
                "alternatives": alternatives, "options": options, "correct_labels": labels,
                "original_correct": original_correct, "attainable_strict": attainable,
                "completion_tokens": len(token_ids), "terminal": terminal,
                "selector_messages": [message.model_dump(mode="json") for message in
                                      __import__("mini_agent_rl.answer_selector", fromlist=["selector_messages"]).selector_messages(task["question"], documents, options)]})
        except Exception as exc:
            row.update({"status": "infrastructure_error", "error": f"{type(exc).__name__}: {exc}"})
        output_rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in output_rows), encoding="utf-8")
        print(f"{index}/{len(rollout_by_task)} {row['status']}", flush=True)
        if row["status"] == "infrastructure_error":
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in output_rows), encoding="utf-8")
    usable = [row for row in output_rows if row["status"] == "ok"]
    wrong = [row for row in usable if not row["original_correct"]]
    report = {"version": "v108", "phase": "candidate_coverage", "source_db_sha256": source_hash,
        "source_db_unchanged": source_hash == hashlib.sha256(args.db.read_bytes()).hexdigest(),
        "adapter": args.adapter, "expected": len(rollout_by_task), "completed": len(output_rows),
        "usable": len(usable), "infrastructure_errors": sum(row["status"] == "infrastructure_error" for row in output_rows),
        "original_strict_correct": sum(row["original_correct"] for row in usable),
        "wrong_original": len(wrong), "wrong_with_strict_correct_alternative": sum(row["attainable_strict"] for row in wrong),
        "strict_repair_coverage": (sum(row["attainable_strict"] for row in wrong) / len(wrong) if wrong else None),
        "attainable_strict_correct": sum(row["attainable_strict"] for row in usable),
        "splits": {name: {"tasks": sum(row["split"] == name for row in usable),
                           "wrong": sum(row["split"] == name and not row["original_correct"] for row in usable),
                           "wrong_covered": sum(row["split"] == name and not row["original_correct"] and row["attainable_strict"] for row in usable)}
                   for name in ("train", "validation")},
        "elapsed_seconds": time.perf_counter() - started, "memory": client.gpu_memory_stats()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
