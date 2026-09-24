"""Reconstruct frozen prompts offline; expose evidence only after exact replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.domain import Message
from mini_agent_rl.hybrid_answer import evidence_hash
from mini_agent_rl.model.context import budget_messages


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_hash(ids):
    """Exactly the serialization in model/local.py::_inputs."""
    return hashlib.sha256(bytes(str(ids), "utf-8")).hexdigest()


def unique(rows, key):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError(f"duplicate {key}: {row[key]}")
        result[row[key]] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("db", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--observation-encoding", default="compact-v3")
    args = parser.parse_args()
    from transformers import AutoProcessor

    before = file_hash(args.db)
    connection = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        tables = {name: [json.loads(row[0]) for row in connection.execute(f"SELECT data FROM {name}")]
                  for name in ("tasks", "rollouts", "transitions")}
    finally:
        connection.close()
    tasks = unique(tables["tasks"], "id")
    rollouts = unique(tables["rollouts"], "task_id")
    grouped = {}
    for transition in tables["transitions"]:
        grouped.setdefault(transition["rollout_id"], []).append(transition)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    def render(messages):
        encoded = processor.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=False)
        ids = encoded["input_ids"][0].tolist()
        mask = encoded["attention_mask"][0].tolist() if "attention_mask" in encoded else [1] * len(ids)
        return ids, mask

    records = []
    for task_id in sorted(tasks.keys() - rollouts.keys()):
        records.append({"task_id": task_id, "provenance_valid": False,
                        "prompt_replay_exact": False, "e0": [], "e1": [],
                        "provenance_error": "missing rollout for source task"})
    for task_id, rollout in sorted(rollouts.items()):
        row = {"task_id": task_id, "rollout_id": rollout["id"], "provenance_valid": False,
               "prompt_replay_exact": False, "e0": [], "e1": []}
        try:
            steps = sorted(grouped[rollout["id"]], key=lambda x: x["step"])
            unique(steps, "step")
            final = steps[-1]
            messages = [Message.model_validate(m) for m in final["messages"]]
            budgeted = budget_messages(messages, args.max_input_tokens,
                                       lambda ms: len(render(ms)[0]),
                                       lambda s: len(tokenizer.encode(s, add_special_tokens=False)),
                                       args.observation_encoding)
            ids, mask = render(budgeted.messages)
            response = final.get("response") or {}
            saved_ids = response.get("prompt_token_ids")
            saved_mask = response.get("prompt_attention_mask")
            saved_hash = (response.get("usage") or {}).get("prompt_sha256")
            row.update(question=tasks[task_id]["question"], gold=tasks[task_id]["answer"],
                       original_answer=str(rollout.get("final_answer") or ""),
                       prompt_token_ids=saved_ids, prompt_attention_mask=saved_mask,
                       replayed_prompt_token_ids=ids, replayed_prompt_attention_mask=mask,
                       saved_prompt_sha256=saved_hash, replayed_prompt_sha256=prompt_hash(ids),
                       actual_prompt_sha256=prompt_hash(saved_ids),
                       prompt_attention_mask_sha256=prompt_hash(saved_mask),
                       replayed_attention_mask_sha256=prompt_hash(mask), budget_usage=budgeted.usage)
            retained = (response.get("usage") or {}).get("retained_tool_documents")
            row.update(saved_retained_tool_documents=retained,
                       replayed_retained_tool_documents=budgeted.retained_tool_documents,
                       retained_tool_documents_sha256=evidence_hash(budgeted.retained_tool_documents),
                       document_changes=budgeted.document_changes,
                       replayed_messages=[m.model_dump(mode="json") for m in budgeted.messages])
            if not saved_ids or not isinstance(saved_ids, list) or not all(type(x) is int for x in saved_ids):
                raise ValueError("missing or invalid saved prompt IDs")
            if not isinstance(saved_mask, list) or not all(type(x) is int for x in saved_mask):
                raise ValueError("missing or invalid saved prompt mask")
            if ids != saved_ids or mask != saved_mask or len(ids) != len(mask):
                raise ValueError("prompt ID or attention mask replay mismatch")
            if saved_hash != prompt_hash(saved_ids) or saved_hash != prompt_hash(ids):
                raise ValueError("saved prompt hash missing or mismatched")
            if retained is None or retained != budgeted.retained_tool_documents:
                raise ValueError("retained tool document provenance missing or mismatched")
            e0 = visible_documents(budgeted.messages)
            e1 = visible_documents(messages)
            row.update(provenance_valid=True, prompt_replay_exact=True, e0=e0, e1=e1,
                       e0_sha256=evidence_hash(e0), e1_sha256=evidence_hash(e1),
                       evidence_provenance="verified_prompt_messages", source_db_sha256=before)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            row["provenance_error"] = f"{type(exc).__name__}: {exc}"
        records.append(row)
    after = file_hash(args.db)
    if before != after:
        for row in records:
            row.update(provenance_valid=False, e0=[], e1=[], provenance_error="source database changed")
            row.pop("e0_sha256", None)
            row.pop("e1_sha256", None)
    report = {"version": "v110", "source_db": str(args.db), "source_db_sha256": before,
              "source_db_sha256_after": after, "source_unchanged": before == after,
              "tasks": len(records), "exact_prompt_replay": sum(r["prompt_replay_exact"] for r in records),
              "provenance_valid": sum(r["provenance_valid"] for r in records),
              "errors": [{"task_id": r["task_id"], "error": r["provenance_error"]}
                         for r in records if "provenance_error" in r],
              "e0_e1_different": sum(r.get("e0_sha256") != r.get("e1_sha256") for r in records),
              "model_path": args.model_path, "max_input_tokens": args.max_input_tokens,
              "observation_encoding": args.observation_encoding, "enable_thinking": False}
    for path in (args.output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not records or report["provenance_valid"] != len(records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
