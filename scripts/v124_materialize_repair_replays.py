"""Materialize separate, non-trainable v117 repair states after CPU replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoProcessor

from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.training.sft import encode_assistant_only
from scripts.prepare_v116_review import PROCESSOR, render
from scripts.v123_cpu_core import ROOT, digest, readl, write
from scripts.v124_forensic import OUT

INPUT = OUT / "repair-draft-replay-inputs-v2-private.jsonl"
OUTPUT = OUT / "repair-draft-states-cpu-v2-private.jsonl"
REPORT = OUT / "repair-draft-state-replay-report-v2-private.json"


def action_at(row: dict) -> dict:
    return json.loads(row["messages"][row["target_assistant_index"]]["content"])


def main() -> None:
    originals = readl(ROOT / "data/evidence-sft-v117/frozen/train.jsonl")
    by_id = {f"v117_train:{i:04d}": r for i, r in enumerate(originals, 1)}
    replay_groups = readl(INPUT)
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    output, summary = [], []
    for group in replay_groups:
        ids = group["record_ids"]
        rows = {}
        for rid in ids:
            if rid not in by_id:
                raise RuntimeError("missing_source_state:" + rid)
            original = by_id[rid]
            row = json.loads(json.dumps(original, ensure_ascii=False))
            row["messages"][1]["content"] = group["question"]
            state = next(s for s in group["states"] if s["record_id"] == rid)
            row["messages"][row["target_assistant_index"]]["content"] = json.dumps(
                state["target_action"], ensure_ascii=False, separators=(",", ":"))
            rows[rid] = row

        state_reports = []
        group_tokens_pass = True
        for rid in ids:
            row = rows[rid]
            target_index = row["target_assistant_index"]
            prompt = row["messages"][:target_index]
            prompt_ids = render(processor, [SimpleNamespace(**m) for m in prompt])
            target = {"role": "assistant", "content": row["messages"][target_index]["content"]}
            encoded = encode_assistant_only(processor, prompt + [target], 2176,
                                            len(prompt), False)
            target_tokens = sum(token != -100 for token in encoded["labels"])
            within = len(prompt_ids) <= 2048 and target_tokens <= 128 and len(encoded["input_ids"]) <= 2176
            group_tokens_pass &= within
            state_report = {"record_id": rid, "original_row_sha256": digest(by_id[rid]),
                "revised_row_sha256": digest(row), "question": group["question"],
                "visible_messages": prompt, "before_action_evidence": visible_documents(prompt),
                "proposed_action": action_at(row), "retrieved_after_search": next(
                    (s["results"] for s in group["search_replays"] if s["record_id"] == rid), None),
                "prompt_tokens": len(prompt_ids), "target_tokens": target_tokens,
                "total_tokens": len(encoded["input_ids"]), "within_token_limits": within,
                "prompt_ids_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
                "target_mask_sha256": hashlib.sha256(json.dumps(encoded["labels"]).encode()).hexdigest(),
                "training_allowed": False}
            state_reports.append(state_report)
        complete_actions = {s["proposed_action"]["action"] for s in state_reports}
        item = {"question": group["question"], "record_ids": ids, "states": state_reports,
                "search_replay_scope": "each action state retains its own exact pre-action context; same-question search and answer states are not assumed to be consecutive turns",
                "changed_searches": [s for s in group["search_replays"] if s["query_changed"]],
                "contains_search_and_answer": {"search", "answer"}.issubset(complete_actions),
                "token_limits_passed": group_tokens_pass,
                "status": "pending_independent_semantic_double_review",
                "training_allowed": False}
        output.append(item)
        summary.append({"record_ids": ids, "token_limits_passed": group_tokens_pass,
            "query_edits": sum(s["query_changed"] for s in group["search_replays"]),
            "search_results_replayed_per_state": True,
            "contains_search_and_answer": item["contains_search_and_answer"],
            "status": item["status"], "training_allowed": False})
    OUTPUT.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                                   for r in output), encoding="utf-8")
    report = {"version": "v124-repair-state-replay-v2", "groups": len(output),
        "states": sum(len(g["states"]) for g in output), "summary": summary,
        "all_original_rows_unchanged": True,
        "semantic_review": "not_run", "independent_double_review": "not_run",
        "training_allowed": False}
    write(REPORT, report)
    print(json.dumps({"groups": len(output), "states": report["states"],
        "search_results_replayed_per_state": True,
        "token_limit_failures": sum(not g["token_limits_passed"] for g in output),
        "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
