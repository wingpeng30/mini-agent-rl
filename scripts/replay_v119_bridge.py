"""One-step v118 replay after an observed v112 bridge search on shared inputs.

The five cases are selected retrospectively from exposed internal results, so
this is a mechanism diagnostic only. V112's recorded search and observation
are inserted into v118's history; gold is used only after model generation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from mini_agent_rl.domain import Message
from mini_agent_rl.evaluation import answer_metrics
from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient
from mini_agent_rl.tools import SearchTool, ToolRegistry
from scripts.compare_v118_internal import ensure_same
from scripts.merge_v118_semantic_reviews import sha

EXPECTED_V118_ADAPTER = "ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    target = Path(path)
    temporary = target.with_name(target.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def model_files(path):
    root = Path(path)
    return {file.relative_to(root).as_posix(): sha(file) for file in sorted(root.rglob("*"))
            if file.is_file() and ".cache" not in file.relative_to(root).parts}


def selected_pairs(v112, v118):
    """Require identical decision input before the observed second-search split."""
    a = {r["task_id"]: r for r in v112["rows"]}
    b = {r["task_id"]: r for r in v118["rows"]}
    if len(a) != 50 or set(a) != set(b):
        raise ValueError("paired_50_required")
    pairs = []
    for task_id, ref in a.items():
        new = b[task_id]
        if ref["normalized_exact_match"] != 1 or new["normalized_exact_match"] == 1:
            continue
        if len(ref["transitions"]) < 3 or len(new["transitions"]) < 2:
            continue
        first_ref, second_ref, third_ref = ref["transitions"][:3]
        first_new, second_new = new["transitions"][:2]
        if (first_ref["response"]["prompt_token_ids"] != first_new["response"]["prompt_token_ids"] or
                second_ref["response"]["prompt_token_ids"] != second_new["response"]["prompt_token_ids"]):
            continue
        if (not second_ref["response"].get("tool_calls") or second_new["response"].get("tool_calls") or
                not second_new["response"].get("content", "").strip() or
                third_ref["response"].get("tool_calls")):
            continue
        observation = second_ref.get("observation") or {}
        if not observation.get("ok") or not observation.get("result"):
            continue
        pairs.append({"task_id": task_id,
                      "pre_split_prompt_sha256": second_ref["response"]["usage"]["prompt_sha256"],
                      "bridge_query": second_ref["response"]["tool_calls"][0]["arguments"]["query"],
                      "bridge_result_document_ids": [d["id"] for d in observation["result"]],
                      "v112_next_prompt_token_ids": third_ref["response"]["prompt_token_ids"],
                      "v112_next_prompt_sha256": third_ref["response"]["usage"]["prompt_sha256"],
                      "counterfactual_messages": third_ref["messages"]})
    return pairs


async def run(args):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("determinism_environment_required_before_process_start")
    ref_path, new_path, corpus_path = map(Path, (args.v112_tasks, args.v118_tasks, args.corpus))
    ref, new = read(ref_path), read(new_path)
    ensure_same([ref, new], "tasks")
    if sha(corpus_path) != new["corpus_sha256"]:
        raise ValueError("corpus_hash_mismatch")
    adapter_path = Path(args.adapter_path)
    if sha(adapter_path / "adapter_model.safetensors") != EXPECTED_V118_ADAPTER:
        raise ValueError("v118_adapter_hash_mismatch")
    if model_files(args.model_path) != new["model_files"]:
        raise ValueError("base_model_file_hash_mismatch")
    pairs = selected_pairs(ref, new)
    if len(pairs) != 5:
        raise ValueError("expected_five_shared_input_bridge_cases")
    answers = {row["id"]: row["answer"] for row in
               (json.loads(line) for line in Path(args.private_tasks).read_text(encoding="utf-8").splitlines() if line.strip())}
    if set(p["task_id"] for p in pairs) - set(answers):
        raise ValueError("missing_scoring_answer")
    cfg = LocalModelConfig(model_id=args.model_path, adapter_path=args.adapter_path,
                           device="cuda", dtype="bfloat16", load_in_4bit=False,
                           max_input_tokens=2048, max_new_tokens=128, enable_thinking=False,
                           timeout=600, gradient_checkpointing=False,
                           observation_encoding="compact-v3", deterministic=True,
                           local_files_only=True, trust_remote_code=False,
                           logprob_mode="bf16")
    client = TransformersModelClient(cfg)
    corpus = read(corpus_path)
    registry = ToolRegistry()
    registry.register(SearchTool(corpus, "bm25-v2"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    initial = {"version": "v119-bridge-replay-v1", "exploratory_development_only": True,
              "source_sha256": {"v112_tasks": sha(ref_path), "v118_tasks": sha(new_path),
                                "private_tasks": sha(Path(args.private_tasks)), "corpus": sha(corpus_path)},
              "adapter_sha256": EXPECTED_V118_ADAPTER, "rows": [], "complete_coverage": False,
              "independent100_opened": False, "final_test_accessed": False}
    report = read(output) if output.exists() else initial
    if any(report.get(key) != initial[key] for key in
           ("version", "source_sha256", "adapter_sha256", "independent100_opened", "final_test_accessed")):
        raise ValueError("resume_source_or_adapter_changed")
    completed = {row["task_id"] for row in report["rows"]}
    if len(completed) != len(report["rows"]) or not completed <= {pair["task_id"] for pair in pairs}:
        raise ValueError("invalid_resumed_case_set")
    for pair in pairs:
        if pair["task_id"] in completed:
            continue
        response = await client.generate([Message(**m) for m in pair["counterfactual_messages"]],
                                         registry.definitions(), temperature=0, seed=42)
        actual_ids = response.model_dump().get("prompt_token_ids")
        actual_sha = response.usage.get("prompt_sha256")
        if actual_ids != pair["v112_next_prompt_token_ids"] or actual_sha != pair["v112_next_prompt_sha256"]:
            raise ValueError("counterfactual_prompt_token_or_hash_mismatch")
        score = answer_metrics(response.content.strip(), answers[pair["task_id"]]) if response.content.strip() else None
        report["rows"].append({"task_id": pair["task_id"],
                               "pre_split_prompt_sha256": pair["pre_split_prompt_sha256"],
                               "bridge_query": pair["bridge_query"],
                               "bridge_result_document_ids": pair["bridge_result_document_ids"],
                               "counterfactual_prompt_sha256": actual_sha,
                               "action": "search" if response.tool_calls else "answer" if response.content.strip() else "invalid",
                               "output": response.raw_text,
                               "normalized_exact_match_if_answer": score["normalized_exact_match"] if score else None,
                               "token_f1_if_answer": score["token_f1"] if score else None})
        save(output, report)
    report["complete_coverage"] = len(report["rows"]) == 5
    report["answered_correct"] = sum(row["normalized_exact_match_if_answer"] == 1 for row in report["rows"])
    save(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("v112-tasks", "v118-tasks", "private-tasks", "corpus", "model-path", "adapter-path", "output"):
        parser.add_argument("--" + key, required=True)
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps({"version": result["version"], "complete_coverage": result["complete_coverage"],
                      "answered_correct": result["answered_correct"]}, ensure_ascii=False))
