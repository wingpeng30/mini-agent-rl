"""Measure v118's native stop choice and its query/retrieval ability when search is requested.

The five cases are a retrospective development subset. The diagnostic search
instruction is an intervention; its results are not an estimate of native
policy accuracy or a causal end-to-end benchmark.
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
from scripts.replay_v119_bridge import EXPECTED_V118_ADAPTER, model_files, read, save, selected_pairs


SEARCH_DIAGNOSTIC = (
    "For this one-step diagnostic, do not answer yet. Use exactly one search action now. "
    "Form a concise query from the original question and the evidence already visible. "
    "After the search result is shown, answer the original question using the evidence."
)


def action_name(response):
    if response.tool_calls:
        if len(response.tool_calls) != 1:
            return "invalid_multiple_tools"
        return response.tool_calls[0].name
    return "answer" if response.content.strip() else "invalid_empty"


def search_observation_documents(observation):
    result = observation.get("result") or []
    if isinstance(result, dict):
        result = result.get("result") or []
    return result if isinstance(result, list) else []


def context_audit(response):
    usage = response.usage or {}
    return {key: usage.get(key) for key in (
        "prompt_tokens", "prompt_truncated", "dropped_observation_tokens",
        "observation_compacted", "history_dropped", "context_budget_error",
    )}


def require_unmodified_evidence_context(response, phase):
    usage = context_audit(response)
    if (usage["prompt_truncated"] or usage["dropped_observation_tokens"] or
            usage["observation_compacted"] or usage["history_dropped"] or usage["context_budget_error"]):
        raise ValueError(f"{phase}_changed_or_truncated_evidence_context:{usage}")
    return usage


async def run(args):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("determinism_environment_required_before_process_start")
    ref_path, new_path, corpus_path = map(Path, (args.v112_tasks, args.v118_tasks, args.corpus))
    ref, new = read(ref_path), read(new_path)
    ensure_same([ref, new], "tasks")
    if sha(corpus_path) != new["corpus_sha256"]:
        raise ValueError("corpus_hash_mismatch")
    if sha(Path(args.adapter_path) / "adapter_model.safetensors") != EXPECTED_V118_ADAPTER:
        raise ValueError("v118_adapter_hash_mismatch")
    if model_files(args.model_path) != new["model_files"]:
        raise ValueError("base_model_file_hash_mismatch")

    pairs = selected_pairs(ref, new)
    if len(pairs) != 5:
        raise ValueError("expected_five_shared_input_cases")
    answers = {
        row["id"]: row["answer"]
        for row in (json.loads(line) for line in Path(args.private_tasks).read_text(encoding="utf-8").splitlines() if line.strip())
    }
    new_by_id = {row["task_id"]: row for row in new["rows"]}
    if set(pair["task_id"] for pair in pairs) - set(answers):
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
    identity = {
        "version": "v119-self-query-v1",
        "source_sha256": {"v112_tasks": sha(ref_path), "v118_tasks": sha(new_path),
                          "private_tasks": sha(Path(args.private_tasks)), "corpus": sha(corpus_path)},
        "adapter_sha256": EXPECTED_V118_ADAPTER,
        "independent100_opened": False,
        "final_test_accessed": False,
        "rows": [],
    }
    report = read(output) if output.exists() else identity
    if any(report.get(key) != value for key, value in identity.items() if key != "rows"):
        raise ValueError("resume_source_or_adapter_changed")
    allowed = {pair["task_id"] for pair in pairs}
    completed = {row["task_id"] for row in report["rows"]}
    if len(completed) != len(report["rows"]) or not completed <= allowed:
        raise ValueError("invalid_resumed_case_set")

    for pair in pairs:
        task_id = pair["task_id"]
        if task_id in completed:
            continue
        transition = new_by_id[task_id]["transitions"][1]
        native_messages = [Message(**message) for message in transition["messages"]]
        native = await client.generate(native_messages, registry.definitions(), temperature=0, seed=42)
        native_ids = native.model_dump().get("prompt_token_ids")
        native_sha = native.usage.get("prompt_sha256")
        expected = transition["response"]
        if native_ids != expected["prompt_token_ids"] or native_sha != expected["usage"].get("prompt_sha256"):
            raise ValueError("native_input_token_or_hash_mismatch")
        native_score = answer_metrics(native.content.strip(), answers[task_id]) if action_name(native) == "answer" else None

        # Native policy may stop before querying. This explicit diagnostic
        # intervention measures query construction conditional on searching.
        query_messages = native_messages + [Message(role="user", content=SEARCH_DIAGNOSTIC)]
        query_response = await client.generate(query_messages, registry.definitions(), temperature=0, seed=42)
        diagnostic_context = require_unmodified_evidence_context(query_response, "diagnostic_query")
        query = None
        documents = []
        answer_response = None
        observation = None
        if action_name(query_response) == "search" and query_response.tool_calls[0].arguments.get("query", "").strip():
            query = query_response.tool_calls[0].arguments["query"].strip()
            observation = await registry.execute("search", {"query": query})
            documents = search_observation_documents(observation)
            model_observation = {key: value for key, value in observation.items() if key != "latency_ms"}
            query_messages += [
                Message(role="assistant", content=json.dumps({"action": "search", "query": query}, ensure_ascii=False),
                        tool_calls=query_response.tool_calls),
                Message(role="tool", content=json.dumps(model_observation, ensure_ascii=False),
                        tool_call_id=query_response.tool_calls[0].id),
            ]
            answer_response = await client.generate(query_messages, registry.definitions(), temperature=0, seed=42)
            post_search_context = context_audit(answer_response)
        else:
            post_search_context = None

        answer_text = answer_response.content.strip() if answer_response and action_name(answer_response) == "answer" else ""
        score = answer_metrics(answer_text, answers[task_id]) if answer_text else None
        gold_norm = " ".join(str(answers[task_id]).casefold().split())
        exact_hit = any(gold_norm and gold_norm in " ".join(
            f"{doc.get('title', '')} {doc.get('text', '')}".casefold().split()
        ) for doc in documents)
        row = {
            "task_id": task_id,
            "original_prompt_sha256": native_sha,
            "native_action": action_name(native),
            "native_query": native.tool_calls[0].arguments.get("query") if native.tool_calls else None,
            "native_answer_em": native_score["normalized_exact_match"] if native_score else None,
            "diagnostic_prompt_sha256": query_response.usage.get("prompt_sha256"),
            "diagnostic_context": diagnostic_context,
            "diagnostic_action": action_name(query_response),
            "self_generated_query": query,
            "search_ok": bool(observation and observation.get("ok")),
            "top3": [{"id": doc.get("id"), "title": doc.get("title"),
                      "gold_answer_string_present": bool(gold_norm and gold_norm in " ".join(
                          f"{doc.get('title', '')} {doc.get('text', '')}".casefold().split()))}
                     for doc in documents],
            "top3_any_exact_gold_string": exact_hit,
            "post_search_action": action_name(answer_response) if answer_response else None,
            "post_search_context": post_search_context,
            "post_search_answer_em": score["normalized_exact_match"] if score else None,
            "post_search_answer_f1": score["token_f1"] if score else None,
            "post_search_output": answer_response.raw_text if answer_response else None,
        }
        report["rows"].append(row)
        save(output, report)

    report["complete_coverage"] = len(report["rows"]) == 5
    report["native_search_count"] = sum(row["native_action"] == "search" for row in report["rows"])
    report["diagnostic_search_count"] = sum(row["diagnostic_action"] == "search" for row in report["rows"])
    report["diagnostic_top3_exact_gold_hit_count"] = sum(bool(row["top3_any_exact_gold_string"]) for row in report["rows"])
    report["post_search_answer_correct_count"] = sum(row["post_search_answer_em"] == 1 for row in report["rows"])
    save(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("v112-tasks", "v118-tasks", "private-tasks", "corpus", "model-path", "adapter-path", "output"):
        parser.add_argument("--" + key, required=True)
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps({key: result[key] for key in (
        "version", "complete_coverage", "native_search_count", "diagnostic_search_count",
        "diagnostic_top3_exact_gold_hit_count", "post_search_answer_correct_count")}, ensure_ascii=False))
