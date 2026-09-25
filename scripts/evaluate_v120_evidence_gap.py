"""Prompt-only evidence-gap policy replay on the frozen v117 internal 50.

This development diagnostic pairs with the existing v118 baseline. It changes
only the system policy text, allows at most one targeted follow-up search, and
never places private answers in model messages.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

from mini_agent_rl.domain import Message
from mini_agent_rl.evaluation import answer_metrics
from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient
from mini_agent_rl.tools import SearchTool, ToolRegistry

from scripts.replay_v119_bridge import EXPECTED_V118_ADAPTER, model_files, read, save


POLICY = (
    "Before answering, identify the exact fact and relation, time, scope, or quantity the question asks for. "
    "Check whether the currently visible documents directly support that fact. A relevant entity or topic alone "
    "is not enough. If a specific required fact is missing, search once using a concise query that names the "
    "known entity and the missing relation. If the evidence already supports the complete answer, answer now. "
    "Use at most two searches total, and never repeat a query. After the one possible follow-up search, answer "
    "from visible evidence only. Output exactly one JSON search or answer action."
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def action(response):
    if response.parse_error or response.usage.get("output_truncated", False):
        return "invalid"
    calls = response.tool_calls or []
    if calls:
        return "search" if len(calls) == 1 and calls[0].name == "search" and str(calls[0].arguments.get("query", "")).strip() else "invalid"
    return "answer" if response.content.strip() else "invalid"


def query_key(value: str) -> str:
    return " ".join(re.sub(r"\s+", " ", value.casefold()).split())


async def run(args):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("determinism_environment_required_before_process_start")
    baseline_path, corpus_path, private_path = map(Path, (args.baseline_tasks, args.corpus, args.private_tasks))
    baseline = read(baseline_path)
    if baseline.get("version") != "v118" or baseline.get("mode") != "tasks" or baseline.get("complete_coverage") is not True:
        raise ValueError("complete_v118_task_baseline_required")
    if len(baseline.get("rows", [])) != 50 or len({row["task_id"] for row in baseline["rows"]}) != 50:
        raise ValueError("paired_frozen_50_required")
    corpus = read(corpus_path)
    if sha(corpus_path) != baseline.get("corpus_sha256") or len(corpus) != baseline.get("corpus_documents"):
        raise ValueError("frozen_corpus_mismatch")
    adapter_path = Path(args.adapter_path)
    if sha(adapter_path / "adapter_model.safetensors") != EXPECTED_V118_ADAPTER:
        raise ValueError("v118_adapter_hash_mismatch")
    if model_files(args.model_path) != baseline.get("model_files"):
        raise ValueError("base_model_file_hash_mismatch")
    private = {row["id"]: row for row in read_jsonl(private_path)}
    baseline_by_id = {row["task_id"]: row for row in baseline["rows"]}
    if set(private) != set(baseline_by_id):
        raise ValueError("private_task_ids_do_not_match_baseline")

    config = LocalModelConfig(model_id=args.model_path, adapter_path=args.adapter_path, device="cuda", dtype="bfloat16",
        load_in_4bit=False, max_input_tokens=2048, max_new_tokens=128, enable_thinking=False, timeout=600,
        gradient_checkpointing=False, observation_encoding="compact-v3", deterministic=True,
        local_files_only=True, trust_remote_code=False, logprob_mode="bf16")
    client = TransformersModelClient(config)
    registry = ToolRegistry()
    registry.register(SearchTool(corpus, "bm25-v2"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "version": "v120-evidence-gap-prompt-replay-v2",
        "exploratory_development_only": True,
        "source_sha256": {"baseline_tasks": sha(baseline_path), "private_tasks": sha(private_path), "corpus": sha(corpus_path)},
        "adapter_sha256": EXPECTED_V118_ADAPTER,
        "script_sha256": sha(Path(__file__)),
        "policy_sha256": hashlib.sha256(POLICY.encode("utf-8")).hexdigest(),
        "configuration": {"temperature": 0, "seed": 42, "retriever": "bm25-v2 top3", "max_searches": 2,
                          "policy": "initial search plus at most one targeted follow-up"},
        "independent100_opened": False,
        "final_test_accessed": False,
        "rows": [],
    }
    report = read(output) if output.exists() else {**identity, "rows": []}
    if any(report.get(key) != value for key, value in identity.items() if key != "rows"):
        raise ValueError("resume_freeze_mismatch")
    completed = {row["task_id"] for row in report["rows"]}
    if len(completed) != len(report["rows"]) or not completed <= set(baseline_by_id):
        raise ValueError("invalid_resume_cases")

    for task_id, base_row in baseline_by_id.items():
        if task_id in completed:
            continue
        task = private[task_id]
        if not base_row.get("transitions"):
            raise ValueError("baseline_missing_initial_transition")
        messages = [Message(**message) for message in base_row["transitions"][0]["messages"]]
        if not messages or messages[0].role != "system" or messages[-1].role != "user" or messages[-1].content != task["question"]:
            raise ValueError("baseline_question_binding_mismatch")
        messages[0] = Message(role="system", content=messages[0].content + " " + POLICY)
        search_queries = []
        retrieved_ids = set()
        retrieved_docs = []
        transition_rows = []
        final_text = ""
        termination = "max_actions"
        protocol_valid = True
        search_limit_compliance = True
        for step in range(4):
            if len(search_queries) >= 2:
                messages.append(Message(role="user", content=(
                    "Both allowed searches have been used. Do not search again. Review only the evidence now "
                    "visible in the conversation and output the best supported answer action."
                )))
            response = await client.generate(messages, registry.definitions(), temperature=0, seed=42 + step)
            kind = action(response)
            item = {"step": step, "action": kind, "prompt_sha256": response.usage.get("prompt_sha256"),
                    "prompt_tokens": response.usage.get("prompt_tokens"), "raw_output": response.raw_text}
            if kind == "invalid":
                protocol_valid = False
                termination = "invalid_action"
                transition_rows.append(item)
                break
            if kind == "answer":
                final_text = response.content.strip()
                termination = "answer"
                item["answer"] = final_text
                transition_rows.append(item)
                break
            query = response.tool_calls[0].arguments["query"].strip()
            if query_key(query) in {query_key(prev) for prev in search_queries} or len(search_queries) >= 2:
                search_limit_compliance = False
                item.update({"query": query, "search_executed": False,
                             "blocked_reason": "repeat" if query_key(query) in {query_key(prev) for prev in search_queries} else "search_limit"})
                transition_rows.append(item)
                messages += [Message(role="assistant", content=json.dumps({"action": "search", "query": query}, ensure_ascii=False),
                                     tool_calls=response.tool_calls),
                             Message(role="tool", content=json.dumps({"ok": False, "error": "Search limit reached; no search was executed."}),
                                     tool_call_id=response.tool_calls[0].id),
                             Message(role="user", content="Do not search again. Answer now using only evidence already visible.")]
                final_response = await client.generate(messages, registry.definitions(), temperature=0, seed=42 + step + 1)
                final_kind = action(final_response)
                final_item = {"step": step + 1, "action": final_kind, "prompt_sha256": final_response.usage.get("prompt_sha256"),
                              "prompt_tokens": final_response.usage.get("prompt_tokens"), "raw_output": final_response.raw_text,
                              "after_search_limit_block": True}
                if final_kind == "answer":
                    final_text = final_response.content.strip()
                    termination = "answer_after_search_limit_block"
                    final_item["answer"] = final_text
                else:
                    protocol_valid = False
                    termination = "failed_after_search_limit_block"
                transition_rows.append(final_item)
                break
            search_queries.append(query)
            observation = await registry.execute("search", {"query": query})
            docs = observation.get("result") or []
            if not observation.get("ok"):
                protocol_valid = False
                termination = "tool_error"
                item.update({"query": query, "search_ok": False})
                transition_rows.append(item)
                break
            observation["searched_queries"] = sorted(query_key(prev) for prev in search_queries)
            model_observation = {key: value for key, value in observation.items() if key != "latency_ms"}
            item.update({"query": query, "search_ok": True,
                         "top3": [{"id": doc.get("id"), "title": doc.get("title")} for doc in docs],
                         "new_top3_doc_count": sum(doc.get("id") not in retrieved_ids for doc in docs)})
            retrieved_ids.update(doc.get("id") for doc in docs if doc.get("id") is not None)
            retrieved_docs.extend(docs)
            transition_rows.append(item)
            messages += [Message(role="assistant", content=json.dumps({"action": "search", "query": query}, ensure_ascii=False),
                                 tool_calls=response.tool_calls),
                         Message(role="tool", content=json.dumps(model_observation, ensure_ascii=False),
                                 tool_call_id=response.tool_calls[0].id)]
        score = answer_metrics(final_text, task["answer"]) if final_text else {"normalized_exact_match": 0.0, "token_f1": 0.0}
        gold_norm = " ".join(str(task["answer"]).casefold().split())
        exact_answer_hit = any(gold_norm and gold_norm in " ".join(
            f"{doc.get('title', '')} {doc.get('text', '')}".casefold().split()) for doc in retrieved_docs)
        baseline_steps = base_row["transitions"]
        baseline_queries = [step["response"].get("tool_calls", [{}])[0].get("arguments", {}).get("query")
                            for step in baseline_steps if step["response"].get("tool_calls")]
        report["rows"].append({"task_id": task_id,
            "baseline_em": base_row["normalized_exact_match"], "baseline_f1": base_row["token_f1"],
            "baseline_search_count": len(baseline_queries), "baseline_queries": baseline_queries,
            "candidate_em": score["normalized_exact_match"], "candidate_f1": score["token_f1"],
            "candidate_search_count": len(search_queries), "candidate_queries": search_queries,
            "candidate_answered": bool(final_text), "termination": termination, "protocol_valid": protocol_valid,
            "search_limit_compliance": search_limit_compliance,
            "blocked_extra_search_count": sum(item.get("blocked_reason") is not None for item in transition_rows),
            "top3_any_exact_gold_string": exact_answer_hit, "transition_trace": transition_rows,
            "final_output": final_text})
        save(output, report)

    rows = report["rows"]
    base_correct = {row["task_id"] for row in rows if row["baseline_em"] == 1}
    cand_correct = {row["task_id"] for row in rows if row["candidate_em"] == 1}
    report["complete_coverage"] = len(rows) == 50 and {row["task_id"] for row in rows} == set(baseline_by_id)
    report["metrics"] = {
        "baseline_correct": len(base_correct), "candidate_correct": len(cand_correct),
        "gains": len(cand_correct - base_correct), "losses": len(base_correct - cand_correct),
        "net_gain": len(cand_correct - base_correct) - len(base_correct - cand_correct),
        "correct_preservation": len(cand_correct & base_correct) / max(1, len(base_correct)),
        "baseline_em": sum(row["baseline_em"] for row in rows) / 50,
        "candidate_em": sum(row["candidate_em"] for row in rows) / 50,
        "baseline_f1": sum(row["baseline_f1"] for row in rows) / 50,
        "candidate_f1": sum(row["candidate_f1"] for row in rows) / 50,
        "baseline_searches": sum(row["baseline_search_count"] for row in rows),
        "candidate_searches": sum(row["candidate_search_count"] for row in rows),
        "candidate_two_search_cases": sum(row["candidate_search_count"] == 2 for row in rows),
        "candidate_protocol_rate": sum(row["protocol_valid"] for row in rows) / 50,
        "search_limit_compliance_rate": sum(row["search_limit_compliance"] for row in rows) / 50,
        "blocked_extra_search_attempts": sum(row["blocked_extra_search_count"] for row in rows),
        "candidate_completed_answers": sum(row["candidate_answered"] for row in rows),
        "candidate_exact_answer_string_in_top3": sum(row["top3_any_exact_gold_string"] for row in rows),
    }
    save(output, report)
    if not report["complete_coverage"]:
        raise RuntimeError("incomplete_frozen_internal_coverage")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("baseline-tasks", "private-tasks", "corpus", "model-path", "adapter-path", "output"):
        parser.add_argument("--" + name, required=True)
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps({"version": result["version"], "complete_coverage": result["complete_coverage"],
                      "metrics": result["metrics"]}, ensure_ascii=False, indent=2))
