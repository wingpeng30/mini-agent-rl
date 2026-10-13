"""Paired v133 end-to-end development run; importing this module is CPU-only."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from mini_agent_rl.domain import Message, Task, ToolCall
from mini_agent_rl.evaluation import answer_metrics
from mini_agent_rl.prompts import system_prompt
from scripts.evaluate_v121_stop_control import generation_context_record, visible_evidence
from scripts.replay_v119_bridge import EXPECTED_V118_ADAPTER

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data/evidence-sft-v117/frozen"
EXPECTED_CORPUS_SHA256 = "989d7ea08187dfdbf2a07d69cbf6a8d7dd1127ac6d971c9f62b365475f46cea9"
EXPECTED_TASK_SHA256 = "19e6328fa008fb7520c443c293bb4acb6e8f307b569d5a49cb48e17ee56e58d1"
COMMON_A_POLICY = (
    "Before answering, identify the exact fact and relation, time, scope, or quantity the question asks for. "
    "Check whether currently visible documents directly support that fact. A relevant entity or topic alone is not enough. "
    "If a specific required fact is missing, search using a concise query that names the known entity and missing relation. "
    "If evidence already supports the complete answer, answer now. Never repeat a query. "
    "After permitted searches, answer from visible evidence only. Output exactly one JSON search or answer action."
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def query_key(query: str) -> str:
    return " ".join(re.sub(r"\s+", " ", query.casefold()).split())


def build_policy(max_searches: int) -> str:
    if max_searches not in (2, 3):
        raise ValueError("v133_only_compares_A2_and_A3")
    return COMMON_A_POLICY + f" Use at most {max_searches} searches total, including the first search."


def file_hashes(directory: Path) -> dict[str, str]:
    root = Path(directory)
    return {path.relative_to(root).as_posix(): sha(path)
            for path in sorted(root.rglob("*")) if path.is_file() and ".cache" not in path.parts}


def validate_frozen_inputs(tasks_path: Path, corpus_path: Path) -> tuple[list[dict], list[dict]]:
    manifest = json.loads((FROZEN / "internal-environment-manifest.json").read_text(encoding="utf-8"))
    if sha(tasks_path) != EXPECTED_TASK_SHA256 or sha(tasks_path) != manifest.get("tasks_sha256"):
        raise ValueError("frozen_development_tasks_hash_mismatch")
    if sha(corpus_path) != EXPECTED_CORPUS_SHA256 or sha(corpus_path) != manifest.get("corpus_sha256"):
        raise ValueError("frozen_development_corpus_hash_mismatch")
    tasks = read_jsonl(tasks_path)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if len(tasks) != 50 or len({row["id"] for row in tasks}) != 50:
        raise ValueError("development_tasks_must_be_50_unique")
    if len(corpus) != 1986 or len({str(row.get("id")) for row in corpus}) != 1986:
        raise ValueError("development_corpus_must_be_1986_unique_documents")
    if manifest.get("question_count") != 50 or manifest.get("document_count") != 1986:
        raise ValueError("frozen_manifest_count_mismatch")
    return tasks, corpus


def parse_action(response: Any) -> tuple[str, str | None]:
    if getattr(response, "parse_error", None) or (getattr(response, "usage", {}) or {}).get("output_truncated"):
        return "invalid", None
    calls = getattr(response, "tool_calls", None) or []
    if calls:
        if len(calls) != 1 or calls[0].name != "search":
            return "invalid", None
        query = calls[0].arguments.get("query")
        return ("search", query.strip()) if isinstance(query, str) and query.strip() else ("invalid", None)
    answer = str(getattr(response, "content", "")).strip()
    return ("answer", answer) if answer else ("invalid", None)


async def execute_case(client: Any, registry: Any, task: dict[str, Any], max_searches: int,
                       seed: int, on_event=None) -> dict[str, Any]:
    question = str(task["question"])
    messages = [Message(role="system", content=system_prompt(False, "direct-v090") + " " + build_policy(max_searches)),
                Message(role="user", content=question)]
    queries: list[str] = []
    retrievals: list[dict] = []
    calls: list[dict] = []
    repair_used = False
    answer = ""
    termination = "decision_limit"
    protocol_valid = True

    # Search budget is an environment limit; the loop separately allows a final answer and one repair.
    for decision in range(max_searches + 3):
        if len(queries) >= max_searches and not any(m.content == "[v133 search budget exhausted; answer from visible evidence only.]" for m in messages if m.role == "user"):
            messages.append(Message(role="user", content="[v133 search budget exhausted; answer from visible evidence only.]"))
        client.reset_context_records()
        input_messages = [message.model_dump(mode="json") for message in messages]
        try:
            response = await client.generate(messages, registry.definitions(), temperature=0, seed=seed + decision)
        except Exception as exc:
            if on_event:
                on_event({"kind": "generation_error", "role": "policy_action", "decision": decision,
                          "input_messages": input_messages, "error_type": type(exc).__name__})
            raise
        context = generation_context_record(client.context_usage_summary().get("records", []), response.usage)
        action, value = parse_action(response)
        item = {"role": "policy_action", "decision": decision, "action": action,
                "raw_output": response.raw_text, "parse_error": response.parse_error,
                "output_truncated": bool(response.usage.get("output_truncated")),
                "action_token_ids": list(getattr(response, "action_token_ids", []) or []),
                "prompt_sha256": response.usage.get("prompt_sha256"),
                "prompt_token_ids": list(response.prompt_token_ids or []),
                "attention_mask": list(response.prompt_attention_mask or []),
                "context_usage": context, "visible_evidence": visible_evidence(messages, context)}
        if action == "search":
            item["query"] = value
        if on_event:
            on_event({"kind": "generation_result", "input_messages": input_messages, **item})
        if action == "invalid":
            protocol_valid = False
            termination = "invalid_action"
            calls.append(item)
            break
        if action == "answer":
            answer, termination = value or "", "answer"
            item["answer"] = answer
            calls.append(item)
            break

        query = value or ""
        if query_key(query) in {query_key(old) for old in queries}:
            item["blocked_reason"] = "repeated_query"
            calls.append(item)
            if repair_used:
                termination = "repair_exhausted"
                break
            repair_used = True
            repair_message = Message(role="user", content=(
                "The proposed query repeats a previous search. Return one different JSON search action for the "
                "specific missing requested property, or an answer action if visible evidence is sufficient."))
            repair_messages = messages + [repair_message]
            client.reset_context_records()
            repair_input = [message.model_dump(mode="json") for message in repair_messages]
            try:
                repaired = await client.generate(repair_messages, registry.definitions(), temperature=0, seed=seed + decision + 100)
            except Exception as exc:
                if on_event:
                    on_event({"kind": "generation_error", "role": "repeat_repair", "decision": decision,
                              "input_messages": repair_input, "error_type": type(exc).__name__})
                raise
            repair_context = generation_context_record(client.context_usage_summary().get("records", []), repaired.usage)
            repair_action, repair_value = parse_action(repaired)
            repair_row = {"role": "repeat_repair", "decision": decision, "action": repair_action,
                          "raw_output": repaired.raw_text, "parse_error": repaired.parse_error,
                          "prompt_sha256": repaired.usage.get("prompt_sha256"),
                          "prompt_token_ids": list(repaired.prompt_token_ids or []),
                          "attention_mask": list(repaired.prompt_attention_mask or []),
                          "context_usage": repair_context, "visible_evidence": visible_evidence(repair_messages, repair_context)}
            if on_event:
                on_event({"kind": "generation_result", "input_messages": repair_input, **repair_row})
            calls.append(repair_row)
            if repair_action == "answer":
                answer, termination = repair_value or "", "answer_after_repeat_repair"
                repair_row["answer"] = answer
                messages = repair_messages
                break
            if repair_action != "search" or not repair_value or query_key(repair_value) in {query_key(old) for old in queries}:
                protocol_valid &= repair_action != "invalid"
                termination = "repair_invalid"
                break
            query = repair_value
            messages = repair_messages
        if len(queries) >= max_searches:
            termination = "search_limit_insufficient"
            break
        observation = await registry.execute("search", {"query": query})
        if on_event:
            on_event({"kind": "tool_outcome", "query": query, "observation": observation})
        if not observation.get("ok"):
            protocol_valid = False
            termination = "tool_error"
            calls[-1]["tool_error"] = observation
            break
        queries.append(query)
        retrievals.append({"query": query, "documents": observation.get("result", [])})
        observation["searched_queries"] = sorted(query_key(q) for q in queries)
        call_id = f"v133-{max_searches}-{len(queries)}"
        messages += [Message(role="assistant", content=json.dumps({"action": "search", "query": query}, ensure_ascii=False),
                             tool_calls=[ToolCall(id=call_id, name="search", arguments={"query": query})]),
                     Message(role="tool", content=json.dumps({k: v for k, v in observation.items() if k != "latency_ms"}, ensure_ascii=False),
                             tool_call_id=call_id)]
    score = answer_metrics(answer, str(task["answer"])) if answer else {"normalized_exact_match": 0.0, "token_f1": 0.0}
    return {"task_id": task["id"], "question": question, "candidate_answer": answer,
            "normalized_exact_match": score["normalized_exact_match"], "token_f1": score["token_f1"],
            "protocol_valid": protocol_valid, "termination": termination,
            "search_queries": queries, "actual_search_calls": len(queries), "retrieved_docs": retrievals,
            "repair_calls": sum(row["role"] == "repeat_repair" for row in calls), "calls": calls,
            "final_messages": [message.model_dump(mode="json") for message in messages]}


def paired_summary(a2: list[dict], a3: list[dict]) -> dict:
    left, right = {r["task_id"]: r for r in a2}, {r["task_id"]: r for r in a3}
    if len(left) != 50 or set(left) != set(right):
        raise ValueError("paired_complete_50_required")
    a2_correct = {key for key, row in left.items() if row["normalized_exact_match"] == 1}
    a3_correct = {key for key, row in right.items() if row["normalized_exact_match"] == 1}
    retained = len(a2_correct & a3_correct)
    return {"n": 50, "a2_em": len(a2_correct), "a3_em": len(a3_correct),
            "a2_f1": sum(r["token_f1"] for r in left.values()) / 50,
            "a3_f1": sum(r["token_f1"] for r in right.values()) / 50,
            "new_correct": sorted(a3_correct - a2_correct), "lost_correct": sorted(a2_correct - a3_correct),
            "net_em_gain": len(a3_correct) - len(a2_correct),
            "a2_correct_retained": retained,
            "a2_correct_retention_rate": (retained / len(a2_correct)) if a2_correct else None,
            "a2_searches": sum(r["actual_search_calls"] for r in left.values()),
            "a3_searches": sum(r["actual_search_calls"] for r in right.values()),
            "a2_protocol_rate": sum(r["protocol_valid"] for r in left.values()) / 50,
            "a3_protocol_rate": sum(r["protocol_valid"] for r in right.values()) / 50}


async def run(args: argparse.Namespace) -> dict:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("determinism_environment_required_before_process_start")
    tasks, corpus = validate_frozen_inputs(Path(args.tasks), Path(args.corpus))
    adapter = Path(args.adapter_path) / "adapter_model.safetensors"
    if sha(adapter) != EXPECTED_V118_ADAPTER:
        raise ValueError("v118_adapter_hash_mismatch")
    adapter_config = json.loads((Path(args.adapter_path) / "adapter_config.json").read_text(encoding="utf-8"))
    if Path(adapter_config.get("base_model_name_or_path", "")).resolve() != Path(args.model_path).resolve():
        raise ValueError("v118_adapter_base_model_path_mismatch")
    required_model_files = (Path(args.model_path) / "config.json", Path(args.model_path) / "tokenizer.json",
                            Path(args.model_path) / "tokenizer_config.json")
    if any(not path.is_file() for path in required_model_files):
        raise ValueError("base_model_or_tokenizer_files_missing")
    if not file_hashes(Path(args.model_path)) or not file_hashes(Path(args.adapter_path)):
        raise ValueError("model_or_adapter_file_tree_empty")
    if args.preflight_only:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        config = build_config(args, tasks)
        frozen = out / "settings.json"
        if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != config:
            raise ValueError("settings_changed_since_freeze")
        frozen.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report = {"development_preflight_passed": True, "inference_authorized": False,
                  "training_allowed": False, "confirmatory_ready": False,
                  "settings_sha256": sha(frozen), "models_loaded": False, "generation_calls": 0}
        (out / "cpu-preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    cfg = LocalModelConfig(model_id=args.model_path, adapter_path=args.adapter_path, device="cuda", dtype="bfloat16",
        load_in_4bit=False, max_input_tokens=2048, max_new_tokens=128, enable_thinking=False, timeout=600,
        gradient_checkpointing=False, observation_encoding="compact-v3", deterministic=True,
        local_files_only=True, trust_remote_code=False, logprob_mode="bf16")
    client = TransformersModelClient(cfg)
    registry = ToolRegistry()
    registry.register(SearchTool(corpus, "bm25-v2"))
    return await run_with_client(args, tasks, client, registry)


async def run_with_client(args: argparse.Namespace, tasks: list[dict], client: Any, registry: Any) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = build_config(args, tasks)
    frozen_path = out / "settings.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text(encoding="utf-8")) != config:
        raise ValueError("settings_changed_since_freeze")
    frozen_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    arms: dict[str, list[dict]] = {}
    for label, budget in (("A2", 2), ("A3", 3)):
        result_path = out / f"{label}.json"
        events_path = out / f"{label}-events.jsonl"
        report = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
            "version": "v133-search-budget-arm-v1", "arm": label,
            "settings_sha256": sha(frozen_path), "rows": []}
        if report.get("settings_sha256") != sha(frozen_path) or report.get("arm") != label:
            raise ValueError("resume_settings_mismatch")
        done = [row["task_id"] for row in report["rows"]]
        ids = {row["id"] for row in tasks}
        if len(done) != len(set(done)) or not set(done) <= ids:
            raise ValueError("invalid_resume_rows")
        if events_path.exists():
            pending: set[str] = set()
            for event in read_jsonl(events_path):
                if event.get("kind") == "task_started":
                    pending.add(event["task_id"])
                elif event.get("kind") == "task_completed":
                    pending.discard(event["task_id"])
            stranded = pending - set(done)
            if stranded:
                raise ValueError("partial_task_attempt_preserved_and_requires_manual_recovery")
        by_id = {row["id"]: row for row in tasks}
        for offset, task_id in enumerate([row["id"] for row in tasks]):
            if task_id in set(done):
                continue
            append_jsonl(events_path, {"kind": "task_started", "task_id": task_id, "arm": label,
                                       "settings_sha256": report["settings_sha256"]})
            def log_event(event: dict[str, Any]) -> None:
                append_jsonl(events_path, {"task_id": task_id, "arm": label, **event})
            try:
                row = await execute_case(client, registry, by_id[task_id], budget, 42 + offset * 31, log_event)
            except Exception as exc:
                log_event({"kind": "task_error", "error_type": type(exc).__name__})
                raise
            report["rows"].append(row)
            result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            append_jsonl(events_path, {"kind": "task_completed", "task_id": task_id, "arm": label})
        report["complete_coverage"] = len(report["rows"]) == 50 and {r["task_id"] for r in report["rows"]} == ids
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not report["complete_coverage"]:
            raise ValueError("incomplete_arm_coverage")
        arms[label] = report["rows"]
    summary = paired_summary(arms["A2"], arms["A3"])
    summary.update({"version": "v133-paired-summary-v1", "settings_sha256": sha(frozen_path),
                    "exploratory_thresholds": {"net_em_gain_min": 3, "f1_delta_min": 0,
                                               "a2_correct_retention_min": 0.95, "protocol_rate_min": 0.98},
                    "independent100_opened": False, "final_test_accessed": False})
    (out / "paired-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_config(args: argparse.Namespace, tasks: list[dict]) -> dict:
    return {"version": "v133-search-budget-e2e-v1", "tasks_path": str(Path(args.tasks).resolve()),
            "tasks_sha256": sha(Path(args.tasks)), "corpus_path": str(Path(args.corpus).resolve()),
            "corpus_sha256": sha(Path(args.corpus)), "model_path": str(Path(args.model_path).resolve()),
            "adapter_path": str(Path(args.adapter_path).resolve()), "adapter_sha256": EXPECTED_V118_ADAPTER,
            "runner_sha256": sha(Path(__file__)), "common_a_policy_sha256": hashlib.sha256(COMMON_A_POLICY.encode()).hexdigest(),
            "a2_policy_sha256": hashlib.sha256(build_policy(2).encode()).hexdigest(),
            "a3_policy_sha256": hashlib.sha256(build_policy(3).encode()).hexdigest(),
            "dependency_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
                ROOT / "src/mini_agent_rl/prompts.py", ROOT / "src/mini_agent_rl/model/context.py",
                ROOT / "src/mini_agent_rl/model/local.py", ROOT / "src/mini_agent_rl/tools/search.py",
                ROOT / "scripts/evaluate_v121_stop_control.py")},
            "model_files": file_hashes(Path(args.model_path)), "adapter_files": file_hashes(Path(args.adapter_path)),
            "max_searches": {"A2": 2, "A3": 3}, "retriever": "bm25-v2 top3",
            "max_input_tokens": 2048, "max_new_tokens": 128, "temperature": 0, "seed": 42,
            "device": "cuda", "dtype": "bfloat16", "load_in_4bit": False, "timeout_seconds": 600,
            "observation_encoding": "compact-v3", "thinking": False,
            "local_files_only": True, "trust_remote_code": False,
            "max_generation_calls": {"A2": 300, "A3": 350, "total": 650},
            "max_successful_search_calls": 250,
            "task_order_sha256": hashlib.sha256("\n".join(str(row["id"]) for row in tasks).encode()).hexdigest(),
            "decision_thresholds": {"em_net_gain_min": 3, "mean_f1_delta_min": 0.0,
                                    "a2_correct_retention_min": 0.95, "protocol_compliance_min": 0.98},
            "review_scope": {"answer_change_pairs": True, "a3_third_successful_search_pairs": True,
                             "mechanical_protocol_or_evidence_visibility_anomalies": True,
                             "sentinel_max": 5, "sentinel_order": "sha256('v133-sentinel:' + task_id)"},
            "stop_rule": "If any threshold fails, stop A3 promotion; retain diagnostics and do not change prompt, seed, or selected questions.",
            "independent100_opened": False, "final_test_accessed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=str(FROZEN / "internal-tasks-private.jsonl"))
    parser.add_argument("--corpus", default=str(FROZEN / "internal-corpus.json"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true", help="freeze and validate local inputs without loading weights or generating")
    arguments = parser.parse_args()
    print(json.dumps(asyncio.run(run(arguments)), ensure_ascii=False, indent=2))
