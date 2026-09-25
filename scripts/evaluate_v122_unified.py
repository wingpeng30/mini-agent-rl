"""v122 development replay using only the trained search/answer action protocol.

Stages are fail closed: a 50-input protocol preflight for every arm must pass
before fixed-state replay; full end-to-end requires fixed-state selection and
an independent semantic-audit approval. No sealed evaluation source is read.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path

from mini_agent_rl.domain import Message, ToolCall
from mini_agent_rl.evaluation import answer_metrics
from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient, parse_agent_action
from mini_agent_rl.tools import SearchTool, ToolRegistry
from scripts.check_action_protocol_contract import check_probe, check_sft
from scripts.evaluate_v120_evidence_gap import POLICY as BASE_POLICY, action
from scripts.evaluate_v121_stop_control import generation_context_record, visible_evidence
from scripts.replay_v119_bridge import EXPECTED_V118_ADAPTER, model_files, read, save


Q_POLICY = ("For a needed search, query the known entity and the one missing requested property. "
            "Use an entity in the question or a relevant visible relation sentence. Do not substitute a same-name distractor. "
            "Return exactly one ordinary JSON search or answer action.")
S_POLICY = ("After each retrieval, answer only when visible evidence supports every requested relation, comparison side, "
            "time, scope, and unit. Otherwise search for the missing relation. Return exactly one ordinary JSON "
            "search or answer action; do not write a review object.")
REPAIR_POLICY = ("The proposed query is repeated or lacks an entity supported by the question or visible evidence. "
                 "If another search is needed, return one different JSON search action for the missing property. "
                 "Do not include an audit report.")
ARM_PROMPTS = {"A": BASE_POLICY, "B": BASE_POLICY + " " + Q_POLICY,
               "C": BASE_POLICY + " " + S_POLICY,
               "D": BASE_POLICY + " " + Q_POLICY + " " + S_POLICY}
NAMED_SPAN = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:\s+(?:[A-Z][A-Za-z0-9]*|of|the|and)){0,3}\b")
GENERIC = {"Who", "What", "Which", "When", "Where", "How", "The", "A", "An", "Is", "Are", "Was", "Were"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_key(value: str) -> str:
    return " ".join(value.casefold().split())


def grounded_query(query: str, question: str, evidence: list[dict]) -> bool:
    """Surface-form gate only; semantic relation support is audited separately."""
    visible = (question + " " + " ".join(str(d["text"]) for d in evidence)).casefold()
    spans = [m.group() for m in NAMED_SPAN.finditer(query) if m.group() not in GENERIC]
    return bool(spans) and any(span.casefold() in visible for span in spans)


def settings(args, baseline: dict) -> dict:
    paths = {name: Path(getattr(args, name)) for name in
             ("baseline_tasks", "v120_report", "private_tasks", "corpus", "train_data")}
    if sha(Path(args.adapter_path) / "adapter_model.safetensors") != EXPECTED_V118_ADAPTER:
        raise ValueError("v118_adapter_hash_mismatch")
    files = model_files(args.model_path)
    if files != baseline.get("model_files"):
        raise ValueError("base_model_hash_mismatch")
    if sha(paths["corpus"]) != baseline.get("corpus_sha256"):
        raise ValueError("frozen_corpus_hash_mismatch")
    return {"version": "v122-unified-actions-v1", "script_sha256": sha(Path(__file__)),
            "sources": {key: sha(path) for key, path in paths.items()},
            "adapter_sha256": EXPECTED_V118_ADAPTER, "model_files": files,
            "policies_sha256": {arm: hashlib.sha256(prompt.encode()).hexdigest() for arm, prompt in ARM_PROMPTS.items()},
            "repair_sha256": hashlib.sha256(REPAIR_POLICY.encode()).hexdigest(),
            "retriever": "bm25-v2 top3", "observation_encoding": "compact-v3",
            "max_input_tokens": 2048, "max_new_tokens": 128, "max_searches": 2,
            "thinking": False, "temperature": 0, "seed": 42, "dtype": "bfloat16",
            "independent100_opened": False, "final_test_accessed": False}


def add_policy(messages: list[Message], arm: str) -> list[Message]:
    result = list(messages)
    if not result or result[0].role != "system":
        raise ValueError("system_message_missing")
    result[0] = Message(role="system", content=result[0].content + " " + ARM_PROMPTS[arm])
    return result


def make_client(args):
    cfg = LocalModelConfig(model_id=args.model_path, adapter_path=args.adapter_path, device="cuda",
        dtype="bfloat16", load_in_4bit=False, max_input_tokens=2048, max_new_tokens=128,
        enable_thinking=False, timeout=600, gradient_checkpointing=False,
        observation_encoding="compact-v3", deterministic=True, local_files_only=True,
        trust_remote_code=False, logprob_mode="bf16")
    return TransformersModelClient(cfg)


async def generate(client, messages: list[Message], tools, seed: int) -> tuple[object, dict]:
    client.reset_context_records()
    response = await client.generate(messages, tools, temperature=0, seed=seed)
    context = generation_context_record(client.context_usage_summary().get("records", []), response.usage)
    evidence = visible_evidence(messages, context)
    ids = list(response.prompt_token_ids or [])
    mask = list(response.prompt_attention_mask or [])
    if not ids or len(ids) > 2048 or len(mask) != len(ids):
        raise RuntimeError("actual_prompt_token_or_mask_invalid")
    tokens = list(response.action_token_ids or [])
    if len(tokens) > 128:
        raise RuntimeError("action_token_cap_exceeded")
    item = {"raw_output": response.raw_text, "action": action(response),
            "parse_error": response.parse_error, "output_truncated": bool(response.usage.get("output_truncated")),
            "generated_tokens": len(tokens), "prompt_sha256": response.usage.get("prompt_sha256"),
            "prompt_token_ids": ids, "attention_mask": mask, "prompt_tokens": response.usage.get("prompt_tokens"),
            "context_usage": context, "visible_evidence": evidence}
    return response, item


def select_preflight(train_path: Path, dev_questions: set[str]) -> dict[str, list[Message]]:
    groups = {"search": [], "answer": []}
    for line in train_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row["messages"]
        index = row["target_assistant_index"]
        question = next((m["content"] for m in messages if m["role"] == "user"), None)
        if not question or query_key(question) in dev_questions:
            continue
        answer, tool = parse_agent_action(messages[index]["content"])
        kind = "search" if tool else "answer"
        digest = hashlib.sha256(("v122-preflight:" + question + ":" + kind).encode()).hexdigest()
        groups[kind].append((digest, question, [Message(**m) for m in messages[:index]]))
    selected = {}
    for kind, rows in groups.items():
        dedup = {}
        for digest, question, messages in sorted(rows, key=lambda row: (row[0], row[1])):
            dedup.setdefault(query_key(question), (digest, messages))
        if len(dedup) < 25:
            raise ValueError(f"preflight_{kind}_needs_25_distinct_questions")
        selected[kind] = [messages for _, messages in sorted(dedup.values())[:25]]
    return selected


async def preflight(args, client, registry, config: dict, config_hash: str, dev_questions: set[str], output: Path):
    selected = select_preflight(Path(args.train_data), dev_questions)
    for arm in "ABCD":
        path = output / f"probe-{arm}.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != 50:
                raise ValueError(f"partial_probe_{arm}_requires_exact_resume")
        else:
            rows = []
            for index, messages in enumerate(selected["search"] + selected["answer"]):
                response, call = await generate(client, add_policy(messages, arm), registry.definitions(), 42 + index)
                rows.append({"settings_sha256": config_hash, "expected_state": "search" if index < 25 else "answer",
                             **call})
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
        if len(rows) != 50 or {r.get("expected_state") for r in rows} != {"search", "answer"}:
            raise ValueError(f"probe_coverage_invalid_{arm}")
        if (sum(r["expected_state"] == "search" for r in rows) != 25 or
                sum(r["expected_state"] == "answer" for r in rows) != 25 or
                any(not isinstance(r.get("generated_tokens"), int) or
                    not isinstance(r.get("output_truncated"), bool) or
                    "parse_error" not in r or not r.get("prompt_sha256") for r in rows)):
            raise ValueError(f"probe_record_incomplete_{arm}")
        check_probe(path, config_hash, .98)
    result = {"version": "v122-preflight-v1", "settings_sha256": config_hash,
              "counts": {arm: 50 for arm in "ABCD"}, "passed": True}
    save(output / "preflight-report.json", result)
    return result


async def repair_query(client, messages, registry, previous: set[str], seed: int, calls: list[dict]):
    repair = messages + [Message(role="user", content=REPAIR_POLICY)]
    response, item = await generate(client, repair, registry.definitions(), seed)
    item["role"] = "repair"; calls.append(item)
    if item["action"] != "search":
        return None
    query = response.tool_calls[0].arguments["query"].strip()
    return query if query_key(query) not in previous else None


async def run_case(client, registry, arm: str, source_messages: list[Message], question: str,
                   first_query: str | None, seed: int, expected_top3: list[str] | None = None) -> dict:
    messages = add_policy(source_messages, arm)
    state = {"calls": [], "search_queries": [], "retrieved_docs": [], "protocol_valid": True,
             "termination": "max_actions", "candidate_answer": "", "repair_used": False}
    if first_query is not None:
        obs = await registry.execute("search", {"query": first_query})
        if not obs.get("ok"):
            raise RuntimeError("fixed_first_search_failed")
        if expected_top3 is not None and [str(d.get("id")) for d in obs.get("result", [])] != expected_top3:
            raise RuntimeError("fixed_first_search_top3_mismatch")
        state["search_queries"].append(first_query)
        state["retrieved_docs"].append({"query": first_query, "documents": obs.get("result", [])})
        obs["searched_queries"] = [query_key(first_query)]
        tid = "v122-fixed-first"
        messages += [Message(role="assistant", content=json.dumps({"action":"search","query":first_query}),
                             tool_calls=[ToolCall(id=tid, name="search", arguments={"query":first_query})]),
                     Message(role="tool", content=json.dumps({k:v for k,v in obs.items() if k != "latency_ms"}, ensure_ascii=False),
                             tool_call_id=tid)]
    for decision in range(5):
        response, item = await generate(client, messages, registry.definitions(), seed + decision * 5)
        item["role"] = "policy_action"; state["calls"].append(item)
        if item["action"] == "invalid":
            state["protocol_valid"] = False; state["termination"] = "invalid_action"; break
        if item["action"] == "answer":
            state["candidate_answer"] = response.content.strip(); state["termination"] = "answer"; break
        query = response.tool_calls[0].arguments["query"].strip()
        previous = {query_key(q) for q in state["search_queries"]}
        if len(previous) >= 2:
            state["termination"] = "search_limit_insufficient"; break
        grounded = grounded_query(query, question, item["visible_evidence"]) if arm in "BD" else True
        if query_key(query) in previous or not grounded:
            item["blocked_reason"] = "repeat" if query_key(query) in previous else "unsupported_query_anchor"
            if state["repair_used"]:
                state["termination"] = "repair_exhausted"; break
            state["repair_used"] = True
            query = await repair_query(client, messages, registry, previous, seed + decision * 5 + 1, state["calls"])
            if not query:
                state["termination"] = "repair_invalid"; break
            if arm in "BD" and not grounded_query(query, question, item["visible_evidence"]):
                state["termination"] = "repair_ungrounded"; break
        obs = await registry.execute("search", {"query": query})
        if not obs.get("ok"):
            state["protocol_valid"] = False; state["termination"] = "tool_error"; break
        state["search_queries"].append(query)
        state["retrieved_docs"].append({"query": query, "documents": obs.get("result", [])})
        obs["searched_queries"] = sorted(query_key(q) for q in state["search_queries"])
        tid = f"v122-{arm}-{decision}"
        messages += [Message(role="assistant", content=json.dumps({"action":"search","query":query},ensure_ascii=False),
                             tool_calls=[ToolCall(id=tid,name="search",arguments={"query":query})]),
                     Message(role="tool", content=json.dumps({k:v for k,v in obs.items() if k != "latency_ms"},ensure_ascii=False),
                             tool_call_id=tid)]
    return state


def metrics(rows: list[dict]) -> dict:
    correct = {r["task_id"] for r in rows if r["candidate_em"] == 1}
    v118 = {r["task_id"] for r in rows if r["baseline_em"] == 1}
    v120 = {r["task_id"] for r in rows if r["v120_em"] == 1}
    return {"n": len(rows), "correct": len(correct),
            "mean_f1": sum(r["candidate_f1"] for r in rows) / max(1,len(rows)),
            "preserve_v118": len(correct & v118), "preserve_v120": len(correct & v120),
            "protocol_rate": sum(r["protocol_valid"] for r in rows) / max(1,len(rows)),
            "searches": sum(len(r["search_queries"]) for r in rows),
            "terminations": {name: sum(r["termination"] == name for r in rows)
                             for name in sorted({r["termination"] for r in rows})}}


def choose(reports: dict) -> str | None:
    base = reports["A"]["metrics"]
    passing = []
    for arm in "BCD":
        m = reports[arm]["metrics"]
        if (m["n"] == 50 and m["correct"] >= 31 and m["correct"] - base["correct"] >= 3 and
                m["mean_f1"] >= base["mean_f1"] and m["preserve_v118"] == 14 and
                m["preserve_v120"] >= 25 and m["protocol_rate"] >= .98):
            passing.append((arm, m["correct"], -m["searches"], -len(ARM_PROMPTS[arm])))
    return max(passing, key=lambda row: row[1:])[0] if passing else None


def full_numeric_gate(candidate: dict, v120_f1: float) -> dict[str, bool]:
    """Development decision only; independent semantic review remains required."""
    return {"complete_50": candidate["n"] == 50,
            "correct_gain_3_vs_v120": candidate["correct"] >= 29,
            "f1_no_decrease_vs_v120": candidate["mean_f1"] >= v120_f1,
            "v118_retention_14_of_14": candidate["preserve_v118"] == 14,
            "v120_retention_25_of_26": candidate["preserve_v120"] >= 25,
            "protocol_98": candidate["protocol_rate"] >= .98}


def require_preflight(output: Path, config_hash: str) -> None:
    gate = output / "preflight-report.json"
    expected = {"version":"v122-preflight-v1","settings_sha256":config_hash,
                "counts":{arm:50 for arm in "ABCD"},"passed":True}
    if not gate.is_file() or read(gate) != expected:
        raise ValueError("passing_matching_preflight_required")
    for arm in "ABCD":
        probe = output / f"probe-{arm}.jsonl"
        if not probe.is_file():
            raise ValueError(f"probe_missing_{arm}")
        rows = [json.loads(line) for line in probe.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 50 or sum(r.get("expected_state") == "search" for r in rows) != 25:
            raise ValueError(f"probe_coverage_invalid_{arm}")
        check_probe(probe, config_hash, .98)


async def run(args):
    import os
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("determinism_environment_required_before_process_start")
    baseline, v120, corpus = read(args.baseline_tasks), read(args.v120_report), read(args.corpus)
    if (baseline.get("version") != "v118" or baseline.get("mode") != "tasks" or
            not baseline.get("complete_coverage") or not v120.get("complete_coverage") or
            len(baseline.get("rows", [])) != 50 or len(v120.get("rows", [])) != 50 or len(corpus) != 1986):
        raise ValueError("frozen_development_inputs_required")
    tasks = {r["id"]: r for r in (json.loads(line) for line in Path(args.private_tasks).read_text(encoding="utf-8").splitlines() if line.strip())}
    base_rows = {r["task_id"]: r for r in baseline["rows"]}
    v120_rows = {r["task_id"]: r for r in v120["rows"]}
    if set(tasks) != set(base_rows) or set(tasks) != set(v120_rows):
        raise ValueError("paired_50_question_ids_required")
    check_sft(Path(args.train_data))
    config = settings(args, baseline)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    config_path = output / "settings.json"
    if config_path.exists() and read(config_path) != config:
        raise ValueError("settings_changed_since_freeze")
    save(config_path, config)
    config_hash = sha(config_path)
    if args.stage != "preflight":
        require_preflight(output, config_hash)
    client = make_client(args)
    registry = ToolRegistry(); registry.register(SearchTool(corpus, "bm25-v2"))
    if args.stage == "preflight":
        return await preflight(args, client, registry, config, config_hash,
                               {query_key(t["question"]) for t in tasks.values()}, output)
    reports = {}
    for arm in "ABCD":
        path = output / f"fixed-{arm}.json"
        report = read(path) if path.exists() else {"version":"v122-fixed-v1", "settings_sha256":config_hash,
                                                   "arm":arm,"rows":[]}
        if report["settings_sha256"] != config_hash or report["arm"] != arm:
            raise ValueError("resume_settings_mismatch")
        done = {r["task_id"] for r in report["rows"]}
        if len(done) != len(report["rows"]) or not done <= set(tasks):
            raise ValueError("resume_rows_invalid")
        for offset,(task_id, base) in enumerate(base_rows.items()):
            if task_id in done:
                continue
            query = v120_rows[task_id]["candidate_queries"][0]
            messages = [Message(**m) for m in base["transitions"][0]["messages"]]
            expected = [str(d.get("id")) for d in (v120_rows[task_id]["transition_trace"][0].get("top3") or [])]
            row = await run_case(client, registry, arm, messages, tasks[task_id]["question"], query,
                                 42 + offset * 17, expected)
            score = answer_metrics(row["candidate_answer"], tasks[task_id]["answer"]) if row["candidate_answer"] else {"normalized_exact_match":0.0,"token_f1":0.0}
            row.update({"task_id":task_id,"candidate_em":score["normalized_exact_match"],
                        "candidate_f1":score["token_f1"],"baseline_em":base["normalized_exact_match"],
                        "v120_em":v120_rows[task_id]["candidate_em"]})
            report["rows"].append(row); save(path, report)
        report["complete_coverage"] = len(report["rows"]) == 50
        report["metrics"] = metrics(report["rows"]); save(path, report)
        reports[arm] = report
    winner = choose(reports)
    summary = {"version":"v122-fixed-summary-v1","settings_sha256":config_hash,
               "arms":{arm:report["metrics"] for arm,report in reports.items()},
               "selected_arm":winner,"independent100_opened":False,"final_test_accessed":False}
    save(output / "fixed-summary.json", summary)
    if args.stage != "full-e2e":
        return summary
    if winner is None:
        raise ValueError("no_fixed_state_candidate_passed")
    audit_path = Path(args.audit_report or "")
    if not args.audit_report or not audit_path.is_file():
        raise ValueError("independent_semantic_audit_required")
    audit = read(audit_path)
    if (audit.get("settings_sha256") != config_hash or audit.get("arm") != winner or
            audit.get("complete_coverage") is not True or audit.get("unresolved_support_disputes") != 0 or
            audit.get("confirmed_semantic_degradation") is True or audit.get("approved_for_full_e2e") is not True):
        raise ValueError("semantic_audit_gate_failed")
    result = {"version":"v122-full-e2e-v1","settings_sha256":config_hash,"arm":winner,"rows":[]}
    full_path = output / "full-e2e.json"
    for offset,(task_id,base) in enumerate(base_rows.items()):
        messages = [Message(**m) for m in base["transitions"][0]["messages"]]
        row = await run_case(client, registry, winner, messages, tasks[task_id]["question"], None, 42 + offset * 19)
        score = answer_metrics(row["candidate_answer"], tasks[task_id]["answer"]) if row["candidate_answer"] else {"normalized_exact_match":0.0,"token_f1":0.0}
        row.update({"task_id":task_id,"candidate_em":score["normalized_exact_match"],
                    "candidate_f1":score["token_f1"],"baseline_em":base["normalized_exact_match"],
                    "v120_em":v120_rows[task_id]["candidate_em"]})
        result["rows"].append(row); save(full_path,result)
    result["complete_coverage"] = len(result["rows"]) == 50
    result["metrics"] = metrics(result["rows"]); save(full_path,result)
    summary["full_e2e"] = {"metrics":result["metrics"],
                           "numeric_gates":full_numeric_gate(result["metrics"],
                               sum(float(row["candidate_f1"]) for row in v120_rows.values()) / 50),
                           "semantic_review_pending":True}
    save(output / "fixed-summary.json",summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("baseline-tasks","v120-report","private-tasks","corpus","train-data",
                 "model-path","adapter-path","output-dir"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--stage",choices=("preflight","fixed","full-e2e"),required=True)
    parser.add_argument("--audit-report")
    print(json.dumps(asyncio.run(run(parser.parse_args())),ensure_ascii=False,indent=2))
