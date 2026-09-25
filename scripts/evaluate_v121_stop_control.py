"""Frozen v117 development ablation for evidence sufficiency and query constraints.

No training and no sealed evaluation data. All policy variants share the v118
adapter and fixed v117 corpus/tasks. This script records the fully rendered
prompt token IDs and the actual compacted evidence excerpts used at each call.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from mini_agent_rl.domain import Message, ToolCall
from mini_agent_rl.evaluation import answer_metrics
from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient
from mini_agent_rl.tools import SearchTool, ToolRegistry
from scripts.replay_v119_bridge import EXPECTED_V118_ADAPTER, model_files, read, save
from scripts.evaluate_v120_evidence_gap import POLICY as V120_POLICY, action


CHECKER_PROMPT = (
    "Review the question against only the evidence visible in this conversation. "
    "Return an answer action whose answer is a compact JSON object: "
    '{"status":"complete|missing|conflict","facts":[{"fact":"supported required fact",'
    '"evidence_id":"visible document ID or question","quote":"verbatim evidence sentence"}],'
    '"anchor":"entity explicitly supported by a cited relation","anchor_evidence_id":"visible document ID or question",'
    '"anchor_quote":"verbatim evidence sentence","required":"all requested properties including time/scope/unit",'
    '"missing":"the one missing relation, or empty","query":"search query naming anchor and missing property"}. '
    "Cite only quotes visible before this action. Entity names in unrelated documents do not support the anchor. "
    "An answer word appearing without the requested relation does not make evidence complete."
)

QUERY_REVIEW_PROMPT = (
    "Check whether the proposed search query targets a fact still missing for the question. "
    "Use only this conversation. Return an answer action whose answer is a JSON object: "
    '{"valid":true|false,"facts":[{"fact":"known fact","evidence_id":"visible document ID or question",'
    '"quote":"verbatim sentence"}],'
    '"anchor":"supported entity","anchor_evidence_id":"visible document ID or question",'
    '"anchor_quote":"verbatim evidence sentence",'
    '"missing":"requested property not yet supported","query":"concise query with anchor and missing property"}. '
    "The anchor must appear in the question or in anchor_quote; do not substitute a same-name entity. "
    "A query that repeats an already-supported relation is invalid."
)

CONTROLLER_SYSTEM = (
    "You are an evidence auditor, not an acting search agent. Never issue a search or any tool call. "
    "Return exactly one answer action with a JSON object in its answer field. Follow the requested schema. "
    "Use only the question and evidence already present in the conversation."
)

FORCE_ANSWER = (
    "Evidence review marks the requested facts complete. Answer only the requested entity or property, "
    "with its required time, scope, and units. Cite no unsupported detail. Return one JSON answer action."
)
FORCE_SEARCH = (
    "Evidence review found a missing required relation. Search once for that relation using the supported "
    "anchor entity and requested property stated below. Do not answer from a related entity alone."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qkey(text: str) -> str:
    return " ".join(re.sub(r"\s+", " ", text.casefold()).split())


def parse_envelope(response) -> dict[str, Any] | None:
    if response.parse_error or response.usage.get("output_truncated") or response.tool_calls:
        return None
    try:
        # The model client already parses the outer action envelope. For an
        # answer action, response.content is the answer payload itself.
        inner = json.loads(response.content)
        return inner if isinstance(inner, dict) else None
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


def visible_evidence(messages: list[Message], context_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover only post-budget document text using the model's exact truncation audit."""
    changes = context_record.get("document_changes") or []
    tool_messages = [m for m in messages if m.role == "tool"]
    out: list[dict[str, Any]] = []
    for change in changes:
        if not change.get("visible") or change.get("deleted") or change.get("reused_reference"):
            continue
        index = change.get("tool_index", -1)
        if index < 0 or index >= len(tool_messages):
            continue
        raw = json.loads(tool_messages[index].content)
        doc = next((d for d in raw.get("result", []) if str(d.get("id", "")) == change.get("document_id")), None)
        if doc is None:
            continue
        text = str(change.get("original_text", ""))
        spans = change.get("visible_segments") or []
        excerpt = "".join(text[int(a):int(b)] for a, b in spans)
        if not excerpt:
            continue
        out.append({"id": change["document_id"], "title": change.get("title", ""), "text": excerpt,
                    "content_sha256": change.get("content_sha256"), "visible_segments": spans})
    return out


def valid_anchor(check: dict[str, Any], question: str, evidence: list[dict[str, Any]]) -> bool:
    anchor = str(check.get("anchor", "")).strip().casefold()
    quote = str(check.get("anchor_quote", "")).strip()
    source = str(check.get("anchor_evidence_id", "")).strip()
    if not anchor or not quote or not source:
        return False
    if anchor not in question.casefold() and not any(anchor in d["text"].casefold() for d in evidence):
        return False
    quote_is_supported = any(str(d["id"]) == source and quote in d["text"] and anchor in d["text"].casefold()
                             for d in evidence)
    if source == "question":
        quote_is_supported = quote in question and anchor in question.casefold()
    if not quote_is_supported:
        return False
    facts = check.get("facts")
    if not isinstance(facts, list):
        return False
    for fact in facts:
        if not isinstance(fact, dict) or not fact.get("fact") or not fact.get("quote") or not fact.get("evidence_id"):
            return False
        fact_quote = str(fact["quote"])
        fact_source = str(fact["evidence_id"])
        if fact_source == "question":
            valid_fact_quote = fact_quote in question
        else:
            valid_fact_quote = any(str(d["id"]) == fact_source and fact_quote in d["text"] for d in evidence)
        if not valid_fact_quote:
            return False
    if check.get("status") == "complete" and (not facts or not evidence):
        return False
    return True


def record_call(response, context_record: dict[str, Any], role: str, action_value: str | None = None) -> dict[str, Any]:
    ids = list(response.prompt_token_ids or [])
    mask = list(response.prompt_attention_mask or [])
    if not ids or len(ids) > 2048 or len(mask) != len(ids) or any(x not in (0, 1) for x in mask):
        raise RuntimeError("actual_prompt_token_or_attention_mask_invalid")
    action_ids = list(response.action_token_ids or [])
    if len(action_ids) > (256 if role in {"sufficiency_check", "query_review"} else 128):
        raise RuntimeError("generated_action_exceeds_frozen_token_cap")
    return {"role": role, "raw_output": response.raw_text,
            "action": action_value or ("search" if response.tool_calls else "answer" if response.content.strip() else "invalid"),
            "prompt_sha256": response.usage.get("prompt_sha256"),
            "prompt_token_ids": ids, "attention_mask": mask,
            "prompt_tokens": response.usage.get("prompt_tokens"),
            "context_usage": context_record,
            "visible_evidence": context_record.get("visible_evidence", [])}


def generation_context_record(records: list[dict[str, Any]], response_usage: dict[str, Any]) -> dict[str, Any]:
    """Select the actual generated prompt, not the later old-logprob replay."""
    for record in reversed(records):
        if record.get("phase") == "rollout_sampling":
            return dict(record)
    return dict(response_usage or {})


def build_controller_input(client, messages: list[Message], instruction: str):
    """Isolate an audit turn from the policy's prior action momentum.

    Evidence is first reconstructed from the exact compact-v3 view available to
    the acting policy. The checker receives only those post-truncation excerpts,
    with IDs for quote binding and without titles or later search results.
    """
    question = next(m.content for m in messages if m.role == "user")
    source_messages = list(messages) + [Message(role="user", content=instruction)]
    source_encoded = client._inputs(source_messages, "v121_evidence_snapshot")
    evidence = visible_evidence(source_messages, source_encoded._context_usage)
    packet = [{"id": d["id"], "text": d["text"]} for d in evidence]
    neutral = [Message(role="system", content=CONTROLLER_SYSTEM),
               Message(role="user", content="Question:\n" + question +
                   "\nVisible evidence excerpts (use only these):\n" +
                   json.dumps(packet, ensure_ascii=False, separators=(",", ":")) +
                   "\nReview request:\n" + instruction)]
    # Fail closed if the reconstructed audit packet does not fit intact.
    encoded = client._inputs(neutral, "v121_controller_input_check")
    if int(encoded._context_usage.get("actual_prompt_tokens", 0)) > 2048:
        raise RuntimeError("controller_evidence_packet_exceeds_input_budget")
    return neutral, evidence


async def model_call(client, messages, tools, state, role, seed):
    # One client/model is reused; controller calls are capped at 256 tokens and
    # normal actions at 128. Calls are sequential, so the mutable cap cannot race.
    client.config.max_new_tokens = 256 if role in {"sufficiency_check", "query_review"} else 128
    client.reset_context_records()
    response = await client.generate(messages, tools, temperature=0, seed=seed)
    rows = client.context_usage_summary().get("records", [])
    context = generation_context_record(rows, response.usage)
    context["visible_evidence"] = visible_evidence(messages, context)
    item = record_call(response, context, role)
    state["calls"].append(item)
    return response, item


async def controller(client, messages, registry, arm, state, seed):
    tools = registry.definitions()
    question = next(m.content for m in messages if m.role == "user")
    check = None
    evidence = []
    has_prior_results = any(m.role == "tool" for m in messages)
    if arm in {"C", "D"} and has_prior_results:
        check_messages, check_evidence = build_controller_input(client, messages, CHECKER_PROMPT)
        response, item = await model_call(client, check_messages, [], state, "sufficiency_check", seed)
        check = parse_envelope(response)
        evidence = check_evidence
        item["visible_evidence"] = evidence
        valid = bool(check and valid_anchor(check, question, evidence) and
                     check.get("status") in {"complete", "missing", "conflict"} and check.get("required"))
        item["parsed_check"] = check
        item["check_valid"] = valid
        if not valid:
            state["controller_failure"] = "invalid_sufficiency_check"
            return None, "invalid_controller"
        if check["status"] == "conflict":
            state["controller_failure"] = "evidence_conflict"
            return None, "insufficient_evidence"

    # Propose an action with the unchanged v120 policy prompt.
    action_messages = list(messages)
    if check and check["status"] == "missing":
        action_messages.append(Message(role="user", content=FORCE_SEARCH + " " + json.dumps(check, ensure_ascii=False)))
    elif check and check["status"] == "complete":
        action_messages.append(Message(role="user", content=FORCE_ANSWER + " " + json.dumps(check, ensure_ascii=False)))
    response, item = await model_call(client, action_messages, tools, state, "policy_action", seed + 1)
    kind = action(response)
    item["action"] = kind
    if kind == "invalid":
        return None, "invalid_action"
    if check and check["status"] == "complete" and kind == "search":
        # S says all required facts are supported; stop and ask the same model to answer.
        answer_messages = messages + [Message(role="user", content=FORCE_ANSWER + " " + json.dumps(check, ensure_ascii=False))]
        response, item = await model_call(client, answer_messages, tools, state, "answer_after_complete", seed + 2)
        kind = action(response)
        item["action"] = kind
        if kind != "answer":
            return None, "invalid_after_complete"
        return response.content.strip(), "answer"
    if check and check["status"] == "missing" and kind == "answer":
        # Enforce the missing-fact decision once; a second disagreement is a failure.
        search_messages = messages + [Message(role="user", content=FORCE_SEARCH + " " + json.dumps(check, ensure_ascii=False))]
        response, item = await model_call(client, search_messages, tools, state, "search_after_missing", seed + 2)
        kind = action(response)
        item["action"] = kind
        if kind != "search":
            return None, "insufficient_evidence"
    if kind == "answer":
        return response.content.strip(), "answer"
    query = response.tool_calls[0].arguments["query"].strip()

    if arm in {"B", "D"}:
        source_check = check
        if source_check is None:
            query_messages, query_evidence = build_controller_input(
                client, messages, QUERY_REVIEW_PROMPT + " Proposed query: " + query)
            qresponse, qitem = await model_call(client, query_messages, [], state, "query_review", seed + 3)
            source_check = parse_envelope(qresponse)
            evidence = query_evidence
            qitem["visible_evidence"] = evidence
            valid = bool(source_check and valid_anchor(source_check, question, evidence) and
                         source_check.get("missing") and source_check.get("query"))
            qitem["parsed_query_review"] = source_check
            qitem["query_review_valid"] = valid
            if not valid:
                state["controller_failure"] = "invalid_query_review"
                return None, "invalid_controller"
        if not source_check.get("missing"):
            return None, "invalid_query_review"
        parts = [str(source_check.get("anchor", "")).strip(),
                 str(source_check.get("missing", "")).strip()]
        required = str(source_check.get("required", "")).strip()
        if required and required.casefold() not in " ".join(parts).casefold():
            parts.append(required)
        rewritten = " ".join(part for part in parts if part)
        if not rewritten:
            return None, "invalid_query_review"
        qkey_prev = qkey(query)
        query = rewritten
        state["query_rewrites"].append({"from": qkey_prev, "to": qkey(query), "reason": "program_assembled_anchor_missing_required"})

    if not query:
        return None, "invalid_action"
    return query, "search"


async def run(args):
    import os
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("determinism_environment_required_before_process_start")
    base_path, corpus_path, private_path = map(Path, (args.baseline_tasks, args.corpus, args.private_tasks))
    baseline, corpus = read(base_path), read(corpus_path)
    v120 = read(args.v120_report)
    if baseline.get("version") != "v118" or baseline.get("mode") != "tasks" or not baseline.get("complete_coverage"):
        raise ValueError("complete_v118_task_baseline_required")
    if len(baseline.get("rows", [])) != 50 or len(corpus) != 1986 or len(v120.get("rows", [])) != 50:
        raise ValueError("frozen_v117_50_and_1986_corpus_required")
    if sha(corpus_path) != baseline.get("corpus_sha256"):
        raise ValueError("frozen_corpus_hash_mismatch")
    adapter_path = Path(args.adapter_path)
    if sha(adapter_path / "adapter_model.safetensors") != EXPECTED_V118_ADAPTER:
        raise ValueError("v118_adapter_hash_mismatch")
    if model_files(args.model_path) != baseline.get("model_files"):
        raise ValueError("base_model_hash_mismatch")
    tasks = {r["id"]: r for r in (json.loads(line) for line in private_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    rows = {r["task_id"]: r for r in baseline["rows"]}
    v120_rows = {r["task_id"]: r for r in v120["rows"]}
    if set(tasks) != set(rows) or set(v120_rows) != set(rows) or not v120.get("complete_coverage"):
        raise ValueError("private_task_id_mismatch")

    cfg = LocalModelConfig(model_id=args.model_path, adapter_path=args.adapter_path, device="cuda", dtype="bfloat16",
        load_in_4bit=False, max_input_tokens=2048, max_new_tokens=128, enable_thinking=False, timeout=600,
        gradient_checkpointing=False, observation_encoding="compact-v3", deterministic=True,
        local_files_only=True, trust_remote_code=False, logprob_mode="bf16")
    client = TransformersModelClient(cfg)
    registry = ToolRegistry(); registry.register(SearchTool(corpus, "bm25-v2"))
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    identity = {"version": "v121-stop-control-dev-v1", "exploratory_development_only": True,
        "smoke_only": bool(args.smoke_only),
        "source_sha256": {"v118_tasks": sha(base_path), "v120_report": sha(Path(args.v120_report)), "private_tasks": sha(private_path), "corpus": sha(corpus_path)},
        "adapter_sha256": EXPECTED_V118_ADAPTER, "model_files": baseline["model_files"],
        "script_sha256": sha(Path(__file__)),
        "policy_sha256": {"v120": hashlib.sha256(V120_POLICY.encode()).hexdigest(),
            "controller_system": hashlib.sha256(CONTROLLER_SYSTEM.encode()).hexdigest(),
            "query_review": hashlib.sha256(QUERY_REVIEW_PROMPT.encode()).hexdigest(),
            "sufficiency_check": hashlib.sha256(CHECKER_PROMPT.encode()).hexdigest(),
            "force_answer": hashlib.sha256(FORCE_ANSWER.encode()).hexdigest(),
            "force_search": hashlib.sha256(FORCE_SEARCH.encode()).hexdigest()},
        "seed": 42, "temperature": 0, "retriever": "bm25-v2 top3", "max_input_tokens": 2048,
        "max_searches": 2, "thinking": False,
        "controller_max_new_tokens": 256, "action_max_new_tokens": 128,
        "arms": list(args.arms), "independent100_opened": False, "final_test_accessed": False}

    if args.stage == "full-e2e" and args.smoke_only:
        raise ValueError("smoke_only_cannot_run_full_e2e")
    task_rows = list(rows.items())[:1] if args.smoke_only else list(rows.items())
    reports = {}
    for arm in identity["arms"]:
        path = output_dir / f"arm-{arm}.json"
        report = read(path) if path.exists() else {**identity, "arm": arm, "rows": []}
        if any(report.get(k) != v for k, v in identity.items()):
            raise ValueError(f"resume_freeze_mismatch_{arm}")
        complete = {r["task_id"] for r in report["rows"]}
        if len(complete) != len(report["rows"]) or not complete <= set(rows):
            raise ValueError(f"invalid_resume_set_{arm}")
        for offset, (task_id, baseline_row) in enumerate(task_rows):
            if task_id in complete:
                continue
            if len(baseline_row["transitions"]) < 2:
                raise ValueError("first_search_state_missing")
            transitions = baseline_row["transitions"]
            v120_row = v120_rows[task_id]
            if not v120_row.get("candidate_queries") or not transitions[0].get("messages"):
                raise ValueError("v120_first_search_state_missing")
            messages = [Message(**m) for m in transitions[0]["messages"]]
            if messages[0].role != "system":
                raise ValueError("system_message_missing")
            messages[0] = Message(role="system", content=messages[0].content + " " + V120_POLICY)
            first_query = str(v120_row["candidate_queries"][0]).strip()
            first_obs = await registry.execute("search", {"query": first_query})
            if not first_obs.get("ok"):
                raise ValueError("v120_first_search_replay_failed")
            expected_top3 = [d.get("id") for d in (v120_row["transition_trace"][0].get("top3") or [])]
            actual_top3 = [d.get("id") for d in (first_obs.get("result") or [])]
            if expected_top3 != actual_top3:
                raise ValueError("v120_first_search_retrieval_mismatch")
            first_obs["searched_queries"] = [qkey(first_query)]
            tool_id = f"v121-first-{arm}-{task_id}"
            messages += [Message(role="assistant", content=json.dumps({"action":"search","query":first_query},ensure_ascii=False),
                          tool_calls=[ToolCall(id=tool_id,name="search",arguments={"query":first_query})]),
                         Message(role="tool",content=json.dumps({k:v for k,v in first_obs.items() if k!="latency_ms"},ensure_ascii=False),
                          tool_call_id=tool_id)]
            state = {"task_id": task_id, "calls": [], "query_rewrites": [], "search_queries": [first_query],
                     "retrieved_docs": [{"query":first_query,"documents":first_obs.get("result",[])}],
                     "step_trace": [{"query": first_query, "top3": [{"id":d.get("id"),"title":d.get("title")} for d in first_obs.get("result",[])],
                                     "document_sha256":[hashlib.sha256(str(d.get("text","")).encode()).hexdigest() for d in first_obs.get("result",[])]}],
                     "protocol_valid": True, "termination": "max_actions"}
            final = ""
            # Shared fixed first search; at most one new search follows it.
            for decision in range(4):
                outcome, kind = await controller(client, messages, registry, arm, state, 42 + offset * 17 + decision * 5)
                if kind == "answer":
                    final = outcome or ""; state["termination"] = "answer"; break
                if kind != "search":
                    state["termination"] = kind
                    if kind in {"invalid_action", "invalid_after_complete", "invalid_controller", "invalid_query_review"}:
                        state["protocol_valid"] = False
                    break
                query = outcome or ""
                previous = {qkey(q) for q in state["search_queries"]}
                if len(state["search_queries"]) >= 2:
                    state["termination"] = "search_limit_insufficient"; break
                if qkey(query) in previous:
                    state["blocked_repeat"] = state.get("blocked_repeat", 0) + 1
                    # A repeat is unexecuted and consumes no retrieval; permit one repair only.
                    if state["blocked_repeat"] > 1:
                        state["termination"] = "repeat_repair_exhausted"; break
                    repair_messages, repair_evidence = build_controller_input(
                        client, messages, QUERY_REVIEW_PROMPT + " The proposed query repeats an executed search. Propose one distinct query for the missing relation.")
                    repair, ritem = await model_call(client, repair_messages, [], state, "repeat_repair", 42 + offset * 17 + decision * 5 + 4)
                    ritem["visible_evidence"] = repair_evidence
                    parsed = parse_envelope(repair)
                    ritem["parsed_repair"] = parsed
                    if (not parsed or not valid_anchor(parsed, tasks[task_id]["question"], ritem.get("visible_evidence", [])) or
                        not parsed.get("query") or qkey(parsed["query"]) in previous):
                        state["termination"] = "repeat_repair_invalid"; break
                    query = str(parsed["query"]).strip()
                state["search_queries"].append(query)
                observation = await registry.execute("search", {"query": query})
                if not observation.get("ok"):
                    state["termination"] = "tool_error"; state["protocol_valid"] = False; break
                docs = observation.get("result") or []
                state["retrieved_docs"].append({"query": query, "documents": docs})
                state["step_trace"] = state.get("step_trace", []) + [{"query": query,
                    "top3": [{"id": d.get("id"), "title": d.get("title")} for d in docs],
                    "document_sha256": [hashlib.sha256(str(d.get("text", "")).encode()).hexdigest() for d in docs]}]
                observation["searched_queries"] = sorted(qkey(q) for q in state["search_queries"])
                model_observation = {k: v for k, v in observation.items() if k != "latency_ms"}
                # SearchTool observations carry a tool-call id only when the policy
                # issued the call. Reuse the most recent valid tool call from trace.
                tool_response = next((call for call in reversed(state["calls"]) if call["action"] == "search"), None)
                tool_id = f"v121-{arm}-{task_id}-{decision}"
                messages += [Message(role="assistant", content=json.dumps({"action": "search", "query": query}, ensure_ascii=False),
                                     tool_calls=[ToolCall(id=tool_id, name="search", arguments={"query": query})]),
                             Message(role="tool", content=json.dumps(model_observation, ensure_ascii=False), tool_call_id=tool_id)]
                state["termination"] = "search_then_decide"
                if len(state["search_queries"]) == 1 and kind == "search":
                    continue
            metrics = answer_metrics(final, tasks[task_id]["answer"]) if final else {"normalized_exact_match": 0.0, "token_f1": 0.0}
            state.update({"candidate_answer": final, "candidate_em": metrics["normalized_exact_match"],
                          "candidate_f1": metrics["token_f1"], "candidate_answered": bool(final),
                          "baseline_em": baseline_row["normalized_exact_match"], "baseline_f1": baseline_row["token_f1"],
                          "v120_em": v120_row["candidate_em"]})
            report["rows"].append(state)
            save(path, report)
        report["complete_coverage"] = (not args.smoke_only and len(report["rows"]) == 50 and
                                        {r["task_id"] for r in report["rows"]} == set(rows))
        report["metrics"] = summarize(report["rows"])
        save(path, report)
        reports[arm] = report

    winner = choose_candidate(reports)
    summary = {"version": "v121-stop-control-summary-v1", "identity": identity,
        "arm_metrics": {arm: report["metrics"] for arm, report in reports.items()},
        "selected_arm_for_full_e2e": winner, "selection_gate_passed": winner is not None,
        "independent100_opened": False, "final_test_accessed": False,
        "selection_rule": "EM >= max(29,26), +3 versus A, F1 nondecrease, v118 correct preservation >=14/14, v120 correct preservation >=25/26, protocol >=98%"}
    save(output_dir / "summary.json", summary)
    if args.smoke_only:
        return summary
    if args.stage == "full-e2e":
        if winner is None:
            raise RuntimeError("no_arm_passed_fixed_state_selection_gate")
        if not args.audit_report:
            raise ValueError("offline_audit_report_required_before_full_e2e")
        audit = read(args.audit_report)
        if (audit.get("complete_coverage") is not True or
                audit.get("unresolved_support_disputes") != 0 or
                audit.get("confirmed_semantic_degradation") is True or
                audit.get("approved_for_full_e2e") is not True):
            raise RuntimeError("offline_semantic_audit_did_not_approve_full_e2e")
        full = await run_full_e2e(args, client, registry, tasks, baseline, v120_rows, identity, winner, output_dir)
        summary["full_e2e"] = full
        summary["audit_report_sha256"] = sha(Path(args.audit_report))
        save(output_dir / "summary.json", summary)
    return summary


def summarize(rows):
    em = {r["task_id"] for r in rows if r["candidate_em"] == 1}
    v118 = {r["task_id"] for r in rows if r["baseline_em"] == 1}
    v120 = {r["task_id"] for r in rows if r["v120_em"] == 1}
    return {"n": len(rows), "correct": len(em), "em": len(em) / max(1, len(rows)),
        "mean_f1": sum(r["candidate_f1"] for r in rows) / max(1, len(rows)),
        "v118_correct": len(v118), "preserve_v118": len(em & v118) / max(1, len(v118)),
        "preserve_v120": len(em & v120) / max(1, len(v120)),
        "gains_vs_v118": len(em - v118), "losses_vs_v118": len(v118 - em),
        "net_vs_v118": len(em - v118) - len(v118 - em),
        "protocol_rate": sum(bool(r["protocol_valid"]) for r in rows) / max(1, len(rows)),
        "searches": sum(len(r["search_queries"]) for r in rows),
        "abstentions": sum(r["termination"] == "search_limit_insufficient" for r in rows),
        "terminations": {k: sum(r["termination"] == k for r in rows) for k in sorted({r["termination"] for r in rows})}}


def choose_candidate(reports):
    passing = []
    for arm, report in reports.items():
        m = report["metrics"]
        a = reports["A"]["metrics"]
        if (m["correct"] >= 29 and m["correct"] >= 26 and m["correct"] - a["correct"] >= 3 and
            m["mean_f1"] >= a["mean_f1"] and m["preserve_v118"] >= 1.0 and
            m["preserve_v120"] >= 25/26 and m["protocol_rate"] >= .98):
            passing.append((arm, m["correct"], -m["searches"], -len(report["rows"])))
    return max(passing, key=lambda x: x[1:])[0] if passing else None


async def run_full_e2e(args, client, registry, tasks, baseline, v120_rows, identity, arm, outdir):
    # This secondary run is gated on the preregistered continuation result.
    # The same frozen task order, model, corpus, and two-search cap are used.
    result = {"version": "v121-full-e2e-v1", "arm": arm, "exploratory_development_only": True,
              "independent100_opened": False, "final_test_accessed": False, "rows": []}
    baseline_by_id = {r["task_id"]: r for r in baseline["rows"]}
    for offset, (task_id, base_row) in enumerate(baseline_by_id.items()):
        # Reuse each task's actual pre-first-search messages; do not use private labels in prompts.
        messages = [Message(**m) for m in base_row["transitions"][0]["messages"]]
        messages[0] = Message(role="system", content=messages[0].content + " " + V120_POLICY)
        state = {"task_id": task_id, "calls": [], "query_rewrites": [], "search_queries": [],
                 "retrieved_docs": [], "protocol_valid": True, "termination": "max_actions", "step_trace": []}
        final = ""
        for decision in range(5):
            outcome, kind = await controller(client, messages, registry, arm, state, 42 + offset * 19 + decision * 5)
            if kind == "answer": final = outcome or ""; state["termination"] = "answer"; break
            if kind != "search":
                state["termination"] = kind
                if kind in {"invalid_action", "invalid_after_complete", "invalid_controller", "invalid_query_review"}:
                    state["protocol_valid"] = False
                break
            if len(state["search_queries"]) >= 2:
                state["termination"] = "search_limit_insufficient"
                state["protocol_valid"] = False
                break
            q = str(outcome or "")
            if qkey(q) in {qkey(x) for x in state["search_queries"]}:
                state["blocked_repeat"] = state.get("blocked_repeat", 0) + 1
                if state["blocked_repeat"] > 1: state["termination"] = "repeat_repair_exhausted"; break
                repair_messages, repair_evidence = build_controller_input(
                    client, messages, QUERY_REVIEW_PROMPT +
                    " The proposed query repeats an executed search. Propose one distinct query for the missing relation.")
                repair, ritem = await model_call(client, repair_messages, [], state, "repeat_repair",
                                                  42 + offset * 19 + decision * 5 + 4)
                ritem["visible_evidence"] = repair_evidence
                parsed = parse_envelope(repair); ritem["parsed_repair"] = parsed
                if (not parsed or not valid_anchor(parsed, tasks[task_id]["question"], ritem.get("visible_evidence", [])) or
                    not parsed.get("query") or qkey(parsed["query"]) in {qkey(x) for x in state["search_queries"]}):
                    state["termination"] = "repeat_repair_invalid"; break
                q = str(parsed["query"]).strip()
            state["search_queries"].append(q)
            obs = await registry.execute("search", {"query": q})
            if not obs.get("ok"): state["termination"] = "tool_error"; state["protocol_valid"] = False; break
            state["retrieved_docs"].append({"query": q, "documents": obs.get("result", [])})
            state["step_trace"].append({"query": q, "top3": [{"id":d.get("id"),"title":d.get("title")} for d in obs.get("result", [])]})
            obs["searched_queries"] = sorted(qkey(x) for x in state["search_queries"])
            tid=f"v121-full-{task_id}-{decision}"
            messages += [Message(role="assistant", content=json.dumps({"action":"search","query":q}),
                          tool_calls=[ToolCall(id=tid,name="search",arguments={"query":q})]),
                         Message(role="tool", content=json.dumps({k:v for k,v in obs.items() if k!="latency_ms"},ensure_ascii=False),tool_call_id=tid)]
        score=answer_metrics(final,tasks[task_id]["answer"]) if final else {"normalized_exact_match":0.0,"token_f1":0.0}
        base = baseline_by_id[task_id]
        state.update({"candidate_answer":final,"candidate_em":score["normalized_exact_match"],"candidate_f1":score["token_f1"],
                      "baseline_em":base["normalized_exact_match"],"baseline_f1":base["token_f1"]})
        state["v120_em"] = v120_rows[task_id]["candidate_em"]
        state["candidate_answered"] = bool(final)
        result["rows"].append(state); save(outdir/"full-e2e.json",result)
    result["complete_coverage"] = len(result["rows"]) == 50
    result["metrics"] = summarize(result["rows"])
    save(outdir/"full-e2e.json",result)
    return {k:v for k,v in result.items() if k!="rows"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("baseline-tasks", "v120-report", "private-tasks", "corpus", "model-path", "adapter-path", "output-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--stage", choices=("arms", "full-e2e"), default="arms")
    parser.add_argument("--audit-report")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=("A", "B", "C", "D"), default=("A", "B", "C", "D"))
    print(json.dumps(asyncio.run(run(parser.parse_args())), ensure_ascii=False, indent=2))
