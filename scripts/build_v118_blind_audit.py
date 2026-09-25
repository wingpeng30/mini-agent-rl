"""Make anonymous, action-time evidence packets for the v117 internal evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

MODELS = ("v101", "v112", "v118")


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def readl(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(s): return " ".join(re.sub(r"[^\w]+", " ", str(s).casefold()).split())
def dump_jsonl(path, rows): Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"))+"\n" for r in rows), encoding="utf-8")


def anonymous_order(case_id):
    return sorted(MODELS, key=lambda model: hashlib.sha256(("v118-blind:"+case_id+":"+model).encode()).hexdigest())


def clean_message(message):
    result = {"role": message.get("role", "")}
    if message.get("content"): result["content"] = message["content"]
    return result


def action_time_visible_messages(state):
    """Return the frozen pre-action conversation without dropping its final prompt/evidence."""
    messages = state.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") == "assistant":
        raise ValueError("invalid_action_time_messages")
    return [clean_message(message) for message in messages]


def verify(reports, mode):
    for report in reports:
        if report.get("version") != "v118" or report.get("mode") != mode or report.get("complete_coverage") is not True:
            raise ValueError("incomplete_or_wrong_report:"+mode)
    base = reports[0]
    for report in reports[1:]:
        for key in ("corpus_sha256", "corpus_documents", "retriever", "runner_config", "model_files", "code_sha256", "private_inputs", "determinism_env"):
            if report.get(key) != base.get(key): raise ValueError("audit_environment_mismatch:"+key)
        cfg1, cfg2 = dict(report["config"]), dict(base["config"])
        cfg1.pop("adapter_path", None); cfg2.pop("adapter_path", None)
        if cfg1 != cfg2: raise ValueError("model_config_mismatch")


def build(args):
    state_reports = [read(getattr(args, model+"_states")) for model in MODELS]
    task_reports = [read(getattr(args, model+"_tasks")) for model in MODELS]
    verify(state_reports, "states"); verify(task_reports, "tasks")
    states = {row["state_id"]: row for row in readl(args.states)}
    tasks = {row["id"]: row for row in readl(args.tasks)}
    if len(states) != 100 or len(tasks) != 50: raise ValueError("frozen_internal_coverage_required")
    state_maps = [{row["state_id"]: row for row in report["rows"]} for report in state_reports]
    task_maps = [{row["task_id"]: row for row in report["rows"]} for report in task_reports]
    if any(set(mapping) != set(states) for mapping in state_maps) or any(set(mapping) != set(tasks) for mapping in task_maps):
        raise ValueError("paired_report_coverage_mismatch")
    state_packets, task_packets, private = [], [], {"version": "v118-private-review-map-v1", "states": {}, "tasks": {}}
    for state_id, state in sorted(states.items()):
        audit_id = "s-"+hashlib.sha256(("v118-review:"+state_id).encode()).hexdigest()[:12]
        order, outputs = anonymous_order(state_id), {}
        private["states"][audit_id] = {"state_id": state_id, "task_id": state["task_id"],
            "anonymous_to_model": {chr(65+i): model for i, model in enumerate(order)}}
        for index, model in enumerate(order):
            observed = state_maps[MODELS.index(model)][state_id]
            outputs[chr(65+index)] = {"action": observed.get("actual_action"), "raw_output": observed.get("raw_output"),
                "parse_error": observed.get("parse_error"),
                "pre_action_visible_messages": action_time_visible_messages(state),
                "post_action_retrieval_results": observed.get("retrieval_results", [])}
        state_packets.append({"audit_id": audit_id, "question": state.get("question"), "anonymous_outputs": outputs,
            "review_request": "Judge each action using only pre_action_visible_messages. For searches separately assess necessity, query justification, and whether post_action_retrieval_results add relevant missing facts; these later results cannot support the prior action. For answers check entity, relation, time, quantity/unit and scope. Quote exact text for any support claim; mark uncertain when evidence does not decide. Ignore protocol_valid as a recommendation and independently assess the shown action."})
    changed = []
    for task_id, task in sorted(tasks.items()):
        rows = [task_maps[MODELS.index(model)][task_id] for model in MODELS]
        answers = [row.get("final_answer") or "" for row in rows]
        if len({norm(answer) for answer in answers}) == 1: continue
        audit_id = "t-"+hashlib.sha256(("v118-review:"+task_id).encode()).hexdigest()[:12]
        order, outputs = anonymous_order(task_id), {}
        private["tasks"][audit_id] = {"task_id": task_id, "anonymous_to_model": {chr(65+i): model for i, model in enumerate(order)}}
        for index, model in enumerate(order):
            row = task_maps[MODELS.index(model)][task_id]
            transitions = row.get("transitions", [])
            answer_step = next((t for t in reversed(transitions)
                if (t.get("response") or {}).get("content", "").strip() and not (t.get("response") or {}).get("tool_calls")), None)
            outputs[chr(65+index)] = {"final_answer": row.get("final_answer"),
                "answer_time_visible_messages": [clean_message(m) for m in (answer_step or {}).get("messages", [])],
                "no_answer_transition": answer_step is None}
        changed.append({"audit_id": audit_id, "question": task["question"], "anonymous_outputs": outputs,
            "review_request": "Assess each final answer using only that output's answer_time_visible_messages, which are the messages visible immediately before its answer. Do not use any evidence fetched later or infer the hidden gold/score. Check requested entity, relation, time, quantity/unit and scope; quote exact support or contradiction and mark uncertainty explicitly."})
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out/"states-blind.jsonl", state_packets); dump_jsonl(out/"changed-tasks-blind.jsonl", changed)
    (out/"private-mapping.json").write_text(json.dumps(private, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    manifest = {"version": "v118-blind-audit-packets-v1", "state_cases": len(state_packets), "changed_task_cases": len(changed),
        "state_packet_sha256": sha(out/"states-blind.jsonl"), "changed_task_packet_sha256": sha(out/"changed-tasks-blind.jsonl"),
        "inputs": {model+"_states": sha(getattr(args, model+"_states")) for model in MODELS}
            | {model+"_tasks": sha(getattr(args, model+"_tasks")) for model in MODELS}}
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("states", "tasks", "output"): parser.add_argument("--"+name, required=True)
    for model in MODELS:
        parser.add_argument("--"+model+"-states", required=True); parser.add_argument("--"+model+"-tasks", required=True)
    print(json.dumps(build(parser.parse_args()), ensure_ascii=False, indent=2))
