"""Recover AC correction-path construction after finding sorted JSONL key order."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import run_v133_goal_ac_failure_state_repair as ac


def product_order_messages_for(case: dict, until_step: int, tok) -> list:
    from mini_agent_rl.domain import Message, ToolCall
    from mini_agent_rl.prompts import system_prompt
    pc = ac.q.load_config()["product"]
    messages = [Message(role="system", content=system_prompt(False, pc["system_prompt"])),
                Message(role="user", content=case["question"])]
    for item in case["trace"][:until_step]:
        if item.get("action") != "search":
            raise RuntimeError("A correction prefix cannot continue after an answer")
        query = item["query"]
        raw = json.dumps({"action": "search", "query": query}, ensure_ascii=False)
        observation = {k: v for k, v in item["observation"].items() if k != "latency_ms"}
        if "result" in observation:
            # The stored rollout JSON sorts dictionary keys. The product's live
            # SearchTool order is id,title,text,score and is part of token replay.
            observation["result"] = [{k: doc[k] for k in ("id", "title", "text", "score") if k in doc}
                                      for doc in observation["result"]]
        call = ToolCall(name="search", arguments={"query": query})
        messages.extend([Message(role="assistant", content=raw, tool_calls=[call]),
            Message(role="tool", content=json.dumps(observation, ensure_ascii=False), tool_call_id=call.id)])
    return messages


def exact_action_ids(tok, action: str, query: str | None = None, answer: str | None = None):
    if action == "search":
        raw = json.dumps({"action": "search", "query": query}, ensure_ascii=False)
    else:
        raw = json.dumps({"action": "answer", "answer": answer}, ensure_ascii=False)
    ids = tok.encode(raw, add_special_tokens=False)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(end, int) or end < 0:
        raise RuntimeError("Tokenizer message terminator missing")
    return raw, ids + [end]


def main() -> None:
    state = ac.z.read(ac.STATE)
    if state["status"] not in ("awaiting_extra_audit", "repair_path_construction"):
        raise RuntimeError(f"Unexpected AC stage: {state['status']}")
    empty = ac.OUT / "repair-candidates.jsonl"
    if empty.exists():
        if empty.stat().st_size:
            raise RuntimeError("Expected no successful path rows before replay repair")
        archived = ac.OUT / "repair-candidates-context-replay-failed.jsonl"
        if archived.exists():
            raise FileExistsError(archived)
        empty.rename(archived)
    for name in ("repair-material-decision.json", "selected-repair-paths.jsonl"):
        src = ac.OUT / name
        dest = ac.OUT / (name.replace(".json", "-before-replay-fix.json"))
        if src.exists():
            if dest.exists():
                raise FileExistsError(dest)
            src.rename(dest)
    ac.messages_for = product_order_messages_for
    ac.action_ids = exact_action_ids
    original_audits = ac.audited_inputs
    audit_values = original_audits()
    eligible = [(key, item) for key, item in audit_values.items() if item.get("repairable_within_budget")
        and item.get("error_supported") is True and item.get("event") in
        ("premature_answer", "wrong_bridge_query", "unnecessary_search")]
    eligible.sort(key=lambda p: hashlib.sha256(("goal-ac-repair-v1\0" + p[0]).encode()).hexdigest())
    # The previous attempt executed one real search, then raised before its
    # result was persisted because action_ids called the SFT processor API on a
    # bare tokenizer. Exclude that one attempted row so it is never rerun.
    attempted = eligible[0][0]
    audit_values[attempted]["repairable_within_budget"] = False
    audit_values[attempted]["main_adjudication_note"] = "excluded after one correction search was executed but not persisted during a code error; no retry"
    ac.audited_inputs = lambda: audit_values
    ac.write(ac.OUT / "correction-attempt-ledger.json", {"case_id": attempted,
        "event": audit_values[attempted]["event"], "attempt_count": 1,
        "search_output_persisted": False, "disposition": "excluded without retry"})
    ac.set_state("repair_path_construction", replay_fix="tool-result field order restored")
    result = ac.select_and_build_repairs()
    ac.write(ac.OUT / "replay-fix-manifest.json", {
        "utility_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repair_runner_sha256": ac.digest(Path(ac.__file__)),
        "source": "Goal AC frozen original AB on-policy trajectories only",
        "extra_candidate_rollouts_already_collected_due_to_prior_validation_shortfall": 128,
        "extra_candidates_used": False,
        "result": result})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
