"""One-time supplemental failure audit and repair path builder for Goal AC."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import run_v133_goal_ac_failure_state_repair as ac
import continue_v133_goal_ac_replay_fix as replay


def package_audit() -> dict:
    if ac.z.read(ac.STATE)["status"] != "awaiting_extra_audit":
        raise RuntimeError("Supplemental failure audit is not active")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ac.q.p.BASE, local_files_only=True)
    pool = {x["id"]: x for x in ac.z.read(ac.OUT / "extra-candidate-pool.json")["cases"]}
    outputs = ac.rows(ac.OUT / "extra-on-policy.jsonl")
    if len(outputs) != 128 or len(pool) != 128:
        raise RuntimeError("Supplemental candidate cap mismatch")
    target = ac.OUT / "extra-failure-audit-packet.jsonl"
    if target.exists():
        raise FileExistsError(target)
    count = 0
    for result in outputs:
        case_id = result["case_id"]
        if case_id not in pool:
            raise RuntimeError("Candidate and rollout IDs differ")
        steps = []
        for i, step in enumerate(result["trace"]):
            prompt = tok.decode(step["prompt_token_ids"], skip_special_tokens=False)
            visible_docs = []
            for prior in result["trace"][:i]:
                for doc in prior.get("observation", {}).get("result", []):
                    if ac.source.norm(doc.get("id", "")) in ac.source.norm(prompt) and \
                       ac.source.norm(doc.get("text", "")) in ac.source.norm(prompt):
                        visible_docs.append({"id": doc["id"], "title": doc["title"], "text": doc["text"]})
            steps.append({"step": i, "action": step.get("action"), "query": step.get("query"),
                "answer": step.get("answer"), "actual_prompt": prompt,
                "prompt_token_sha256": hashlib.sha256(json.dumps(step["prompt_token_ids"],
                    separators=(",", ":")).encode()).hexdigest(),
                "visible_documents": visible_docs})
        ac.append(target, {"case_id": case_id, "question": result["question"],
            "steps": steps, "final_answer": result.get("answer"), "final_reason": result.get("reason"),
            "instructions": "Use only each step's actual_prompt and its visible_documents. Classify the earliest decision error and give up to two corrective search queries grounded in the question/current evidence. Do not use outside knowledge or infer hidden answer/support mappings. Query entities must already appear in the question or that query's prior visible evidence."})
        count += 1
    manifest = {"count": count, "sha256": ac.digest(target), "created_at": time.time(),
        "rollout_sha256": ac.digest(ac.OUT / "extra-on-policy.jsonl"),
        "corpus_sha256": ac.digest(ac.OUT / "corpus-extra-train.json")}
    ac.write(ac.OUT / "extra-failure-audit-manifest.json", manifest)
    return manifest


def resume_supplement() -> dict:
    state = ac.z.read(ac.STATE)
    if state["status"] == "collecting_extra_rollouts":
        path = ac.OUT / "extra-on-policy.jsonl"
        if not path.exists() or len(ac.rows(path)) != 128:
            raise RuntimeError("Supplemental rollouts are incomplete; preserve and stop")
        if ac.z.read(ac.OUT / "repair-material-decision.json")["built_paths"] >= ac.MIN_PAIRS:
            raise RuntimeError("Do not use supplemental pool when original material reaches minimum")
        ac.set_state("awaiting_extra_audit", extra_rollouts=128,
            source_material_shortfall_after_exact_replay=ac.z.read(ac.OUT / "repair-material-decision.json"))
    return ac.z.read(ac.STATE)


def combine_extra_audits() -> dict:
    packet = {x["case_id"]: x for x in ac.rows(ac.OUT / "extra-failure-audit-packet.jsonl")}
    a, b = (ac.rows(ac.OUT / name) for name in ("extra-audit-a.jsonl", "extra-audit-b.jsonl"))
    maps = [{x["case_id"]: x for x in items} for items in (a, b)]
    if any(set(m) != set(packet) for m in maps):
        raise RuntimeError("Both reviewers must cover exactly the frozen 128 supplemental cases")
    merged = []
    for case_id, item in packet.items():
        va, vb = maps[0][case_id], maps[1][case_id]
        agreed = va.get("event") == vb.get("event") and va.get("error_step") == vb.get("error_step") and \
                 va.get("repairable_within_budget") is vb.get("repairable_within_budget") and \
                 va.get("error_supported") is vb.get("error_supported")
        allowed = {"premature_answer", "unnecessary_search", "wrong_bridge_query", "retrieval_failure",
                   "sufficient_evidence_wrong_answer", "none", "ambiguous", "no_recoverable_evidence"}
        if va.get("event") not in allowed or vb.get("event") not in allowed:
            raise RuntimeError(f"Invalid reviewer event for {case_id}")
        if agreed and va.get("repairable_within_budget"):
            needs_query = va.get("event") in ("premature_answer", "wrong_bridge_query")
            if not isinstance(va.get("error_step"), int) or not va.get("reason") or \
               (needs_query and not va.get("repair_queries")):
                raise RuntimeError(f"Repairable vote lacks step, reason, or query for {case_id}")
        vote = {"case_id": case_id, "event": va.get("event") if agreed else "ambiguous",
            "error_step": va.get("error_step") if agreed else None,
            "repairable_within_budget": bool(agreed and va.get("repairable_within_budget")),
            "missing_relation": va.get("missing_relation") if agreed else "",
            "bridge_entity": va.get("bridge_entity") if agreed else "",
            "bridge_source": va.get("bridge_source") if agreed else "",
            "repair_queries": va.get("repair_queries", []) if agreed else [],
            "reason": "reviewer A: " + str(va.get("reason", "")) + "\nreviewer B: " + str(vb.get("reason", "")),
            "reviewer_agreement": agreed, "reviewer_a": va, "reviewer_b": vb}
        merged.append(vote)
    ac.write_jsonl(ac.OUT / "extra-failure-audit-final.jsonl", merged)
    summary = {"cases": len(merged), "events": {k: sum(x["event"] == k for x in merged)
        for k in sorted({x["event"] for x in merged})},
        "repairable": sum(x["repairable_within_budget"] for x in merged),
        "stop_error_repairable": sum(x["repairable_within_budget"] and
            x["event"] in ("premature_answer", "unnecessary_search") for x in merged),
        "reviewer_agreements": sum(x["reviewer_agreement"] for x in merged)}
    ac.write(ac.OUT / "extra-failure-audit-summary.json", summary)
    return summary


def build_extra_paths() -> dict:
    import torch
    from transformers import AutoTokenizer
    from mini_agent_rl.domain import Message, ToolCall
    state = ac.z.read(ac.STATE)
    if state["status"] != "awaiting_extra_audit":
        raise RuntimeError("Wrong Goal AC supplemental state")
    if not (ac.OUT / "extra-failure-audit-final.jsonl").exists():
        combine_extra_audits()
    votes = {x["case_id"]: x for x in ac.rows(ac.OUT / "extra-failure-audit-final.jsonl")}
    pool = {x["id"]: x for x in ac.z.read(ac.OUT / "extra-candidate-pool.json")["cases"]}
    rollouts = {x["case_id"]: x for x in ac.rows(ac.OUT / "extra-on-policy.jsonl")}
    tok = AutoTokenizer.from_pretrained(ac.q.p.BASE, local_files_only=True)
    client, _, registry = ac.z.make_client(ac.SOURCE, ac.OUT / "corpus-extra-train.json")
    client._load()
    built, failures = [], {}
    for case_id, vote in votes.items():
        if not vote["repairable_within_budget"] or vote["event"] not in (
            "premature_answer", "wrong_bridge_query", "unnecessary_search"):
            continue
        result, case = rollouts[case_id], pool[case_id]
        trace = result["trace"]
        try:
            index = int(vote["error_step"])
            if index < 0 or index >= len(trace):
                raise ValueError("error_step_invalid")
            expected = trace[index]
            messages = replay.product_order_messages_for({"question": result["question"], "trace": trace}, index, tok)
            encoded = client._inputs(messages, "goal_ac_extra_failure_state")
            prompt_ids, prompt_mask = encoded["input_ids"][0].tolist(), encoded["attention_mask"][0].tolist()
            if prompt_ids != expected["prompt_token_ids"] or prompt_mask != expected["prompt_attention_mask"]:
                raise ValueError("exact_saved_prefix_mismatch")
            answer = case["answer"]
            answer_norm = ac.source.norm(answer)
            actions = []
            prefix_end = index
            if vote["event"] == "unnecessary_search":
                first = next((i for i, x in enumerate(trace[:index]) if x.get("action") == "search"), None)
                if first is None:
                    raise ValueError("missing_insufficient_search_state")
                prefix_end = first
                insufficient = trace[first]
                actions.extend({k: insufficient[k] for k in ("action", "query", "answer",
                    "prompt_token_ids", "prompt_attention_mask", "action_token_ids") if k in insufficient})
                actions.extend({k: x[k] for k in ("action", "query", "answer", "prompt_token_ids",
                    "prompt_attention_mask", "action_token_ids") if k in x}
                    for x in trace[first + 1:index])
                final_ids, final_mask = prompt_ids, prompt_mask
                insuff = {"action": "search", "query": insufficient["query"],
                    "prompt_token_ids": insufficient["prompt_token_ids"],
                    "prompt_attention_mask": insufficient["prompt_attention_mask"],
                    "action_token_ids": insufficient["action_token_ids"], "evidence_source": "actual_prefix"}
            else:
                query_specs = vote.get("repair_queries", [])
                if not query_specs or len(query_specs) > 2:
                    raise ValueError("reviewed_queries_absent_or_over_limit")
                current = list(messages)
                seen = {" ".join(str(x.get("query", "")).casefold().split())
                        for x in trace[:index] if x.get("action") == "search"}
                first_search = None
                for j, spec in enumerate(query_specs):
                    query = spec.get("query", "")
                    if not isinstance(query, str) or not query.strip():
                        raise ValueError("empty_query")
                    context = tok.decode(client._inputs(current, "goal_ac_extra_query")["input_ids"][0].tolist(),
                                         skip_special_tokens=False)
                    normalized = ac.source.norm(context)
                    bridge_source = str(spec.get("bridge_source", ""))
                    if len(answer_norm) >= 5 and answer_norm in ac.source.norm(query) and answer_norm not in normalized:
                        raise ValueError("query_discloses_hidden_answer")
                    if bridge_source:
                        entities = re.findall(r"[A-Z][\w’'-]+(?:\s+[A-Z][\w’'-]+)*", bridge_source)
                        if any(ac.source.norm(x) not in normalized for x in entities):
                            raise ValueError("bridge_not_currently_visible")
                    if " ".join(query.casefold().split()) in seen:
                        raise ValueError("repeat_query")
                    before = client._inputs(current, "goal_ac_extra_search")
                    pids, pmask = before["input_ids"][0].tolist(), before["attention_mask"][0].tolist()
                    raw, action_ids = replay.exact_action_ids(tok, "search", query=query)
                    if j == 0:
                        executed = ac.execute_search_path(client, registry, result["question"], current, query,
                            trace, {"bridge_source": vote.get("bridge_source"),
                                    "missing_relation": vote.get("missing_relation")})
                        obs = executed["observation"]; action_ids = executed["action_token_ids"]
                    else:
                        obs = asyncio.run(registry.execute("search", {"query": query}))
                        if not obs.get("ok"):
                            raise RuntimeError("supplement_search_failed")
                    seen.add(" ".join(query.casefold().split()))
                    obs["searched_queries"] = sorted(seen)
                    call = ToolCall(name="search", arguments={"query": query})
                    current.extend([Message(role="assistant", content=raw, tool_calls=[call]),
                        Message(role="tool", content=json.dumps({k: v for k, v in obs.items() if k != "latency_ms"},
                            ensure_ascii=False), tool_call_id=call.id)])
                    after = client._inputs(current, "goal_ac_extra_answer")
                    action = {"action": "search", "query": query, "prompt_token_ids": pids,
                        "prompt_attention_mask": pmask, "action_token_ids": action_ids,
                        "observation": obs, "query_source": bridge_source}
                    actions.append(action)
                    if first_search is None: first_search = action
                    final_ids, final_mask = after["input_ids"][0].tolist(), after["attention_mask"][0].tolist()
                insuff = {"action": "search", "query": first_search["query"],
                    "prompt_token_ids": first_search["prompt_token_ids"],
                    "prompt_attention_mask": first_search["prompt_attention_mask"],
                    "action_token_ids": first_search["action_token_ids"], "evidence_source": vote.get("bridge_source")}
            raw_answer, answer_ids = replay.exact_action_ids(tok, "answer", answer=answer)
            actions.append({"action": "answer", "answer": answer, "raw": raw_answer,
                "prompt_token_ids": final_ids, "prompt_attention_mask": final_mask,
                "action_token_ids": answer_ids})
            searches = sum(x.get("action") == "search" for x in trace[:prefix_end]) + \
                       sum(x.get("action") == "search" for x in actions)
            calls = prefix_end + len(actions)
            if searches > 3 or calls > 4 or any(len(x["prompt_token_ids"]) > 2048 or
                len(x["action_token_ids"]) > 128 for x in actions):
                raise ValueError("product_budget_exceeded")
            final_text = ac.source.norm(tok.decode(final_ids, skip_special_tokens=False))
            docs_by_step = []
            for step in trace[:prefix_end]:
                if step.get("action") == "search":
                    docs_by_step.append({"step": step.get("step"), "query": step.get("query"),
                        "documents": step.get("observation", {}).get("result", [])})
            for action in actions:
                if action.get("action") == "search":
                    docs_by_step.append({"step": "repair", "query": action.get("query"),
                        "documents": action.get("observation", {}).get("result", [])})
            for group in docs_by_step:
                group["documents"] = [d for d in group["documents"] if
                    ac.source.norm(d.get("id", "")) in final_text and
                    ac.source.norm(d.get("text", "")) and ac.source.norm(d["text"]) in final_text]
            pair = {"case_id": case_id, "event": vote["event"], "error_supported": True,
                "targeted_stop_error": vote["event"] in ("premature_answer", "unnecessary_search"),
                "original_position_case_id": None, "source": case["source"], "question": case["question"],
                "training_answer": answer, "support_titles": [x["title"] for x in case["support_docs"]],
                "insufficient_state": insuff, "sufficient_state": actions[-1], "repair_actions": actions,
                "repair_searches": searches, "repair_calls": calls, "visible_documents": docs_by_step,
                "actual_final_prompt": tok.decode(final_ids, skip_special_tokens=False),
                "repair_path_sha256": hashlib.sha256(json.dumps(actions, sort_keys=True,
                    ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                "evidence_audit_required": True, "review_reason": vote["reason"]}
            built.append(pair)
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            failures[str(exc)] = failures.get(str(exc), 0) + 1
    out = ac.OUT / "extra-repair-candidates.jsonl"
    ac.write_jsonl(out, built)
    del client, registry
    import gc
    gc.collect(); torch.cuda.empty_cache()
    result = {"built": len(built), "stop_repairs": sum(x["targeted_stop_error"] for x in built),
        "failures": failures, "sha256": ac.digest(out)}
    ac.write(ac.OUT / "extra-repair-candidate-summary.json", result)
    return result


def combine_paths_for_evidence() -> dict:
    original = ac.rows(ac.OUT / "selected-repair-paths.jsonl")
    extras = ac.rows(ac.OUT / "extra-repair-candidates.jsonl")
    by_id = {x["case_id"]: x for x in original}
    for x in extras:
        if x["case_id"] in by_id:
            raise RuntimeError("Supplemental path duplicates a base repair case")
        by_id[x["case_id"]] = x
    # Preserve the original 31 path artifact and present one fixed blind packet.
    backup = ac.OUT / "original-only-repair-paths.jsonl"
    if not backup.exists():
        ac.write_jsonl(backup, original)
    combined = list(by_id.values())
    combined.sort(key=lambda x: hashlib.sha256(("goal-ac-combined:" + x["case_id"]).encode()).hexdigest())
    ac.write_jsonl(ac.OUT / "selected-repair-paths.jsonl", combined)
    packet = ac.build_repair_evidence_packet(combined)
    ac.set_state("awaiting_repair_evidence_review", combined_path_count=len(combined),
        supplement_candidates=len(extras), repair_evidence_packet=packet,
        supplemental_rollouts_sampled_early_due_to_replay_implementation_defect=True)
    return {"combined_candidates": len(combined), "original_paths": len(original),
        "extra_paths": len(extras), "packet": packet}


def finalize_material_and_bind() -> dict:
    cfg = ac.verify()
    selected_before = ac.rows(ac.OUT / "selected-repair-paths.jsonl")
    selected_ids = {x["case_id"] for x in selected_before}
    original_ids = set(cfg["order"]["42"])
    extra_ids = selected_ids - original_ids
    if len(extra_ids) > 128:
        raise RuntimeError("Supplemental material exceeds fixed pool")
    def bind(selected: list[dict]) -> dict:
        selected_ids_now = {x["case_id"] for x in selected}
        original = selected_ids_now & original_ids
        extra = sorted(selected_ids_now - original_ids,
            key=lambda x: hashlib.sha256(("goal-ac-extra-slot:" + x).encode()).hexdigest())
        free = [x for x in cfg["order"]["42"] if x not in original]
        free.sort(key=lambda x: hashlib.sha256(("goal-ac-free-slot:" + x).encode()).hexdigest())
        if len(extra) > len(free):
            raise RuntimeError("Not enough frozen update slots for supplemental repairs")
        slot_to_repair = {x: x for x in original}
        slot_to_repair.update(dict(zip(free, extra)))
        result = {}
        for seed in ("42", "43"):
            result[seed] = [{"update": i, "case_id": slot, "repair_case_id": repair,
                "kind": "original" if slot == repair else "supplemental"}
                for i, slot in enumerate(cfg["order"][seed], 1) if slot in slot_to_repair
                for repair in [slot_to_repair[slot]]]
        if {x["case_id"] for x in result["42"]} != {x["case_id"] for x in result["43"]}:
            raise RuntimeError("Seed repair update-slot mapping differs")
        ac.write(ac.OUT / "repair-update-binding.json", result)
        return result
    ac.freeze_repair_binding = bind
    result = ac.finalize_repair_evidence()
    if result["passed"]:
        binding = ac.z.read(ac.OUT / "repair-update-binding.json")
        result["binding_updates"] = {seed: len(items) for seed, items in binding.items()}
    ac.write(ac.OUT / "supplemental-finalization.json", {"result": result,
        "extension_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "replay_fix_sha256": hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest(),
        "selected_supplemental_count": len(extra_ids)})
    return result


def finalize_confirmation() -> dict:
    """Validate evidence citations before applying the frozen confirmation gate."""
    packet = ac.rows(ac.OUT / "confirmation-review-packet.jsonl")
    votes = ac.rows(ac.OUT / "confirmation-review-final.jsonl")
    by_vote = {x["review_id"]: x for x in votes}
    if len(by_vote) != len(votes) or {x["review_id"] for x in packet} != set(by_vote):
        raise RuntimeError("Confirmation review does not cover the frozen packet exactly")
    for item in packet:
        vote = by_vote[item["review_id"]]
        if vote.get("category") not in ac.REVIEW_CATEGORIES or not vote.get("reason") or \
           not set(vote.get("evidence_ids", [])).issubset(item.get("visible_document_ids", [])):
            raise RuntimeError("Invalid confirmation evidence review")
        if vote["category"] == "correct_sufficient" and (item.get("reason") != "answer" or
                                                            not vote.get("evidence_ids")):
            raise RuntimeError("Confirmation success lacks an answer-time evidence citation")
    return ac.confirmation_gate()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("resume-supplement", "package-audit", "combine-audits", "build-paths", "combine-paths", "finalize", "finalize-confirmation"))
    command = p.parse_args().command
    result = {"resume-supplement": resume_supplement, "package-audit": package_audit, "combine-audits": combine_extra_audits,
        "build-paths": build_extra_paths, "combine-paths": combine_paths_for_evidence,
        "finalize": finalize_material_and_bind, "finalize-confirmation": finalize_confirmation}[command]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
