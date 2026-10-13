"""Goal Q: bounded live-search SFT and GRPO from the Goal J adapter."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import run_v133_goal_p_system as p
from run_v133_goal_p_variant import SearchTextAliasClient
from evaluate_v133_goal_c_sft import normalize_answer, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-q-search-success"
J = ROOT / "runs/v133-goal-j-transfer/adapter-final"
OLD_RL = ROOT / "runs/v133-goal-p-system/p3-training/rl/adapter-final"
CFG = OUT / "config.json"
STATE = OUT / "state.json"
CAPS = {"baseline": 900, "signal": 900, "direct": 2100, "sft": 900,
        "warm_signal": 900, "warm_rl": 2100, "evaluation": 900,
        "confirm": 1800, "final": 600}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def append(path, item):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def load_config():
    value = read(CFG)
    for key, path in (("corpus", p.CORPUS), ("cases", p.CASES), ("j", J / "adapter_model.safetensors"),
                      ("old_rl", OLD_RL / "adapter_model.safetensors")):
        if sha(path) != value["hashes"][key]:
            raise RuntimeError(f"frozen asset changed: {key}")
    return value


def prepare():
    if OUT.exists():
        raise FileExistsError("Goal Q output already exists")
    pc = p.config()
    train = p.J / "train-frozen.jsonl"
    if len(rows(train)) != 16 or len(rows(p.CASES)) != 48:
        raise RuntimeError("frozen cases incomplete")
    hashes = {"corpus": sha(p.CORPUS), "cases": sha(p.CASES),
              "j": sha(J / "adapter_model.safetensors"),
              "old_rl": sha(OLD_RL / "adapter_model.safetensors"),
              "train": sha(train), "runner": sha(__file__),
              "search": sha(ROOT / "src/mini_agent_rl/tools/search.py"),
              "local": sha(ROOT / "src/mini_agent_rl/model/local.py")}
    config = {"version": "goal-q-v1", "hashes": hashes, "seed": 42, "confirm_seed": 43,
              "group_size": 4, "passes": 2, "max_groups": 32, "max_updates": 32,
              "mixed_gate": 4, "demo_gate": 8, "learning_rate": 5e-6,
              "kl_beta": .02, "clip_epsilon": .2, "max_grad_norm": 1.,
              "reference": "goal_j", "temperature": 1., "top_p": 1., "top_k": 0,
              "product": {key: pc[key] for key in ("retriever", "top_k", "system_prompt", "thinking",
                         "max_input_tokens", "max_new_tokens", "max_model_calls_per_case",
                         "max_searches_per_case", "observation_encoding")},
              "caps": CAPS, "total_gpu_seconds": 10800}
    OUT.mkdir(parents=True)
    write(CFG, config)
    write(STATE, {"status": "prepared", "stage": "baseline", "gpu_seconds": 0., "stages": []})
    print(json.dumps({"prepared": str(OUT), "hashes": hashes}))


def make_client(adapter, *, deterministic, product_config=None, model_path=None, corpus=None):
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    pc = product_config if product_config is not None else load_config()["product"]
    explicit = ({key: pc[key] for key in ("device", "dtype", "load_in_4bit", "timeout", "action_completion", "observation_presentation", "action_constraint", "context_aware_decoding")
                 if key in pc} if product_config is not None else {})
    client = TransformersModelClient(LocalModelConfig(model_id=str(model_path or p.BASE), adapter_path=str(adapter),
        max_input_tokens=pc["max_input_tokens"], max_new_tokens=pc["max_new_tokens"],
        enable_thinking=pc.get("thinking", False), deterministic=deterministic,
        observation_encoding=pc["observation_encoding"], recompute_old_logprobs=False, **explicit))
    registry = ToolRegistry()
    registry.register(SearchTool(read(corpus or p.CORPUS), pc["retriever"]))
    return client, SearchTextAliasClient(client), registry


def score(answer, gold):
    answer = answer or ""
    return {"em": int(bool(answer) and normalize_answer(answer) == normalize_answer(gold)),
            "f1": f1_score(answer, gold) if answer else 0.}


async def rollout(question, wrapped, registry, *, sample, seed, product_config=None, decision_pipeline=None):
    """The one search loop used by Q sampling, evaluation, and demo."""
    from mini_agent_rl.domain import Message
    from mini_agent_rl.prompts import system_prompt
    # Optional per-run override is used by isolated prompt A/B evaluations.
    # Existing callers keep the frozen Goal Q product configuration unchanged.
    pc = product_config if product_config is not None else load_config()["product"]
    if product_config is not None:
        base = getattr(wrapped, "base", wrapped)
        actual = getattr(base, "config", None)
        for key in ("max_input_tokens", "max_new_tokens", "observation_encoding", "action_completion", "observation_presentation", "action_constraint"):
            if key in pc and (actual is None or getattr(actual, key, None) != pc[key]):
                raise RuntimeError(f"product/client configuration mismatch: {key}")
        if "thinking" in pc and (actual is None or actual.enable_thinking != pc["thinking"]):
            raise RuntimeError("product/client configuration mismatch: thinking")
        if "context_aware_decoding" in pc:
            from mini_agent_rl.model.context_aware_decoding import ContextAwareDecodingConfig
            expected = ContextAwareDecodingConfig(**pc["context_aware_decoding"]) if pc["context_aware_decoding"] else None
            if actual is None or actual.context_aware_decoding != expected:
                raise RuntimeError("product/client configuration mismatch: context_aware_decoding")
    messages = [Message(role="system", content=system_prompt(False, pc["system_prompt"])),
                Message(role="user", content=question)]
    trace, seen, reason, answer = [], set(), "max_steps", None
    started = time.monotonic()
    for step in range(pc["max_model_calls_per_case"]):
        before = time.monotonic()
        base_client = getattr(wrapped, "base", None)
        context_start = len(getattr(base_client, "_context_records", []))
        generate = decision_pipeline if decision_pipeline is not None else wrapped.generate
        response = await generate(messages, registry.definitions(),
                                          temperature=1. if sample else 0., seed=seed + step)
        item = {"step": step, "raw": response.raw_text, "parse_error": response.parse_error,
                "prompt_token_ids": response.prompt_token_ids, "prompt_attention_mask": response.prompt_attention_mask,
                "action_token_ids": response.action_token_ids, "old_logprobs": response.old_logprobs,
                "usage": response.usage, "model_seconds": time.monotonic() - before}
        if base_client is not None:
            records = getattr(base_client, "_context_records", [])
            if len(records) > context_start:
                item["context_usage"] = records[-1]
                if response.usage.get("final_review"):
                    fingerprint = response.usage.get("prompt_sha256")
                    matching = [r for r in records[context_start:] if r.get("prompt_sha256") == fingerprint]
                    if matching:
                        item["context_usage"] = matching[-1]
        if response.parse_error:
            reason = "invalid_action"; trace.append(item); break
        if response.tool_call:
            query = str(response.tool_call.arguments.get("query", ""))
            key = " ".join(query.casefold().split())
            item.update(action="search", query=query, repeated=key in seen)
            review = response.usage.get("final_review", {})
            if review.get("accepted") and (step + 1 >= pc["max_model_calls_per_case"] or
                    sum("observation" in x for x in trace) >= pc["max_searches_per_case"]):
                reason = "insufficient_evidence"; trace.append(item); break
            if step >= pc["max_searches_per_case"]:
                reason = "max_steps"; trace.append(item); break
            if key in seen:
                reason = "repeated_action"; trace.append(item); break
            seen.add(key)
            observation = await registry.execute("search", {"query": query})
            observation["searched_queries"] = sorted(seen)
            item["observation"] = observation
            trace.append(item)
            if not observation.get("ok"):
                reason = "tool_error"; break
            canonical = json.dumps({"action": "search", "query": query}, ensure_ascii=False)
            visible = {key: value for key, value in observation.items() if key != "latency_ms"}
            messages.extend([Message(role="assistant", content=canonical, tool_calls=[response.tool_call]),
                Message(role="tool", content=json.dumps(visible, ensure_ascii=False), tool_call_id=response.tool_call.id)])
        else:
            answer = response.content
            item.update(action="answer", answer=answer)
            trace.append(item); reason = "answer"; break
    return {"question": question, "answer": answer, "reason": reason, "trace": trace,
            "model_calls": decision_pipeline.model_calls if decision_pipeline is not None else len(trace),
            "action_decisions": len(trace),
            "auxiliary_calls": decision_pipeline.auxiliary_calls if decision_pipeline is not None else 0,
            "searches": sum("observation" in x for x in trace),
            "elapsed_seconds": time.monotonic() - started}


def eval_model(label, adapter, cases, path):
    if path.exists(): raise FileExistsError(path)
    client, wrapped, registry = make_client(adapter, deterministic=True)
    for row in cases:
        value = asyncio.run(rollout(row["question"], wrapped, registry, sample=False, seed=42))
        observed = {doc["id"] for item in value["trace"]
                    for doc in item.get("observation", {}).get("result", [])}
        value.update(case_id=row["case_id"], split=row["split"], model=label,
                     both_support_retrieved=set(row["support_ids"]).issubset(observed),
                     **score(value["answer"], row["official_answer"]))
        append(path, value)
        print(json.dumps({"model": label, "case": row["case_id"], "em": value["em"]}), flush=True)
    del client, wrapped, registry
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def dev_cases():
    return [row for row in rows(p.CASES) if row["split"] != "train"]


def train_cases():
    return [row for row in rows(p.CASES) if row["split"] == "train"]


def signal(adapter, prefix, seed):
    client, wrapped, registry = make_client(adapter, deterministic=False)
    path = OUT / f"{prefix}-signal.jsonl"
    if path.exists(): raise FileExistsError(path)
    for row_index, row in enumerate(train_cases()):
        for candidate in range(4):
            value = asyncio.run(rollout(row["question"], wrapped, registry, sample=True,
                                        seed=seed * 100000 + row_index * 100 + candidate))
            observed = {doc["id"] for item in value["trace"]
                        for doc in item.get("observation", {}).get("result", [])}
            value.update(case_id=row["case_id"], candidate=candidate,
                         both_support_retrieved=set(row["support_ids"]).issubset(observed),
                         **score(value["answer"], row["official_answer"]))
            append(path, value)
    data = rows(path)
    mixed = sum(len({x["em"] for x in data if x["case_id"] == row["case_id"]}) == 2 for row in train_cases())
    summary = {"rows": len(data), "mixed": mixed, "all_correct": 0, "all_wrong": 0}
    for row in train_cases():
        ems = [x["em"] for x in data if x["case_id"] == row["case_id"]]
        summary["all_correct" if all(ems) else "all_wrong" if not any(ems) else "mixed_total"] = \
            summary.get("all_correct" if all(ems) else "all_wrong" if not any(ems) else "mixed_total", 0) + 1
    write(OUT / f"{prefix}-signal-summary.json", summary)
    del client, wrapped, registry
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return summary


def build_demos():
    """Only training questions and genuine tool observations enter demonstrations."""
    from mini_agent_rl.domain import Message, ToolCall
    from mini_agent_rl.prompts import system_prompt
    source = rows(OUT / "j-signal.jsonl")
    _, _, registry = make_client(J, deterministic=True)
    corpus = {doc["id"]: doc for doc in read(p.CORPUS)}
    output = []
    for row in train_cases():
        possibilities = [x for x in source if x["case_id"] == row["case_id"]
                         and x["em"] and x["both_support_retrieved"]]
        possibilities.sort(key=lambda x: (x["searches"], sum(len(t["action_token_ids"] or []) for t in x["trace"]), x["candidate"]))
        if possibilities:
            chosen = possibilities[0]
            actions = [x["query"] for x in chosen["trace"] if x.get("action") == "search"]
            origin = "sample"
        else:
            question = row["question"]
            titles = [corpus[doc_id]["title"] for doc_id in row["support_ids"]]
            actions = [title for title in titles if title.casefold() in question.casefold()]
            if not actions: actions = [question]
            actions = list(dict.fromkeys(actions))[:3]
            origin = "question_title"
        messages = [Message(role="system", content=system_prompt(False, "direct-v090")),
                    Message(role="user", content=row["question"])]
        seen = set(); observed = set(); valid = True
        for query in actions:
            key = " ".join(query.casefold().split())
            if key in seen: valid = False; break
            seen.add(key)
            observation = asyncio.run(registry.execute("search", {"query": query}))
            observation["searched_queries"] = sorted(seen)
            if not observation.get("ok"): valid = False; break
            observed.update(doc["id"] for doc in observation.get("result", []))
            call = ToolCall(name="search", arguments={"query": query})
            messages.append(Message(role="assistant", content=json.dumps({"action": "search", "query": query}, ensure_ascii=False),
                                    tool_calls=[call]))
            visible = {key: value for key, value in observation.items() if key != "latency_ms"}
            messages.append(Message(role="tool", content=json.dumps(visible, ensure_ascii=False), tool_call_id=call.id))
        if not valid or not set(row["support_ids"]).issubset(observed):
            output.append({"case_id": row["case_id"], "eligible": False, "reason": "support_not_retrieved"})
            continue
        messages.append(Message(role="assistant", content=json.dumps({"action": "answer", "answer": row["official_answer"]}, ensure_ascii=False)))
        output.append({"case_id": row["case_id"], "eligible": True, "origin": origin,
                       "messages": [message.model_dump(mode="json") for message in messages],
                       "queries": actions, "support_retrieved": True})
    write(OUT / "demonstrations.json", output)
    return sum(item["eligible"] for item in output)


def sft(seed, destination):
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.sft import encode_assistant_only
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as o
    set_seed(seed)
    demos = [x for x in read(OUT / "demonstrations.json") if x["eligible"]]
    if len(demos) < 8: raise RuntimeError("insufficient demonstrations")
    folder = OUT / destination
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir()
    client, _, _ = make_client(J, deterministic=True)
    named = client.prepare_for_training(); params = [value for _, value in named]
    base_before = o.parameter_hashes(client._model, lora=False)
    lora_before = o.parameter_hashes(client._model, lora=True)
    optimizer = torch.optim.AdamW(params, lr=5e-6, betas=(.9, .999), eps=1e-8, weight_decay=0)
    updates = 0
    for epoch in range(2):
        for item in demos:
            messages = [{"role": x["role"], "content": x["content"]} for x in item["messages"]]
            indexes = [i for i, x in enumerate(messages) if x["role"] == "assistant"]
            encoded = encode_assistant_only(client._processor, messages, 2176, None, False)
            # A trajectory is one optimizer step, regardless of its number of actions.
            labels = encoded["labels"]
            if len(indexes) < 2 or not all(any(x != -100 for x in encode_assistant_only(
                    client._processor, messages, 2176, i, False)["labels"]) for i in indexes):
                raise RuntimeError("missing assistant action labels")
            if sum(x != -100 for x in labels) < len(indexes):
                raise RuntimeError("incomplete target mask")
            batch = {key: torch.tensor([value], dtype=torch.long, device=client._model.device)
                     for key, value in encoded.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = client._model(**batch).loss
            if not torch.isfinite(loss): raise RuntimeError("nonfinite SFT loss")
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(params, 1.))
            if not math.isfinite(grad) or grad <= 0: raise RuntimeError("invalid SFT gradient")
            guarded_optimizer_step(optimizer, params, client.gpu_memory_stats(), 21.)
            updates += 1
            append(folder / "updates.jsonl", {"epoch": epoch + 1, "case_id": item["case_id"],
                                               "loss": float(loss.detach()), "grad": grad,
                                               "target_tokens": sum(x != -100 for x in labels)})
    if updates > 32 or o.parameter_hashes(client._model, lora=False) != base_before:
        raise RuntimeError("SFT budget/base gate failed")
    changed = sum(o.parameter_hashes(client._model, lora=True)[key] != value for key, value in lora_before.items())
    if not changed: raise RuntimeError("SFT LoRA unchanged")
    checkpoint = folder / "adapter-final"
    client._model.save_pretrained(checkpoint)
    del client, optimizer, params
    gc.collect(); torch.cuda.empty_cache()
    loaded, _, _ = make_client(checkpoint, deterministic=True)
    if not loaded.prepare_for_training(): raise RuntimeError("SFT reload failed")
    probe = demos[0]["messages"]
    probe_messages = [{"role": x["role"], "content": x["content"]} for x in probe]
    encoded = encode_assistant_only(loaded._processor, probe_messages, 2176, None, False)
    batch = {key: torch.tensor([value], dtype=torch.long, device=loaded._model.device)
             for key, value in encoded.items()}
    with torch.no_grad(): after = float(loaded._model(**batch).loss)
    if not math.isfinite(after): raise RuntimeError("SFT reload loss invalid")
    write(folder / "summary.json", {"updates": updates, "changed_lora": changed,
                                    "checkpoint_sha256": sha(checkpoint / "adapter_model.safetensors"),
                                    "reload": True, "reload_loss": after, "base_unchanged": True})


def rl(start_adapter, seed, destination):
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import clipped_grpo_loss, guarded_optimizer_step, require_initial_ratio, ratio_statistics
    import run_v133_goal_o_budget as o
    set_seed(seed)
    folder = OUT / destination
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir()
    ref_client, _, _ = make_client(J, deterministic=True)
    reference = {name: value.detach().clone() for name, value in ref_client.prepare_for_training()}
    del ref_client
    gc.collect(); torch.cuda.empty_cache()
    client, wrapped, registry = make_client(start_adapter, deterministic=False)
    named = client.prepare_for_training(); params = [value for _, value in named]
    base_before = o.parameter_hashes(client._model, lora=False)
    lora_before = o.parameter_hashes(client._model, lora=True)
    ref_before = o.reference_hashes(reference)
    optimizer = torch.optim.AdamW(params, lr=5e-6, betas=(.9, .999), eps=1e-8, weight_decay=0)
    groups = updates = samples = 0
    check_action = None
    for epoch in range(2):
        for row in train_cases():
            group = []
            client._model.eval()
            for candidate in range(4):
                value = asyncio.run(rollout(row["question"], wrapped, registry, sample=True,
                                            seed=seed * 100000 + epoch * 10000 + groups * 10 + candidate))
                value.update(case_id=row["case_id"], candidate=candidate, epoch=epoch + 1,
                             **score(value["answer"], row["official_answer"]))
                append(folder / "rollouts.jsonl", value)
                group.append(value)
            samples += 4; groups += 1
            rewards = [x["em"] for x in group]
            item = {"case_id": row["case_id"], "epoch": epoch + 1, "group": groups,
                    "rewards": rewards, "updated": False}
            if min(rewards) != max(rewards):
                mean = statistics.mean(rewards); std = statistics.pstdev(rewards)
                frozen = []
                for trajectory in group:
                    actions = []
                    for step in trajectory["trace"]:
                        action = step["action_token_ids"]
                        prompt = step["prompt_token_ids"]
                        mask = step["prompt_attention_mask"]
                        if not action or not prompt: raise RuntimeError("missing sampled action tokens")
                        old = client.score_action_tensor([], action, False, prompt, mask).detach()
                        ref = client.reference_logprobs([], action, reference, prompt, mask).detach()
                        actions.append((step, old, ref))
                    frozen.append(actions)
                optimizer.zero_grad(set_to_none=True)
                client.set_training_forward_mode()
                ratios = []; losses = []; kls = []
                for trajectory, actions in zip(group, frozen):
                    total_tokens = sum(len(x[0]["action_token_ids"]) for x in actions)
                    advantage = (trajectory["em"] - mean) / max(std, 1e-8)
                    for step, old, ref in actions:
                        current = client.score_action_tensor([], step["action_token_ids"], True,
                                                             step["prompt_token_ids"], step["prompt_attention_mask"])
                        ratios.append(torch.exp(current.detach() - old).float().cpu())
                        loss, kl, _ = clipped_grpo_loss(current, old, ref, advantage, .2, .02)
                        if not torch.isfinite(loss): raise RuntimeError("nonfinite RL loss")
                        (loss * len(step["action_token_ids"]) / total_tokens / 4).backward()
                        losses.append(float(loss.detach())); kls.append(float(kl))
                        check_action = step
                require_initial_ratio(ratio_statistics(ratios), label=f"{destination} group {groups}")
                grad = float(torch.nn.utils.clip_grad_norm_(params, 1.))
                if not math.isfinite(grad) or grad <= 0: raise RuntimeError("invalid RL gradient")
                guarded_optimizer_step(optimizer, params, client.gpu_memory_stats(), 21.)
                updates += 1
                item.update(updated=True, loss=statistics.mean(losses), kl=statistics.mean(kls), grad=grad)
            append(folder / "groups.jsonl", item)
            if groups > 32 or updates > 32 or samples > 128: raise RuntimeError("RL budget exceeded")
    if groups != 32 or not updates or o.reference_hashes(reference) != ref_before:
        raise RuntimeError("RL group/reference gate failed")
    if o.parameter_hashes(client._model, lora=False) != base_before: raise RuntimeError("base changed")
    changed = sum(o.parameter_hashes(client._model, lora=True)[key] != value for key, value in lora_before.items())
    if not changed: raise RuntimeError("RL LoRA unchanged")
    checkpoint = folder / "adapter-final"
    client._model.save_pretrained(checkpoint)
    before = client.score_action_tensor([], check_action["action_token_ids"], False,
                                        check_action["prompt_token_ids"], check_action["prompt_attention_mask"]).detach().float().cpu()
    del client, wrapped, registry, optimizer, params, reference
    gc.collect(); torch.cuda.empty_cache()
    loaded, _, _ = make_client(checkpoint, deterministic=True)
    after = loaded.score_action_tensor([], check_action["action_token_ids"], False,
                                       check_action["prompt_token_ids"], check_action["prompt_attention_mask"]).detach().float().cpu()
    delta = float((before - after).abs().max())
    if delta > .001: raise RuntimeError(f"reload mismatch: {delta}")
    write(folder / "summary.json", {"groups": groups, "updates": updates, "samples": samples,
                                    "changed_lora": changed, "base_unchanged": True,
                                    "reference_unchanged": True, "reload_max_abs": delta,
                                    "checkpoint_sha256": sha(checkpoint / "adapter_model.safetensors")})


def stage(name):
    load_config()
    cases = dev_cases()
    if name == "baseline":
        eval_model("goal_j", J, cases, OUT / "eval-goal_j.jsonl")
        eval_model("old_rl", OLD_RL, cases, OUT / "eval-old_rl.jsonl")
    elif name == "signal":
        signal(J, "j", 42)
        count = build_demos()
        write(OUT / "demo-summary.json", {"eligible": count, "total": 16})
    elif name == "direct":
        rl(J, 42, "direct")
    elif name == "sft":
        sft(42, "sft")
    elif name == "warm_signal":
        signal(OUT / "sft/adapter-final", "warm", 42)
    elif name == "warm_rl":
        rl(OUT / "sft/adapter-final", 42, "warm_rl")
    elif name == "evaluation":
        for label in ("direct", "sft", "warm_rl"):
            adapter = OUT / label / "adapter-final"
            if adapter.exists(): eval_model(label, adapter, cases, OUT / f"eval-{label}.jsonl")
    elif name == "confirm":
        winner = read(OUT / "provisional.json")["winner"]
        if winner in ("sft", "warm_rl"):
            sft(43, "confirm_sft")
        if winner == "direct": rl(J, 43, "confirm_rl")
        elif winner == "warm_rl": rl(OUT / "confirm_sft/adapter-final", 43, "confirm_rl")
        adapter = OUT / ("confirm_rl" if winner in ("direct", "warm_rl") else "confirm_sft") / "adapter-final"
        eval_model("confirm", adapter, cases, OUT / "eval-confirm.jsonl")
    elif name == "final":
        decision = read(OUT / "decision.json")
        adapter = Path(decision["adapter"])
        if not adapter.exists(): raise RuntimeError("final adapter unavailable")
        eval_model("selected_reloaded", adapter, train_cases()[:1], OUT / "final-smoke.jsonl")
    else: raise ValueError(name)


def metrics(label):
    path = OUT / f"eval-{label}.jsonl"
    if not path.exists(): return None
    data = rows(path)
    if len(data) != 32 or len({x["case_id"] for x in data}) != 32:
        raise RuntimeError(f"incomplete evaluation: {label}")
    result = {"model": label, "total": sum(x["em"] for x in data),
              "invalid": sum(x["reason"] == "invalid_action" for x in data),
              "searches": sum(x["searches"] for x in data),
              "both_support": sum(x["both_support_retrieved"] for x in data),
              "f1": statistics.mean(x["f1"] for x in data)}
    for split in ("new_eval", "goal_G_regression"):
        part = [x for x in data if x["split"] == split]
        if len(part) != 16: raise RuntimeError(f"missing split {label} {split}")
        result[split] = sum(x["em"] for x in part)
    return result


def choose():
    j = metrics("goal_j"); old = metrics("old_rl")
    if not j or not old: raise RuntimeError("baselines incomplete")
    incumbent = old if old["goal_G_regression"] >= j["goal_G_regression"] and old["total"] > j["total"] else j
    candidates = [metrics(name) for name in ("direct", "sft", "warm_rl")]
    eligible = [x for x in candidates if x and x["total"] > incumbent["total"]
                and x["goal_G_regression"] >= j["goal_G_regression"] and x["invalid"] <= j["invalid"]]
    if eligible:
        ranked = sorted(eligible, key=lambda x: (-x["total"], -x["goal_G_regression"],
                                                 x["searches"], 1 if x["model"] == "sft" else 2))
        winner = ranked[0]["model"]
    else: winner = None
    write(OUT / "provisional.json", {"goal_j": j, "old_rl": old, "incumbent": incumbent,
                                     "candidates": candidates, "eligible": eligible, "winner": winner})
    return winner


def decide():
    provisional = read(OUT / "provisional.json")
    j = provisional["goal_j"]
    winner = provisional["winner"]
    confirmed = metrics("confirm") if winner else None
    accepted = bool(confirmed and confirmed["total"] > provisional["incumbent"]["total"]
                    and confirmed["goal_G_regression"] >= j["goal_G_regression"]
                    and confirmed["invalid"] <= j["invalid"])
    selected = winner if accepted else provisional["incumbent"]["model"]
    adapter = {"goal_j": J, "old_rl": OLD_RL}.get(selected, OUT / selected / "adapter-final")
    write(OUT / "decision.json", {"selected": selected, "adapter": str(adapter),
                                  "confirmed": accepted, "replication": confirmed,
                                  "reason": "new_model_confirmed" if accepted else "incumbent_retained"})


def run_stage(name, state):
    cap = CAPS[name]
    if state["gpu_seconds"] + cap > 10800:
        raise RuntimeError(f"insufficient total GPU budget for {name}")
    started = time.monotonic()
    state["stage"] = name
    write(STATE, state)
    command = [sys.executable, str(Path(__file__).resolve()), "stage", "--name", name]
    try:
        subprocess.run(command, cwd=ROOT, check=True, timeout=cap, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    finally:
        elapsed = time.monotonic() - started
        state["gpu_seconds"] += elapsed
        state["stages"].append({"name": name, "seconds": elapsed})
        write(STATE, state)


def run():
    load_config()
    state = read(STATE)
    if state["status"] != "prepared": raise RuntimeError("run already started")
    state["status"] = "running"
    write(STATE, state)
    try:
        run_stage("baseline", state)
        run_stage("signal", state)
        if read(OUT / "j-signal-summary.json")["mixed"] >= 4:
            run_stage("direct", state)
        else: state["stages"].append({"name": "direct", "skipped": "mixed_groups_below_four"})
        if read(OUT / "demo-summary.json")["eligible"] >= 8:
            run_stage("sft", state)
            run_stage("warm_signal", state)
            if read(OUT / "warm-signal-summary.json")["mixed"] >= 4:
                run_stage("warm_rl", state)
            else: state["stages"].append({"name": "warm_rl", "skipped": "mixed_groups_below_four"})
        else: state["stages"].append({"name": "sft_and_warm_rl", "skipped": "fewer_than_eight_demos"})
        run_stage("evaluation", state)
        if choose(): run_stage("confirm", state)
        decide()
        run_stage("final", state)
        state.update(status="complete", stage="complete")
    except BaseException as exc:
        state.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(STATE, state)
        summarize()


def summarize():
    if not OUT.exists(): raise FileNotFoundError(OUT)
    state = read(STATE)
    table = []
    for name in ("goal_j", "old_rl", "direct", "sft", "warm_rl", "confirm"):
        try: table.append(metrics(name))
        except RuntimeError: table.append({"model": name, "status": "partial"})
    result = {"status": state["status"], "gpu_seconds": state["gpu_seconds"],
              "stages": state["stages"], "models": [x for x in table if x],
              "decision": read(OUT / "decision.json") if (OUT / "decision.json").exists() else None}
    write(OUT / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "summarize", "stage", "demo"])
    parser.add_argument("--name")
    parser.add_argument("--question")
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "stage": stage(args.name)
    elif args.command == "run": run()
    elif args.command == "summarize": summarize()
    elif args.command == "demo":
        adapter = Path(read(OUT / "decision.json")["adapter"])
        client, wrapped, registry = make_client(adapter, deterministic=True)
        print(json.dumps(asyncio.run(rollout(args.question or input("Question: "), wrapped, registry,
                                            sample=False, seed=42)), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
