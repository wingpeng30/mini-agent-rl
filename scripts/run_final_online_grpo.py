"""One bounded AM continuation experiment, with real hybrid-search rollouts.

No new datasets, quality gates, teacher calls, or checkpoint selection.  A group
with constant reward is skipped as part of GRPO, not used as a prerequisite.
The original AM adapter and all existing experimental ledgers stay untouched.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PARENT_SHA = "3bcd1f85382c1d7775fa674c1a6e1e68c80911a69b86de209d20f2ab6269280c"
PRODUCT = dict(max_input_tokens=16384, max_new_tokens=2048,
               max_model_calls_per_case=4, max_searches_per_case=3,
               system_prompt="direct-v090", thinking=False,
               action_completion="first-json-v1", observation_encoding="legacy",
               observation_presentation="legacy-json", action_constraint="none",
               context_aware_decoding=None, retriever="hybrid-rrf-v1", top_k=3,
               device="cuda", dtype="bfloat16", load_in_4bit=False, timeout=None)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    path = Path(path)
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log(event, **values):
    print(json.dumps(dict(event=event, utc_epoch=time.time(), **values), ensure_ascii=False), flush=True)


class DeadlineReached(RuntimeError):
    pass


class Budget:
    def __init__(self, args):
        self.deadline = args.deadline_epoch
        self.training_deadline = min(args.train_deadline_epoch, self.deadline)

    def check(self, training=False):
        limit = self.training_deadline if training else self.deadline
        if time.time() >= limit:
            raise DeadlineReached("training cutoff" if training else "experiment deadline")


def apply_alias(response):
    """Exactly the historical search.text compatibility, applied per batch row."""
    if not response.parse_error:
        return response
    from mini_agent_rl.domain import ToolCall
    try:
        value = json.loads((response.raw_text or "").removesuffix("<|im_end|>").strip())
    except ValueError:
        return response
    if (isinstance(value, dict) and set(value) == {"action", "text"}
            and value.get("action") == "search" and isinstance(value.get("text"), str)
            and value["text"].strip()):
        usage = dict(response.usage, protocol_recovery="search_text_alias_v1")
        return response.model_copy(update=dict(content="", parse_error=None,
            tool_calls=[ToolCall(name="search", arguments={"query": value["text"].strip()})], usage=usage))
    return response


async def wavefront(client, engine, cases, *, sample, seed, policy_version, budget=None, training=False, journal_folder=None):
    """Real GPU batches of active episodes, with independent tool histories.

    One optimizer update is never allowed between members of the same group.
    Search/termination behavior follows Goal Q's production loop exactly.
    """
    from mini_agent_rl.domain import Message
    from mini_agent_rl.prompts import system_prompt
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    registry = ToolRegistry()
    registry.register(SearchTool(engine.documents, "hybrid-rrf-v1", retrieval=engine))
    started = time.monotonic()
    episodes = [dict(case=c, messages=[Message(role="system", content=system_prompt(False, PRODUCT["system_prompt"])),
                    Message(role="user", content=c["question"])], seen=set(), trace=[], answer=None,
                    reason="max_steps", finished=False, started=started) for c in cases]
    def partial_snapshot(reason):
        if journal_folder is not None:
            for candidate, episode in enumerate(episodes):
                append(Path(journal_folder) / "partial-rollouts.jsonl", dict(case_id=episode["case"]["id"],
                    candidate=candidate, policy_version=policy_version, complete=episode["finished"],
                    partial_reason=reason, answer=episode["answer"], reason=episode["reason"],
                    trace=episode["trace"], elapsed_seconds=time.monotonic() - started))

    def checked_budget():
        if budget:
            try:
                budget.check(training)
            except DeadlineReached:
                partial_snapshot("cutoff")
                raise

    for step in range(PRODUCT["max_model_calls_per_case"]):
        active = [e for e in episodes if not e["finished"]]
        if not active:
            break
        checked_budget()
        requests = [dict(messages=e["messages"], tools=registry.definitions(), sample=sample,
                         temperature=1.0 if sample else 0.0) for e in active]
        batch_started = time.monotonic()
        if journal_folder is not None:
            append(Path(journal_folder) / "generation-attempts.jsonl", dict(event="started", policy_version=policy_version,
                step=step, batch_seed=seed + step, rows=len(active), started_at=time.time()))
        try:
            responses = await client.generate_batch(requests, batch_seed=seed + step, policy_version=policy_version)
        except BaseException:
            partial_snapshot("generation_failure")
            raise
        seconds = time.monotonic() - batch_started
        if len(responses) != len(active):
            raise RuntimeError("batch response count differs from active episodes")
        if journal_folder is not None:
            append(Path(journal_folder) / "generation-attempts.jsonl", dict(event="completed", policy_version=policy_version,
                step=step, batch_seed=seed + step, rows=len(active), elapsed_seconds=seconds))

        async def process_response(episode, response):
            response = apply_alias(response)
            item = dict(step=step, raw=response.raw_text, parse_error=response.parse_error,
                prompt_token_ids=response.prompt_token_ids, prompt_attention_mask=response.prompt_attention_mask,
                action_token_ids=response.action_token_ids, old_logprobs=response.old_logprobs,
                usage=response.usage, context_usage=response.usage.get("context_usage", {}),
                model_seconds=seconds / len(active), batch_seconds=seconds, batch_active_rows=len(active))
            if response.parse_error:
                episode.update(reason="invalid_action", finished=True)
            elif response.tool_call:
                query = str(response.tool_call.arguments.get("query", ""))
                key = " ".join(query.casefold().split())
                item.update(action="search", query=query, repeated=key in episode["seen"])
                if step >= PRODUCT["max_searches_per_case"]:
                    episode.update(reason="max_steps", finished=True)
                elif key in episode["seen"]:
                    episode.update(reason="repeated_action", finished=True)
                else:
                    episode["seen"].add(key)
                    observation = await registry.execute("search", {"query": query})
                    observation["searched_queries"] = sorted(episode["seen"])
                    item["observation"] = observation
                    if not observation.get("ok"):
                        episode.update(reason="tool_error", finished=True)
                    else:
                        visible = {k: v for k, v in observation.items() if k != "latency_ms"}
                        canonical = json.dumps(dict(action="search", query=query), ensure_ascii=False)
                        episode["messages"].extend([Message(role="assistant", content=canonical, tool_calls=[response.tool_call]),
                            Message(role="tool", content=json.dumps(visible, ensure_ascii=False), tool_call_id=response.tool_call.id)])
            else:
                episode.update(answer=response.content, reason="answer", finished=True)
                item.update(action="answer", answer=response.content)
            episode["trace"].append(item)
            if episode["finished"]:
                episode["finished_seconds"] = time.monotonic() - episode["started"]
        try:
            await asyncio.gather(*(process_response(e, r) for e, r in zip(active, responses)))
        except BaseException:
            partial_snapshot("tool_or_processing_failure")
            raise
        checked_budget()
    elapsed = time.monotonic() - started
    return [dict(question=e["case"]["question"], answer=e["answer"], reason=e["reason"], trace=e["trace"],
                 model_calls=len(e["trace"]), action_decisions=len(e["trace"]), auxiliary_calls=0,
                 searches=sum("observation" in s for s in e["trace"]),
                 elapsed_seconds=e.get("finished_seconds", elapsed), wavefront_wall_seconds=elapsed) for e in episodes]


def metrics(answer, case):
    from mini_agent_rl.evaluation import answer_metrics
    answers = [case["answer"]] + case.get("answer_aliases", case.get("aliases", []))
    values = [answer_metrics(answer or "", a) for a in answers]
    result = {key: max(v[key] for v in values) for key in values[0]}
    # Preserve the evaluator's canonical names and the existing report aliases.
    result.update(exact_match=result["strict_exact_match"], f1=result["token_f1"])
    return result


def make_client(args, adapter, *, seed=42):
    import torch
    from transformers import set_seed
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    set_seed(seed)
    client = TransformersModelClient(LocalModelConfig(model_id=str(args.base_model), adapter_path=str(adapter),
        max_input_tokens=PRODUCT["max_input_tokens"], max_new_tokens=PRODUCT["max_new_tokens"],
        enable_thinking=False, device="cuda", dtype="bfloat16", load_in_4bit=False,
        gradient_checkpointing=True, deterministic=True, timeout=None, recompute_old_logprobs=False,
        action_completion="first-json-v1", observation_encoding="legacy", observation_presentation="legacy-json"))
    started = time.monotonic()
    client._load()
    torch.cuda.synchronize()
    # Full-softmax T=1 sampling matches the existing raw-policy scorer.
    generation = client._model.generation_config
    for key, value in dict(top_p=1.0, top_k=0, typical_p=1.0, repetition_penalty=1.0,
                           no_repeat_ngram_size=0, min_new_tokens=0, min_length=0,
                           suppress_tokens=None, begin_suppress_tokens=None, bad_words_ids=None).items():
        setattr(generation, key, value)
    client._model.eval()
    return client, dict(model_loading_seconds=time.monotonic() - started,
                        generation_config=generation.to_dict(), gpu_memory=client.gpu_memory_stats())


def release(client):
    import torch
    del client
    gc.collect()
    torch.cuda.empty_cache()


def engines(args):
    from mini_agent_rl.retrieval.local import LocalRetrieval
    evaluate = read(args.eval_retrieval_config)
    if args.train_retrieval_config:
        train = read(args.train_retrieval_config)
    else:
        train = dict(evaluate, corpus=str(args.train_corpus or args.train_cases.parent / "corpus-train.json"),
                     index_dir=str(args.train_index or args.train_cases.parent / "training-index"))
        train["corpus_sha256"] = sha(train["corpus"])
    for cfg in (train, evaluate):
        if cfg.get("candidate_k") != 20 or cfg.get("rrf_k") != 60 or cfg.get("top_k") != 3:
            raise ValueError("this experiment preserves existing hybrid candidate20/RRF60/top3")
    started = time.monotonic()
    a, b = LocalRetrieval(train), LocalRetrieval(evaluate)
    a.dense_index(); b.dense_index()
    write(args.run_dir / "retrieval-configs.json", dict(train=train, evaluation=evaluate,
          loading_seconds=time.monotonic() - started, train_metadata=a.metadata("hybrid-rrf-v1"),
          evaluation_metadata=b.metadata("hybrid-rrf-v1")))
    return a, b


def evaluate(args, label, adapter, engine, budget, *, batch_size=1):
    import torch
    import run_v133_goal_q_search as q
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    folder = args.run_dir / "branches" / label
    folder.mkdir(parents=True, exist_ok=True)
    cases = rows(args.eval_cases)
    path = folder / "evaluation.jsonl"
    done = {r["case_id"] for r in rows(path)}
    if len(done) == len(cases):
        return read(folder / "evaluation-summary.json") if (folder / "evaluation-summary.json").exists() else None
    budget.check()
    client, loading = make_client(args, adapter)
    write(folder / "evaluation-loading.json", loading)
    registry = ToolRegistry()
    registry.register(SearchTool(engine.documents, "hybrid-rrf-v1", retrieval=engine))
    wrapped = q.SearchTextAliasClient(client)
    todo = [c for c in cases if c["id"] not in done]
    wall_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    digest = sha(Path(adapter) / "adapter_model.safetensors")
    try:
        for offset in range(0, len(todo), batch_size):
            budget.check()
            chunk = todo[offset:offset + batch_size]
            if batch_size == 1:
                results = [asyncio.run(q.rollout(chunk[0]["question"], wrapped, registry,
                           sample=False, seed=42, product_config=PRODUCT))]
            else:
                results = asyncio.run(wavefront(client, engine, chunk, sample=False, seed=42,
                                               policy_version=label, budget=budget))
            for case, result in zip(chunk, results):
                trace = result["trace"]
                value = dict(result, case_id=case["id"], id=case["id"], gold=case["answer"],
                    stratum=case.get("stratum", case.get("source", "unspecified")),
                    adapter_sha256=digest, policy_version=label, batch_size=batch_size,
                    retrieval=engine.metadata("hybrid-rrf-v1"), **metrics(result["answer"], case),
                    repeated_actions=sum(bool(s.get("repeated")) for s in trace),
                    invalid_actions=sum(bool(s.get("parse_error")) for s in trace),
                    input_tokens=sum(s.get("usage", {}).get("prompt_tokens", len(s.get("prompt_token_ids") or [])) for s in trace),
                    output_tokens=sum(s.get("usage", {}).get("completion_tokens", len(s.get("action_token_ids") or [])) for s in trace))
                append(path, value)
                log("evaluation", branch=label, done=len(rows(path)), total=len(cases),
                    case_id=case["id"], normalized_exact_match=value["normalized_exact_match"])
            budget.check()
        elapsed = time.monotonic() - wall_started
        all_rows = rows(path)
        summary = dict(status="COMPLETED", branch=label, case_count=len(all_rows),
            current_process_cases=len(todo), evaluation_wall_seconds=elapsed, batch_size=batch_size,
            questions_per_second=len(todo) / max(elapsed, 1e-9),
            model_calls=sum(r["model_calls"] for r in all_rows), searches=sum(r["searches"] for r in all_rows),
            input_tokens=sum(r["input_tokens"] for r in all_rows), output_tokens=sum(r["output_tokens"] for r in all_rows),
            exact_match=sum(r["exact_match"] for r in all_rows),
            normalized_exact_match=sum(r["normalized_exact_match"] for r in all_rows),
            f1=statistics.mean(r["f1"] for r in all_rows), adapter_sha256=digest,
            gpu_memory=client.gpu_memory_stats(), resumed=bool(done))
        write(folder / "evaluation-summary.json", summary)
        return summary
    finally:
        del wrapped, registry
        release(client)
        del client


def check_action(action):
    ids, prompt, mask = action["action_token_ids"], action["prompt_token_ids"], action["prompt_attention_mask"]
    if not ids or not prompt or len(prompt) != len(mask) or not any(mask) or any(v not in (0, 1) for v in mask):
        raise RuntimeError("invalid exact sampled action/prompt/attention mask")
    return ids, prompt, mask


def finite_optimizer_step(optimizer, params):
    """Necessary numerical assertions only; resource usage is telemetry."""
    import torch
    if any(not torch.isfinite(p).all() for p in params):
        raise RuntimeError("nonfinite parameter before optimizer step")
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
        raise RuntimeError("nonfinite gradient before optimizer step")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in params):
        raise RuntimeError("nonfinite parameter after optimizer step")


def recovery(folder, client, optimizer, *, order, next_index, updates, groups=0, zero=0,
             elapsed=0., failure=False, pending=None, required=True):
    """Periodic boundary snapshots are resumable; failure scenes are separate."""
    import numpy as np
    import torch
    name = "failure" if failure else "recovery"
    try:
        adapter = folder / ("adapter-" + name)
        adapter.mkdir(exist_ok=True)
        client._model.save_pretrained(adapter)
        state = dict(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            order=order, next_index=next_index, updates=updates, groups=groups, zero_variance=zero,
            elapsed_seconds=elapsed, pending_group=pending, resume_safe=not failure)
        path = folder / (name + "-state.pt")
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary); temporary.replace(path)
        write(folder / (name + ".json"), dict(adapter=str(adapter), next_index=next_index,
            updates=updates, groups=groups, zero_variance=zero, order=order, elapsed_seconds=elapsed,
            resume_safe=not failure, pending_group=bool(pending), saved_at=time.time(),
            adapter_sha256=sha(adapter / "adapter_model.safetensors")))
        log("recovery_saved", branch=folder.name, kind=name, next_index=next_index, updates=updates)
    except BaseException as exc:
        log("recovery_save_failed", branch=folder.name, kind=name, error=repr(exc))
        if required:
            raise


def restore_recovery(folder, client, optimizer, order):
    import numpy as np
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    state = torch.load(folder / "recovery-state.pt", map_location="cpu", weights_only=False)
    if state["order"] != order or not state.get("resume_safe"):
        raise RuntimeError("recovery order differs or snapshot is not a completed-group boundary")
    result = set_peft_model_state_dict(client._model, load_file(folder / "adapter-recovery/adapter_model.safetensors"), adapter_name="default")
    if result.unexpected_keys:
        raise RuntimeError(f"recovery adapter unexpected keys: {result.unexpected_keys}")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
    # Original ledgers remain complete history. Resume rows identify the rewind.
    append(folder / "resume-ledger.jsonl", dict(resumed_at=time.time(), next_index=state["next_index"],
           restored_updates=state["updates"], note="restart from last completed periodic boundary; prior attempts remain charged"))
    return state


def save_and_verify(args, folder, client, *, check, baseline_hashes, reference=None, reference_digest=None):
    from run_v133_goal_o_budget import parameter_hashes, reference_hashes
    base_before, lora_before = baseline_hashes
    if parameter_hashes(client._model, lora=False) != base_before:
        raise RuntimeError("base parameters changed")
    if reference is not None and reference_hashes(reference) != reference_digest:
        raise RuntimeError("reference parameters changed")
    changed = sum(value != lora_before[name] for name, value in parameter_hashes(client._model, lora=True).items())
    adapter = folder / "adapter-final"
    adapter.mkdir(exist_ok=False)
    client._model.save_pretrained(adapter)
    return adapter, dict(changed_lora_tensors=changed, base_unchanged=True,
                        reference_unchanged=True if reference is not None else None,
                        adapter_sha256=sha(adapter / "adapter_model.safetensors"),
                        reload_max_abs=None, reload_note="final evaluation reloads saved checkpoint; no additional scoring probe")


def sft(args, budget):
    import torch
    from run_v133_goal_o_budget import parameter_hashes
    folder = args.run_dir / "branches" / "sft42"
    if (folder / "summary.json").exists():
        return folder / "adapter-final"
    restarting = (folder / "training-started.json").exists()
    if restarting and not args.resume_training:
        raise RuntimeError("SFT has already started: refuse an implicit restart")
    folder.mkdir(parents=True, exist_ok=True)
    data = rows(args.train_inputs)
    random.Random(42).shuffle(data)
    order = [c["id"] for c in data]
    if not restarting:
        write(folder / "training-started.json", dict(started_at=time.time(), order=order, seed=42))
    client, loading = make_client(args, args.parent_adapter, seed=42)
    write(folder / "training-loading.json", loading)
    named = client.prepare_for_training(); params = [p for _, p in named]
    original = (parameter_hashes(client._model, lora=False), parameter_hashes(client._model, lora=True))
    optimizer = torch.optim.AdamW(params, lr=5e-6, betas=(.9, .999), eps=1e-8, weight_decay=0)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic(); updates = 0; next_index = 0; previous_elapsed = 0.
    check = data[0]["actions"][-1]; cutoff = False; finished = False
    if restarting:
        restored = restore_recovery(folder, client, optimizer, order)
        next_index, updates, previous_elapsed = restored["next_index"], restored["updates"], restored["elapsed_seconds"]
    try:
        for index in range(next_index, len(data)):
            trajectory = data[index]
            try:
                budget.check(training=True)
            except DeadlineReached:
                cutoff = True; break
            actions = trajectory["actions"]
            total = sum(len(a["action_token_ids"]) for a in actions)
            optimizer.zero_grad(set_to_none=True); client.set_training_forward_mode()
            scalar = 0.
            for action in actions:
                ids, prompt, mask = check_action(action)
                current = client.score_action_tensor([], ids, True, prompt, mask)
                loss = -current.sum() / total
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite SFT loss")
                loss.backward(); scalar += float(loss.detach())
                del current, loss
            grad = float(torch.nn.utils.clip_grad_norm_(params, 1.))
            if not math.isfinite(grad) or grad <= 0:
                raise RuntimeError("invalid SFT gradient")
            try:
                budget.check(training=True)
            except DeadlineReached:
                optimizer.zero_grad(set_to_none=True); cutoff = True; break
            finite_optimizer_step(optimizer, params)
            updates += 1; check = actions[-1]
            next_index = index + 1
            append(folder / "updates.jsonl", dict(update=updates, step=updates, case_id=trajectory["id"],
                loss=scalar, grad=grad, supervised_tokens=total, gpu_memory=client.gpu_memory_stats(),
                elapsed_seconds=time.monotonic() - started))
            log("sft_update", branch="sft42", update=updates, loss=scalar)
            if updates % 16 == 0:
                recovery(folder, client, optimizer, order=order, next_index=next_index, updates=updates,
                         elapsed=previous_elapsed + time.monotonic() - started)
        elapsed = previous_elapsed + time.monotonic() - started
        memory = client.gpu_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        adapter, verification = save_and_verify(args, folder, client, check=check, baseline_hashes=original)
        finished = True
    finally:
        if not finished:
            recovery(folder, client, optimizer, order=order, next_index=next_index, updates=updates,
                     elapsed=previous_elapsed + time.monotonic() - started, failure=True, required=False)
        del optimizer, params, named
        release(client)
        del client
    write(folder / "summary.json", dict(method="continued_trajectory_sft", seed=42,
        status="CUTOFF" if cutoff else "COMPLETED", training_complete=updates == 128,
        updates=updates, groups=0, zero_variance=0, training_seconds=elapsed,
        peak_reserved_gb=memory.get("peak_reserved_gb"), gpu_memory=memory,
        **verification))
    return adapter


def grpo(args, seed, engine, budget):
    import torch
    from run_v133_goal_o_budget import parameter_hashes, reference_hashes
    from mini_agent_rl.training.grpo import clipped_grpo_loss, ratio_statistics
    folder = args.run_dir / "branches" / f"grpo{seed}"
    if (folder / "summary.json").exists():
        return folder / "adapter-final"
    restarting = (folder / "training-started.json").exists()
    if restarting and not args.resume_training:
        raise RuntimeError("GRPO has already started: refuse an implicit restart")
    folder.mkdir(parents=True, exist_ok=True)
    data = rows(args.train_cases)
    random.Random(seed).shuffle(data)
    order = [c["id"] for c in data]
    if not restarting:
        write(folder / "training-started.json", dict(started_at=time.time(), order=order, seed=seed))
    client, loading = make_client(args, args.parent_adapter, seed=seed)
    write(folder / "training-loading.json", loading)
    named = client.prepare_for_training(); params = [p for _, p in named]
    original = (parameter_hashes(client._model, lora=False), parameter_hashes(client._model, lora=True))
    reference = {name: p.detach().clone() for name, p in named}
    reference_digest = reference_hashes(reference)
    optimizer = torch.optim.AdamW(params, lr=5e-6, betas=(.9, .999), eps=1e-8, weight_decay=0)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic(); groups = updates = zero = next_index = 0; previous_elapsed = 0.
    check = rows(args.train_inputs)[0]["actions"][-1]; cutoff = False; finished = False; pending = None
    if restarting:
        restored = restore_recovery(folder, client, optimizer, order)
        next_index, updates, groups, zero, previous_elapsed = (restored["next_index"], restored["updates"],
            restored["groups"], restored["zero_variance"], restored["elapsed_seconds"])
    try:
        for index in range(next_index, len(data)):
            case = data[index]
            try:
                budget.check(training=True)
                group = asyncio.run(wavefront(client, engine, [case] * 4, sample=True,
                    seed=seed * 100000 + index * 10, policy_version=f"grpo{seed}:update{updates}",
                    budget=budget, training=True, journal_folder=folder))
            except DeadlineReached:
                cutoff = True; break
            for candidate, trajectory in enumerate(group):
                trajectory.update(case_id=case["id"], candidate=candidate, group=index + 1,
                    policy_version=f"grpo{seed}:update{updates}", **metrics(trajectory["answer"], case))
                trajectory["reward"] = float(trajectory["normalized_exact_match"])
                append(folder / "rollouts.jsonl", trajectory)
            rewards = [r["reward"] for r in group]
            mean, std = statistics.mean(rewards), statistics.pstdev(rewards)
            groups += 1
            record = dict(group=groups, case_id=case["id"], rewards=rewards, reward_mean=mean,
                reward_std=std, policy_version=f"grpo{seed}:update{updates}", updated=False,
                optimizer_steps=0, policy_epochs=2, metrics=[])
            if std == 0:
                zero += 1; record["skip_reason"] = "zero_variance"
            else:
                client._model.eval()
                frozen = []
                for trajectory in group:
                    actions = []
                    for action in trajectory["trace"]:
                        budget.check(training=True)
                        ids, prompt, mask = check_action(action)
                        old = client.score_action_tensor([], ids, False, prompt, mask).detach().float().cpu()
                        ref = client.reference_logprobs([], ids, reference, prompt, mask).detach().float().cpu()
                        if old.numel() != len(ids) or ref.numel() != len(ids) or not torch.isfinite(old).all() or not torch.isfinite(ref).all():
                            raise RuntimeError("invalid old/reference exact-token scores")
                        actions.append((action, old, ref)); check = action
                    frozen.append(actions)
                for epoch in range(2):
                    pending = dict(case_id=case["id"], index=index, next_policy_epoch=epoch, group=group,
                        frozen_scores=[[(a, old.tolist(), ref.tolist()) for a, old, ref in actions] for actions in frozen])
                    budget.check(training=True)
                    optimizer.zero_grad(set_to_none=True); client.set_training_forward_mode()
                    ratios = []; losses = []; kls = []; clips = []
                    for trajectory, actions in zip(group, frozen):
                        count = sum(len(a[0]["action_token_ids"]) for a in actions)
                        advantage = (trajectory["reward"] - mean) / max(std, 1e-8)
                        for action, old, ref in actions:
                            budget.check(training=True)
                            ids, prompt, mask = check_action(action)
                            current = client.score_action_tensor([], ids, True, prompt, mask)
                            old_device, ref_device = old.to(current.device), ref.to(current.device)
                            ratios.append((current.detach() - old_device).exp().float().cpu())
                            loss, kl, clip = clipped_grpo_loss(current, old_device, ref_device, advantage, .2, .02)
                            if not torch.isfinite(loss):
                                raise RuntimeError("nonfinite GRPO loss")
                            weight = len(ids) / count / 4
                            (loss * weight).backward()
                            losses.append(float(loss.detach()) * weight); kls.append(float(kl) * weight)
                            clips.append(float(clip) * weight)
                            del current, loss, old_device, ref_device
                    ratio = ratio_statistics(ratios)
                    grad = float(torch.nn.utils.clip_grad_norm_(params, 1.))
                    if not math.isfinite(grad) or grad <= 0:
                        raise RuntimeError("invalid GRPO gradient")
                    budget.check(training=True)
                    finite_optimizer_step(optimizer, params)
                    updates += 1
                    epoch_row = dict(update=updates, group=groups, policy_epoch=epoch + 1,
                        loss=sum(losses), kl=sum(kls), clip_fraction=sum(clips), grad=grad,
                        ratio=ratio, gpu_memory=client.gpu_memory_stats(), elapsed_seconds=time.monotonic() - started)
                    append(folder / "updates.jsonl", epoch_row)
                    record["metrics"].append(epoch_row); record["optimizer_steps"] += 1; record["updated"] = True
                    log("grpo_update", branch=f"grpo{seed}", update=updates, group=groups,
                        policy_epoch=epoch + 1, rewards=rewards, kl=sum(kls), clip_fraction=sum(clips))
                del frozen
            next_index = index + 1; pending = None
            record["elapsed_seconds"] = time.monotonic() - started
            record["gpu_memory"] = client.gpu_memory_stats()
            append(folder / "group-metrics.jsonl", record)
            log("grpo_group", branch=f"grpo{seed}", group=groups, updated=record["updated"], rewards=rewards)
            if groups % 16 == 0:
                recovery(folder, client, optimizer, order=order, next_index=next_index, updates=updates,
                    groups=groups, zero=zero, elapsed=previous_elapsed + time.monotonic() - started)
        elapsed = previous_elapsed + time.monotonic() - started
        memory = client.gpu_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        adapter, verification = save_and_verify(args, folder, client, check=check, baseline_hashes=original,
                                reference=reference, reference_digest=reference_digest)
        finished = True
    except DeadlineReached:
        cutoff = True
        elapsed = previous_elapsed + time.monotonic() - started
        memory = client.gpu_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        if "record" in locals():
            record.update(cutoff=True, elapsed_seconds=elapsed, gpu_memory=memory)
            append(folder / "group-metrics.jsonl", record)
        adapter, verification = save_and_verify(args, folder, client, check=check, baseline_hashes=original,
                                reference=reference, reference_digest=reference_digest)
        finished = True
    finally:
        if not finished:
            recovery(folder, client, optimizer, order=order, next_index=next_index, updates=updates,
                groups=groups, zero=zero, elapsed=previous_elapsed + time.monotonic() - started,
                failure=True, pending=pending, required=False)
        del optimizer, params, named, reference
        release(client)
        del client
    write(folder / "summary.json", dict(method="online_hybrid_grpo", seed=seed,
        status="CUTOFF" if cutoff else "COMPLETED" if updates else "NO_UPDATE",
        training_complete=groups == 128 and not cutoff, groups=groups, samples=len(rows(folder / "rollouts.jsonl")),
        partial_trajectory_attempts=len(rows(folder / "partial-rollouts.jsonl")),
        model_generation_attempts=sum(r["rows"] for r in rows(folder / "generation-attempts.jsonl") if r["event"] == "started"),
        updates=updates, zero_variance=zero, policy_epochs=2, training_seconds=elapsed,
        effective_groups=sum(bool(r.get("updated")) for r in rows(folder / "group-metrics.jsonl")),
        peak_reserved_gb=memory.get("peak_reserved_gb"), gpu_memory=memory,
        **verification))
    return adapter


def prepare(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    inputs = dict(base_model=args.base_model, parent_adapter=args.parent_adapter,
        train_cases=args.train_cases, train_inputs=args.train_inputs, eval_cases=args.eval_cases,
        eval_retrieval_config=args.eval_retrieval_config)
    identity = {k: str(v.resolve()) for k, v in inputs.items()}
    identity["hashes"] = {k: sha(v) for k, v in inputs.items() if v.is_file()}
    identity["parent_sha256"] = sha(args.parent_adapter / "adapter_model.safetensors")
    if identity["parent_sha256"] != PARENT_SHA:
        raise ValueError("parent adapter is not the frozen latest AM seed42-A")
    train, evaluate_rows, trajectories = rows(args.train_cases), rows(args.eval_cases), rows(args.train_inputs)
    if len(train) != 128 or len(evaluate_rows) != 128 or len(trajectories) != 128:
        raise ValueError("existing 128 training cases /128 demonstrations /128 evaluation cases required")
    if {c["id"] for c in train} & {c["id"] for c in evaluate_rows}:
        raise ValueError("training/evaluation case IDs overlap")
    if {c["id"] for c in train} != {c["id"] for c in trajectories}:
        raise ValueError("training question and demonstration IDs differ")
    config = dict(identity=identity, product=PRODUCT, group_size=4, policy_epochs=2,
        train_passes=1, seeds=[42, 43], sft_updates=128, grpo_groups_per_seed=128,
        grpo_max_updates_per_seed=256, reward="existing aliases normalized_exact_match; terminal binary",
        temperature=1., top_p=1., top_k=0, learning_rate=5e-6, clip_epsilon=.2, kl_beta=.02,
        adam_betas=[.9, .999], adam_eps=1e-8, weight_decay=0., max_grad_norm=1.,
        reference="fixed initial AM LoRA snapshot", batch_size=args.batch_size,
        final_evaluation_batch_size=args.eval_batch_size, train_deadline_epoch=args.train_deadline_epoch,
        deadline_epoch=args.deadline_epoch, no_new_quality_gate=True,
        evaluation_identity="already exposed 128 cases; no independent generalization claim",
        runner_sha256=sha(__file__))
    manifest = args.run_dir / "manifest.json"
    if manifest.exists():
        if read(manifest) != config:
            raise RuntimeError("existing run manifest differs; use a new isolated run directory")
    else:
        write(manifest, config)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "base-model", "parent-adapter", "train-cases", "train-inputs", "eval-cases", "eval-retrieval-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--train-retrieval-config", type=Path)
    parser.add_argument("--train-corpus", type=Path)
    parser.add_argument("--train-index", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--train-deadline-epoch", type=float, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--stage", choices=["all", "prepare", "baseline", "efficiency", "sft", "grpo42", "grpo43", "evaluate"], default="all")
    parser.add_argument("--resume-training", action="store_true", help="explicitly restore last completed periodic boundary; keep the original absolute deadline")
    args = parser.parse_args(argv)
    if args.batch_size != 4 or args.eval_batch_size < 1:
        parser.error("fixed training group/batch size is 4; evaluation batch size must be positive")
    prepare(args)
    if args.stage == "prepare":
        log("prepared", run_dir=str(args.run_dir)); return
    budget = Budget(args); budget.check()
    status_path = args.run_dir / "status.json"
    status = read(status_path) if status_path.exists() else dict(status="RUNNING", started_at=time.time(),
        deadline_epoch=args.deadline_epoch, train_deadline_epoch=args.train_deadline_epoch, stages={})
    write(status_path, status)
    try:
        train_engine, eval_engine = engines(args)
        tasks = ["baseline", "efficiency", "sft", "grpo42", "grpo43"] if args.stage == "all" else [args.stage]
        for stage in tasks:
            budget.check()
            status.update(status="RUNNING", current_stage=stage)
            write(status_path, status)
            log("stage_started", stage=stage)
            started = time.monotonic()
            if stage == "baseline":
                evaluate(args, "am-serial", args.parent_adapter, eval_engine, budget)
            elif stage == "efficiency":
                evaluate(args, "am-batch", args.parent_adapter, eval_engine, budget, batch_size=args.batch_size)
            elif stage == "sft":
                adapter = sft(args, budget)
                evaluate(args, "sft42", adapter, eval_engine, budget, batch_size=args.eval_batch_size)
            elif stage in ("grpo42", "grpo43"):
                adapter = grpo(args, int(stage[-2:]), train_engine, budget)
                evaluate(args, stage, adapter, eval_engine, budget, batch_size=args.eval_batch_size)
            elif stage == "evaluate":
                for label in ("sft42", "grpo42", "grpo43"):
                    adapter = args.run_dir / "branches" / label / "adapter-final"
                    if adapter.exists():
                        evaluate(args, label, adapter, eval_engine, budget, batch_size=args.eval_batch_size)
            status["stages"][stage] = dict(status="COMPLETED", elapsed_seconds=time.monotonic() - started,
                                          completed_at=time.time())
            write(status_path, status)
            log("stage_completed", stage=stage)
        status.update(status="COMPLETED", completed_at=time.time(), current_stage=None)
        write(status_path, status)
    except BaseException as exc:
        status.update(status="CUTOFF" if isinstance(exc, DeadlineReached) else "FAILED", error=repr(exc),
                      error_type=type(exc).__name__, stopped_at=time.time())
        write(status_path, status)
        log("stopped", **status)
        raise


if __name__ == "__main__":
    main()
