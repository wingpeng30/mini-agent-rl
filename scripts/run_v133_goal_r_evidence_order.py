"""Goal R: answer-only SFT with matched document-order controls."""
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
import subprocess
import sys
import time
from pathlib import Path

import run_v133_goal_q_search as q

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-r-evidence-order"
CFG = OUT / "config.json"
STATE = OUT / "state.json"
START = q.OLD_RL
CAPS = {"baseline": 600, "collect": 600, "a": 1200, "b": 1200,
        "evaluation": 600, "confirm": 1800, "diagnostic": 900, "final": 300}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def prepare():
    if OUT.exists(): raise FileExistsError(OUT)
    if read(q.OUT / "decision.json")["selected"] != "old_rl":
        raise RuntimeError("Goal Q selected model changed")
    if len(q.train_cases()) != 16 or len(q.dev_cases()) != 32:
        raise RuntimeError("frozen data incomplete")
    paths = {"adapter": START / "adapter_model.safetensors", "corpus": q.p.CORPUS,
             "cases": q.p.CASES, "train": q.p.J / "train-frozen.jsonl",
             "runner": Path(__file__), "q_runner": Path(q.__file__),
             "model": ROOT / "src/mini_agent_rl/model/local.py",
             "tokenizer": q.p.BASE / "tokenizer.json", "base": q.p.BASE / "config.json"}
    hashes = {key: sha(path) for key, path in paths.items()}
    order = {}
    original = [x["case_id"] for x in q.train_cases()]
    for seed in (42, 43):
        order[str(seed)] = []
        for epoch in range(2):
            ids = original.copy(); random.Random(seed * 100 + epoch).shuffle(ids)
            order[str(seed)].append(ids)
    OUT.mkdir(parents=True)
    write(CFG, {"version": "goal-r-v1", "hashes": hashes, "seed": 42, "confirm_seed": 43,
                "order": order, "passes": 2, "max_updates_per_arm": 64,
                "learning_rate": 5e-6, "betas": [.9, .999], "eps": 1e-8,
                "weight_decay": 0, "max_grad_norm": 1., "input_limit": 2048,
                "new_token_limit": 128, "min_eligible": 8, "baseline_total": 19,
                "baseline_regression": 7, "baseline_invalid": 0,
                "gpu_total_cap": 7200, "caps": CAPS})
    write(STATE, {"status": "prepared", "stage": "baseline", "gpu_seconds": 0., "stages": []})
    print(json.dumps({"prepared": str(OUT), "hashes": hashes}))


def config():
    c = read(CFG)
    paths = {"adapter": START / "adapter_model.safetensors", "corpus": q.p.CORPUS,
             "cases": q.p.CASES, "train": q.p.J / "train-frozen.jsonl",
             "runner": Path(__file__), "q_runner": Path(q.__file__),
             "model": ROOT / "src/mini_agent_rl/model/local.py",
             "tokenizer": q.p.BASE / "tokenizer.json", "base": q.p.BASE / "config.json"}
    for key, path in paths.items():
        if sha(path) != c["hashes"][key]: raise RuntimeError(f"frozen asset changed: {key}")
    return c


def answer_target(processor, answer):
    raw = json.dumps({"action": "answer", "answer": answer}, ensure_ascii=False)
    ids = processor.tokenizer.encode(raw, add_special_tokens=False)
    end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(end, int) or end < 0: raise RuntimeError("missing assistant terminator")
    return ids + [end], raw


def rebuild(question, trace, *, reverse):
    """Recreate the Q product messages up to the first final answer."""
    from mini_agent_rl.domain import Message, ToolCall
    from mini_agent_rl.prompts import system_prompt
    messages = [Message(role="system", content=system_prompt(False, "direct-v090")),
                Message(role="user", content=question)]
    for item in trace:
        if item.get("action") == "answer": break
        if item.get("action") != "search" or "observation" not in item:
            raise RuntimeError("cannot reconstruct incomplete search trajectory")
        query = item["query"]
        observed = json.loads(json.dumps(item["observation"], ensure_ascii=False))
        if reverse: observed["result"] = list(reversed(observed["result"]))
        call = ToolCall(name="search", arguments={"query": query})
        messages.append(Message(role="assistant", content=json.dumps({"action": "search", "query": query}, ensure_ascii=False),
                                tool_calls=[call]))
        visible = {key: value for key, value in observed.items() if key != "latency_ms"}
        messages.append(Message(role="tool", content=json.dumps(visible, ensure_ascii=False), tool_call_id=call.id))
    return messages


def prompt_ids(client, messages):
    inputs = client._inputs(messages, "goal_r_input")
    ids = inputs["input_ids"][0].tolist()
    mask = inputs["attention_mask"][0].tolist()
    if len(ids) > 2048: raise RuntimeError("input limit exceeded")
    return ids, mask


def collect():
    import torch
    c = config()
    output = OUT / "train-inputs.jsonl"
    if output.exists(): raise FileExistsError(output)
    client, wrapped, registry = q.make_client(START, deterministic=True)
    count = 0
    for case in q.train_cases():
        rollout = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False, seed=42))
        observed = {doc["id"] for item in rollout["trace"]
                    for doc in item.get("observation", {}).get("result", [])}
        final = next((x for x in rollout["trace"] if x.get("action") == "answer"), None)
        entry = {"case_id": case["case_id"], "eligible": False, "reason": "unknown",
                 "rollout": rollout}
        if not final or rollout["reason"] != "answer" or not rollout["searches"]:
            entry["reason"] = "no_final_answer_after_search"
        elif not set(case["support_ids"]).issubset(observed):
            entry["reason"] = "support_not_retrieved"
        elif final["usage"].get("prompt_truncated"):
            entry["reason"] = "original_input_truncated"
        else:
            original = rebuild(case["question"], rollout["trace"], reverse=False)
            reversed_messages = rebuild(case["question"], rollout["trace"], reverse=True)
            ids, mask = prompt_ids(client, original)
            if ids != final["prompt_token_ids"] or mask != final["prompt_attention_mask"]:
                raise RuntimeError(f"product prompt mismatch {case['case_id']}")
            reverse_ids, reverse_mask = prompt_ids(client, reversed_messages)
            if len(reverse_ids) > 2048:
                entry["reason"] = "reverse_input_over_limit"
            else:
                target, raw = answer_target(client._processor, case["official_answer"])
                with torch.no_grad():
                    values = client.score_action_tensor([], target, False, ids, mask)
                if not torch.isfinite(values).all(): raise RuntimeError("nonfinite teacher target score")
                entry.update(eligible=True, reason="eligible", input_ids=ids, input_mask=mask,
                             reverse_ids=reverse_ids, reverse_mask=reverse_mask,
                             answer_token_ids=target, target_raw=raw,
                             answer_target_tokens=len(target))
                count += 1
        append(output, entry)
    write(OUT / "input-summary.json", {"eligible": count, "total": 16,
                                       "reasons": {reason: sum(x["reason"] == reason for x in rows(output))
                                                   for reason in sorted({x["reason"] for x in rows(output)})}})
    del client, wrapped, registry
    gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"eligible": count}))


def train(arm, seed, destination):
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as o
    c = config(); set_seed(seed)
    data = {x["case_id"]: x for x in rows(OUT / "train-inputs.jsonl") if x["eligible"]}
    if len(data) < 8: raise RuntimeError("insufficient eligible training examples")
    folder = OUT / destination
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir(parents=True)
    client, _, _ = q.make_client(START, deterministic=True)
    named = client.prepare_for_training(); params = [value for _, value in named]
    base_before = o.parameter_hashes(client._model, lora=False)
    lora_before = o.parameter_hashes(client._model, lora=True)
    optimizer = torch.optim.AdamW(params, lr=5e-6, betas=(.9, .999), eps=1e-8, weight_decay=0)
    updates = 0; check = None
    for epoch, order in enumerate(c["order"][str(seed)], 1):
        for case_id in order:
            if case_id not in data: continue
            row = data[case_id]
            for variant in (0, 1):
                use_reverse = arm == "b" and variant == 1
                prompt = row["reverse_ids"] if use_reverse else row["input_ids"]
                mask = row["reverse_mask"] if use_reverse else row["input_mask"]
                target = row["answer_token_ids"]
                optimizer.zero_grad(set_to_none=True)
                client.set_training_forward_mode()
                values = client.score_action_tensor([], target, True, prompt, mask)
                loss = -values.float().mean()
                if not torch.isfinite(loss): raise RuntimeError("nonfinite SFT loss")
                loss.backward()
                grad = float(torch.nn.utils.clip_grad_norm_(params, 1.))
                if not math.isfinite(grad) or grad <= 0: raise RuntimeError("invalid SFT gradient")
                guarded_optimizer_step(optimizer, params, client.gpu_memory_stats(), 21.)
                updates += 1
                check = (prompt, mask, target)
                append(folder / "updates.jsonl", {"case_id": case_id, "pass": epoch,
                    "variant": variant, "reversed": use_reverse, "loss": float(loss.detach()),
                    "grad": grad, "input_tokens": len(prompt), "target_tokens": len(target),
                    "gpu_memory": client.gpu_memory_stats()})
                if updates > 64: raise RuntimeError("update cap exceeded")
    if updates != 4 * len(data): raise RuntimeError("missing matched SFT updates")
    if o.parameter_hashes(client._model, lora=False) != base_before: raise RuntimeError("base changed")
    lora_after = o.parameter_hashes(client._model, lora=True)
    changed = sum(lora_after[key] != value for key, value in lora_before.items())
    if not changed: raise RuntimeError("LoRA unchanged")
    checkpoint = folder / "adapter-final"
    client._model.save_pretrained(checkpoint)
    before = client.score_action_tensor([], check[2], False, check[0], check[1]).detach().float().cpu()
    del client, optimizer, params
    gc.collect(); torch.cuda.empty_cache()
    loaded, _, _ = q.make_client(checkpoint, deterministic=True)
    after = loaded.score_action_tensor([], check[2], False, check[0], check[1]).detach().float().cpu()
    delta = float((before - after).abs().max())
    if delta > .001: raise RuntimeError(f"reload mismatch {delta}")
    write(folder / "summary.json", {"status": "PASSED", "arm": arm, "seed": seed,
          "eligible": len(data), "updates": updates, "base_unchanged": True,
          "changed_lora": changed, "reload_max_abs": delta,
          "checkpoint_sha256": sha(checkpoint / "adapter_model.safetensors")})
    print(json.dumps({"arm": arm, "seed": seed, "updates": updates, "reload_delta": delta}))


def eval_model(label, adapter):
    q.eval_model(label, adapter, q.dev_cases(), OUT / f"eval-{label}.jsonl")


def stats(label):
    path = OUT / f"eval-{label}.jsonl"
    if not path.exists(): return None
    data = rows(path)
    if len(data) != 32 or len({x["case_id"] for x in data}) != 32:
        raise RuntimeError(f"incomplete eval {label}")
    return {"model": label, "total": sum(x["em"] for x in data),
            "new_eval": sum(x["em"] for x in data if x["split"] == "new_eval"),
            "regression": sum(x["em"] for x in data if x["split"] == "goal_G_regression"),
            "invalid": sum(x["reason"] == "invalid_action" for x in data),
            "f1": statistics.mean(x["f1"] for x in data),
            "support": sum(x["both_support_retrieved"] for x in data),
            "searches": sum(x["searches"] for x in data),
            "seconds": sum(x["elapsed_seconds"] for x in data)}


def select():
    baseline = stats("incumbent")
    if (baseline["total"], baseline["regression"], baseline["invalid"]) != (19, 7, 0):
        raise RuntimeError("baseline shifted; selection forbidden")
    candidates = [stats("a"), stats("b")]
    eligible = [x for x in candidates if x and x["total"] > 19 and x["regression"] >= 7
                and x["invalid"] == 0]
    eligible.sort(key=lambda x: (-x["total"], -x["regression"], x["searches"], x["model"]))
    winner = eligible[0]["model"] if eligible else None
    write(OUT / "provisional.json", {"baseline": baseline, "candidates": candidates,
                                      "eligible": eligible, "winner": winner})
    return winner


def decide():
    provisional = read(OUT / "provisional.json")
    winner = provisional["winner"]
    confirm = stats("confirm") if winner else None
    accepted = bool(confirm and confirm["total"] > 19 and confirm["regression"] >= 7
                    and confirm["invalid"] == 0)
    selected = winner if accepted else "incumbent"
    adapter = OUT / selected / "adapter-final" if accepted else START
    write(OUT / "decision.json", {"selected": selected, "adapter": str(adapter),
                                   "confirmed": accepted, "replication": confirm,
                                   "reason": "candidate_confirmed" if accepted else "incumbent_retained"})


def diagnostic():
    from mini_agent_rl.domain import Message
    from evaluate_v133_goal_c_sft import normalize_answer, f1_score
    config()
    baseline = rows(OUT / "eval-incumbent.jsonl")
    gold = {x["case_id"]: x for x in q.dev_cases()}
    if len(baseline) != 32: raise RuntimeError("baseline contexts incomplete")
    for label, adapter in (("incumbent", START), ("a", OUT / "a/adapter-final"), ("b", OUT / "b/adapter-final")):
        client, wrapped, registry = q.make_client(adapter, deterministic=True)
        client._load()
        for row in baseline:
            if not row["trace"] or row["trace"][-1].get("action") != "answer":
                # The diagnostic is defined only for actual final-answer states.
                continue
            for mode in ("original", "reversed"):
                messages = rebuild(row["question"], row["trace"], reverse=mode == "reversed")
                ids, mask = prompt_ids(client, messages)
                if mode == "original" and ids != row["trace"][-1]["prompt_token_ids"]:
                    raise RuntimeError("diagnostic input differs from product baseline")
                response = asyncio.run(wrapped.generate(messages, registry.definitions(), temperature=0., seed=42))
                answer = response.content if not response.parse_error and not response.tool_call else None
                actual = gold[row["case_id"]]["official_answer"]
                append(OUT / "order-diagnostic.jsonl", {"model": label, "case_id": row["case_id"],
                    "mode": mode, "input_tokens": ids, "raw": response.raw_text,
                    "answer": answer, "em": int(bool(answer) and normalize_answer(answer) == normalize_answer(actual)),
                    "f1": f1_score(answer, actual) if answer else 0.})
        del client, wrapped, registry
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def stage(name):
    config()
    if name == "baseline": eval_model("incumbent", START)
    elif name == "collect": collect()
    elif name in ("a", "b"): train(name, 42, name)
    elif name == "evaluation":
        eval_model("a", OUT / "a/adapter-final")
        eval_model("b", OUT / "b/adapter-final")
    elif name == "confirm":
        arm = read(OUT / "provisional.json")["winner"]
        if arm not in ("a", "b"): raise RuntimeError("no winner to confirm")
        train(arm, 43, "confirm")
        eval_model("confirm", OUT / "confirm/adapter-final")
    elif name == "diagnostic": diagnostic()
    elif name == "final":
        decision = read(OUT / "decision.json")
        if not Path(decision["adapter"]).exists(): raise RuntimeError("selected adapter missing")
        client, wrapped, registry = q.make_client(decision["adapter"], deterministic=True)
        example = q.train_cases()[0]
        value = asyncio.run(q.rollout(example["question"], wrapped, registry, sample=False, seed=42))
        write(OUT / "final-demo.json", {"selected": decision["selected"], "trace": value})
    else: raise ValueError(name)


def run_stage(name, state):
    cap = CAPS[name]
    if state["gpu_seconds"] + cap > 7200: raise RuntimeError("total GPU budget insufficient")
    state["stage"] = name; write(STATE, state)
    started = time.monotonic()
    command = [sys.executable, str(Path(__file__).resolve()), "stage", "--name", name]
    try:
        subprocess.run(command, cwd=ROOT, check=True, timeout=cap,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    finally:
        seconds = time.monotonic() - started
        state["gpu_seconds"] += seconds
        state["stages"].append({"name": name, "seconds": seconds})
        write(STATE, state)


def run():
    config(); state = read(STATE)
    if state["status"] != "prepared": raise RuntimeError("run already started")
    state["status"] = "running"; write(STATE, state)
    try:
        run_stage("baseline", state)
        if (stats("incumbent")["total"], stats("incumbent")["regression"],
            stats("incumbent")["invalid"]) != (19, 7, 0):
            raise RuntimeError("Goal Q baseline did not reproduce")
        run_stage("collect", state)
        if read(OUT / "input-summary.json")["eligible"] >= 8:
            run_stage("a", state); run_stage("b", state)
            run_stage("evaluation", state)
            if select(): run_stage("confirm", state)
            decide()
            run_stage("diagnostic", state)
        else:
            state["stages"].append({"name": "training", "skipped": "fewer_than_eight_eligible"})
            write(OUT / "decision.json", {"selected": "incumbent", "adapter": str(START),
                                           "confirmed": False, "reason": "insufficient_training_contexts"})
        run_stage("final", state)
        state.update(status="complete", stage="complete")
    except BaseException as exc:
        state.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write(STATE, state)
        summarize()


def summarize():
    state = read(STATE)
    models = []
    for label in ("incumbent", "a", "b", "confirm"):
        try:
            result = stats(label)
            if result: models.append(result)
        except RuntimeError:
            models.append({"model": label, "status": "partial"})
    write(OUT / "summary.json", {"status": state["status"], "gpu_seconds": state["gpu_seconds"],
                                  "stages": state["stages"], "models": models,
                                  "decision": read(OUT / "decision.json") if (OUT / "decision.json").exists() else None})
    print(json.dumps(read(OUT / "summary.json"), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "stage", "summarize"])
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "run": run()
    elif args.command == "stage": stage(args.name)
    else: summarize()


if __name__ == "__main__": main()
