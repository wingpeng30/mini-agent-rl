"""Goal T: paired 32/64-update checkpoints from continuous answer SFT."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import pickle
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import run_v133_goal_q_search as q
import run_v133_goal_r_evidence_order_v2 as r
import run_v133_goal_s_label_granularity as s

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("GOAL_T_OUT", str(ROOT / "runs/v133-goal-t-update-length"))).resolve()
CFG = OUT / "config.json"
STATE = OUT / "state.json"
SOURCE = q.OLD_RL
STAGES = ("baseline", "train42", "train43", "evaluation", "diagnostics", "final")
CHECKPOINTS = (32, 64)
SEEDS = (42, 43)


def read(path): return s.read(path)
def rows(path): return s.rows(path)
def write(path, value): return s.write(path, value)
def append(path, value): return s.append(path, value)
def sha(path): return s.sha(path)


def asset_paths():
    weights = sorted(q.p.BASE.glob("*.safetensors"))
    if not weights: raise RuntimeError("Base weight files missing")
    return {"adapter": SOURCE / "adapter_model.safetensors",
            "tokenizer": q.p.BASE / "tokenizer.json", "corpus": q.p.CORPUS,
            "cases": q.p.CASES, "labels": s.LABELS,
            "s_config": s.CFG, "training_inputs": r.OUT / "train-inputs.jsonl",
            "q_runner": Path(q.__file__), "r_runner": Path(r.__file__),
            "s_runner": Path(s.__file__), "model_code": ROOT / "src/mini_agent_rl/model/local.py",
            "runner": Path(__file__),
            **{f"base_weight_{i}": path for i, path in enumerate(weights)}}


def software_environment():
    import torch
    import transformers
    import peft
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "env": {key: os.environ.get(key) for key in (
                "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")}}


def prepare():
    if OUT.exists(): raise FileExistsError(OUT)
    if read(q.OUT / "decision.json")["selected"] != "old_rl":
        raise RuntimeError("Goal Q incumbent changed")
    if read(s.OUT / "decision.json")["selected"] != "incumbent":
        raise RuntimeError("Goal S selected product changed")
    if len(q.train_cases()) != 16 or len(q.dev_cases()) != 32:
        raise RuntimeError("Frozen split counts changed")
    labels = read(s.LABELS)
    inputs = rows(r.OUT / "train-inputs.jsonl")
    if labels["changes"] != 2 or len(labels["cases"]) != 16 or len(inputs) != 16:
        raise RuntimeError("Goal S labels or Goal R inputs changed")
    if labels["source_sha256"] != sha(r.OUT / "train-inputs.jsonl"):
        raise RuntimeError("Training labels do not match visible input evidence")
    if any(not item["eligible"] for item in inputs):
        raise RuntimeError("Training context ineligible")
    order = read(s.CFG)["order"]
    ids = {item["case_id"] for item in inputs}
    if set(order) != {"42", "43"}:
        raise RuntimeError("Goal S training permutations missing")
    for seed in SEEDS:
        if len(order[str(seed)]) != 2 or any(len(pass_ids) != 16 or set(pass_ids) != ids
                                               for pass_ids in order[str(seed)]):
            raise RuntimeError("Invalid Goal S training permutation")
    environment = software_environment()
    if not environment["gpu"]: raise RuntimeError("GPU unavailable")
    hashes = {key: sha(path) for key, path in asset_paths().items()}
    OUT.mkdir(parents=True)
    write(CFG, {"version": "goal-t-v1", "hashes": hashes, "environment": environment,
                "model_settings": q.load_config()["product"], "order": order,
                "seeds": list(SEEDS), "checkpoint_updates": list(CHECKPOINTS),
                "max_updates_per_seed": 64, "max_total_updates": 128,
                "learning_rate": 5e-6, "betas": [.9, .999], "eps": 1e-8,
                "weight_decay": 0, "grad_clip": 1., "passes": 2, "copies_per_case_pass": 2,
                "baseline": {"total": 19, "regression": 7, "invalid": 0},
                "gpu_time_limit": None, "stage_timeouts": None,
                "semantic_rubric_sha256": sha(s.OUT / "semantic-rubric.json")})
    write(STATE, {"status": "prepared", "gpu_seconds": 0., "stages": []})


def config():
    c = read(CFG)
    if {key: sha(path) for key, path in asset_paths().items()} != c["hashes"]:
        raise RuntimeError("Frozen asset changed")
    if software_environment() != c["environment"]:
        raise RuntimeError("Software or GPU environment changed")
    if sha(s.OUT / "semantic-rubric.json") != c["semantic_rubric_sha256"]:
        raise RuntimeError("Frozen semantic rubric changed")
    return c


def checkpoint_dir(seed, updates): return OUT / f"seed{seed}/update{updates}"


def checkpoint_due(updates, epoch):
    if updates not in CHECKPOINTS: return False
    if epoch != CHECKPOINTS.index(updates) + 1:
        raise RuntimeError("Checkpoint at wrong pass")
    return True


def rng_snapshot(torch, np):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]}


def rng_equal(torch, np, left, right):
    return (left["python"] == right["python"]
            and left["numpy"][0] == right["numpy"][0]
            and np.array_equal(left["numpy"][1], right["numpy"][1])
            and left["numpy"][2:] == right["numpy"][2:]
            and torch.equal(left["cpu"], right["cpu"])
            and len(left["cuda"]) == len(right["cuda"])
            and all(torch.equal(x, y) for x, y in zip(left["cuda"], right["cuda"])))


def restore_rng(torch, np, snapshot):
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["cpu"])
    torch.cuda.set_rng_state_all(snapshot["cuda"])


def optimizer_hash(optimizer):
    return hashlib.sha256(pickle.dumps(optimizer.state_dict(), protocol=5)).hexdigest()


def save_checkpoint(client, optimizer, seed, updates, case_id, prompt, mask, target, base_before, lora_before):
    import torch
    import numpy as np
    import run_v133_goal_o_budget as o

    if updates not in CHECKPOINTS: raise RuntimeError("Unplanned checkpoint")
    folder = checkpoint_dir(seed, updates)
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir(parents=True)
    rng_before = rng_snapshot(torch, np)
    optimizer_before = optimizer_hash(optimizer)
    training_before = client._model.training
    lora_saved = o.parameter_hashes(client._model, lora=True)
    if o.parameter_hashes(client._model, lora=False) != base_before:
        raise RuntimeError("Base changed before checkpoint")
    changed = sum(lora_saved[key] != value for key, value in lora_before.items())
    if changed == 0: raise RuntimeError("LoRA unchanged")
    checkpoint = folder / "adapter-final"
    try:
        client._model.save_pretrained(checkpoint)
        expected = client.score_action_tensor([], target, False, prompt, mask).detach().float().cpu().tolist()
    finally:
        restore_rng(torch, np, rng_before)
        client._model.train(training_before)
        client.set_training_forward_mode()
    if not rng_equal(torch, np, rng_before, rng_snapshot(torch, np)):
        raise RuntimeError("RNG state changed while saving checkpoint")
    if optimizer_hash(optimizer) != optimizer_before:
        raise RuntimeError("Optimizer state changed while saving checkpoint")
    if o.parameter_hashes(client._model, lora=True) != lora_saved:
        raise RuntimeError("LoRA changed while saving checkpoint")
    write(folder / "checkpoint-check.json", {"seed": seed, "updates": updates,
          "case_id": case_id, "prompt_token_ids": prompt, "prompt_attention_mask": mask,
          "target_token_ids": target, "expected_logprobs": expected,
          "checkpoint_sha256": sha(checkpoint / "adapter_model.safetensors"),
          "changed_lora": changed, "base_unchanged": True,
          "rng_unchanged": True, "optimizer_unchanged": True,
          "training_mode_unchanged": client._model.training == training_before,
          "gpu_memory": client.gpu_memory_stats()})


def train(seed):
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as o

    c = config()
    set_seed(seed)
    data = {row["case_id"]: row for row in rows(r.OUT / "train-inputs.jsonl")}
    label_map = {row["case_id"]: row["revised_answer"] for row in read(s.LABELS)["cases"]}
    if len(data) != 16 or set(data) != set(label_map): raise RuntimeError("Training data mismatch")
    folder = OUT / f"seed{seed}"
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir(parents=True)
    client, _, _ = q.make_client(SOURCE, deterministic=True)
    params = [value for _, value in client.prepare_for_training()]
    base_before = o.parameter_hashes(client._model, lora=False)
    lora_before = o.parameter_hashes(client._model, lora=True)
    optimizer = torch.optim.AdamW(params, lr=5e-6, betas=(.9, .999), eps=1e-8, weight_decay=0)
    updates = 0
    for epoch, order in enumerate(c["order"][str(seed)], 1):
        for case_id in order:
            row = data[case_id]
            prompt, mask = row["input_ids"], row["input_mask"]
            if len(prompt) != len(mask) or len(prompt) > 2048: raise RuntimeError("Invalid frozen prompt")
            target = s.target_ids(client._processor, label_map[case_id])
            for copy in range(2):
                optimizer.zero_grad(set_to_none=True)
                client.set_training_forward_mode()
                logps = client.score_action_tensor([], target, True, prompt, mask)
                loss = -logps.float().mean()
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
                loss.backward()
                grad = float(torch.nn.utils.clip_grad_norm_(params, 1.))
                if not math.isfinite(grad) or grad <= 0: raise RuntimeError("Invalid gradient")
                guarded_optimizer_step(optimizer, params, client.gpu_memory_stats(), 21.)
                updates += 1
                if updates > 64: raise RuntimeError("Update cap exceeded")
                append(folder / "updates.jsonl", {"case_id": case_id, "pass": epoch,
                    "copy": copy, "update": updates, "loss": float(loss.detach()), "grad": grad,
                    "input_tokens": len(prompt), "target_tokens": len(target),
                    "gpu_memory": client.gpu_memory_stats()})
                if updates % 8 == 0: print(json.dumps({"seed": seed, "updates": updates}), flush=True)
                if checkpoint_due(updates, epoch):
                    save_checkpoint(client, optimizer, seed, updates, case_id, prompt, mask,
                                    target, base_before, lora_before)
    if updates != 64 or o.parameter_hashes(client._model, lora=False) != base_before:
        raise RuntimeError("Training update count or base invariant failed")
    del client, optimizer, params
    gc.collect(); torch.cuda.empty_cache()
    for count in CHECKPOINTS:
        details = read(checkpoint_dir(seed, count) / "checkpoint-check.json")
        loaded, _, _ = q.make_client(checkpoint_dir(seed, count) / "adapter-final", deterministic=True)
        actual = loaded.score_action_tensor([], details["target_token_ids"], False,
                    details["prompt_token_ids"], details["prompt_attention_mask"]).detach().float().cpu()
        expected = torch.tensor(details["expected_logprobs"], dtype=torch.float32)
        delta = float((actual - expected).abs().max())
        if delta > .001: raise RuntimeError(f"Independent reload mismatch {seed}/{count}: {delta}")
        details["reload_max_abs"] = delta
        details["status"] = "passed"
        write(checkpoint_dir(seed, count) / "checkpoint-check.json", details)
        del loaded
        gc.collect(); torch.cuda.empty_cache()
    write(folder / "summary.json", {"status": "passed", "seed": seed,
          "updates": updates, "checkpoints": list(CHECKPOINTS)})


def stats(label):
    path = OUT / f"eval-{label}.jsonl"
    if not path.exists(): return None
    items = rows(path)
    if len(items) != 32 or len({item["case_id"] for item in items}) != 32:
        raise RuntimeError(f"Incomplete evaluation: {label}")
    return {"model": label, "main": sum(x["em"] for x in items if x["split"] == "new_eval"),
            "regression": sum(x["em"] for x in items if x["split"] == "goal_G_regression"),
            "total": sum(x["em"] for x in items),
            "f1": statistics.mean(x["f1"] for x in items),
            "support": sum(x["both_support_retrieved"] for x in items),
            "invalid": sum(x["reason"] == "invalid_action" for x in items),
            "searches": sum(x["searches"] for x in items),
            "mean_latency": statistics.mean(x["elapsed_seconds"] for x in items)}


def diagnostics():
    from evaluate_v133_goal_c_sft import normalize_answer
    baseline = rows(OUT / "eval-incumbent.jsonl")
    gold = {row["case_id"]: row["official_answer"] for row in q.dev_cases()}
    if len(baseline) != 32: raise RuntimeError("Baseline contexts incomplete")
    for label, adapter in (("incumbent", SOURCE),
                           *((f"{seed}-{count}", checkpoint_dir(seed, count) / "adapter-final")
                             for seed in SEEDS for count in CHECKPOINTS)):
        output = OUT / f"fixed-{label}.jsonl"
        if output.exists(): raise FileExistsError(output)
        client, wrapped, registry = q.make_client(adapter, deterministic=True)
        client._load()
        for item in baseline:
            final = item["trace"][-1]
            if final.get("action") != "answer": raise RuntimeError("Baseline did not answer")
            result = s.fixed_answer(client, wrapped, registry, item["question"],
                                    item["trace"], final["prompt_token_ids"])
            answer = result["answer"]
            append(output, {"model": label, "case_id": item["case_id"],
                "answer": answer, "raw": result["raw"], "invalid": result["invalid"],
                "em": int(bool(answer) and normalize_answer(answer) == normalize_answer(gold[item["case_id"]]))})
        del client, wrapped, registry
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def eligible(result):
    return bool(result and result["total"] > 19 and result["regression"] >= 7 and result["invalid"] == 0)


def decide():
    baseline = stats("incumbent")
    if (baseline["total"], baseline["regression"], baseline["invalid"]) != (19, 7, 0):
        raise RuntimeError("Baseline changed")
    short = {str(seed): stats(f"{seed}-32") for seed in SEEDS}
    long = {str(seed): stats(f"{seed}-64") for seed in SEEDS}
    for seed in SEEDS:
        if read(OUT / f"seed{seed}/summary.json")["status"] != "passed":
            raise RuntimeError("Training check failed")
        for count in CHECKPOINTS:
            if read(checkpoint_dir(seed, count) / "checkpoint-check.json")["status"] != "passed":
                raise RuntimeError("Checkpoint check failed")
    accepted = all(eligible(short[str(seed)]) for seed in SEEDS)
    selected = "42-32" if accepted else "incumbent"
    adapter = checkpoint_dir(42, 32) / "adapter-final" if accepted else SOURCE
    write(OUT / "decision.json", {"selected": selected, "adapter": str(adapter),
         "baseline": baseline, "short": short, "long": long,
         "reason": "both_short_models_qualified" if accepted else "incumbent_retained"})


def stage(name):
    config()
    if name == "baseline":
        q.eval_model("incumbent", SOURCE, q.dev_cases(), OUT / "eval-incumbent.jsonl")
        result = stats("incumbent")
        if (result["total"], result["regression"], result["invalid"]) != (19, 7, 0):
            raise RuntimeError("Current product baseline not reproduced")
    elif name in ("train42", "train43"):
        train(int(name[-2:]))
    elif name == "evaluation":
        if any(read(OUT / f"seed{seed}/summary.json")["status"] != "passed" for seed in SEEDS):
            raise RuntimeError("Both seeds must finish training before evaluation")
        for seed in SEEDS:
            for count in CHECKPOINTS:
                label = f"{seed}-{count}"
                q.eval_model(label, checkpoint_dir(seed, count) / "adapter-final",
                             q.dev_cases(), OUT / f"eval-{label}.jsonl")
    elif name == "diagnostics": diagnostics()
    elif name == "final":
        decision = read(OUT / "decision.json")
        client, wrapped, registry = q.make_client(decision["adapter"], deterministic=True)
        case = q.train_cases()[0]
        result = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False, seed=42))
        write(OUT / "final-demo.json", {"selected": decision["selected"], "trace": result})
    else: raise ValueError(name)


def run_stage(name, state):
    if name not in STAGES: raise ValueError(name)
    state["current_stage"] = name
    write(STATE, state)
    started = time.monotonic()
    try:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "stage", "--name", name],
                       cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")},
                       check=True)
    finally:
        elapsed = time.monotonic() - started
        state["gpu_seconds"] += elapsed
        state["stages"].append({"name": name, "seconds": elapsed})
        write(STATE, state)


def run():
    config()
    state = read(STATE)
    if state["status"] != "prepared": raise RuntimeError("Run already started")
    state["status"] = "running"
    write(STATE, state)
    try:
        for name in STAGES[:-1]: run_stage(name, state)
        decide()
        run_stage("final", state)
        state.update(status="complete", current_stage="complete")
    except BaseException as error:
        state.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(STATE, state)
        summarize()


def summarize():
    state = read(STATE)
    models = []
    for label in ("incumbent", "42-32", "42-64", "43-32", "43-64"):
        try:
            item = stats(label)
            if item: models.append(item)
        except RuntimeError:
            models.append({"model": label, "status": "partial"})
    write(OUT / "summary.json", {"status": state["status"], "gpu_seconds": state["gpu_seconds"],
        "stages": state["stages"], "models": models,
        "decision": read(OUT / "decision.json") if (OUT / "decision.json").exists() else None})
    print(json.dumps(read(OUT / "summary.json"), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "stage", "summarize"])
    parser.add_argument("--name", choices=STAGES)
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "run": run()
    elif args.command == "stage": stage(args.name)
    else: summarize()


if __name__ == "__main__": main()
