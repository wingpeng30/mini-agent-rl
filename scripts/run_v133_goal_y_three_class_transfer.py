"""Goal Y: audited 25-pair small-scale SFT/DPO search-transfer experiment."""
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
import run_v133_goal_s_label_granularity as s
import run_v133_goal_t_update_length as t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-y-three-class-transfer"
SOURCE = t.checkpoint_dir(42, 32) / "adapter-final"
CFG, STATE = OUT / "config.json", OUT / "state.json"
SEEDS, ARMS, UPDATES = (42, 43), ("a", "b"), 25


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(v) for v in stream if v.strip()]
def write(path, value):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pairs():
    values = read(OUT / "pairs.json")["pairs"]
    if len(values) != UPDATES or len({v["case_id"] for v in values}) != UPDATES:
        raise RuntimeError("Audited pair count changed")
    return values


def dev_cases():
    values = rows(OUT / "dev-cases.jsonl")
    if len(values) != 48 or len({v["case_id"] for v in values}) != 48:
        raise RuntimeError("Clean dev count changed")
    return values


def frozen_files():
    weights = sorted(q.p.BASE.glob("*.safetensors"))
    if not weights: raise RuntimeError("Missing actual base weights")
    return {"adapter": SOURCE / "adapter_model.safetensors",
            "tokenizer": q.p.BASE / "tokenizer.json",
            "old_corpus": q.p.CORPUS, "old_cases": q.p.CASES,
            "pairs": OUT / "pairs.json", "pair_review": OUT / "pair-main-review.jsonl",
            "dev_cases": OUT / "dev-cases.jsonl", "dev_corpus": OUT / "dev-corpus.json",
            "dev_freeze": OUT / "dev-freeze.json", "goal_q_runner": Path(q.__file__),
            "model_code": ROOT / "src/mini_agent_rl/model/local.py",
            "search_code": ROOT / "src/mini_agent_rl/tools/search.py",
            "runner": Path(__file__), **{f"base_{i}": p for i, p in enumerate(weights)}}


def software():
    import torch, transformers, peft
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "env": {k: os.environ.get(k) for k in
                    ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")}}


def prepare():
    if CFG.exists() or STATE.exists(): raise FileExistsError("Goal Y already prepared")
    if read(t.OUT / "decision.json")["selected"] != "42-32":
        raise RuntimeError("Goal T source winner changed")
    training = pairs()
    dev_cases()
    if not software()["gpu"]: raise RuntimeError("GPU unavailable")
    order = {}
    for seed in SEEDS:
        ids = [v["case_id"] for v in training]
        random.Random(seed).shuffle(ids)
        order[str(seed)] = ids
    write(CFG, {"version": "goal-y-25-pair-v1", "source": str(SOURCE),
                "files": {k: sha(v) for k, v in frozen_files().items()},
                "software": software(), "product": q.load_config()["product"],
                "seeds": list(SEEDS), "arms": list(ARMS), "order": order,
                "updates_per_run": UPDATES, "learning_rate": 5e-6,
                "adam_betas": [.9, .999], "adam_eps": 1e-8,
                "weight_decay": 0, "gradient_clip": 1.0, "dpo_beta": .1,
                "gpu_time_limit": None, "stage_timeouts": None,
                "dev_gate": {"success_gain": 3, "old_semantic_noninferiority": True},
                "confirm_gate": {"success_gain": 3}})
    write(STATE, {"status": "prepared", "stages": [], "gpu_seconds": 0.0})


def config():
    value = read(CFG)
    if {k: sha(v) for k, v in frozen_files().items()} != value["files"]:
        raise RuntimeError("Frozen source changed")
    if software() != value["software"]:
        raise RuntimeError("Software/GPU environment changed")
    return value


def eval_dev(label, adapter):
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    target = OUT / f"dev-{label}.jsonl"
    if target.exists(): raise FileExistsError(target)
    pc = config()["product"]
    client = TransformersModelClient(LocalModelConfig(
        model_id=str(q.p.BASE), adapter_path=str(adapter),
        max_input_tokens=pc["max_input_tokens"], max_new_tokens=pc["max_new_tokens"],
        enable_thinking=False, deterministic=True, observation_encoding=pc["observation_encoding"],
        recompute_old_logprobs=False))
    wrapped = q.SearchTextAliasClient(client)
    registry = ToolRegistry()
    registry.register(SearchTool(read(OUT / "dev-corpus.json"), pc["retriever"]))
    for case in dev_cases():
        begin = time.monotonic()
        result = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False, seed=42))
        observed = {doc["id"] for step in result["trace"]
                    for doc in step.get("observation", {}).get("result", [])}
        result.update(case_id=case["case_id"], category=case["attribute_category"],
                      model=label, both_support_retrieved=set(case["support_ids"]).issubset(observed),
                      wall_seconds=time.monotonic() - begin,
                      **q.score(result["answer"], case["official_answer"]))
        append(target, result)
        print(json.dumps({"dev_model": label, "case": case["case_id"], "em": result["em"]}), flush=True)
    del client, wrapped, registry
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def baseline():
    config()
    target = OUT / "old-incumbent.jsonl"
    if target.exists(): raise FileExistsError(target)
    q.eval_model("goal-y-incumbent", SOURCE, q.dev_cases(), target)
    old = rows(target)
    summary = {"total": sum(v["em"] for v in old),
               "regression": sum(v["em"] for v in old if v["split"] == "goal_G_regression"),
               "invalid": sum(v["reason"] == "invalid_action" for v in old)}
    if len(old) != 32 or summary != {"total": 20, "regression": 7, "invalid": 0}:
        raise RuntimeError(f"Old baseline changed: {summary}")
    eval_dev("incumbent", SOURCE)


def reference_scores():
    import torch
    target = OUT / "reference-scores.json"
    if target.exists(): raise FileExistsError(target)
    client, _, _ = q.make_client(SOURCE, deterministic=True)
    client._load()
    client.set_training_forward_mode()
    scores = {}
    for row in pairs():
        chosen = s.target_ids(client._processor, row["chosen"])
        rejected = s.target_ids(client._processor, row["rejected"])
        if chosen == rejected: raise RuntimeError("Equal answer target tokens")
        prompt, mask = row["input_ids"], row["input_mask"]
        good = float(client.score_action_tensor([], chosen, False, prompt, mask).float().sum())
        bad = float(client.score_action_tensor([], rejected, False, prompt, mask).float().sum())
        if not math.isfinite(good) or not math.isfinite(bad): raise RuntimeError("Nonfinite reference")
        scores[row["case_id"]] = {"chosen": good, "rejected": bad,
                                   "chosen_tokens": chosen, "rejected_tokens": rejected}
    write(target, {"adapter_sha256": sha(SOURCE / "adapter_model.safetensors"), "scores": scores})
    del client
    gc.collect(); torch.cuda.empty_cache()


def dpo_loss(torch, chosen, rejected, ref_chosen, ref_rejected, beta=.1):
    return -torch.nn.functional.logsigmoid(beta * ((chosen - rejected) - (ref_chosen - ref_rejected)))


def train(seed, arm):
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as o
    cfg = config()
    ref = read(OUT / "reference-scores.json")
    if ref["adapter_sha256"] != sha(SOURCE / "adapter_model.safetensors"):
        raise RuntimeError("DPO reference changed")
    folder = OUT / f"seed{seed}-{arm}"
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir()
    set_seed(seed)
    client, _, _ = q.make_client(SOURCE, deterministic=True)
    params = [p for _, p in client.prepare_for_training()]
    base_before = o.parameter_hashes(client._model, lora=False)
    lora_before = o.parameter_hashes(client._model, lora=True)
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], betas=tuple(cfg["adam_betas"]),
                                  eps=cfg["adam_eps"], weight_decay=0)
    data = {v["case_id"]: v for v in pairs()}
    last = None
    try:
        order = cfg["order"][str(seed)]
        if len(order) != UPDATES or set(order) != set(data):
            raise RuntimeError("Frozen order invalid")
        for step, case_id in enumerate(order, 1):
            row = data[case_id]
            prompt, mask = row["input_ids"], row["input_mask"]
            reference = ref["scores"][case_id]
            chosen, rejected = reference["chosen_tokens"], reference["rejected_tokens"]
            optimizer.zero_grad(set_to_none=True)
            client.set_training_forward_mode()
            pc = client.score_action_tensor([], chosen, True, prompt, mask).float()
            pr = None
            if arm == "a":
                loss = -pc.mean()
            else:
                pr = client.score_action_tensor([], rejected, True, prompt, mask).float()
                loss = dpo_loss(torch, pc.sum(), pr.sum(), reference["chosen"],
                                reference["rejected"], cfg["dpo_beta"])
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"]))
            if not math.isfinite(grad) or grad <= 0: raise RuntimeError("Invalid gradient")
            guarded_optimizer_step(optimizer, params, client.gpu_memory_stats(), 21.)
            append(folder / "updates.jsonl", {"step": step, "case_id": case_id,
                "loss": float(loss.detach()), "grad": grad, "input_tokens": len(prompt),
                "chosen_tokens": len(chosen), "rejected_tokens": len(rejected),
                "margin": float(pc.detach().sum() - pr.detach().sum()) if pr is not None else None,
                "gpu_memory": client.gpu_memory_stats()})
            last = (prompt, mask, chosen)
            if step % 5 == 0: print(json.dumps({"seed": seed, "arm": arm, "updates": step}), flush=True)
        if o.parameter_hashes(client._model, lora=False) != base_before:
            raise RuntimeError("Base weights changed")
        changed = sum(o.parameter_hashes(client._model, lora=True)[k] != v
                      for k, v in lora_before.items())
        if not changed: raise RuntimeError("LoRA unchanged")
        adapter = folder / "adapter-final"
        client._model.save_pretrained(adapter)
        expected = client.score_action_tensor([], last[2], False, last[0], last[1]).detach().float().cpu()
        del client, params, optimizer
        gc.collect(); torch.cuda.empty_cache()
        loaded, _, _ = q.make_client(adapter, deterministic=True)
        actual = loaded.score_action_tensor([], last[2], False, last[0], last[1]).detach().float().cpu()
        delta = float((actual - expected).abs().max())
        if delta > .001: raise RuntimeError(f"Independent reload mismatch {delta}")
        write(folder / "summary.json", {"updates": UPDATES, "changed_lora": changed,
              "base_unchanged": True, "reload_max_abs": delta,
              "adapter_sha256": sha(adapter / "adapter_model.safetensors")})
        del loaded
        gc.collect(); torch.cuda.empty_cache()
    except BaseException as exc:
        write(folder / "failure.json", {"error": str(exc),
              "completed_updates": len(rows(folder / "updates.jsonl")) if (folder / "updates.jsonl").exists() else 0})
        raise


def evaluate():
    config()
    for seed in SEEDS:
        for arm in ARMS:
            if read(OUT / f"seed{seed}-{arm}/summary.json")["updates"] != UPDATES:
                raise RuntimeError("Training incomplete")
    for seed in SEEDS:
        for arm in ARMS:
            label = f"{seed}-{arm}"
            adapter = OUT / f"seed{label}/adapter-final"
            eval_dev(label, adapter)
            q.eval_model(label, adapter, q.dev_cases(), OUT / f"old-{label}.jsonl")


def stage(name, function):
    state = read(STATE)
    if name in {item["name"] for item in state["stages"]}:
        raise RuntimeError(f"Stage already run: {name}")
    started = time.monotonic()
    try:
        function()
    except BaseException as exc:
        state = read(STATE)
        state["status"] = "stopped"
        state["failure"] = {"stage": name, "error": str(exc)}
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    state = read(STATE)
    state["stages"].append({"name": name, "seconds": time.monotonic() - started})
    state["gpu_seconds"] += state["stages"][-1]["seconds"]
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run():
    state = read(STATE)
    if state["status"] == "prepared":
        state["status"] = "running_baseline"
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        def execute_baseline():
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "scripts")))
            subprocess.run([sys.executable, str(Path(__file__)), "baseline"], cwd=ROOT, env=env, check=True)
        stage("baseline", execute_baseline)
        state = read(STATE)
        state["status"] = "awaiting_baseline_review"
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if state["status"] != "awaiting_baseline_review":
        raise RuntimeError("Run already started or wrong stage")
    review_path = OUT / "baseline-semantic-review.json"
    review = read(review_path)
    if review["source_sha256"] != sha(OUT / "dev-incumbent.jsonl") or review["cases"] != 48:
        raise RuntimeError("Incomplete or mismatched baseline semantic review")
    if review["sufficient"] > 45:
        state["status"] = "complete_evaluation_ceiling"
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    state["status"] = "running_training"
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    commands = [("reference", ["reference"])]
    commands += [(f"train-{seed}-{arm}", ["train", str(seed), arm]) for seed in SEEDS for arm in ARMS]
    commands += [("evaluate", ["evaluate"])]
    for name, args in commands:
        def execute():
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "scripts")))
            subprocess.run([sys.executable, str(Path(__file__)), *args], cwd=ROOT, env=env, check=True)
        stage(name, execute)
    state = read(STATE)
    state["status"] = "awaiting_semantic_review"
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize():
    state = read(STATE)
    result = {"status": state["status"], "stages": state["stages"], "gpu_seconds": state["gpu_seconds"]}
    for split, count in (("dev", 48), ("old", 32)):
        for label in ("incumbent", "42-a", "42-b", "43-a", "43-b"):
            path = OUT / f"{split}-{label}.jsonl"
            if not path.exists(): continue
            values = rows(path)
            if len(values) != count: raise RuntimeError(f"Incomplete {path.name}")
            gold = {v["case_id"]: v["official_answer"] for v in
                    (dev_cases() if split == "dev" else q.dev_cases())}
            if {v["case_id"] for v in values} != set(gold): raise RuntimeError("Evaluation case mismatch")
            for value in values:
                score = q.score(value["answer"], gold[value["case_id"]])
                if value["em"] != score["em"] or abs(value["f1"] - score["f1"]) > 1e-9:
                    raise RuntimeError("Recomputed score mismatch")
                if value["model_calls"] > 4 or value["searches"] > 3:
                    raise RuntimeError("Search call budget exceeded")
            result[f"{split}-{label}"] = {"em": sum(v["em"] for v in values),
                "f1": statistics.mean(v["f1"] for v in values),
                "invalid": sum(v["reason"] == "invalid_action" for v in values),
                "support": sum(v.get("both_support_retrieved", False) for v in values),
                "searches": sum(v["searches"] for v in values),
                "mean_seconds": statistics.mean(v["elapsed_seconds"] for v in values)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "summarize", "baseline", "reference", "train", "evaluate"))
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument("arm", nargs="?", choices=ARMS)
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "run": run()
    elif args.command == "summarize": print(json.dumps(summarize(), ensure_ascii=False, sort_keys=True))
    elif args.command == "baseline": baseline()
    elif args.command == "reference": reference_scores()
    elif args.command == "train":
        if args.seed not in SEEDS or args.arm not in ARMS: raise ValueError("Invalid train seed or arm")
        train(args.seed, args.arm)
    else: evaluate()
