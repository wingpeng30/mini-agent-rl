"""Goal U/V: frozen answer preference training and real-search evaluation.

The run stops after the old development evaluations. A separately frozen,
blind evidence review supplies the semantic gate before fresh evaluation.
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

import run_v133_goal_q_search as q
import run_v133_goal_r_evidence_order_v2 as r
import run_v133_goal_s_label_granularity as s
import run_v133_goal_t_update_length as t

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("GOAL_UV_OUT", ROOT / "runs/v133-goal-u-v-evidence-preference")).resolve()
SOURCE = t.checkpoint_dir(42, 32) / "adapter-final"
CFG, STATE = OUT / "config.json", OUT / "state.json"
PAIRS = OUT / "pairs.json"
FRESH_CORPUS = OUT / "fresh-corpus.json"
FRESH_CASES = OUT / "fresh-cases.jsonl"
RUBRIC = OUT / "review-rubric.json"
SEEDS = (42, 43)
ARMS = ("a", "b")
UPDATES = 32


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def append(path, obj):
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(1 << 20), b""): h.update(part)
    return h.hexdigest()


def source_files():
    weights = sorted(q.p.BASE.glob("*.safetensors"))
    if not weights: raise RuntimeError("Base weights missing")
    return {"adapter": SOURCE / "adapter_model.safetensors",
            "tokenizer": q.p.BASE / "tokenizer.json", "corpus": q.p.CORPUS,
            "old_cases": q.p.CASES, "training_inputs": r.OUT / "train-inputs.jsonl",
            "labels": s.LABELS, "pairs": PAIRS,
            "pair_audit": OUT / "pair-audit.json",
            "pair_builder": ROOT / "scripts/prepare_v133_goal_u_pairs.py",
            "fresh_corpus": FRESH_CORPUS, "fresh_cases": FRESH_CASES,
            "fresh_sources": OUT / "fresh-sources.jsonl", "review_rubric": RUBRIC,
            "fresh_freeze": OUT / "fresh-set-freeze.json",
            "fresh_validator": ROOT / "scripts/validate_v133_goal_v_fresh_set.py",
            "q_runner": Path(q.__file__), "r_runner": Path(r.__file__),
            "s_runner": Path(s.__file__),
            "model_code": ROOT / "src/mini_agent_rl/model/local.py",
            "search_code": ROOT / "src/mini_agent_rl/tools/search.py",
            "runner": Path(__file__),
            **{f"base_{i}": file for i, file in enumerate(weights)}}


def software():
    import torch, transformers, peft
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")}}


def pairs():
    value = read(PAIRS)
    values = value["pairs"]
    if not 8 <= len(values) <= 16: raise RuntimeError("Preference pair count outside 8..16")
    ids = {row["case_id"] for row in q.train_cases()}
    if len({v["case_id"] for v in values}) != len(values) or not {v["case_id"] for v in values}.issubset(ids):
        raise RuntimeError("Invalid training case ids")
    originals = {v["case_id"]: v for v in rows(r.OUT / "train-inputs.jsonl")}
    labels = {v["case_id"]: v["revised_answer"] for v in read(s.LABELS)["cases"]}
    for item in values:
        original = originals[item["case_id"]]
        if item["chosen"] != labels[item["case_id"]] or not item["rejected"] or item["chosen"] == item["rejected"]:
            raise RuntimeError("Pair answer/label mismatch")
        if item["input_ids"] != original["input_ids"] or item["input_mask"] != original["input_mask"]:
            raise RuntimeError("Pair input differs from saved real context")
        if len(item["input_ids"]) > 2048: raise RuntimeError("Training context over limit")
    return values


def fresh_cases():
    cases = rows(FRESH_CASES)
    documents = read(FRESH_CORPUS)
    if len(cases) != 32 or len(documents) != 96: raise RuntimeError("Fresh set size mismatch")
    if len({v["case_id"] for v in cases}) != 32 or len({v["id"] for v in documents}) != 96:
        raise RuntimeError("Fresh IDs not unique")
    doc_ids = {v["id"] for v in documents}
    support_ids = [doc_id for value in cases for doc_id in value["support_ids"]]
    if len(set(support_ids)) != 64 or len(doc_ids - set(support_ids)) != 32:
        raise RuntimeError("Fresh support/distractor allocation mismatch")
    old_titles = {" ".join(str(v["title"]).casefold().split()) for v in read(q.p.CORPUS)}
    fresh_titles = {" ".join(str(v["title"]).casefold().split()) for v in documents}
    if fresh_titles & old_titles:
        raise RuntimeError("Fresh documents reuse old corpus titles")
    if any(len(v["support_ids"]) != 2 or not set(v["support_ids"]).issubset(doc_ids)
           or not v["question"] or not v["official_answer"] for v in cases):
        raise RuntimeError("Fresh case malformed")
    if {v["stratum"] for v in cases} != {"one_sided", "ordinary"}:
        raise RuntimeError("Fresh strata missing")
    if sum(v["stratum"] == "one_sided" for v in cases) != 16:
        raise RuntimeError("Fresh strata not balanced")
    for category in ("occupation", "nationality_geography", "entity_type", "other"):
        group = [v for v in cases if v.get("attribute_category") == category]
        if len(group) != 8 or sum(v["stratum"] == "one_sided" for v in group) != 4:
            raise RuntimeError(f"Fresh category quota mismatch: {category}")
    return cases


def prepare():
    if CFG.exists() or STATE.exists(): raise FileExistsError("Goal U/V run already prepared")
    if read(t.OUT / "decision.json")["selected"] != "42-32": raise RuntimeError("Goal T winner changed")
    if read(t.STATE)["status"] != "complete": raise RuntimeError("Goal T incomplete")
    training = pairs()
    fresh_cases()
    if RUBRIC.exists(): raise FileExistsError("Review rubric already exists")
    write(RUBRIC, {"version": "goal-uv-rubric-v1",
         "categories": ["sufficient", "missing_qualifier", "unsupported_or_wrong", "undetermined"],
         "rule": "Judge the asked common attribute using both actually visible source passages. "
                 "Mark sufficient only with both-sided support and all required information. "
                 "Equivalent phrasing is allowed only in semantic review. "
                 "A one-sided property is unsupported, not necessarily false in the world. "
                 "Invalid actions, no answer, and exhausted budgets are failures. "
                 "Do not infer sufficient by default; cite actual passages for each judgment.",
         "reviewer": "blind automatic agent; explanatory evidence review, not independent human gold"})
    hashes = {key: sha(path) for key, path in source_files().items()}
    if not software()["gpu"]: raise RuntimeError("GPU unavailable")
    order = {}
    for seed in SEEDS:
        pool = [x["case_id"] for x in training]
        rng = random.Random(seed)
        result = []
        while len(result) < UPDATES:
            cycle = pool[:]
            rng.shuffle(cycle)
            result.extend(cycle)
        order[str(seed)] = result[:UPDATES]
    write(CFG, {"version": "goal-uv-v1", "files": hashes, "software": software(),
                "source_adapter": str(SOURCE), "product": q.load_config()["product"],
                "seeds": list(SEEDS), "arms": list(ARMS), "order": order,
                "updates_per_arm": UPDATES, "lr": 5e-6, "betas": [.9, .999],
                "eps": 1e-8, "weight_decay": 0, "clip_grad": 1., "dpo_beta": .1,
                "gpu_time_limit": None, "stage_timeouts": None,
                "old_gate": {"total_gt": 20, "regression_gte": 7, "invalid": 0}})
    write(STATE, {"status": "prepared", "stages": [], "gpu_seconds": 0.})


def config():
    c = read(CFG)
    if {key: sha(path) for key, path in source_files().items()} != c["files"]:
        raise RuntimeError("Frozen source changed")
    if software() != c["software"]: raise RuntimeError("Software/GPU environment changed")
    return c


def count_stats(label, fresh=False):
    path = OUT / (f"fresh-{label}.jsonl" if fresh else f"eval-{label}.jsonl")
    if not path.exists(): return None
    data = rows(path)
    if len(data) != 32 or len({v["case_id"] for v in data}) != 32:
        raise RuntimeError(f"Incomplete evaluation {label}")
    truth = {v["case_id"]: v for v in (fresh_cases() if fresh else q.dev_cases())}
    if {v["case_id"] for v in data} != set(truth):
        raise RuntimeError(f"Evaluation case IDs changed {label}")
    for item in data:
        case = truth[item["case_id"]]
        recomputed = q.score(item["answer"], case["official_answer"])
        if item["em"] != recomputed["em"] or abs(item["f1"] - recomputed["f1"]) > 1e-9:
            raise RuntimeError(f"Score mismatch {label}/{item['case_id']}")
        if item["model_calls"] > 4 or item["searches"] > 3 or len(item["trace"]) != item["model_calls"]:
            raise RuntimeError(f"Search budget exceeded {label}/{item['case_id']}")
        if any(len(step.get("action_token_ids") or []) > 128 or
               len(step.get("prompt_token_ids") or []) > 2048 for step in item["trace"]):
            raise RuntimeError(f"Token budget exceeded {label}/{item['case_id']}")
        if fresh and item["stratum"] != case["stratum"]:
            raise RuntimeError("Fresh stratum mismatch")
    result = {"model": label, "total": sum(v["em"] for v in data),
              "f1": statistics.mean(v["f1"] for v in data),
              "invalid": sum(v["reason"] == "invalid_action" for v in data),
              "support": sum(v["both_support_retrieved"] for v in data),
              "searches": sum(v["searches"] for v in data),
              "mean_latency": statistics.mean(v["elapsed_seconds"] for v in data)}
    if fresh:
        for stratum in ("one_sided", "ordinary"):
            result[stratum] = sum(v["em"] for v in data if v["stratum"] == stratum)
    else:
        result["main"] = sum(v["em"] for v in data if v["split"] == "new_eval")
        result["regression"] = sum(v["em"] for v in data if v["split"] == "goal_G_regression")
    return result


def baseline():
    q.eval_model("incumbent", SOURCE, q.dev_cases(), OUT / "eval-incumbent.jsonl")
    result = count_stats("incumbent")
    if (result["total"], result["regression"], result["invalid"]) != (20, 7, 0):
        raise RuntimeError(f"Goal T product baseline changed: {result}")


def dpo_loss(torch, policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=.1):
    return -torch.nn.functional.logsigmoid(beta * ((policy_chosen - policy_rejected)
                                                - (ref_chosen - ref_rejected)))


def reference_scores(items):
    import torch
    client, _, _ = q.make_client(SOURCE, deterministic=True)
    client._load()
    client.set_training_forward_mode()
    cached = {}
    for item in items:
        prompt, mask = item["input_ids"], item["input_mask"]
        chosen = s.target_ids(client._processor, item["chosen"])
        rejected = s.target_ids(client._processor, item["rejected"])
        if chosen == rejected: raise RuntimeError("Equal target token ids")
        score_chosen = float(client.score_action_tensor([], chosen, False, prompt, mask).float().sum())
        score_rejected = float(client.score_action_tensor([], rejected, False, prompt, mask).float().sum())
        if not math.isfinite(score_chosen) or not math.isfinite(score_rejected):
            raise RuntimeError("Nonfinite reference logprob")
        cached[item["case_id"]] = {"chosen": score_chosen, "rejected": score_rejected,
                                    "chosen_tokens": chosen, "rejected_tokens": rejected}
    write(OUT / "reference-scores.json", {"adapter_sha256": sha(SOURCE / "adapter_model.safetensors"),
                                           "scores": cached})
    del client
    gc.collect(); torch.cuda.empty_cache()


def train(seed, arm):
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as o

    c = config()
    data = {v["case_id"]: v for v in pairs()}
    ref = read(OUT / "reference-scores.json")
    if ref["adapter_sha256"] != sha(SOURCE / "adapter_model.safetensors"):
        raise RuntimeError("Reference is not frozen Goal T")
    folder = OUT / f"seed{seed}-{arm}"
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir(parents=True)
    set_seed(seed)
    client, _, _ = q.make_client(SOURCE, deterministic=True)
    params = [v for _, v in client.prepare_for_training()]
    base_before = o.parameter_hashes(client._model, lora=False)
    lora_before = o.parameter_hashes(client._model, lora=True)
    optimizer = torch.optim.AdamW(params, lr=c["lr"], betas=tuple(c["betas"]), eps=c["eps"], weight_decay=0)
    last = None
    try:
        order = c["order"][str(seed)]
        if len(order) != UPDATES or set(order) != set(data):
            raise RuntimeError("Frozen training schedule malformed")
        for step, case_id in enumerate(order, 1):
            if step > UPDATES: raise RuntimeError("Update cap exceeded")
            row = data[case_id]
            prompt, mask = row["input_ids"], row["input_mask"]
            reference = ref["scores"][case_id]
            chosen, rejected = reference["chosen_tokens"], reference["rejected_tokens"]
            optimizer.zero_grad(set_to_none=True)
            client.set_training_forward_mode()
            p_chosen = client.score_action_tensor([], chosen, True, prompt, mask).float()
            if arm == "a":
                loss = -p_chosen.mean()
                p_rejected = None
            else:
                p_rejected = client.score_action_tensor([], rejected, True, prompt, mask).float()
                loss = dpo_loss(torch, p_chosen.sum(), p_rejected.sum(),
                                reference["chosen"], reference["rejected"], c["dpo_beta"])
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(params, c["clip_grad"]))
            if not math.isfinite(grad) or grad <= 0: raise RuntimeError("Invalid gradient")
            guarded_optimizer_step(optimizer, params, client.gpu_memory_stats(), 21.)
            append(folder / "updates.jsonl", {"step": step, "case_id": case_id,
                   "loss": float(loss.detach()), "grad": grad, "chosen_tokens": len(chosen),
                   "rejected_tokens": len(rejected), "input_tokens": len(prompt),
                   "margin": float(p_chosen.detach().sum() - p_rejected.detach().sum()) if p_rejected is not None else None,
                   "gpu_memory": client.gpu_memory_stats()})
            last = (prompt, mask, chosen)
            if step % 8 == 0: print(json.dumps({"seed": seed, "arm": arm, "updates": step}), flush=True)
        if step != UPDATES: raise RuntimeError("Training ended before 32 updates")
        if o.parameter_hashes(client._model, lora=False) != base_before:
            raise RuntimeError("Base weights changed")
        changed = sum(o.parameter_hashes(client._model, lora=True)[key] != value for key, value in lora_before.items())
        if not changed: raise RuntimeError("LoRA unchanged")
        target = folder / "adapter-final"
        client._model.save_pretrained(target)
        expected = client.score_action_tensor([], last[2], False, last[0], last[1]).detach().float().cpu()
        del client, params, optimizer
        gc.collect(); torch.cuda.empty_cache()
        loaded, _, _ = q.make_client(target, deterministic=True)
        actual = loaded.score_action_tensor([], last[2], False, last[0], last[1]).detach().float().cpu()
        delta = float((actual - expected).abs().max())
        if delta > .001: raise RuntimeError(f"Independent reload mismatch {delta}")
        write(folder / "summary.json", {"status": "passed", "updates": UPDATES,
              "changed_lora": changed, "base_unchanged": True, "reload_max_abs": delta,
              "checkpoint_sha256": sha(target / "adapter_model.safetensors")})
        del loaded
        gc.collect(); torch.cuda.empty_cache()
    except BaseException:
        write(folder / "failure.json", {"status": "stopped", "completed_updates": len(rows(folder / "updates.jsonl"))
                                             if (folder / "updates.jsonl").exists() else 0})
        raise


def old_evaluations():
    for seed in SEEDS:
        for arm in ARMS:
            label = f"{seed}-{arm}"
            folder = OUT / f"seed{label}"
            if read(folder / "summary.json")["status"] != "passed": raise RuntimeError("Training incomplete")
    for seed in SEEDS:
        for arm in ARMS:
            label = f"{seed}-{arm}"
            q.eval_model(label, OUT / f"seed{label}/adapter-final", q.dev_cases(), OUT / f"eval-{label}.jsonl")


def fixed_diagnostics():
    """Replay the exact incumbent answer context for all five models."""
    from evaluate_v133_goal_c_sft import normalize_answer
    baseline_rows = rows(OUT / "eval-incumbent.jsonl")
    gold = {item["case_id"]: item["official_answer"] for item in q.dev_cases()}
    if len(baseline_rows) != 32: raise RuntimeError("Baseline trace incomplete")
    candidates = [("incumbent", SOURCE)] + [
        (f"{seed}-{arm}", OUT / f"seed{seed}-{arm}/adapter-final")
        for seed in SEEDS for arm in ARMS]
    for label, adapter in candidates:
        target = OUT / f"fixed-{label}.jsonl"
        if target.exists(): raise FileExistsError(target)
        client, wrapped, registry = q.make_client(adapter, deterministic=True)
        client._load()
        for item in baseline_rows:
            final = item["trace"][-1]
            if final.get("action") != "answer": raise RuntimeError("Baseline did not answer")
            result = s.fixed_answer(client, wrapped, registry, item["question"],
                                    item["trace"], final["prompt_token_ids"])
            answer = result["answer"]
            append(target, {"model": label, "case_id": item["case_id"], "answer": answer,
                            "raw": result["raw"], "invalid": result["invalid"],
                            "em": int(bool(answer) and normalize_answer(answer) ==
                                      normalize_answer(gold[item["case_id"]]))})
        del client, wrapped, registry
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def eval_new(label, adapter):
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    pc = config()["product"]
    client = TransformersModelClient(LocalModelConfig(model_id=str(q.p.BASE), adapter_path=str(adapter),
        max_input_tokens=pc["max_input_tokens"], max_new_tokens=pc["max_new_tokens"],
        enable_thinking=False, deterministic=True, observation_encoding=pc["observation_encoding"],
        recompute_old_logprobs=False))
    wrapped = q.SearchTextAliasClient(client)
    registry = ToolRegistry()
    registry.register(SearchTool(read(FRESH_CORPUS), pc["retriever"]))
    path = OUT / f"fresh-{label}.jsonl"
    if path.exists(): raise FileExistsError(path)
    for case in fresh_cases():
        value = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False, seed=42))
        observed = {doc["id"] for action in value["trace"]
                    for doc in action.get("observation", {}).get("result", [])}
        value.update(case_id=case["case_id"], stratum=case["stratum"], model=label,
                     both_support_retrieved=set(case["support_ids"]).issubset(observed),
                     **q.score(value["answer"], case["official_answer"]))
        append(path, value)
        print(json.dumps({"fresh_model": label, "case": case["case_id"], "em": value["em"]}), flush=True)
    del client, wrapped, registry
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def old_gate(review):
    baseline = count_stats("incumbent")
    base_sem = review["models"]["incumbent"]
    options = []
    for arm in ARMS:
        stats = [count_stats(f"{seed}-{arm}") for seed in SEEDS]
        sem = [review["models"][f"{seed}-{arm}"] for seed in SEEDS]
        passed = all(v["total"] > 20 and v["regression"] >= 7 and v["invalid"] == 0
                     and z["sufficient"] > base_sem["sufficient"]
                     and z["unsupported_or_wrong"] < base_sem["unsupported_or_wrong"]
                     and z["one_sided_error"] < base_sem["one_sided_error"]
                     and z["missing_qualifier"] <= base_sem["missing_qualifier"]
                     for v, z in zip(stats, sem))
        if passed:
            options.append((min(v["total"] for v in stats),
                            statistics.mean(v["total"] for v in stats),
                            statistics.mean(z["sufficient"] for z in sem),
                            1 if arm == "a" else 0, arm))
    return max(options)[-1] if options else None


def advance():
    c = config()
    state = read(STATE)
    if state["status"] != "awaiting_old_review": raise RuntimeError("Wrong stage for advance")
    path = OUT / "old-semantic-review.json"
    review = read(path)
    if review["source_hashes"] != {label: sha(OUT / f"eval-{label}.jsonl")
                                   for label in ("incumbent", "42-a", "42-b", "43-a", "43-b")}:
        raise RuntimeError("Old semantic review not tied to exact outputs")
    if any(sum(review["models"][label][k] for k in
               ("sufficient", "missing_qualifier", "unsupported_or_wrong", "undetermined")) != 32
           for label in ("incumbent", "42-a", "42-b", "43-a", "43-b")):
        raise RuntimeError("Incomplete semantic review")
    baseline_review = read(OUT / "baseline-semantic-review.json")
    if baseline_review["source_hash"] != sha(OUT / "eval-incumbent.jsonl") or \
       review["models"]["incumbent"] != baseline_review["counts"]:
        raise RuntimeError("Baseline semantic rubric changed after candidate evaluation")
    selected = old_gate(review)
    write(OUT / "old-gate.json", {"selected_arm": selected,
                                  "reason": "old_development_gate_passed" if selected else "old_development_gate_failed"})
    if selected is None:
        state["status"] = "complete"
        write(OUT / "decision.json", {"selected": "incumbent", "adapter": str(SOURCE),
                                        "reason": "old_development_gate_failed"})
        write(STATE, state)
        return
    state["status"] = "running_fresh"
    write(STATE, state)
    try:
        run_stage("fresh", lambda: (eval_new("incumbent", SOURCE),
                                    *(eval_new(f"{seed}-{selected}", OUT / f"seed{seed}-{selected}/adapter-final")
                                      for seed in SEEDS)))
        state = read(STATE)
        state["status"] = "awaiting_fresh_review"
    except BaseException as error:
        state = read(STATE)
        state.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally: write(STATE, state)


def finish():
    config()
    state = read(STATE)
    if state["status"] != "awaiting_fresh_review": raise RuntimeError("Wrong stage for finish")
    arm = read(OUT / "old-gate.json")["selected_arm"]
    labels = ("incumbent", f"42-{arm}", f"43-{arm}")
    review = read(OUT / "fresh-semantic-review.json")
    if review["source_hashes"] != {label: sha(OUT / f"fresh-{label}.jsonl") for label in labels}:
        raise RuntimeError("Fresh review does not match outputs")
    if any(sum(review["models"][label][k] for k in
               ("sufficient", "missing_qualifier", "unsupported_or_wrong", "undetermined")) != 32
           for label in labels): raise RuntimeError("Incomplete fresh semantic review")
    base = count_stats("incumbent", True)
    sem_base = review["models"]["incumbent"]
    qualified = fresh_gate(arm, review, base, sem_base)
    selected = f"42-{arm}" if qualified else "incumbent"
    adapter = OUT / f"seed42-{arm}/adapter-final" if qualified else SOURCE
    write(OUT / "decision.json", {"selected": selected, "adapter": str(adapter),
        "reason": "old_and_fresh_gates_passed" if qualified else "fresh_gate_failed",
        "old": {label: count_stats(label) for label in ("incumbent", "42-a", "42-b", "43-a", "43-b")},
        "fresh": {label: count_stats(label, True) for label in labels}})
    state["status"] = "complete"
    write(STATE, state)


def fresh_gate(arm, review, base, sem_base):
    qualified = True
    for seed in SEEDS:
        label = f"{seed}-{arm}"
        result, sem = count_stats(label, True), review["models"][label]
        qualified &= (result["total"] > base["total"] and result["invalid"] == 0
                      and sem["sufficient"] > sem_base["sufficient"]
                      and sem["unsupported_or_wrong"] <= sem_base["unsupported_or_wrong"]
                      and sem["one_sided_error"] <= sem_base["one_sided_error"]
                      and sem["missing_qualifier"] <= sem_base["missing_qualifier"])
        for stratum in ("one_sided", "ordinary"):
            qualified &= sem["strata"][stratum]["sufficient"] >= sem_base["strata"][stratum]["sufficient"]
    return bool(qualified)


def run_stage(name, fn):
    state = read(STATE)
    start = time.monotonic()
    state["current_stage"] = name
    write(STATE, state)
    try: fn()
    finally:
        state = read(STATE)
        seconds = time.monotonic() - start
        state["gpu_seconds"] += seconds
        state["stages"].append({"name": name, "seconds": seconds})
        write(STATE, state)


def run():
    config()
    state = read(STATE)
    if state["status"] != "prepared": raise RuntimeError("Run already started")
    state["status"] = "running"
    write(STATE, state)
    try:
        run_stage("baseline", baseline)
        state = read(STATE)
        state["status"] = "awaiting_baseline_review"
    except BaseException as error:
        state = read(STATE)
        state.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(STATE, state)
        summarize()


def continue_training():
    config()
    state = read(STATE)
    if state["status"] != "awaiting_baseline_review":
        raise RuntimeError("Baseline review required before training")
    review = read(OUT / "baseline-semantic-review.json")
    if review["source_hash"] != sha(OUT / "eval-incumbent.jsonl") or \
       sum(review["counts"][key] for key in
           ("sufficient", "missing_qualifier", "unsupported_or_wrong", "undetermined")) != 32:
        raise RuntimeError("Baseline review incomplete or mismatched")
    state["status"] = "running"
    write(STATE, state)
    try:
        run_stage("reference", lambda: reference_scores(pairs()))
        for seed in SEEDS:
            for arm in ARMS:
                run_stage(f"train-{seed}-{arm}", lambda seed=seed, arm=arm: train(seed, arm))
        run_stage("old_evaluation", old_evaluations)
        run_stage("fixed_diagnostics", fixed_diagnostics)
        state = read(STATE)
        state["status"] = "awaiting_old_review"
    except BaseException as error:
        state = read(STATE)
        state.update(status="stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(STATE, state)
        summarize()


def summarize():
    state = read(STATE)
    def safe(label, fresh=False):
        try: return count_stats(label, fresh)
        except RuntimeError as error: return {"model": label, "status": "partial", "error": str(error)}
    output = {"status": state["status"], "gpu_seconds": state["gpu_seconds"],
              "stages": state["stages"],
              "old": {label: safe(label) for label in ("incumbent", "42-a", "42-b", "43-a", "43-b")},
              "decision": read(OUT / "decision.json") if (OUT / "decision.json").exists() else None}
    if (OUT / "old-gate.json").exists():
        arm = read(OUT / "old-gate.json")["selected_arm"]
        if arm:
            output["fresh"] = {label: safe(label, True)
                               for label in ("incumbent", f"42-{arm}", f"43-{arm}")}
    write(OUT / "summary.json", output)
    print(json.dumps(output, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "continue", "advance", "finish", "summarize"))
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "run": run()
    elif args.command == "continue": continue_training()
    elif args.command == "advance": advance()
    elif args.command == "finish": finish()
    else: summarize()


if __name__ == "__main__": main()
