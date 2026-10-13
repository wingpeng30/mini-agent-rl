"""Goal S: matched answer-label SFT and live-search evaluation."""
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

import run_v133_goal_q_search as q
import run_v133_goal_r_evidence_order_v2 as r
from prepare_v133_goal_s_labels import build as build_labels

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("GOAL_S_OUT", str(ROOT / "runs/v133-goal-s-label-granularity"))).resolve()
CFG = OUT / "config.json"
STATE = OUT / "state.json"
LABELS = OUT / "labels.json"
SOURCE = q.OLD_RL
CAPS = {"baseline": 600, "train42_a": 1200, "train42_b": 1200,
        "eval42": 900, "train43_a": 1200, "train43_b": 1200,
        "eval43": 600, "final": 300}
LIMIT = 7200


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def asset_paths():
    base_weights = sorted(q.p.BASE.glob("*.safetensors"))
    if not base_weights:
        raise RuntimeError("Base model weight files missing")
    return {"adapter": SOURCE / "adapter_model.safetensors", "tokenizer": q.p.BASE / "tokenizer.json",
            "corpus": q.p.CORPUS, "cases": q.p.CASES,
            "q_runner": Path(q.__file__), "r_runner": Path(r.__file__),
            "runner": Path(__file__), "labels": LABELS,
            "training_inputs": r.OUT / "train-inputs.jsonl",
            **{f"base_weight_{index}": path for index, path in enumerate(base_weights)}}


def prepare():
    if CFG.exists() or STATE.exists(): raise FileExistsError("Goal S run already prepared")
    if read(q.OUT / "decision.json")["selected"] != "old_rl":
        raise RuntimeError("Goal Q product choice changed")
    if len(q.train_cases()) != 16 or len(q.dev_cases()) != 32:
        raise RuntimeError("Frozen split counts changed")
    planned = build_labels(r.OUT / "train-inputs.jsonl")
    if planned["changes"] == 0:
        write(OUT / "decision.json", {"selected": "incumbent", "reason": "no_defensible_label_changes"})
        return
    if not LABELS.exists(): raise FileNotFoundError(LABELS)
    existing = read(LABELS)
    if existing != planned: raise RuntimeError("Label audit differs from frozen training-only evidence")
    asset_hashes = {key: sha(path) for key, path in asset_paths().items()}
    original_order = read(r.OUT / "config.json")["order"]
    audit_rule = {
        "categories": ["sufficient", "missing_qualifier", "unsupported_or_wrong", "undetermined"],
        "instructions": "Assess the question and both actually visible evidence passages. Equivalent phrasing may be sufficient. Mark a qualifier missing only when needed to answer the asked attribute. New qualifiers need both-sided evidence. Keep original strict EM unchanged. Blind model identities and deduplicate identical question-evidence-answer triples.",
        "reviewer": "automatic agent review; explanatory, not independent human gold",
    }
    write(OUT / "semantic-rubric.json", audit_rule)
    write(CFG, {"version": "goal-s-v1", "hashes": asset_hashes, "order": original_order,
                "seeds": [42, 43], "passes": 2, "copies_per_case_pass": 2,
                "updates_per_arm": 64, "learning_rate": 5e-6,
                "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0,
                "grad_clip": 1., "baseline": {"total": 19, "regression": 7, "invalid": 0},
                "caps": CAPS, "gpu_total_cap": LIMIT,
                "semantic_rubric_sha256": sha(OUT / "semantic-rubric.json")})
    write(STATE, {"status": "prepared", "gpu_seconds": 0., "stages": []})


def config():
    c = read(CFG)
    if {key: sha(path) for key, path in asset_paths().items()} != c["hashes"]:
        raise RuntimeError("A frozen asset changed")
    if sha(OUT / "semantic-rubric.json") != c["semantic_rubric_sha256"]:
        raise RuntimeError("Semantic rubric changed")
    return c


def answers():
    audited = read(LABELS)["cases"]
    if len(audited) != 16:
        raise RuntimeError("Label audit incomplete")
    return {row["case_id"]: row for row in audited}


def target_ids(processor, answer):
    token_ids, raw = r.answer_target(processor, answer)
    if not token_ids or raw != json.dumps({"action": "answer", "answer": answer}, ensure_ascii=False):
        raise RuntimeError("Answer target encoding mismatch")
    return token_ids


def train(seed, arm):
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as o

    c = config()
    set_seed(seed)
    data = {row["case_id"]: row for row in rows(r.OUT / "train-inputs.jsonl")}
    labels = answers()
    if set(data) != set(labels) or any(not row["eligible"] for row in data.values()):
        raise RuntimeError("Training inputs do not match 16 eligible labels")
    folder = OUT / f"seed{seed}-{arm}"
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir(parents=True)
    client, _, _ = q.make_client(SOURCE, deterministic=True)
    named = client.prepare_for_training()
    params = [parameter for _, parameter in named]
    base_before = o.parameter_hashes(client._model, lora=False)
    lora_before = o.parameter_hashes(client._model, lora=True)
    optimizer = torch.optim.AdamW(params, lr=5e-6, betas=(.9, .999), eps=1e-8, weight_decay=0)
    updates = 0
    last = None
    for epoch, order in enumerate(c["order"][str(seed)], 1):
        if len(order) != 16 or set(order) != set(data): raise RuntimeError("Training permutation invalid")
        for case_id in order:
            row = data[case_id]
            prompt, mask = row["input_ids"], row["input_mask"]
            if len(prompt) != len(mask) or len(prompt) > 2048:
                raise RuntimeError("Frozen prompt invalid")
            answer = labels[case_id]["original_answer" if arm == "a" else "revised_answer"]
            target = target_ids(client._processor, answer)
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
                last = (prompt, mask, target)
                append(folder / "updates.jsonl", {"case_id": case_id, "pass": epoch, "copy": copy,
                    "loss": float(loss.detach()), "grad": grad, "input_tokens": len(prompt),
                    "target_tokens": len(target), "gpu_memory": client.gpu_memory_stats()})
                if updates > 64: raise RuntimeError("Update cap exceeded")
    if updates != 64: raise RuntimeError("Missing training updates")
    if o.parameter_hashes(client._model, lora=False) != base_before: raise RuntimeError("Base changed")
    changed = sum(o.parameter_hashes(client._model, lora=True)[key] != value for key, value in lora_before.items())
    if not changed: raise RuntimeError("LoRA unchanged")
    checkpoint = folder / "adapter-final"
    client._model.save_pretrained(checkpoint)
    before = client.score_action_tensor([], last[2], False, last[0], last[1]).detach().float().cpu()
    del client, optimizer, params
    gc.collect(); torch.cuda.empty_cache()
    loaded, _, _ = q.make_client(checkpoint, deterministic=True)
    after = loaded.score_action_tensor([], last[2], False, last[0], last[1]).detach().float().cpu()
    delta = float((before - after).abs().max())
    if delta > .001: raise RuntimeError(f"Reload mismatch: {delta}")
    write(folder / "summary.json", {"status": "passed", "updates": updates, "changed_lora": changed,
          "base_unchanged": True, "reload_max_abs": delta,
          "checkpoint_sha256": sha(checkpoint / "adapter_model.safetensors")})


def stats(label):
    path = OUT / f"eval-{label}.jsonl"
    if not path.exists(): return None
    data = rows(path)
    if len(data) != 32 or len({x["case_id"] for x in data}) != 32:
        raise RuntimeError(f"Incomplete product evaluation {label}")
    return {"model": label, "total": sum(x["em"] for x in data),
            "main": sum(x["em"] for x in data if x["split"] == "new_eval"),
            "regression": sum(x["em"] for x in data if x["split"] == "goal_G_regression"),
            "invalid": sum(x["reason"] == "invalid_action" for x in data),
            "support": sum(x["both_support_retrieved"] for x in data),
            "searches": sum(x["searches"] for x in data),
            "mean_f1": statistics.mean(x["f1"] for x in data),
            "mean_latency": statistics.mean(x["elapsed_seconds"] for x in data)}


def restored_trace(trace):
    rebuilt = json.loads(json.dumps(trace, ensure_ascii=False))
    for step in rebuilt:
        if "observation" not in step: continue
        restored = []
        for doc in step["observation"].get("result", []):
            if set(doc) != {"id", "title", "text", "score"}:
                raise RuntimeError("Unexpected persisted document fields")
            restored.append({key: doc[key] for key in ("id", "title", "text", "score")})
        step["observation"]["result"] = restored
    return rebuilt


def fixed_answer(client, wrapped, registry, question, trace, expected_ids):
    messages = r.rebuild(question, restored_trace(trace), reverse=False)
    ids, mask = r.prompt_ids(client, messages)
    if ids != expected_ids or len(ids) != len(mask):
        raise RuntimeError("Persisted context differs from product input")
    response = asyncio.run(wrapped.generate(messages, registry.definitions(), temperature=0., seed=42))
    return {"answer": response.content if not response.parse_error and not response.tool_call else None,
            "raw": response.raw_text, "input_tokens": len(ids),
            "invalid": bool(response.parse_error or response.tool_call)}


def diagnostics(seed, arms):
    from evaluate_v133_goal_c_sft import normalize_answer
    baseline = rows(OUT / "eval-incumbent.jsonl")
    if len(baseline) != 32: raise RuntimeError("Baseline contexts incomplete")
    train_data = rows(r.OUT / "train-inputs.jsonl")
    label_map = answers()
    gold = {row["case_id"]: row for row in q.dev_cases()}
    for arm in arms:
        adapter = SOURCE if arm == "incumbent" else OUT / f"seed{seed}-{arm}/adapter-final"
        client, wrapped, registry = q.make_client(adapter, deterministic=True)
        fixed_path = OUT / f"fixed-{seed}-{arm}.jsonl"
        fit_path = OUT / f"fit-{seed}-{arm}.jsonl"
        if fixed_path.exists() or fit_path.exists(): raise FileExistsError("Diagnostic output exists")
        client._load()
        for row in baseline:
            final = row["trace"][-1]
            if final.get("action") != "answer": raise RuntimeError("Baseline has no final answer")
            result = fixed_answer(client, wrapped, registry, row["question"], row["trace"], final["prompt_token_ids"])
            actual = gold[row["case_id"]]["official_answer"]
            append(fixed_path, {"model": f"{seed}-{arm}", "case_id": row["case_id"],
                 "answer": result["answer"], "raw": result["raw"], "invalid": result["invalid"],
                 "em": int(bool(result["answer"]) and normalize_answer(result["answer"]) == normalize_answer(actual))})
        if arm != "incumbent":
            for row in train_data:
                result = fixed_answer(client, wrapped, registry, row["rollout"]["question"],
                                      row["rollout"]["trace"], row["input_ids"])
                labels = label_map[row["case_id"]]
                answer = result["answer"]
                append(fit_path, {"model": f"{seed}-{arm}", "case_id": row["case_id"],
                    "answer": answer, "raw": result["raw"],
                    "original_em": int(bool(answer) and normalize_answer(answer) == normalize_answer(labels["original_answer"])),
                    "revised_em": int(bool(answer) and normalize_answer(answer) == normalize_answer(labels["revised_answer"]))})
        del client, wrapped, registry
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def evaluate(seed):
    for arm in ("a", "b"):
        label = f"{seed}-{arm}"
        q.eval_model(label, OUT / f"seed{seed}-{arm}/adapter-final", q.dev_cases(), OUT / f"eval-{label}.jsonl")
    diagnostics(seed, ("incumbent", "a", "b") if seed == 42 else ("a", "b"))


def replication_gate():
    a, b = stats("42-a"), stats("42-b")
    if b["total"] > a["total"] and b["regression"] >= a["regression"]:
        return {"replicate": True, "reason": "product_better_than_control"}
    semantic_path = OUT / "semantic-review.json"
    if not semantic_path.exists(): return {"replicate": None, "reason": "semantic_review_pending"}
    review = read(semantic_path)
    if review.get("rubric_sha256") != sha(OUT / "semantic-rubric.json"):
        raise RuntimeError("Semantic review rubric mismatch")
    counts = review["fixed_counts"]
    if set(counts) != {"42-a", "42-b"}:
        raise RuntimeError("Both fixed-context models must be reviewed")
    allowed = set(read(OUT / "semantic-rubric.json")["categories"])
    for arm in ("42-a", "42-b"):
        if sum(counts[arm].values()) != 32 or set(counts[arm]) != allowed:
            raise RuntimeError("Semantic review incomplete")
    improved = (counts["42-b"]["missing_qualifier"] < counts["42-a"]["missing_qualifier"]
                and counts["42-b"]["unsupported_or_wrong"] <= counts["42-a"]["unsupported_or_wrong"])
    return {"replicate": improved, "reason": "semantic_gate_passed" if improved else "no_gate_passed"}


def eligible_product(result):
    return bool(result and result["total"] > 19 and result["regression"] >= 7 and result["invalid"] == 0)


def decide():
    baseline, a42, b42 = stats("incumbent"), stats("42-a"), stats("42-b")
    if (baseline["total"], baseline["regression"], baseline["invalid"]) != (19, 7, 0):
        raise RuntimeError("Baseline changed")
    gate = replication_gate()
    if gate["replicate"] is None: raise RuntimeError("Replication decision pending semantic review")
    b43 = stats("43-b") if gate["replicate"] else None
    accepted = bool(gate["replicate"] and eligible_product(b42) and eligible_product(b43))
    adapter = OUT / "seed42-b/adapter-final" if accepted else SOURCE
    decision = {"selected": "42-b" if accepted else "incumbent", "adapter": str(adapter),
                "reason": "both_seeds_pass_product_gate" if accepted else "incumbent_retained",
                "baseline": baseline, "a42": a42, "b42": b42, "a43": stats("43-a"), "b43": b43,
                "replication_gate": gate}
    write(OUT / "decision.json", decision)


def stage(name):
    config()
    if name == "baseline":
        q.eval_model("incumbent", SOURCE, q.dev_cases(), OUT / "eval-incumbent.jsonl")
        result = stats("incumbent")
        if (result["total"], result["regression"], result["invalid"]) != (19, 7, 0):
            raise RuntimeError("Goal R baseline did not reproduce")
    elif name.startswith("train"):
        train(int(name[5:7]), name[-1])
    elif name.startswith("eval"):
        evaluate(int(name[4:]))
    elif name == "final":
        decision = read(OUT / "decision.json")
        if not Path(decision["adapter"]).exists(): raise FileNotFoundError(decision["adapter"])
        client, wrapped, registry = q.make_client(decision["adapter"], deterministic=True)
        case = q.train_cases()[0]
        result = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False, seed=42))
        write(OUT / "final-demo.json", {"selected": decision["selected"], "trace": result})
    else: raise ValueError(name)


def run_stage(name, state):
    cap = CAPS[name]
    if state["gpu_seconds"] + cap > LIMIT: raise RuntimeError("GPU budget insufficient")
    state["current_stage"] = name
    write(STATE, state)
    started = time.monotonic()
    try:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "stage", "--name", name],
                       cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                       check=True, timeout=cap)
    finally:
        elapsed = time.monotonic() - started
        state["gpu_seconds"] += elapsed
        state["stages"].append({"name": name, "seconds": elapsed})
        write(STATE, state)


def run():
    config()
    state = read(STATE)
    if state["status"] not in {"prepared", "awaiting_semantic_review"}:
        raise RuntimeError("Run cannot be restarted or overwritten")
    try:
        if state["status"] == "prepared":
            state["status"] = "running"
            write(STATE, state)
            for name in ("baseline", "train42_a", "train42_b", "eval42"):
                run_stage(name, state)
        gate = replication_gate()
        if gate["replicate"] is None:
            state["status"] = "awaiting_semantic_review"
            write(STATE, state)
            return
        write(OUT / "replication-gate.json", gate)
        if gate["replicate"]:
            for name in ("train43_a", "train43_b", "eval43"):
                run_stage(name, state)
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
    for label in ("incumbent", "42-a", "42-b", "43-a", "43-b"):
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
    parser.add_argument("--name", choices=list(CAPS))
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "run": run()
    elif args.command == "stage": stage(args.name)
    else: summarize()


if __name__ == "__main__":
    main()
