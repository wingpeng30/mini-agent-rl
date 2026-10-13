"""Goal Z: true two-search SFT comparison, with frozen source and evidence gates."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import run_v133_goal_q_search as q
import run_v133_goal_s_label_granularity as s
import run_v133_goal_t_update_length as t
import run_v133_goal_y_three_class_transfer as y

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-z-multihop-sft"
SOURCE = t.checkpoint_dir(42, 32) / "adapter-final"
CFG, STATE = OUT / "config.json", OUT / "state.json"
SEEDS, ARMS = (42, 43), ("answer", "trajectory")
COUNTS = {"train": 128, "dev": 64, "confirm": 64}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append(path: Path, value) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_files() -> dict[str, Path]:
    files = {"source_adapter": SOURCE / "adapter_model.safetensors",
             "tokenizer": q.p.BASE / "tokenizer.json",
             "old_corpus": q.p.CORPUS, "old_cases": q.p.CASES,
             "goal_y_cases": y.OUT / "dev-cases.jsonl", "goal_y_corpus": y.OUT / "dev-corpus.json",
             "source_audit": OUT / "source-audit.json", "curation": OUT / "curation.json",
             "data_review": OUT / "data-review.json",
             "train_cases": OUT / "selected-train.jsonl", "dev_cases": OUT / "selected-dev.jsonl",
             "confirm_cases": OUT / "selected-confirm.jsonl",
             "train_corpus": OUT / "corpus-train.json", "dev_corpus": OUT / "corpus-dev.json",
             "confirm_corpus": OUT / "corpus-confirm.json",
             "runner": Path(__file__), "q_runner": Path(q.__file__),
             "model_code": ROOT / "src/mini_agent_rl/model/local.py",
             "search_code": ROOT / "src/mini_agent_rl/tools/search.py"}
    files.update({f"base_weights_{i}": path for i, path in enumerate(sorted(q.p.BASE.glob("*.safetensors")))})
    return files


def software():
    import torch, transformers, peft
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "env": {key: os.environ.get(key) for key in
                    ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")}}


def cases(split: str) -> list[dict]:
    path = OUT / ("selected-confirm.jsonl" if split == "confirm" else f"selected-{split}.jsonl")
    values = rows(path)
    if len(values) != COUNTS[split] or len({v["id"] for v in values}) != len(values):
        raise RuntimeError(f"Frozen {split} case count or IDs invalid")
    return values


def prepare() -> None:
    if CFG.exists() or STATE.exists():
        raise FileExistsError("Goal Z already prepared")
    if read(t.OUT / "decision.json")["selected"] != "42-32":
        raise RuntimeError("Goal T selected source changed")
    curation = read(OUT / "curation.json")
    if curation["shortfalls"]:
        raise RuntimeError("Source/data gate failed")
    review = read(OUT / "data-review.json")
    if review.get("status") != "approved" or review.get("selected_sha256") != {
        split: sha(OUT / ("selected-confirm.jsonl" if split == "confirm" else f"selected-{split}.jsonl"))
        for split in COUNTS}:
        raise RuntimeError("Selected data lack frozen evidence review")
    for split in COUNTS:
        cases(split)
    sw = software()
    if not sw["gpu"]:
        raise RuntimeError("GPU unavailable")
    files = frozen_files()
    if not any(k.startswith("base_weights_") for k in files):
        raise RuntimeError("Actual base weight file absent")
    order = {}
    for seed in SEEDS:
        ids = [v["id"] for v in cases("train")]
        random.Random(seed).shuffle(ids)
        order[str(seed)] = ids
    write(CFG, {"version": "goal-z-multihop-sft-v1", "files": {k: sha(v) for k, v in files.items()},
                "software": sw, "product": q.load_config()["product"], "seeds": SEEDS, "arms": ARMS,
                "order": order, "updates_per_run": 128, "learning_rate": 5e-6,
                "adam_betas": [.9, .999], "adam_eps": 1e-8, "weight_decay": 0.,
                "gradient_clip": 1., "gpu_time_limit": None, "stage_timeouts": None,
                "dev_gate": {"success_gain": 4, "old_semantic_noninferiority": True},
                "confirm_gate": {"success_gain": 4}})
    write(STATE, {"status": "prepared", "stages": [], "gpu_seconds": 0.})


def config():
    value = read(CFG)
    if {k: sha(v) for k, v in frozen_files().items()} != value["files"]:
        raise RuntimeError("Frozen asset changed")
    if software() != value["software"]:
        raise RuntimeError("GPU/software changed")
    return value


def make_client(adapter: Path, corpus: Path):
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    pc = config()["product"]
    client = TransformersModelClient(LocalModelConfig(
        model_id=str(q.p.BASE), adapter_path=str(adapter),
        max_input_tokens=pc["max_input_tokens"], max_new_tokens=pc["max_new_tokens"],
        enable_thinking=False, deterministic=True, observation_encoding=pc["observation_encoding"],
        recompute_old_logprobs=False))
    registry = ToolRegistry()
    registry.register(SearchTool(read(corpus), pc["retriever"]))
    return client, q.SearchTextAliasClient(client), registry


def evaluate_new(label: str, adapter: Path, split: str) -> None:
    if split == "confirm" and read(STATE)["status"] != "confirmation_authorized":
        raise RuntimeError("Confirmation set is still sealed to evaluation")
    path = OUT / f"eval-{split}-{label}.jsonl"
    if path.exists():
        raise FileExistsError(path)
    client, wrapped, registry = make_client(adapter, OUT / f"corpus-{split}.json")
    for case in cases(split):
        result = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False, seed=42))
        found = {d["title"] for step in result["trace"]
                 for d in step.get("observation", {}).get("result", [])}
        result.update(case_id=case["id"], source=case["source"], split=split,
                      model=label, support_retrieved=set(case["support_titles"]).issubset(found),
                      **q.score(result["answer"], case["answer"]))
        append(path, result)
        print(json.dumps({"eval": split, "model": label, "completed": len(rows(path))}), flush=True)
    del client, wrapped, registry
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def evaluate_y(label: str, adapter: Path) -> None:
    path = OUT / f"eval-y-{label}.jsonl"
    if path.exists():
        raise FileExistsError(path)
    client, wrapped, registry = make_client(adapter, y.OUT / "dev-corpus.json")
    for case in y.dev_cases():
        result = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False, seed=42))
        result.update(case_id=case["case_id"], split="goal_y", model=label,
                      **q.score(result["answer"], case["official_answer"]))
        append(path, result)
    del client, wrapped, registry
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def baseline() -> None:
    config()
    old = OUT / "eval-old-incumbent.jsonl"
    q.eval_model("goal-z-incumbent", SOURCE, q.dev_cases(), old)
    values = rows(old)
    total = sum(v["em"] for v in values)
    regression = sum(v["em"] for v in values if v["split"] == "goal_G_regression")
    invalid = sum(v["reason"] == "invalid_action" for v in values)
    if len(values) != 32 or (total, regression, invalid) != (20, 7, 0):
        raise RuntimeError(f"Old baseline mismatch: {(total, regression, invalid)}")
    evaluate_new("incumbent", SOURCE, "dev")
    evaluate_y("incumbent", SOURCE)


def materialize() -> None:
    """Encode authored actions using each product call's actual packed context."""
    from mini_agent_rl.domain import Message, ToolCall
    from mini_agent_rl.prompts import system_prompt
    import torch
    cfg = config()
    target = OUT / "training-inputs.jsonl"
    if target.exists():
        raise FileExistsError(target)
    client, _, registry = make_client(SOURCE, OUT / "corpus-train.json")
    client._load()
    tokenizer = client._processor.tokenizer
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(end, int) or end < 0:
        raise RuntimeError("Missing assistant terminator")
    for case in cases("train"):
        actions = []
        if case["source"] == "old":
            for step in case["trace"]:
                if step["action"] == "search":
                    if not step.get("prompt_token_ids") or not step.get("action_token_ids"):
                        raise RuntimeError("Old search token replay incomplete")
                    actions.append({"action": "search", "query": step["query"],
                                    "prompt_token_ids": step["prompt_token_ids"],
                                    "prompt_attention_mask": step["prompt_attention_mask"],
                                    "action_token_ids": step["action_token_ids"],
                                    "observation": step["observation"]})
            actions.append({"action": "answer", "answer": case["answer"],
                            "prompt_token_ids": case["answer_prompt_token_ids"],
                            "prompt_attention_mask": case["answer_prompt_attention_mask"],
                            "action_token_ids": s.target_ids(client._processor, case["answer"])})
        else:
            messages = [Message(role="system", content=system_prompt(False, cfg["product"]["system_prompt"])),
                        Message(role="user", content=case["question"])]
            seen = set()
            for query in case["teacher"]["queries"]:
                encoded = client._inputs(messages, "goal_z_teacher")
                prompt = encoded["input_ids"][0].tolist()
                mask = encoded["attention_mask"][0].tolist()
                raw = json.dumps({"action": "search", "query": query}, ensure_ascii=False)
                tokens = tokenizer.encode(raw, add_special_tokens=False) + [end]
                observation = asyncio.run(registry.execute("search", {"query": query}))
                key = " ".join(query.casefold().split())
                if key in seen or not observation.get("ok"):
                    raise RuntimeError("Teacher search stopped")
                seen.add(key)
                observation["searched_queries"] = sorted(seen)
                if observation.get("result") != case["teacher"]["retrieved"][len(actions)]:
                    raise RuntimeError("Teacher tool result changed")
                actions.append({"action": "search", "query": query, "raw": raw,
                                "prompt_token_ids": prompt, "prompt_attention_mask": mask,
                                "action_token_ids": tokens, "observation": observation})
                call = ToolCall(name="search", arguments={"query": query})
                visible = {k: v for k, v in observation.items() if k != "latency_ms"}
                messages.extend([Message(role="assistant", content=raw, tool_calls=[call]),
                                 Message(role="tool", content=json.dumps(visible, ensure_ascii=False),
                                         tool_call_id=call.id)])
            encoded = client._inputs(messages, "goal_z_teacher_answer")
            actions.append({"action": "answer", "answer": case["answer"],
                            "prompt_token_ids": encoded["input_ids"][0].tolist(),
                            "prompt_attention_mask": encoded["attention_mask"][0].tolist(),
                            "action_token_ids": s.target_ids(client._processor, case["answer"])})
            if any(len(step["prompt_token_ids"]) > cfg["product"]["max_input_tokens"] or
                   len(step["action_token_ids"]) > cfg["product"]["max_new_tokens"] for step in actions):
                raise RuntimeError(f"Teacher exceeds product token budget: {case['id']}")
            if not set(case["teacher"]["support_titles"]).issubset({
                    d["title"] for action in actions for d in action.get("observation", {}).get("result", [])}):
                raise RuntimeError("Both supports not actually observed")
        if len(actions) > 4 or len([a for a in actions if a["action"] == "search"]) > 3:
            raise RuntimeError("Teacher exceeded product call budget")
        append(target, {"id": case["id"], "source": case["source"], "question": case["question"],
                        "answer": case["answer"], "actions": actions})
        print(json.dumps({"materialized": case["id"], "actions": len(actions)}), flush=True)
    del client, registry
    gc.collect(); torch.cuda.empty_cache()


def train(seed: int, arm: str) -> None:
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as o
    cfg = config()
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError((seed, arm))
    data = {v["id"]: v for v in rows(OUT / "training-inputs.jsonl")}
    if len(data) != 128 or set(data) != set(cfg["order"][str(seed)]):
        raise RuntimeError("Training input IDs mismatch")
    folder = OUT / f"seed{seed}-{arm}"
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir()
    set_seed(seed)
    client, _, _ = make_client(SOURCE, OUT / "corpus-train.json")
    params = [p for _, p in client.prepare_for_training()]
    base_before = o.parameter_hashes(client._model, lora=False)
    lora_before = o.parameter_hashes(client._model, lora=True)
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], betas=tuple(cfg["adam_betas"]),
                                  eps=cfg["adam_eps"], weight_decay=cfg["weight_decay"])
    last = None
    try:
        for index, case_id in enumerate(cfg["order"][str(seed)], 1):
            actions = data[case_id]["actions"]
            supervised = actions[-1:] if arm == "answer" else actions
            total_tokens = sum(len(a["action_token_ids"]) for a in supervised)
            optimizer.zero_grad(set_to_none=True)
            client.set_training_forward_mode()
            scalar = 0.
            for action in supervised:
                values = client.score_action_tensor([], action["action_token_ids"], True,
                    action["prompt_token_ids"], action["prompt_attention_mask"]).float()
                loss = -values.sum() / total_tokens
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                scalar += float(loss.detach())
            grad = float(torch.nn.utils.clip_grad_norm_(params, cfg["gradient_clip"]))
            if not math.isfinite(grad) or grad <= 0:
                raise RuntimeError("Invalid gradient")
            guarded_optimizer_step(optimizer, params, client.gpu_memory_stats(), 21.)
            append(folder / "updates.jsonl", {"step": index, "case_id": case_id, "loss": scalar,
                "grad": grad, "supervised_tokens": total_tokens,
                "prompt_tokens": [len(a["prompt_token_ids"]) for a in supervised],
                "gpu_memory": client.gpu_memory_stats()})
            last = supervised[-1]
            if index % 8 == 0:
                print(json.dumps({"seed": seed, "arm": arm, "updates": index}), flush=True)
        if o.parameter_hashes(client._model, lora=False) != base_before:
            raise RuntimeError("Base weights changed")
        changed = sum(o.parameter_hashes(client._model, lora=True)[k] != v for k, v in lora_before.items())
        if not changed:
            raise RuntimeError("LoRA unchanged")
        adapter = folder / "adapter-final"
        client._model.save_pretrained(adapter)
        expected = client.score_action_tensor([], last["action_token_ids"], False,
            last["prompt_token_ids"], last["prompt_attention_mask"]).detach().float().cpu()
        del client, optimizer, params
        gc.collect(); torch.cuda.empty_cache()
        loaded, _, _ = make_client(adapter, OUT / "corpus-train.json")
        actual = loaded.score_action_tensor([], last["action_token_ids"], False,
            last["prompt_token_ids"], last["prompt_attention_mask"]).detach().float().cpu()
        difference = float((actual - expected).abs().max())
        if difference > .001:
            raise RuntimeError(f"Reload mismatch: {difference}")
        write(folder / "summary.json", {"updates": 128, "changed_lora_tensors": changed,
            "base_unchanged": True, "reload_max_abs": difference,
            "adapter_sha256": sha(adapter / "adapter_model.safetensors")})
        del loaded
        gc.collect(); torch.cuda.empty_cache()
    except BaseException as exc:
        write(folder / "failure.json", {"error": str(exc),
            "completed_updates": len(rows(folder / "updates.jsonl")) if (folder / "updates.jsonl").exists() else 0})
        raise


def evaluate_all() -> None:
    config()
    for seed in SEEDS:
        for arm in ARMS:
            label = f"{seed}-{arm}"
            adapter = OUT / f"seed{label}/adapter-final"
            if read(OUT / f"seed{label}/summary.json")["updates"] != 128:
                raise RuntimeError("Training incomplete")
            evaluate_new(label, adapter, "dev")
            q.eval_model(label, adapter, q.dev_cases(), OUT / f"eval-old-{label}.jsonl")
            evaluate_y(label, adapter)


def run_subprocess(name: str, args: list[str]) -> None:
    state = read(STATE)
    if state["status"] == "stopped" or any(x["name"] == name for x in state["stages"]):
        raise RuntimeError("Stage already run or experiment stopped")
    started = time.monotonic()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT / "scripts")))
    try:
        subprocess.run([sys.executable, str(Path(__file__)), *args], cwd=ROOT, env=env, check=True)
    except BaseException as exc:
        state = read(STATE)
        state["status"] = "stopped"
        state["failure"] = {"stage": name, "error": str(exc)}
        STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    state = read(STATE)
    seconds = time.monotonic() - started
    state["stages"].append({"name": name, "seconds": seconds})
    state["gpu_seconds"] += seconds
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run() -> None:
    state = read(STATE)
    if state["status"] == "prepared":
        run_subprocess("baseline", ["baseline"])
        state = read(STATE)
        state["status"] = "awaiting_baseline_evidence_review"
        STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if state["status"] == "awaiting_baseline_evidence_review":
        review = read(OUT / "baseline-evidence-review.json")
        if review.get("dev_output_sha256") != sha(OUT / "eval-dev-incumbent.jsonl") or \
           review.get("goal_y_output_sha256") != sha(OUT / "eval-y-incumbent.jsonl") or \
           review.get("cases") != {"dev": 64, "goal_y": 48}:
            raise RuntimeError("Baseline evidence review incomplete")
        if review["dev_sufficient"] > 60:
            state["status"] = "complete_evaluation_ceiling"
            STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return
        state["status"] = "training"
        STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run_subprocess("materialize", ["materialize"])
        for seed in SEEDS:
            for arm in ARMS:
                run_subprocess(f"train-{seed}-{arm}", ["train", str(seed), arm])
        run_subprocess("evaluate", ["evaluate"])
        state = read(STATE)
        state["status"] = "awaiting_dev_evidence_review"
        STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if state["status"] == "confirmation_authorized":
        choice = read(OUT / "development-decision.json")["selected_route"]
        run_subprocess("confirm-incumbent", ["confirm", "incumbent"])
        for seed in SEEDS:
            run_subprocess(f"confirm-{seed}-{choice}", ["confirm", f"{seed}-{choice}"])
        state = read(STATE)
        state["status"] = "awaiting_confirmation_evidence_review"
        STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    raise RuntimeError(f"No automatic action from state {state['status']}")


def confirm(label: str) -> None:
    state = read(STATE)
    if state["status"] != "confirmation_authorized":
        raise RuntimeError("Confirmation gate closed")
    adapter = SOURCE if label == "incumbent" else OUT / f"seed{label}/adapter-final"
    evaluate_new(label, adapter, "confirm")


def summarize() -> dict:
    state = read(STATE)
    result = {"status": state["status"], "stages": state["stages"], "gpu_seconds": state["gpu_seconds"]}
    for split, count in (("dev", 64), ("confirm", 64), ("old", 32), ("y", 48)):
        for label in ("incumbent", "42-answer", "42-trajectory", "43-answer", "43-trajectory"):
            path = OUT / f"eval-{split}-{label}.jsonl"
            if not path.exists():
                continue
            values = rows(path)
            if len(values) != count:
                raise RuntimeError(f"Incomplete evaluation {path}")
            gold = ({v["id"]: v["answer"] for v in cases(split)} if split in ("dev", "confirm")
                    else {v["case_id"]: v["official_answer"] for v in
                          (q.dev_cases() if split == "old" else y.dev_cases())})
            for value in values:
                score = q.score(value["answer"], gold[value["case_id"]])
                if score["em"] != value["em"] or abs(score["f1"] - value["f1"]) > 1e-9:
                    raise RuntimeError("Score recomputation mismatch")
                if value["model_calls"] > 4 or value["searches"] > 3:
                    raise RuntimeError("Product call budget exceeded")
            result[f"{split}-{label}"] = {"em": sum(v["em"] for v in values),
                "invalid": sum(v["reason"] == "invalid_action" for v in values),
                "searches": sum(v["searches"] for v in values),
                "support": sum(v.get("support_retrieved", False) for v in values),
                "seconds": sum(v["elapsed_seconds"] for v in values)}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "summarize", "baseline", "materialize",
                                           "train", "evaluate", "confirm"))
    parser.add_argument("seed", nargs="?")
    parser.add_argument("arm", nargs="?")
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "run": run()
    elif args.command == "summarize": print(json.dumps(summarize(), ensure_ascii=False))
    elif args.command == "baseline": baseline()
    elif args.command == "materialize": materialize()
    elif args.command == "train": train(int(args.seed), args.arm)
    elif args.command == "evaluate": evaluate_all()
    elif args.command == "confirm": confirm(args.seed)
