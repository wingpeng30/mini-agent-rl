"""Goal N: one bounded fixed-evidence v118 GRPO experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-n-v118-grpo"
J = ROOT / "runs/v133-goal-j-transfer"
G = ROOT / "runs/v133-goal-g-cpu/evaluation-frozen.jsonl"
V118 = ROOT / "backups/v118/extracted/runs/v118/formal/adapter"
J_ADAPTER = J / "adapter-final"
BASE = Path("/root/autodl-tmp/qwen_08b")
CONFIG = OUT / "config-frozen.json"
STATUS = OUT / "status.json"
INPUT_HASHES = {
    "train": "d4b41cb3413c9fca736d26b30e6d5e5292c1b47308d21c01f174f8957e0b263b",
    "new_eval": "42531fc1dbd2c3cb1fbec12e85523483c1c0b40546c55a0a6d622474d9648d29",
    "goal_G_regression": "f689f94b598a1c0d89f1b87f0b008c8a03158a7ab479028368285564244aad69",
    "v118": "ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5",
    "goal_J_sft": "21b24dc960c3915e9db89626358b2271d8e37ff8b6e0205d3ab40ddb5eca0fc2",
}
PATHS = {"train": J / "train-frozen.jsonl", "new_eval": J / "evaluation-frozen.jsonl",
         "goal_G_regression": G, "v118": V118 / "adapter_model.safetensors",
         "goal_J_sft": J_ADAPTER / "adapter_model.safetensors"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()


def load_config():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for name, path in PATHS.items():
        if digest(path) != cfg["input_hashes"][name]:
            raise RuntimeError(f"frozen input changed: {name}")
    if digest(Path(__file__)) != cfg["runner_sha256"]:
        raise RuntimeError("runner changed after freeze")
    if digest(ROOT / "src/mini_agent_rl/training/grpo.py") != cfg["grpo_sha256"]:
        raise RuntimeError("GRPO implementation changed after freeze")
    if digest(ROOT / "src/mini_agent_rl/model/local.py") != cfg["client_sha256"]:
        raise RuntimeError("scorer implementation changed after freeze")
    return cfg


def cpu_contract():
    from transformers import AutoProcessor
    from mini_agent_rl.training.sft import encode_assistant_only, AssistantOnlyCollator
    from evaluate_v133_goal_c_sft import normalize_answer, f1_score
    proc = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    result = []
    for split in ("train", "new_eval", "goal_G_regression"):
        rows = read_rows(PATHS[split])
        if len(rows) != 16 or len({row["case_id"] for row in rows}) != 16:
            raise RuntimeError(f"bad frozen set: {split}")
        for row in rows:
            index = row["target_assistant_index"]
            prompt = row["messages"][:index]
            ids = proc.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            ids = list(ids[0] if ids and isinstance(ids[0], list) else ids)
            encoded = encode_assistant_only(proc, row["messages"], 2176, index, False)
            positions = [n for n, label in enumerate(encoded["labels"]) if label != -100]
            if (encoded["input_ids"][:len(ids)] != ids or any(label != -100 for label in encoded["labels"][:len(ids)])
                    or len(ids) > 2048 or len(positions) > 128 or len(encoded["input_ids"]) > 2176
                    or positions[0] != len(ids) or positions[-1] != len(encoded["labels"]) - 2
                    or encoded["labels"][positions[-1]] != proc.tokenizer.convert_tokens_to_ids("<|im_end|>")):
                raise RuntimeError(f"prefix or target mismatch: {split}:{row['case_id']}")
            if split != "goal_G_regression":
                prompt_hash = hashlib.sha256(b"".join(int(x).to_bytes(4, "little") for x in ids)).hexdigest()
                if prompt_hash != row["cpu_check"]["prompt_sha256"]:
                    raise RuntimeError(f"prompt hash mismatch: {split}:{row['case_id']}")
            result.append({"split": split, "case_id": row["case_id"], "prompt_tokens": len(ids),
                           "target_tokens": len(positions)})
    collated = AssistantOnlyCollator(proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id)([
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}])
    if collated["labels"].tolist()[1][-1] != -100 or collated["attention_mask"].tolist()[1][-1] != 0:
        raise RuntimeError("padding labels or mask incorrect")
    if normalize_answer("The Film") != normalize_answer("film") or f1_score("film director", "director") <= 0:
        raise RuntimeError("frozen scoring check failed")
    if classify('{"action":"search","query":"x"}')[0] != "search" or classify("bad")[0] != "invalid":
        raise RuntimeError("action classification check failed")
    write_json(OUT / "cpu-contract.json", {"status": "pass", "rows": result,
                                             "max_prompt": max(x["prompt_tokens"] for x in result)})


def prepare():
    if CONFIG.exists() or STATUS.exists():
        raise FileExistsError("Goal N already prepared; refusing to overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, path in PATHS.items():
        if digest(path) != INPUT_HASHES[name]:
            raise RuntimeError(f"frozen input changed: {name}")
    cpu_contract()
    import torch, transformers, peft, tokenizers
    versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                "peft": peft.__version__, "tokenizers": tokenizers.__version__}
    expected = {"torch": "2.5.1+cu124", "transformers": "5.17.0", "peft": "0.20.0", "tokenizers": "0.23.2"}
    if versions != expected:
        raise RuntimeError(f"runtime versions changed: {versions}")
    cfg = {"version": "v133-goal-n-v1", "input_hashes": INPUT_HASHES,
           "runner_sha256": digest(Path(__file__)),
           "grpo_sha256": digest(ROOT / "src/mini_agent_rl/training/grpo.py"),
           "client_sha256": digest(ROOT / "src/mini_agent_rl/model/local.py"),
           "base_config_sha256": digest(BASE / "config.json"),
           "tokenizer_sha256": digest(BASE / "tokenizer.json"),
           "versions": versions, "seed": 42, "group_size": 4, "temperature": 0.7, "top_p": 1.0,
           "top_k": 0, "max_input_tokens": 2048, "max_new_tokens": 128, "max_total_tokens": 2176,
           "thinking": False, "dtype": "bfloat16", "reward_em": 1.0, "reward_f1": 0.25,
           "reward_search": 0.0, "reward_invalid": 0.0,
           "min_variable_groups": 4, "min_correctness_variable_groups": 2,
           "min_formal_updates": 4, "max_formal_updates": 16,
           "learning_rate": 5e-6, "betas": [0.9, 0.999], "optimizer_eps": 1e-8,
           "weight_decay": 0.0, "clip_epsilon": 0.2, "kl_beta": 0.02,
           "policy_epochs": 1, "max_grad_norm": 1.0, "max_reserved_memory_gb": 21.0,
           "probe_cap_seconds": 1800, "train_cap_seconds": 3600, "eval_cap_seconds": 1800,
           "total_gpu_cap_seconds": 7200}
    write_json(CONFIG, cfg)
    write_json(STATUS, {"status": "CPU_READY", "stage": "N1", "config_sha256": digest(CONFIG),
                        "gpu_seconds_used": 0.0, "input_hashes": INPUT_HASHES})
    print(json.dumps({"status": "CPU_READY", "config_sha256": digest(CONFIG), "rows": 48}))


def classify(raw):
    try:
        value = json.loads(raw.strip())
    except (ValueError, TypeError):
        return "invalid", ""
    if not isinstance(value, dict):
        return "invalid", ""
    if value.get("action") == "answer" and set(value) == {"action", "answer"} and isinstance(value.get("answer"), str) and value["answer"].strip():
        return "answer", value["answer"].strip()
    if value.get("action") == "search" and set(value) == {"action", "query"} and isinstance(value.get("query"), str) and value["query"].strip():
        return "search", value["query"].strip()
    return "invalid", ""


def score(kind, answer, gold, cfg):
    from evaluate_v133_goal_c_sft import normalize_answer, f1_score
    em = int(kind == "answer" and normalize_answer(answer) == normalize_answer(gold))
    f1 = f1_score(answer, gold) if kind == "answer" else 0.0
    reward = cfg["reward_em"] * em + cfg["reward_f1"] * f1 if kind == "answer" else 0.0
    return em, f1, reward


def load_client(adapter, cfg, *, training=False):
    import torch
    from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    if torch.cuda.get_device_properties(0).total_memory < 23.5 * 1024**3:
        raise RuntimeError("GPU below 24GB class")
    settings = LocalModelConfig(model_id=str(BASE), adapter_path=str(adapter), dtype=cfg["dtype"],
                                device="cuda", deterministic=True, enable_thinking=False,
                                max_input_tokens=2048, max_new_tokens=128,
                                gradient_checkpointing=training, recompute_old_logprobs=True,
                                observation_encoding="legacy")
    client = TransformersModelClient(settings)
    client._load()
    active = client._model.active_adapters() if callable(getattr(client._model, "active_adapters", None)) else [client._model.active_adapter]
    modules = sum(1 for _, module in client._model.named_modules() if hasattr(module, "lora_A") and "default" in module.lora_A)
    if "default" not in active or modules != 96:
        raise RuntimeError(f"adapter not active: {active}, {modules}")
    client._model.eval()
    return client


def generate(client, row, cfg, *, sample):
    import torch
    from mini_agent_rl.model.local import generation_terminator_ids, trim_generation_terminators
    messages = row["messages"][:row["target_assistant_index"]]
    processor = client._processor
    batch = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True, enable_thinking=False)
    prompt_ids = batch["input_ids"][0].tolist()
    if len(prompt_ids) > cfg["max_input_tokens"]:
        raise RuntimeError("prompt over 2048; no truncation")
    if row.get("cpu_check"):
        h = hashlib.sha256(b"".join(int(x).to_bytes(4, "little") for x in prompt_ids)).hexdigest()
        if h != row["cpu_check"]["prompt_sha256"]:
            raise RuntimeError("frozen prompt changed")
    batch = batch.to(client._model.device)
    kwargs = {"do_sample": sample, "max_new_tokens": cfg["max_new_tokens"],
              "pad_token_id": processor.tokenizer.eos_token_id,
              "eos_token_id": generation_terminator_ids(processor.tokenizer)}
    if sample:
        kwargs.update(temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"])
    start = time.monotonic()
    client._model.config.use_cache = True
    with torch.inference_mode():
        output = client._model.generate(**batch, **kwargs)
    action_ids = output[0, len(prompt_ids):].tolist()
    if not action_ids or len(prompt_ids) + len(action_ids) > cfg["max_total_tokens"]:
        raise RuntimeError("empty or oversized action")
    parse_ids, terminal = trim_generation_terminators(action_ids, processor.tokenizer)
    raw = processor.tokenizer.decode(parse_ids, skip_special_tokens=True).strip()
    full_raw = processor.tokenizer.decode(action_ids, skip_special_tokens=False)
    kind, answer = classify(raw)
    em, f1, reward = score(kind, answer, row["official_answer"], cfg)
    result = {"case_id": row["case_id"], "raw": raw, "raw_with_special_tokens": full_raw,
              "action": kind, "prediction": answer, "em": em, "f1": f1, "reward": reward,
              "prompt_token_ids": prompt_ids, "prompt_attention_mask": batch["attention_mask"][0].tolist(),
              "action_token_ids": action_ids, "action_mask": [1] * len(action_ids),
              "prompt_tokens": len(prompt_ids), "action_tokens": len(action_ids),
              "terminated": terminal, "reached_max_new_tokens": len(action_ids) == cfg["max_new_tokens"] and not any(terminal.values()),
              "generation_seconds": round(time.monotonic() - start, 3)}
    return result


def group_summary(group):
    rewards = [float(item["reward"]) for item in group]
    valid = [item for item in group if item["action"] == "answer"]
    return {"case_id": group[0]["case_id"], "reward_variable": len(set(rewards)) > 1,
            "answer_reward_variable": len({item["reward"] for item in valid}) > 1,
            "answer_correctness_variable": {item["em"] for item in valid} == {0, 1},
            "mean_reward": sum(rewards) / len(rewards),
            "reward_std_population": math.sqrt(sum((x - sum(rewards)/len(rewards))**2 for x in rewards)/len(rewards)),
            "actions": {name: sum(item["action"] == name for item in group) for name in ("answer", "search", "invalid")}}


def probe():
    import torch
    from transformers import set_seed
    cfg = load_config()
    if (OUT / "probe-results.jsonl").exists():
        raise FileExistsError("probe already started; no automatic retry")
    set_seed(cfg["seed"])
    client = load_client(V118, cfg)
    rows = read_rows(PATHS["train"])
    started = time.monotonic()
    durations = []
    for row in rows:
        for candidate in range(cfg["group_size"]):
            if time.monotonic() - started >= cfg["probe_cap_seconds"]:
                raise TimeoutError("probe stage cap")
            item = generate(client, row, cfg, sample=True)
            item["candidate"] = candidate
            append_jsonl(OUT / "probe-results.jsonl", item)
            durations.append(item["generation_seconds"])
            if len(durations) == 2 and time.monotonic() - started + sum(durations)/2 * 62 + 120 > cfg["probe_cap_seconds"]:
                raise TimeoutError("probe first-two speed projection")
    results = read_rows(OUT / "probe-results.jsonl")
    if len(results) != 64:
        raise RuntimeError("incomplete probe")
    groups = [[item for item in results if item["case_id"] == row["case_id"]] for row in rows]
    stats = [group_summary(group) for group in groups]
    total_variable = sum(x["reward_variable"] for x in stats)
    correctness_variable = sum(x["answer_correctness_variable"] for x in stats)
    summary = {"status": "COMPLETE", "sample_count": 64, "groups": stats,
               "reward_variable_groups": total_variable,
               "answer_reward_variable_groups": sum(x["answer_reward_variable"] for x in stats),
               "answer_correctness_variable_groups": correctness_variable,
               "zero_variance_groups": 16 - total_variable,
               "action_counts": {name: sum(item["action"] == name for item in results) for name in ("answer", "search", "invalid")},
               "max_action_tokens": max(item["action_tokens"] for item in results),
               "elapsed_seconds": round(time.monotonic()-started, 3),
               "signal_pass": total_variable >= cfg["min_variable_groups"] and correctness_variable >= cfg["min_correctness_variable_groups"],
               "policy_fingerprint": client.policy_fingerprint(), "gpu_peak": client.gpu_memory_stats()}
    write_json(OUT / "probe-summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("sample_count", "reward_variable_groups", "answer_reward_variable_groups", "answer_correctness_variable_groups", "signal_pass")}))


def tensor_hashes(model, *, lora):
    hashes = {}
    for name, value in model.named_parameters():
        if ("lora_" in name) == lora:
            hashes[name] = hashlib.sha256(value.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()
    return hashes


def train():
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import clipped_grpo_loss, guarded_optimizer_step, ratio_statistics, require_initial_ratio
    cfg = load_config()
    probe_summary = json.loads((OUT / "probe-summary.json").read_text(encoding="utf-8"))
    if not probe_summary["signal_pass"]:
        raise RuntimeError("probe signal gate failed")
    if (OUT / "train-groups.jsonl").exists():
        raise FileExistsError("training already started; no automatic retry")
    set_seed(cfg["seed"])
    client = load_client(V118, cfg, training=True)
    params_named = client.prepare_for_training()
    params = [parameter for _, parameter in params_named]
    reference = {name: value.detach().clone() for name, value in params_named}
    reference_hashes = {name: hashlib.sha256(value.detach().float().cpu().numpy().tobytes()).hexdigest() for name, value in reference.items()}
    base_before = tensor_hashes(client._model, lora=False)
    lora_before = tensor_hashes(client._model, lora=True)
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], betas=tuple(cfg["betas"]), eps=cfg["optimizer_eps"], weight_decay=cfg["weight_decay"])
    rows = read_rows(PATHS["train"])
    started = time.monotonic()
    updates = 0
    group_durations = []
    for row in rows:
        if time.monotonic() - started >= cfg["train_cap_seconds"]:
            raise TimeoutError("formal training cap")
        group_started = time.monotonic()
        client._model.eval()
        group = []
        for candidate in range(cfg["group_size"]):
            item = generate(client, row, cfg, sample=True)
            item["candidate"] = candidate
            old = client.score_action_tensor([], item["action_token_ids"], False, item["prompt_token_ids"], item["prompt_attention_mask"]).detach()
            item["old_logprobs"] = old.float().cpu().tolist()
            group.append(item)
            append_jsonl(OUT / "train-rollouts.jsonl", item)
        summary = group_summary(group)
        if summary["reward_variable"]:
            old_rows = []
            ref_rows = []
            for item in group:
                old = client.score_action_tensor([], item["action_token_ids"], False, item["prompt_token_ids"], item["prompt_attention_mask"]).detach()
                reference_lp = client.reference_logprobs([], item["action_token_ids"], reference, item["prompt_token_ids"], item["prompt_attention_mask"]).detach()
                expected = hashlib.sha256(str(item["prompt_token_ids"]).encode()).hexdigest()
                if any(client._context_records[-n]["prompt_sha256"] != expected for n in (1, 2)):
                    raise RuntimeError("old/reference prompt replay mismatch")
                recorded = torch.tensor(item["old_logprobs"], dtype=torch.float32)
                if (old.float().cpu() - recorded).abs().max() > 0.01:
                    raise RuntimeError("old logprob changed across scoring")
                if old.numel() != len(item["action_token_ids"]) or reference_lp.numel() != old.numel() or not torch.isfinite(old).all() or not torch.isfinite(reference_lp).all():
                    raise RuntimeError("nonfinite or wrong-sized old/reference")
                old_rows.append(old)
                ref_rows.append(reference_lp)
            client.set_training_forward_mode()
            initial_ratios = []
            for item, old in zip(group, old_rows):
                current = client.score_action_tensor([], item["action_token_ids"], True, item["prompt_token_ids"], item["prompt_attention_mask"])
                initial_ratios.append(torch.exp(current.detach()-old).float().cpu())
            initial = ratio_statistics(initial_ratios)
            require_initial_ratio(initial, label="initial")
            optimizer.zero_grad(set_to_none=True)
            backwards = []
            losses = []
            kls = []
            for item, old, reference_lp in zip(group, old_rows, ref_rows):
                current = client.score_action_tensor([], item["action_token_ids"], True, item["prompt_token_ids"], item["prompt_attention_mask"])
                expected = hashlib.sha256(str(item["prompt_token_ids"]).encode()).hexdigest()
                if client._context_records[-1]["prompt_sha256"] != expected:
                    raise RuntimeError("current prompt replay mismatch")
                backwards.append(torch.exp(current.detach()-old).float().cpu())
                advantage = (item["reward"] - summary["mean_reward"]) / max(summary["reward_std_population"], 1e-8)
                loss, kl, _ = clipped_grpo_loss(current, old, reference_lp, advantage, cfg["clip_epsilon"], cfg["kl_beta"])
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite GRPO loss")
                (loss / cfg["group_size"]).backward()
                losses.append(float(loss.detach()))
                kls.append(float(kl))
            backwards_stats = ratio_statistics(backwards)
            require_initial_ratio(backwards_stats, label="first backward")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]))
            if not math.isfinite(grad_norm) or grad_norm == 0:
                raise RuntimeError("nonfinite or zero gradient")
            memory = client.gpu_memory_stats()
            guarded_optimizer_step(optimizer, params, memory, cfg["max_reserved_memory_gb"])
            updates += 1
            summary.update({"updated": True, "initial_ratio": initial, "first_backward_ratio": backwards_stats,
                            "loss": sum(losses)/len(losses), "kl": sum(kls)/len(kls),
                            "grad_norm": grad_norm, "gpu_memory": memory})
        else:
            summary["updated"] = False
            summary["skip_reason"] = "zero_variance"
        summary["group_seconds"] = round(time.monotonic()-group_started, 3)
        append_jsonl(OUT / "train-groups.jsonl", summary)
        group_durations.append(summary["group_seconds"])
        if len(group_durations) == 2 and time.monotonic()-started + sum(group_durations)/2 * 14 + 180 > cfg["train_cap_seconds"]:
            raise TimeoutError("first-two group speed projection")
    lora_after = tensor_hashes(client._model, lora=True)
    base_after = tensor_hashes(client._model, lora=False)
    ref_after = {name: hashlib.sha256(value.detach().float().cpu().numpy().tobytes()).hexdigest() for name, value in reference.items()}
    if updates < cfg["min_formal_updates"] or not any(lora_before[name] != lora_after[name] for name in lora_before) or base_before != base_after or reference_hashes != ref_after:
        raise RuntimeError("final update/base/reference gate failed")
    adapter_dir = OUT / "adapter-final"
    client._model.save_pretrained(adapter_dir)
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise RuntimeError("final adapter missing")
    check = {"case_id": rows[0]["case_id"], "prompt_token_ids": group[0]["prompt_token_ids"],
             "prompt_attention_mask": group[0]["prompt_attention_mask"], "action_token_ids": group[0]["action_token_ids"]}
    before_lp = client.score_action_tensor([], check["action_token_ids"], False, check["prompt_token_ids"], check["prompt_attention_mask"]).detach().float().cpu()
    import gc
    del client, optimizer, params, reference
    gc.collect(); torch.cuda.empty_cache()
    reloaded = load_client(adapter_dir, cfg)
    after_lp = reloaded.score_action_tensor([], check["action_token_ids"], False, check["prompt_token_ids"], check["prompt_attention_mask"]).detach().float().cpu()
    reload_max_abs = float((before_lp-after_lp).abs().max())
    if reload_max_abs > 0.001:
        raise RuntimeError(f"final adapter reload mismatch: {reload_max_abs}")
    write_json(OUT / "train-summary.json", {"status": "PASSED", "updates": updates, "groups": len(rows),
                                              "elapsed_seconds": round(time.monotonic()-started, 3),
                                              "changed_lora_parameters": sum(lora_before[name] != lora_after[name] for name in lora_before),
                                              "base_unchanged": base_before == base_after,
                                              "reference_unchanged": reference_hashes == ref_after,
                                              "reload_max_abs": reload_max_abs,
                                              "adapter_sha256": digest(adapter_dir / "adapter_model.safetensors")})
    print(json.dumps({"status": "PASSED", "updates": updates, "reload_max_abs": reload_max_abs}))


def evaluate():
    import torch
    from transformers import set_seed
    cfg = load_config()
    if json.loads((OUT / "train-summary.json").read_text(encoding="utf-8"))["status"] != "PASSED":
        raise RuntimeError("training not passed")
    if (OUT / "evaluation.jsonl").exists():
        raise FileExistsError("evaluation already started")
    started = time.monotonic()
    set_seed(cfg["seed"])
    arms = [("v118", V118), ("goal_J_sft", J_ADAPTER), ("goal_N_grpo", OUT / "adapter-final")]
    for arm, adapter in arms:
        client = load_client(adapter, cfg)
        for split in ("train", "new_eval", "goal_G_regression"):
            for row in read_rows(PATHS[split]):
                if time.monotonic()-started >= cfg["eval_cap_seconds"]:
                    raise TimeoutError("evaluation cap")
                item = generate(client, row, cfg, sample=False)
                compact = {key: item[key] for key in ("case_id", "raw", "action", "prediction", "em", "f1", "prompt_tokens", "action_tokens", "generation_seconds")}
                compact.update({"arm": arm, "split": split})
                append_jsonl(OUT / "evaluation.jsonl", compact)
        del client
        import gc
        gc.collect(); torch.cuda.empty_cache()
    all_rows = read_rows(OUT / "evaluation.jsonl")
    if len(all_rows) != 144:
        raise RuntimeError("evaluation incomplete")
    summary = {"status": "PASSED", "elapsed_seconds": round(time.monotonic()-started, 3), "sets": {}}
    for arm, _ in arms:
        summary["sets"][arm] = {}
        for split in ("train", "new_eval", "goal_G_regression"):
            items = [item for item in all_rows if item["arm"] == arm and item["split"] == split]
            summary["sets"][arm][split] = {"n": len(items), "em": sum(item["em"] for item in items),
                                            "f1": sum(item["f1"] for item in items)/len(items),
                                            "actions": {name: sum(item["action"] == name for item in items) for name in ("answer", "search", "invalid")},
                                            "generation_seconds": round(sum(item["generation_seconds"] for item in items), 3)}
    write_json(OUT / "evaluation-summary.json", summary)
    print(json.dumps(summary))


def run():
    cfg = load_config()
    state = json.loads(STATUS.read_text(encoding="utf-8"))
    if state["status"] != "CPU_READY":
        raise RuntimeError("run already started or not prepared")
    started = time.monotonic()
    state.update(status="RUNNING", stage="N2", gpu_started_unix=time.time())
    write_json(STATUS, state)
    commands = [("N2", "probe", cfg["probe_cap_seconds"]), ("N3", "train", cfg["train_cap_seconds"]),
                ("N4", "evaluate", cfg["eval_cap_seconds"])]
    for stage, command, cap in commands:
        remaining = cfg["total_gpu_cap_seconds"] - (time.monotonic()-started)
        if remaining <= 0:
            state.update(status="TECHNICAL_BLOCKED", stage=stage, error="total GPU cap elapsed")
            break
        state.update(stage=stage, gpu_seconds_used=round(time.monotonic()-started, 3))
        write_json(STATUS, state)
        child = [sys.executable, str(Path(__file__).resolve()), command]
        try:
            done = subprocess.run(child, cwd=ROOT, timeout=min(cap, remaining), check=False)
        except subprocess.TimeoutExpired:
            state.update(status="TECHNICAL_BLOCKED", error=f"{stage} hard timeout")
            break
        if done.returncode:
            state.update(status="TECHNICAL_BLOCKED", error=f"{stage} exit {done.returncode}")
            break
        if stage == "N2":
            summary = json.loads((OUT / "probe-summary.json").read_text(encoding="utf-8"))
            if not summary["signal_pass"]:
                state.update(status="NO_SIGNAL", stage="N5")
                break
    else:
        state.update(status="EXPERIMENT_COMPLETE", stage="N5")
    state["gpu_seconds_used"] = round(time.monotonic()-started, 3)
    write_json(STATUS, state)
    print(json.dumps(state))


def summarize():
    load_config()
    state = json.loads(STATUS.read_text(encoding="utf-8"))
    available = {name: json.loads((OUT / name).read_text(encoding="utf-8")) for name in
                 ("probe-summary.json", "train-summary.json", "evaluation-summary.json") if (OUT / name).exists()}
    print(json.dumps({"state": state, "available": available}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "probe", "train", "evaluate", "run", "summarize"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    if args.config.resolve() != CONFIG.resolve():
        parser.error("only the named frozen Goal N config is accepted")
    globals()[args.command]()


if __name__ == "__main__":
    main()
