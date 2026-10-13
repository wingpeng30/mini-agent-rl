"""Goal L: frozen answer-policy comparison; CPU prepare, GPU-only generation."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-l-answer-policy"
MODEL = Path("/root/autodl-tmp/qwen_08b")
V118 = ROOT / "backups/v118/extracted/runs/v118/formal/adapter"
J_ADAPTER = ROOT / "runs/v133-goal-j-transfer/adapter-final"
J_EVAL = ROOT / "runs/v133-goal-j-transfer/evaluation-frozen.jsonl"
G_EVAL = ROOT / "runs/v133-goal-g-cpu/evaluation-frozen.jsonl"
J_TRACES = ROOT / "runs/v133-goal-j-transfer/trajectories.jsonl"
J_CONFIG = ROOT / "runs/v133-goal-j-transfer/training-config-effective.json"
J_STATE = ROOT / "runs/v133-goal-j-transfer/run-state.json"
K_ATTRIBUTION = ROOT / "runs/v133-goal-k-attribution/autodl-results/attribution-summary.json"
K_BINDINGS = ROOT / "runs/v133-goal-k-attribution/autodl-results/gain-ticket-bindings.json"
K_DATAFLOW = ROOT / "runs/v133-goal-k-attribution/autodl-results/training-dataflow-audit.json"
K_REVIEW_STATUS = ROOT / "runs/v133-goal-k-attribution/review-channel-status.json"
CAP = 7200
V118_SHA = "ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5"
J_EVAL_SHA = "42531fc1dbd2c3cb1fbec12e85523483c1c0b40546c55a0a6d622474d9648d29"
G_EVAL_SHA = "f689f94b598a1c0d89f1b87f0b008c8a03158a7ab479028368285564244aad69"
J_ADAPTER_SHA = "21b24dc960c3915e9db89626358b2271d8e37ff8b6e0205d3ab40ddb5eca0fc2"
P1 = ("For the next response, use only the evidence already shown and produce one final answer action "
      "in the existing JSON format. Do not issue a search action or add explanatory text.")
P2 = ("Keep the answer concise. Preserve modifiers needed by the question to distinguish the requested "
      "category or attribute; do not drop them just to shorten the answer. Do not add unsupported qualifiers "
      "or list alternatives.")
EXPECTED = {"torch": "2.5.1+cu124", "transformers": "5.17.0", "peft": "0.20.0", "tokenizers": "0.23.2"}
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from scripts.run_v133_goal_g_cpu import cpu_check
from scripts.evaluate_v133_goal_c_sft import f1_score, normalize_answer


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def read_rows(path: Path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def classify_action(raw: str):
    """Keep legal search actions distinct from malformed or nonconforming JSON."""
    try:
        value = json.loads(raw.strip())
    except (ValueError, TypeError):
        return "invalid", None
    if not isinstance(value, dict):
        return "invalid", None
    if value.get("action") == "answer" and set(value) == {"action", "answer"} and isinstance(value.get("answer"), str):
        return "answer", value
    if value.get("action") == "search" and set(value) == {"action", "query"} and isinstance(value.get("query"), str):
        return "search", value
    return "invalid", None


def require_sha(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ValueError(f"sha256_mismatch:{label}")


def unique_subset(actual, expected, label: str) -> None:
    if len(actual) != len(set(actual)) or not set(actual) <= set(expected):
        raise ValueError(f"duplicate_or_unexpected_pair:{label}")


def prompt_ids(processor, messages):
    ids = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids]
    digest = hashlib.sha256(b"".join(x.to_bytes(4, "little") for x in ids)).hexdigest()
    return ids, digest


def expected_sources():
    for p, want in ((J_EVAL, J_EVAL_SHA), (G_EVAL, G_EVAL_SHA),
                    (V118 / "adapter_model.safetensors", V118_SHA),
                    (J_ADAPTER / "adapter_model.safetensors", J_ADAPTER_SHA)):
        if not p.is_file():
            raise ValueError(f"frozen_input_missing:{p}")
        require_sha(sha(p), want, str(p))
    for p in (K_ATTRIBUTION, K_BINDINGS, K_DATAFLOW, K_REVIEW_STATUS):
        if not p.is_file():
            raise ValueError(f"required_goal_K_record_missing:{p}")
    state = json.loads(J_STATE.read_text(encoding="utf-8"))
    config = json.loads(J_CONFIG.read_text(encoding="utf-8"))
    if state.get("status") != "complete" or state.get("gpu") != "NVIDIA GeForce RTX 4090 D":
        raise ValueError("historical_run_state_unverified")
    if state.get("adapter_sha256") != J_ADAPTER_SHA or config["trainer_kwargs"].get("seed") != 42:
        raise ValueError("historical_adapter_or_seed_mismatch")
    versions = {k: md.version(k) for k in EXPECTED}
    if versions != EXPECTED:
        raise ValueError(f"runtime_version_mismatch:{versions}")
    return state, config, versions


def prepare():
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("prepare_requires_instance_in_no_gpu_mode")
    if (OUT / "freeze-manifest.json").exists() or (OUT / "trajectories-new.jsonl").exists():
        raise FileExistsError("Goal_L_is_one_shot_no_overwrite")
    state, config, versions = expected_sources()
    regressions = cpu_contract_checks()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    sets = {"new_eval": read_rows(J_EVAL), "goal_G_regression": read_rows(G_EVAL)}
    if any(len(x) != 16 for x in sets.values()):
        raise ValueError("frozen_set_count_mismatch")
    historical = read_rows(J_TRACES)
    by_trace = {(x["split"], x["arm"], x["case_id"]): x for x in historical}
    if len(historical) != 96:
        raise ValueError("historical_trajectory_count_mismatch")

    prompts, gold_rows, p0_rows, cpu_results = [], [], [], []
    for split, rows in sets.items():
        for row in rows:
            check = cpu_check(processor, row)
            if check["prompt_sha256"] != row["cpu_check"]["prompt_sha256"]:
                raise ValueError(f"frozen_original_prompt_changed:{split}:{row['case_id']}")
            idx = row["target_assistant_index"]
            gold_rows.append({"split": split, "case_id": row["case_id"], "official_answer": row["official_answer"]})
            for policy, addition in (("P0", ""), ("P1", P1), ("P2", P1 + "\n\n" + P2)):
                messages = copy.deepcopy(row["messages"][:idx])
                if addition:
                    if messages[0]["role"] != "system":
                        raise ValueError("system_message_missing")
                    messages[0]["content"] += "\n\n" + addition
                ids, digest = prompt_ids(processor, messages)
                if len(ids) > 2048:
                    raise ValueError(f"prompt_over_2048:{split}:{row['case_id']}:{policy}")
                # Re-run the project's real assistant-only encoder on each variant.
                # This checks that modifying the system instruction did not break
                # the generation/supervision prefix, target location, EOS or limits.
                variant = copy.deepcopy(row)
                variant["messages"] = messages + [copy.deepcopy(row["messages"][idx])]
                check_variant = cpu_check(processor, variant)
                if check_variant["prompt_sha256"] != digest:
                    raise ValueError(f"variant_prefix_mismatch:{split}:{row['case_id']}:{policy}")
                if row["messages"][idx]["content"] in json.dumps(messages, ensure_ascii=False):
                    raise ValueError(f"target_answer_leaked:{split}:{row['case_id']}:{policy}")
                item = {"split": split, "case_id": row["case_id"], "policy": policy,
                        "messages": messages, "prompt_sha256": digest, "prompt_tokens": len(ids)}
                prompts.append(item)
                cpu_results.append({"split": split, "case_id": row["case_id"], "policy": policy,
                                    "prompt_sha256": digest, "prompt_tokens": len(ids), "status": "pass"})
                if policy == "P0":
                    for arm, hist_arm in (("v118", "v118_new_eval" if split == "new_eval" else "v118"),
                                          ("goal_J_sft", "goal_J_sft")):
                        old = by_trace.get((split, hist_arm, row["case_id"]))
                        if old is None or old["prompt_sha256"] != digest or old["gold"] != row["official_answer"]:
                            raise ValueError(f"historical_P0_binding_failed:{split}:{arm}:{row['case_id']}")
                        p0_rows.append({**old, "arm": arm, "policy": "P0", "historical_reused": True})
    if len(prompts) != 96 or len(p0_rows) != 64:
        raise ValueError("prompt_or_P0_count_mismatch")
    # Gold labels are kept in a distinct file inaccessible to generation workers.
    dump_rows(OUT / "prompts-frozen.jsonl", prompts)
    dump_rows(OUT / "scores-private.jsonl", gold_rows)
    dump_rows(OUT / "P0-history.jsonl", p0_rows)
    dump_rows(OUT / "cpu-checks.jsonl", cpu_results)
    code_paths = [Path(__file__).resolve(), ROOT / "scripts/run_v133_goal_g_cpu.py",
                  ROOT / "scripts/evaluate_v133_goal_c_sft.py"]
    manifest = {"goal": "L", "status": "cpu_ready", "cpu_ready": True, "gpu_required": True,
        "inference_complete": False, "review_complete": False, "new_generation_started": False,
        "training_started": False, "new_candidate_extraction_started": False,
        "sets": {k: len(v) for k, v in sets.items()}, "prompt_rows": len(prompts),
        "p0_reused_rows": len(p0_rows), "new_generation_planned": 128,
        "new_eval_is_exploratory_and_previously_exposed": True, "goal_G_is_exploratory_regression": True,
        "historical_gpu": state["gpu"], "historical_adapter_sha256": J_ADAPTER_SHA,
        "versions": versions, "generation_config": {"seed": 42, "dtype": "bfloat16", "eval_mode": True,
            "thinking": False, "do_sample": False, "max_new_tokens": 128},
        "cpu_regression_checks": regressions,
        "policies": {"P0": "historical unchanged prompt", "P1": P1, "P2_suffix": P2},
        "inputs": {str(p): sha(p) for p in (J_EVAL, G_EVAL, J_TRACES, J_CONFIG,
            V118 / "adapter_model.safetensors", J_ADAPTER / "adapter_model.safetensors",
            K_ATTRIBUTION, K_BINDINGS, K_DATAFLOW, K_REVIEW_STATUS)},
        "code_sha256": {str(p): sha(p) for p in code_paths},
        "prompts_sha256": sha(OUT / "prompts-frozen.jsonl"), "scores_private_sha256": sha(OUT / "scores-private.jsonl"),
        "P0_history_sha256": sha(OUT / "P0-history.jsonl"), "cpu_checks_sha256": sha(OUT / "cpu-checks.jsonl"),
        "unmet_conditions": []}
    dump(OUT / "freeze-manifest.json", manifest)
    dump(OUT / "runtime-config.json", {"adapter_paths": {"v118": str(V118), "goal_J_sft": str(J_ADAPTER)},
        "base_model": str(MODEL), "output_file": str(OUT / "trajectories-new.jsonl"), "hard_cap_seconds": CAP,
        "sequence": ["v118:P1", "v118:P2", "goal_J_sft:P1", "goal_J_sft:P2"],
        "rows_order": "new_eval then goal_G_regression, frozen file order", "no_training": True})
    dump(OUT / "run-state.json", {"status": "cpu_ready_waiting_for_gpu", "cpu_ready": True,
        "gpu_required": True, "inference_complete": False, "review_complete": False,
        "completed_new_generations": 0, "planned_new_generations": 128, "training_started": False})
    print(json.dumps({"status": "cpu_ready", "new_generation_planned": 128, "p0_reused": 64,
        "prompt_rows": len(prompts), "sets": manifest["sets"], "historical_gpu": state["gpu"],
        "max_prompt_tokens": max(x["prompt_tokens"] for x in cpu_results), "gpu_required": True}))


def load_frozen():
    manifest = json.loads((OUT / "freeze-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "cpu_ready" or manifest.get("new_generation_started") is not False:
        raise ValueError("CPU_freeze_gate_failed")
    for p, digest in manifest["inputs"].items():
        if sha(Path(p)) != digest:
            raise ValueError(f"input_changed_after_freeze:{p}")
    for p, digest in manifest["code_sha256"].items():
        if sha(Path(p)) != digest:
            raise ValueError(f"code_changed_after_freeze:{p}")
    for name, expected in (("prompts-frozen.jsonl", manifest["prompts_sha256"]),
                           ("scores-private.jsonl", manifest["scores_private_sha256"]),
                           ("P0-history.jsonl", manifest["P0_history_sha256"])):
        if sha(OUT / name) != expected:
            raise ValueError(f"frozen_artifact_changed:{name}")
    versions = {k: md.version(k) for k in EXPECTED}
    if versions != manifest["versions"]:
        raise ValueError(f"runtime_version_changed_after_freeze:{versions}")
    return manifest


def run():
    import torch
    manifest = load_frozen()
    if torch.cuda.is_available() is False:
        raise RuntimeError("GPU_mode_required_before_inference")
    if (OUT / "run-state.json").exists() and json.loads((OUT / "run-state.json").read_text()).get("status") != "cpu_ready_waiting_for_gpu":
        raise FileExistsError("Goal_L_one_shot_no_retry")
    try:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker"], cwd=ROOT,
                              timeout=CAP, check=False)
    except subprocess.TimeoutExpired:
        dump(OUT / "controller-stop.json", {"status": "hard_timeout", "seconds": CAP})
        raise SystemExit("Goal_L_hard_timeout_7200_seconds") from None
    if proc.returncode:
        raise SystemExit(f"Goal_L_worker_failed:{proc.returncode}")


def worker():
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.sft import build_lora_model
    manifest = load_frozen()
    if not torch.cuda.is_available():
        raise RuntimeError("authorized_GPU_unavailable")
    if (OUT / "trajectories-new.jsonl").exists():
        raise FileExistsError("Goal_L_one_shot_no_retry")
    rows = read_rows(OUT / "prompts-frozen.jsonl")
    state = {"status": "running", "cpu_ready": True, "gpu_required": True,
        "gpu": torch.cuda.get_device_name(0), "completed_new_generations": 0,
        "planned_new_generations": 128, "elapsed_seconds": 0, "speed_check": None,
        "training_started": False, "review_complete": False}
    if state["gpu"] != manifest["historical_gpu"]:
        raise RuntimeError("GPU_model_differs_from_historical; refusing mixed hardware run")
    (OUT / "run-state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    trajectories = OUT / "trajectories-new.jsonl"
    for arm, adapter in (("v118", V118), ("goal_J_sft", J_ADAPTER)):
        set_seed(42)
        model, processor = build_lora_model(str(MODEL), dtype="bfloat16", gradient_checkpointing=False,
                                            adapter_path=str(adapter))
        model.eval()
        active = model.active_adapters() if callable(getattr(model, "active_adapters", None)) else [model.active_adapter]
        modules = sum(1 for _, m in model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A)
        if "default" not in active or modules != 96 or next(model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError(f"adapter_activation_check_failed:{arm}")
        state.setdefault("adapter_loads", {})[arm] = {"active": active, "lora_modules": modules,
            "dtype": str(next(model.parameters()).dtype), "adapter_sha256": sha(adapter / "adapter_model.safetensors")}
        for policy in (("P1", "P2")):
            group = [r for r in rows if r["policy"] == policy]
            for row in group:
                if time.monotonic() - started >= CAP:
                    raise TimeoutError("hard_cap_before_generation")
                batch = processor.apply_chat_template(row["messages"], tokenize=True,
                    add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False)
                ids = batch["input_ids"][0].tolist()
                ph = hashlib.sha256(b"".join(int(i).to_bytes(4, "little") for i in ids)).hexdigest()
                if ph != row["prompt_sha256"]:
                    raise ValueError(f"runtime_prompt_hash_mismatch:{arm}:{policy}:{row['case_id']}")
                batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items() if hasattr(v, "to")}
                t0 = time.monotonic()
                with torch.inference_mode():
                    output = model.generate(**batch, do_sample=False, max_new_tokens=128,
                                            pad_token_id=processor.tokenizer.eos_token_id)
                elapsed_gen = time.monotonic() - t0
                raw = processor.tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                action_type, action = classify_action(raw)
                is_answer, is_search = action_type == "answer", action_type == "search"
                prediction = action["answer"] if is_answer else ""
                protocol_valid = action_type != "invalid"
                gold = next(x["official_answer"] for x in read_rows(OUT / "scores-private.jsonl")
                            if x["split"] == row["split"] and x["case_id"] == row["case_id"])
                rec = {"arm": arm, "split": row["split"], "case_id": row["case_id"], "policy": policy,
                    "prompt_sha256": ph, "prompt_tokens": len(ids), "raw_output": raw, "action": action,
                    "prediction": prediction, "gold": gold, "answer_valid": is_answer, "search_valid": is_search,
                    "protocol_valid": protocol_valid,
                    "em": int(is_answer and normalize_answer(prediction) == normalize_answer(gold)),
                    "f1": f1_score(prediction, gold) if is_answer else 0.0,
                    "generation_seconds": round(elapsed_gen, 3), "elapsed_seconds": round(time.monotonic()-started, 3),
                    "adapter_sha256": state["adapter_loads"][arm]["adapter_sha256"]}
                with trajectories.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())
                state["completed_new_generations"] += 1
                state["elapsed_seconds"] = round(time.monotonic()-started, 3)
                if state["completed_new_generations"] == 2:
                    first = read_rows(trajectories)
                    mean = sum(x["generation_seconds"] for x in first) / 2
                    projected = time.monotonic()-started + mean * (128-2) * 1.5 + 120
                    state["speed_check"] = {"after_two_formal_results": True, "mean_generation_seconds": mean,
                        "projected_total_seconds": round(projected, 2), "decision": "continue" if projected <= CAP else "stop"}
                    if projected > CAP:
                        state["status"] = "stopped_speed_projection_over_cap"
                        break
                (OUT / "run-state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            if state.get("status", "running") != "running":
                break
        del model, processor
        gc.collect(); torch.cuda.empty_cache()
        if state.get("status", "running") != "running":
            break
    if state.get("status") == "running":
        state["status"] = "inference_complete"
        state["inference_complete"] = True
    state["elapsed_seconds"] = round(time.monotonic()-started, 3)
    (OUT / "run-state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def summarize():
    manifest = load_frozen()
    new = read_rows(OUT / "trajectories-new.jsonl")
    p0 = read_rows(OUT / "P0-history.jsonl")
    gold = {(r["split"], r["case_id"]): r["official_answer"] for r in read_rows(OUT / "scores-private.jsonl")}
    all_rows = p0 + new
    for row in all_rows:
        action_type, parsed = classify_action(row["raw_output"])
        row["action_type"] = action_type
        row["answer_valid"] = action_type == "answer"
        row["search_valid"] = action_type == "search"
        row["protocol_valid"] = action_type != "invalid"
        row["prediction"] = parsed["answer"] if action_type == "answer" else ""
        row["em"] = int(action_type == "answer" and normalize_answer(row["prediction"]) == normalize_answer(row["gold"]))
        row["f1"] = f1_score(row["prediction"], row["gold"]) if action_type == "answer" else 0.0
    expected = {(s, c, a, p) for s in ("new_eval", "goal_G_regression")
                for c in (r["case_id"] for r in read_rows(J_EVAL if s == "new_eval" else G_EVAL))
                for a in ("v118", "goal_J_sft") for p in ("P0", "P1", "P2")}
    actual = {(r["split"], r["case_id"], r["arm"], r["policy"]) for r in all_rows}
    unique_subset(list(actual), expected, "goal_L_results")
    if len(actual) != len(all_rows):
        raise ValueError("duplicate_or_unexpected_result_pair")
    results = {}
    for split in ("new_eval", "goal_G_regression"):
        results[split] = {}
        for arm in ("v118", "goal_J_sft"):
            results[split][arm] = {}
            for pol in ("P0", "P1", "P2"):
                rr = [r for r in all_rows if (r["split"], r["arm"], r["policy"]) == (split, arm, pol)]
                if len(rr) != 16:
                    continue
                results[split][arm][pol] = {"n": 16, "strict_em": sum(r["em"] for r in rr),
                    "mean_f1": sum(r["f1"] for r in rr)/16, "answer_valid": sum(r["answer_valid"] for r in rr),
                    "search_valid": sum(r["search_valid"] for r in rr), "invalid": sum(not r["protocol_valid"] for r in rr)}
        results[split]["paired_changes"] = {}
        for arm in ("v118", "goal_J_sft"):
            for pbefore, pafter in (("P0", "P1"), ("P1", "P2")):
                left = {(r["case_id"]): r for r in all_rows if r["split"] == split and r["arm"] == arm and r["policy"] == pbefore}
                right = {(r["case_id"]): r for r in all_rows if r["split"] == split and r["arm"] == arm and r["policy"] == pafter}
                common = left.keys() & right.keys()
                if len(common) == 16:
                    results[split]["paired_changes"][f"{arm}:{pbefore}->{pafter}"] = {
                        "new_correct": sorted(c for c in common if not left[c]["em"] and right[c]["em"]),
                        "lost_correct": sorted(c for c in common if left[c]["em"] and not right[c]["em"]),
                        "retained_correct": sorted(c for c in common if left[c]["em"] and right[c]["em"]),
                        "action_transitions": {f"{x}->{y}": sum(left[c]["action_type"] == x and right[c]["action_type"] == y for c in common)
                            for x in ("answer", "search", "invalid") for y in ("answer", "search", "invalid")
                            if any(left[c]["action_type"] == x and right[c]["action_type"] == y for c in common)}}
        results[split]["model_comparisons"] = {}
        for policy in ("P1", "P2"):
            left = {r["case_id"]: r for r in all_rows if r["split"] == split and r["arm"] == "v118" and r["policy"] == policy}
            right = {r["case_id"]: r for r in all_rows if r["split"] == split and r["arm"] == "goal_J_sft" and r["policy"] == policy}
            common = left.keys() & right.keys()
            if len(common) == 16:
                results[split]["model_comparisons"][policy] = {
                    "v118_minus_goal_J_em": sum(left[c]["em"] for c in common) - sum(right[c]["em"] for c in common),
                    "goal_J_minus_v118_mean_f1": sum(right[c]["f1"]-left[c]["f1"] for c in common)/16,
                    "new_correct": sorted(c for c in common if not left[c]["em"] and right[c]["em"]),
                    "lost_correct": sorted(c for c in common if left[c]["em"] and not right[c]["em"]),
                    "retained_correct": sorted(c for c in common if left[c]["em"] and right[c]["em"])}
    result = {"status": json.loads((OUT / "run-state.json").read_text())["status"], "sets": results,
        "comparisons_preregistered": ["P1-P0 response timing", "P2-P1 answer granularity", "Goal-J minus v118 in P1/P2"],
        "history_P0_reused": True, "historical_P0_rows": len(p0), "new_generation_rows": len(new),
        "complete": len(new) == 128 and len(actual) == len(expected), "training_started": False,
        "new_candidate_extraction_started": False, "final_test_accessed": False, "sealed_test_accessed": False,
        "unmet_conditions": [] if len(new) == 128 else ["incomplete_generation_or_pairs"]}
    packets, private_map = build_review_packets(all_rows)
    dump(OUT / "review-packets-anonymous.json", packets)
    dump(OUT / "review-alias-map-private.json", private_map)
    dump(OUT / "review-status.json", {"max_cases": 8, "selected_cases": len(packets["packets"]),
        "complete": False, "reason": "awaiting bounded independent reviews after GPU inference",
        "deepseek_retry_of_goal_K_unknown_request": False})
    result["review_packet_count"] = len(packets["packets"])
    result["review_complete"] = False
    dump(OUT / "summary.json", result)
    print(json.dumps({"status": result["status"], "new_generation_rows": len(new), "complete": result["complete"],
                      "primary": results.get("new_eval", {})}, ensure_ascii=False))


def build_review_packets(all_rows):
    """Select at most eight changed answer candidates by the frozen priority order."""
    by = {(r["split"], r["case_id"], r["arm"], r["policy"]): r for r in all_rows}
    candidates = []
    def add(before, after, priority, reason):
        if before is None or after is None or before["raw_output"] == after["raw_output"]:
            return
        if after["action_type"] != "answer":
            return
        try:
            action = {"action": "answer", "answer": after["prediction"]}
        except KeyError:
            return
        candidates.append({"priority": priority, "reason": reason, "before": before,
                           "after": after, "action": action})
    def transition(split, arm, p0, p1, priority_loss, priority_gain, label):
        left = {r["case_id"]: r for r in all_rows if r["split"] == split and r["arm"] == arm and r["policy"] == p0}
        right = {r["case_id"]: r for r in all_rows if r["split"] == split and r["arm"] == arm and r["policy"] == p1}
        for cid in left.keys() & right.keys():
            if left[cid]["em"] and not right[cid]["em"]:
                add(left[cid], right[cid], priority_loss, label + ":strict_loss")
            elif not left[cid]["em"] and right[cid]["em"]:
                add(left[cid], right[cid], priority_gain, label + ":strict_gain")
            elif left[cid]["prediction"] != right[cid]["prediction"]:
                add(left[cid], right[cid], 3, label + ":answer_changed")
    for arm in ("v118", "goal_J_sft"):
        transition("new_eval", arm, "P0", "P1", 0, 1, f"new_eval:{arm}:P0-P1")
        transition("new_eval", arm, "P1", "P2", 0, 1, f"new_eval:{arm}:P1-P2")
        transition("goal_G_regression", arm, "P0", "P1", 2, 3, f"goal_G:{arm}:P0-P1")
        transition("goal_G_regression", arm, "P1", "P2", 2, 3, f"goal_G:{arm}:P1-P2")
    for split in ("new_eval", "goal_G_regression"):
        for policy in ("P1", "P2"):
            a = {r["case_id"]: r for r in all_rows if r["split"] == split and r["arm"] == "v118" and r["policy"] == policy}
            b = {r["case_id"]: r for r in all_rows if r["split"] == split and r["arm"] == "goal_J_sft" and r["policy"] == policy}
            for cid in a.keys() & b.keys():
                if split == "new_eval" and a[cid]["em"] and not b[cid]["em"]:
                    add(a[cid], b[cid], 0, f"new_eval:model:{policy}:strict_loss")
                elif split == "new_eval" and not a[cid]["em"] and b[cid]["em"]:
                    add(a[cid], b[cid], 1, f"new_eval:model:{policy}:strict_gain")
                elif split == "goal_G_regression" and a[cid]["em"] and not b[cid]["em"]:
                    add(a[cid], b[cid], 2, f"goal_G:model:{policy}:strict_loss")
                elif a[cid]["prediction"] != b[cid]["prediction"]:
                    add(a[cid], b[cid], 3, f"{split}:model:{policy}:answer_changed")
    dedup = {}
    for c in candidates:
        key = (c["after"]["split"], c["after"]["prompt_sha256"],
               json.dumps(c["action"], sort_keys=True, ensure_ascii=False))
        if key not in dedup or (c["priority"], c["after"]["case_id"], c["after"]["policy"], c["after"]["arm"]) < (
                dedup[key]["priority"], dedup[key]["after"]["case_id"], dedup[key]["after"]["policy"], dedup[key]["after"]["arm"]):
            dedup[key] = c
    selected = sorted(dedup.values(), key=lambda c: (c["priority"], c["after"]["case_id"],
        c["after"]["policy"], hashlib.sha256(c["action"]["answer"].encode()).hexdigest()))[:8]
    prompt_by = {(x["split"], x["case_id"], x["policy"]): x for x in read_rows(OUT / "prompts-frozen.jsonl")}
    packets, aliases = [], []
    for i, c in enumerate(selected, 1):
        after = c["after"]
        prompt = prompt_by[(after["split"], after["case_id"], after["policy"])]["messages"]
        docs = []
        for msg in prompt:
            if msg["role"] != "tool":
                continue
            blob = json.loads(msg["content"])
            if blob.get("ok") is True:
                docs.extend(blob.get("result", []))
        before_docs = [{"id": x["id"], "title": x.get("title", ""), "text": x["text"]} for x in docs]
        alias = f"L-review-{i:02d}"
        packets.append({"case_id": alias, "question": prompt[1]["content"],
            "visible_messages": prompt, "before_action": before_docs, "retrieved_after_search": [],
            "proposed_action": c["action"], "protocol": {"version": "v130-evidence-review-citations-v1"},
            "review_instructions": "Judge only this candidate against the question and complete visible context. Assess support for both entities, requested property, and necessary answer granularity. Cite exact short text from each supporting document. Do not infer model or condition identity."})
        aliases.append({"case_id": alias, "split": after["split"], "source_case_id": after["case_id"],
            "arm": after["arm"], "policy": after["policy"], "priority": c["priority"], "selection_reason": c["reason"],
            "prediction": after["prediction"], "candidate_context_sha256": after["prompt_sha256"]})
    contract = {"training_allowed": False, "review_authorized": True,
        "max_cases": 8, "packets": packets}
    payload = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    contract["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    return contract, {"aliases": aliases, "condition_identity_disclosed": False,
        "review_priority": ["new-evaluation strict losses", "new-evaluation strict gains",
                            "Goal G strict losses", "other decision-driving answer changes"]}


def cpu_contract_checks():
    assert classify_action('{"action":"search","query":"x"}')[0] == "search"
    assert classify_action('{"action":"answer","answer":"x"}')[0] == "answer"
    for bad in ('not json', '{"action":"search","query":"x","extra":1}',
                '{"action":"answer","answer":"x","reason":"extra"}'):
        assert classify_action(bad)[0] == "invalid"
    try:
        require_sha("bad", "good", "fixture")
    except ValueError:
        pass
    else:
        raise AssertionError("hash_mismatch_not_rejected")
    try:
        unique_subset(["x", "x"], {"x"}, "fixture")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate_pair_not_rejected")
    unique_subset(["x"], {"x"}, "fixture")
    return {"legal_search_separate_from_invalid_json": True, "hash_mismatch_rejected": True,
            "duplicate_pair_rejected": True, "strict_EM_scoring_unchanged": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("prepare", "run", "worker", "summarize", "cpu-check"))
    phase = ap.parse_args().phase
    {"prepare": prepare, "run": run, "worker": worker, "summarize": summarize,
     "cpu-check": lambda: print(json.dumps(cpu_contract_checks(), ensure_ascii=False))}[phase]()
