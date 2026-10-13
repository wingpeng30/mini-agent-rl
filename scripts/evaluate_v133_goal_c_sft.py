"""Paired v118-vs-SFT answer-only development evaluation.

This script is intentionally inference-capable but fail-closed: it requires an
explicit authorization flag and writes incrementally. It was not run during
CPU-only Goal C preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import re
import string
import time
import unicodedata
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(gold).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum(min(pred.count(token), ref.count(token)) for token in set(pred))
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def parse_action(raw: str) -> dict:
    value = json.loads(raw.strip())
    if not isinstance(value, dict) or value.get("action") != "answer" or not isinstance(value.get("answer"), str):
        raise ValueError("invalid_answer_action")
    if set(value) != {"action", "answer"}:
        raise ValueError("unexpected_answer_action_fields")
    return value


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 16:
        raise ValueError(f"expected_16_dev_rows:{len(rows)}")
    return rows


def evaluate_adapter(model_path: str, adapter_path: str, rows: list[dict], arm: str,
                     state: dict, output_path: Path, started: float, max_seconds: int) -> None:
    import torch
    from mini_agent_rl.training.sft import build_lora_model
    from transformers import set_seed

    load_started = time.monotonic()
    set_seed(42)
    model, processor = build_lora_model(model_path, dtype="bfloat16", gradient_checkpointing=False,
                                        adapter_path=adapter_path)
    load_seconds = time.monotonic() - load_started
    state.setdefault("model_load_seconds", {})[arm] = round(load_seconds, 3)
    model.eval()
    tokenizer = processor.tokenizer
    for index, row in enumerate(rows):
        key = f"{arm}:{index:02d}"
        if key in state["results"]:
            continue
        if time.monotonic() - started >= max_seconds:
            state["stop_reason"] = "wall_clock_cap_between_examples"
            break
        target_index = int(row["target_assistant_index"])
        prompt_messages = row["messages"][:target_index]
        encoded = processor.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        )
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        if prompt_tokens > 2048:
            raise ValueError(f"prompt_token_limit:{arm}:{index}:{prompt_tokens}")
        device = next(model.parameters()).device
        inputs = {name: value.to(device) for name, value in encoded.items() if hasattr(value, "to")}
        inference_started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=128,
                                       pad_token_id=tokenizer.eos_token_id)
        generation_seconds = time.monotonic() - inference_started
        raw = tokenizer.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        gold_action = json.loads(row["messages"][target_index]["content"])
        gold = str(gold_action["answer"])
        protocol_valid, parse_error, prediction = True, None, ""
        try:
            prediction = parse_action(raw)["answer"]
        except Exception as exc:
            protocol_valid, parse_error = False, type(exc).__name__
        state["results"][key] = {
            "arm": arm, "row_index": index, "question_sha256": hashlib.sha256(row["messages"][1]["content"].encode("utf-8")).hexdigest(),
            "prompt_tokens": prompt_tokens, "raw_output": raw, "protocol_valid": protocol_valid,
            "parse_error": parse_error, "prediction": prediction, "gold": gold,
            "em": int(protocol_valid and normalize_answer(prediction) == normalize_answer(gold)),
            "f1": f1_score(prediction, gold) if protocol_valid else 0.0,
            "generation_seconds": round(generation_seconds, 3),
        }
        if arm == "v118" and index == 1:
            first_two = [state["results"][f"v118:{i:02d}"]["generation_seconds"] for i in (0, 1)]
            mean_generation = sum(first_two) / 2
            elapsed = time.monotonic() - started
            projected_total = elapsed + mean_generation * 30 + load_seconds
            state["speed_check"] = {
                "after_completed_outputs": 2,
                "mean_generation_seconds": round(mean_generation, 3),
                "elapsed_seconds": round(elapsed, 3),
                "projected_total_seconds": round(projected_total, 3),
                "decision": "within_cap" if projected_total <= max_seconds else "projected_over_cap",
            }
            if projected_total > max_seconds:
                state["stop_reason"] = "early_speed_projection_exceeds_limit"
        output_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if state["stop_reason"] is not None:
            break
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_gpu_floor(minimum_total_gib: float, minimum_free_gib: float) -> None:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; no CPU fallback is permitted for this run")
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free_gb = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
    if total_gb < minimum_total_gib or free_gb < minimum_free_gib:
        raise SystemExit(f"GPU memory total={total_gb:.3f} GiB/free={free_gb:.3f} GiB below frozen floors")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=r"D:\qwen_08b")
    parser.add_argument("--base-adapter", required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--dev-data", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-minutes", type=int, default=60)
    parser.add_argument("--authorize-model-inference", action="store_true")
    args = parser.parse_args()
    if not args.authorize_model_inference:
        raise SystemExit("model inference requires separate explicit user authorization")
    if args.max_minutes <= 0 or args.max_minutes > 60:
        raise SystemExit("max-minutes must be in 1..60")
    dev_path, out_path = Path(args.dev_data), Path(args.output)
    if not dev_path.is_file() or out_path.exists():
        raise SystemExit("missing_dev_data_or_output_already_exists")
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/v133-goal-c-small-sft.json"
    freeze_path = root / "data/evidence-sft-v133-goal-c-private/frozen/freeze-manifest-private.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    run_dir = root / "runs/v133-goal-c-sft-once/autodl-run-20260928"
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    training_start = json.loads((run_dir / "training-start.json").read_text(encoding="utf-8"))
    expected_versions = config["runtime_dependencies"]
    actual_versions = {name: metadata.version(name) for name in expected_versions}
    if actual_versions != expected_versions:
        raise SystemExit("runtime_package_version_mismatch")
    if (training_start.get("status") != "complete"
            or training_start.get("global_step") != config["training"]["max_steps"]
            or training_start.get("speed_check", {}).get("status") != "within_cap"
            or preflight.get("status") != "passed_cpu_only_preflight"):
        raise SystemExit("training_or_preflight_not_complete")
    training_report = json.loads(Path(args.training_report).read_text(encoding="utf-8"))
    model_dir = Path(args.model_path).resolve()
    for name, expected in frozen["baseline"]["tokenizer_sha256"].items():
        if sha(model_dir / name) != expected:
            raise SystemExit(f"tokenizer_asset_hash_mismatch:{name}")
    for name, expected in frozen["baseline"]["model_weight_files_sha256"].items():
        if sha(model_dir / name) != expected:
            raise SystemExit(f"base_model_weight_hash_mismatch:{name}")
    for name, expected in frozen["baseline"]["model_runtime_files_sha256"].items():
        if sha(model_dir / name) != expected:
            raise SystemExit(f"base_model_runtime_asset_hash_mismatch:{name}")
    for name, expected in frozen["baseline"]["v118_adapter_files_sha256"].items():
        if sha(Path(args.base_adapter) / name) != expected:
            raise SystemExit(f"v118_adapter_asset_hash_mismatch:{name}")
    base_hash = sha(Path(args.base_adapter) / "adapter_model.safetensors")
    sft_hash = sha(Path(args.sft_adapter) / "adapter_model.safetensors")
    if base_hash != config["base_adapter_sha256"] or base_hash != frozen["baseline"]["v118_adapter_sha256"]:
        raise SystemExit("v118_adapter_hash_mismatch")
    if sft_hash == base_hash or training_report.get("global_step") != config["training"]["max_steps"]:
        raise SystemExit("missing_or_unexpected_single_sft_checkpoint")
    if training_report.get("train_samples") != 25 or training_report.get("validation_samples") != 16:
        raise SystemExit("training_report_data_count_mismatch")
    expected_base = Path(args.base_adapter).resolve()
    if Path(training_report.get("adapter_path") or "").resolve() != expected_base:
        raise SystemExit("training_report_base_adapter_mismatch")
    if (training_report.get("learning_rate") != config["training"]["learning_rate"]
            or training_report.get("seed") != config["training"]["seed"]
            or training_report.get("max_length") != config["training"]["max_length"]
            or training_report.get("gradient_accumulation_steps") != config["training"]["gradient_accumulation_steps"]):
        raise SystemExit("training_report_config_mismatch")
    if sha(dev_path) != frozen["splits"]["dev_sha256"]:
        raise SystemExit("frozen_dev_hash_mismatch")
    rows = load_rows(dev_path)
    check_gpu_floor(float(config["resource_limits"]["minimum_gpu_vram_gib_for_planning"]),
                    float(config["resource_limits"]["minimum_free_gpu_vram_gib_at_start"]))
    state = {"version": "v133-goal-c-paired-dev-eval-v1", "dev_sha256": sha(dev_path),
             "config_sha256": sha(config_path), "freeze_manifest_sha256": sha(freeze_path),
             "base_model_path": str(model_dir),
             "base_model_weight_files_sha256": frozen["baseline"]["model_weight_files_sha256"],
             "base_adapter": str(Path(args.base_adapter)), "base_adapter_sha256": base_hash,
             "v118_adapter_files_sha256": frozen["baseline"]["v118_adapter_files_sha256"],
             "sft_adapter": str(Path(args.sft_adapter)), "sft_adapter_sha256": sft_hash,
             "training_report_sha256": sha(Path(args.training_report)),
             "status": "running", "results": {}, "stop_reason": None, "speed_check": None,
             "changed_answer_support_review": "not performed; requires separate blind review authorization"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cap = args.max_minutes * 60
    evaluate_adapter(args.model_path, args.base_adapter, rows, "v118", state, out_path, started, cap)
    if state["stop_reason"] is None:
        evaluate_adapter(args.model_path, args.sft_adapter, rows, "sft", state, out_path, started, cap)
    base = [state["results"].get(f"v118:{i:02d}") for i in range(16)]
    tuned = [state["results"].get(f"sft:{i:02d}") for i in range(16)]
    complete = all(x is not None for x in base + tuned)
    state["paired_summary"] = None
    if complete:
        state["paired_summary"] = {
            "n": 16,
            "v118_em": sum(x["em"] for x in base), "sft_em": sum(x["em"] for x in tuned),
            "v118_mean_f1": sum(x["f1"] for x in base) / 16, "sft_mean_f1": sum(x["f1"] for x in tuned) / 16,
            "new_correct": [i for i, (a, b) in enumerate(zip(base, tuned)) if not a["em"] and b["em"]],
            "lost_correct": [i for i, (a, b) in enumerate(zip(base, tuned)) if a["em"] and not b["em"]],
            "changed_answer_indices": [i for i, (a, b) in enumerate(zip(base, tuned)) if a["prediction"] != b["prediction"]],
            "protocol_valid_v118": sum(x["protocol_valid"] for x in base),
            "protocol_valid_sft": sum(x["protocol_valid"] for x in tuned),
            "promotion_eligible_pending_support_review": sum(b["em"] for b in tuned) - sum(a["em"] for a in base) >= 2,
        }
    state["status"] = "complete" if complete else "stopped_or_partial"
    state["elapsed_seconds"] = round(time.monotonic() - started, 3)
    out_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": state["status"], "elapsed_seconds": state["elapsed_seconds"], "paired_summary": state["paired_summary"], "stop_reason": state["stop_reason"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
