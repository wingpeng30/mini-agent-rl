"""Fail-closed one-shot continuation of the existing v133 SFT adapter."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/v133-goal-d-sft-incremental.json"
OUTPUT = ROOT / "runs/v133-goal-d-sft-once/adapter"
RUN_DIR = ROOT / "runs/v133-goal-d-sft-once/autodl-run-20260928"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--cpu-only-preflight", action="store_true")
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    if not args.authorize_training and not args.cpu_only_preflight:
        raise SystemExit("explicit_training_authorization_required")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = Path(args.model_path).resolve()
    train = ROOT / cfg["train_data"]
    dev = ROOT / cfg["dev_data"]
    init_adapter = ROOT / cfg["initial_adapter"]
    v118_adapter = ROOT / cfg["v118_adapter"]
    prior_eval = ROOT / cfg["prior_aligned_eval"]
    prior_training = ROOT / "runs/v133-goal-c-sft-once/autodl-run-20260928/training-start.json"
    if OUTPUT.exists() or RUN_DIR.joinpath("training-start.json").exists():
        raise SystemExit("one_shot_output_or_marker_exists_no_retry")
    checks = {
        "train": (train, cfg["train_sha256"]),
        "dev": (dev, cfg["dev_sha256"]),
        "initial_adapter": (init_adapter / "adapter_model.safetensors", cfg["initial_adapter_sha256"]),
        "v118_adapter": (v118_adapter / "adapter_model.safetensors", cfg["v118_adapter_sha256"]),
        "prior_aligned_eval": (prior_eval, cfg["prior_aligned_eval_sha256"]),
    }
    for name, (path, expected) in checks.items():
        if sha(path) != expected:
            raise SystemExit(f"frozen_hash_mismatch:{name}")
    prior_state = json.loads(prior_training.read_text(encoding="utf-8"))
    if prior_state.get("status") != "complete" or prior_state.get("global_step") != 7:
        raise SystemExit("initial_sft_not_complete")
    prior_metrics = json.loads((init_adapter / "training_metrics.json").read_text(encoding="utf-8"))
    prior_eval_state = json.loads(prior_eval.read_text(encoding="utf-8"))
    if (prior_metrics.get("global_step") != 7
            or prior_eval_state.get("status") != "complete"
            or prior_eval_state.get("paired_summary", {}).get("protocol_valid_sft") != 16):
        raise SystemExit("initial_sft_or_aligned_eval_mismatch")
    for name, expected in prior_state.get("model_runtime_files_sha256", {}).items():
        if sha(model / name) != expected:
            raise SystemExit(f"model_runtime_hash_mismatch:{name}")
    # The original frozen preflight records weight and tokenizer hashes.
    old_preflight = json.loads((ROOT / "runs/v133-goal-c-sft-once/autodl-run-20260928/preflight.json").read_text(encoding="utf-8"))
    if old_preflight.get("status") != "passed_cpu_only_preflight":
        raise SystemExit("original_asset_preflight_missing")
    frozen_path = ROOT / "data/evidence-sft-v133-goal-c-private/frozen/freeze-manifest-private.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    for group in ("model_weight_files_sha256", "tokenizer_sha256", "model_runtime_files_sha256"):
        for name, expected in frozen["baseline"][group].items():
            if sha(model / name) != expected:
                raise SystemExit(f"frozen_model_asset_mismatch:{name}")
    versions = {name: metadata.version(name) for name in cfg["runtime_dependencies"]}
    if versions != cfg["runtime_dependencies"]:
        raise SystemExit("runtime_dependency_mismatch")

    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA_or_BF16_unavailable_no_cpu_fallback")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = torch.cuda.mem_get_info(0)[0] / 1024**3
    floor = cfg["resource_limits"]
    if total < floor["minimum_gpu_vram_gib_for_planning"] or free < floor["minimum_free_gpu_vram_gib_at_start"]:
        raise SystemExit(f"GPU_memory_floor_failed:total={total:.3f},free={free:.3f}")

    if args.cpu_only_preflight:
        from transformers import AutoProcessor
        sys.path.insert(0, str(ROOT / "src"))
        from mini_agent_rl.training.sft import encode_assistant_only, load_jsonl
        processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        train_rows = load_jsonl(train)
        dev_rows = load_jsonl(dev)
        max_length = cfg["training"]["max_length"]
        train_encodings = [encode_assistant_only(processor, row["messages"], max_length,
                                                  row.get("target_assistant_index"),
                                                  bool(row.get("thinking_enabled", False)))
                           for row in train_rows]
        dev_encodings = [encode_assistant_only(processor, row["messages"], max_length,
                                                row.get("target_assistant_index"),
                                                bool(row.get("thinking_enabled", False)))
                         for row in dev_rows]
        print(json.dumps({"status": "cpu_only_training_preflight_passed",
                          "train_rows": len(train_encodings), "dev_rows": len(dev_encodings),
                          "supervised_train_tokens": sum(sum(x != -100 for x in r["labels"]) for r in train_encodings),
                          "supervised_dev_tokens": sum(sum(x != -100 for x in r["labels"]) for r in dev_encodings),
                          "gpu_name": torch.cuda.get_device_name(0), "gpu_total_gib": round(total, 3),
                          "gpu_free_gib": round(free, 3), "runtime_versions": versions,
                          "train_sha256": sha(train), "dev_sha256": sha(dev),
                          "parent_adapter_sha256": sha(init_adapter / "adapter_model.safetensors")}, indent=2))
        return

    if not args.authorize_training:
        raise SystemExit("explicit_training_authorization_required")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    start = {
        "status": "starting", "config_sha256": sha(CONFIG_PATH),
        "runner_sha256": sha(Path(__file__)), "train_sha256": sha(train), "dev_sha256": sha(dev),
        "parent_adapter_sha256": sha(init_adapter / "adapter_model.safetensors"),
        "v118_adapter_sha256": sha(v118_adapter / "adapter_model.safetensors"),
        "prior_aligned_eval_sha256": sha(prior_eval), "model_path": str(model),
        "runtime_versions": versions, "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_gib": round(total, 3), "gpu_free_gib_before_load": round(free, 3),
        "seed": cfg["training"]["seed"], "max_steps": cfg["training"]["max_steps"],
        "learning_rate": cfg["training"]["learning_rate"],
    }
    marker = RUN_DIR / "training-start.json"
    marker.write_text(json.dumps(start, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.path.insert(0, str(ROOT / "src"))
    from mini_agent_rl.training.sft import train_sft
    t = cfg["training"]
    try:
        metrics = train_sft(
            model_path=str(model), train_path=str(train), validation_path=str(dev),
            output_dir=str(OUTPUT), max_steps=t["max_steps"], epochs=t["epochs"],
            max_length=t["max_length"], seed=t["seed"], batch_size=t["batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            gradient_checkpointing=t["gradient_checkpointing"], adapter_path=str(init_adapter),
            enable_thinking=t["enable_thinking"], learning_rate=t["learning_rate"],
            save_steps=t["save_steps"], eval_steps=t["eval_steps"],
            warmup_ratio=t["warmup_steps"] / t["max_steps"],
            wall_clock_cap_seconds=t["wall_clock_cap_seconds"],
        )
        if metrics["global_step"] != t["max_steps"] or metrics["speed_check"].get("status") != "within_cap":
            raise RuntimeError("training_incomplete_or_speed_gate_failed_no_retry")
        start.update({"status": "complete", "global_step": metrics["global_step"],
                      "speed_check": metrics["speed_check"],
                      "metrics_sha256": sha(OUTPUT / "training_metrics.json"),
                      "adapter_sha256": sha(OUTPUT / "adapter_model.safetensors")})
        marker.write_text(json.dumps(start, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(start, ensure_ascii=False, indent=2))
    except BaseException as exc:
        start.update({"status": "failed_no_retry", "error_type": type(exc).__name__, "error": str(exc)[:500]})
        marker.write_text(json.dumps(start, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
