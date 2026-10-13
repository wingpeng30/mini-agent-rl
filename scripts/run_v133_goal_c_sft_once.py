"""Fail-closed one-shot runner for the frozen v133 Goal C SFT plan."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v133-goal-c-small-sft.json"
FREEZE = ROOT / "data/evidence-sft-v133-goal-c-private/frozen/freeze-manifest-private.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--model-path", required=True, help="local model directory on the selected host")
    args = parser.parse_args()
    if not args.authorize_training:
        raise SystemExit("SFT requires separate explicit user authorization")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if sha(CONFIG) != frozen["baseline"]["config_sha256"]:
        raise SystemExit("frozen_config_hash_mismatch")
    run_dir = ROOT / "runs/v133-goal-c-sft-once/autodl-run-20260928"
    preflight_path = run_dir / "preflight.json"
    start_path = run_dir / "training-start.json"
    if not preflight_path.is_file() or start_path.exists():
        raise SystemExit("missing_preflight_or_this_one_shot_already_started")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (preflight.get("status") != "passed_cpu_only_preflight"
            or preflight.get("config_sha256") != sha(CONFIG)
            or preflight.get("freeze_manifest_sha256") != sha(FREEZE)
            or Path(preflight.get("model_path", "")).resolve() != Path(args.model_path).resolve()):
        raise SystemExit("preflight_scope_or_hash_mismatch")
    if sha(ROOT / config["train_data"]) != frozen["splits"]["train_sha256"]:
        raise SystemExit("frozen_train_hash_mismatch")
    if sha(ROOT / config["dev_data"]) != frozen["splits"]["dev_sha256"]:
        raise SystemExit("frozen_dev_hash_mismatch")
    adapter = Path(config["base_adapter"])
    if not adapter.is_absolute():
        adapter = ROOT / adapter
    if sha(adapter / "adapter_model.safetensors") != config["base_adapter_sha256"]:
        raise SystemExit("v118_adapter_hash_mismatch")
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
        if sha(adapter / name) != expected:
            raise SystemExit(f"v118_adapter_asset_hash_mismatch:{name}")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; refusing CPU fallback")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free_gb = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
    floor = config["resource_limits"]["minimum_gpu_vram_gib_for_planning"]
    free_floor = config["resource_limits"]["minimum_free_gpu_vram_gib_at_start"]
    if vram_gb < floor or free_gb < free_floor or not torch.cuda.is_bf16_supported():
        raise SystemExit(f"GPU preflight failed: total={vram_gb:.3f} GiB, free={free_gb:.3f} GiB, BF16={torch.cuda.is_bf16_supported()}")
    runtime_versions = {name: metadata.version(name) for name in config["runtime_dependencies"]}
    if runtime_versions != config["runtime_dependencies"]:
        raise SystemExit(f"runtime_package_version_mismatch:{runtime_versions}")

    output = ROOT / "runs/v133-goal-c-sft-once/adapter"
    if output.exists():
        raise SystemExit("output_exists; automatic overwrite or retry is forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    record = {
        "status": "starting",
        "config_sha256": sha(CONFIG), "freeze_manifest_sha256": sha(FREEZE),
        "train_sha256": sha(ROOT / config["train_data"]), "dev_sha256": sha(ROOT / config["dev_data"]),
        "model_path": str(model_dir), "runtime_versions": runtime_versions,
        "gpu_name": torch.cuda.get_device_name(0), "gpu_total_gib": round(vram_gb, 3),
        "gpu_free_gib_before_load": round(free_gb, 3), "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    start_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source = f'''from mini_agent_rl.training.sft import train_sft
train_sft(
    model_path={str(model_dir)!r}, adapter_path={str(adapter)!r},
    train_path={str(ROOT / config["train_data"])!r},
    validation_path={str(ROOT / config["dev_data"])!r}, output_dir={str(output)!r},
    max_steps={training["max_steps"]}, epochs={training["epochs"]},
    max_length={training["max_length"]}, seed={training["seed"]},
    batch_size={training["batch_size"]},
    gradient_accumulation_steps={training["gradient_accumulation_steps"]},
    gradient_checkpointing={training["gradient_checkpointing"]!r},
    enable_thinking={training["enable_thinking"]!r},
    learning_rate={training["learning_rate"]}, save_steps={training["save_steps"]},
    eval_steps={training["eval_steps"]}, warmup_ratio={training["warmup_ratio"]!r},
    wall_clock_cap_seconds={config["resource_limits"]["wall_clock_cap_minutes"] * 60},
)
'''
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        subprocess.run([sys.executable, "-c", source], cwd=ROOT, env=env, check=True,
                       timeout=config["resource_limits"]["wall_clock_cap_minutes"] * 60)
    except subprocess.TimeoutExpired as exc:
        record["status"] = "stopped_at_wall_clock_cap"
        start_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("60-minute hard limit reached; worker terminated, no retry") from exc
    except subprocess.CalledProcessError as exc:
        record["status"] = "worker_failed_no_retry"
        record["return_code"] = exc.returncode
        start_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("SFT worker failed; no retry") from exc

    metrics_path = output / "training_metrics.json"
    if not metrics_path.is_file():
        raise SystemExit("training returned without training_metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    record["metrics_sha256"] = sha(metrics_path)
    record["global_step"] = metrics.get("global_step")
    record["speed_check"] = metrics.get("speed_check")
    record["status"] = "complete" if metrics.get("global_step") == training["max_steps"] else "incomplete_no_retry"
    start_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if metrics.get("global_step") != training["max_steps"]:
        raise SystemExit("unexpected_step_count; do not retry")
    print(json.dumps({"status": "one_shot_sft_complete", "metrics_path": str(metrics_path),
                      "metrics_sha256": sha(metrics_path), "global_step": metrics["global_step"]}, indent=2))


if __name__ == "__main__":
    main()
