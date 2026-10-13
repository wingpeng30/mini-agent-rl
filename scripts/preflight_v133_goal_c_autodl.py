"""CPU-only AutoDL asset and tokenization preflight; never loads model weights."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/root/autodl-tmp/qwen_08b")
RUN_DIR = ROOT / "runs/v133-goal-c-sft-once/autodl-run-20260928"
CONFIG_PATH = ROOT / "configs/v133-goal-c-small-sft.json"
FREEZE_PATH = ROOT / "data/evidence-sft-v133-goal-c-private/frozen/freeze-manifest-private.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_asset_hashes(config: dict, frozen: dict) -> dict:
    checks: dict[str, str] = {}
    pairs = [
        (CONFIG_PATH, frozen["baseline"]["config_sha256"], "config"),
        (ROOT / config["train_data"], frozen["splits"]["train_sha256"], "train_data"),
        (ROOT / config["dev_data"], frozen["splits"]["dev_sha256"], "dev_data"),
        (ROOT / "src/mini_agent_rl/training/sft.py", frozen["baseline"]["sft_encoder_sha256"], "sft_encoder"),
        (ROOT / config["training_runner"], frozen["baseline"]["training_runner_sha256"], "training_runner"),
        (ROOT / config["evaluation_runner"], frozen["baseline"]["evaluation_runner_sha256"], "evaluation_runner"),
        (ROOT / config["evaluation_deadline_runner"], frozen["baseline"]["evaluation_deadline_runner_sha256"], "evaluation_deadline_runner"),
        (ROOT / config["autodl_preflight_runner"], frozen["baseline"]["autodl_preflight_runner_sha256"], "autodl_preflight_runner"),
    ]
    for path, expected, name in pairs:
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"hash_mismatch:{name}")
        checks[name] = actual
    for name, expected in frozen["baseline"]["tokenizer_sha256"].items():
        actual = sha(MODEL / name)
        if actual != expected:
            raise ValueError(f"hash_mismatch:tokenizer:{name}")
        checks[f"tokenizer:{name}"] = actual
    for group in ("model_weight_files_sha256", "model_runtime_files_sha256"):
        for name, expected in frozen["baseline"][group].items():
            actual = sha(MODEL / name)
            if actual != expected:
                raise ValueError(f"hash_mismatch:{group}:{name}")
            checks[f"{group}:{name}"] = actual
    adapter = ROOT / config["base_adapter"]
    for name, expected in frozen["baseline"]["v118_adapter_files_sha256"].items():
        actual = sha(adapter / name)
        if actual != expected:
            raise ValueError(f"hash_mismatch:v118_adapter:{name}")
        checks[f"v118_adapter:{name}"] = actual
    return checks


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    checked_hashes = check_asset_hashes(config, frozen)
    output = ROOT / "runs/v133-goal-c-sft-once/adapter"
    if output.exists():
        raise ValueError("training_output_already_exists")

    package_versions = {name: metadata.version(name) for name in config["runtime_dependencies"]}
    expected_packages = config["runtime_dependencies"]
    if package_versions != expected_packages:
        raise ValueError(f"unexpected_package_versions:{package_versions}")

    import torch
    from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor
    from mini_agent_rl.training.sft import encode_assistant_only

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA_or_BF16_unavailable")
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_gib = torch.cuda.mem_get_info(0)[0] / (1024**3)
    limits = config["resource_limits"]
    if total_gib < limits["minimum_gpu_vram_gib_for_planning"]:
        raise ValueError(f"gpu_total_below_floor:{total_gib:.3f}")
    if free_gib < limits["minimum_free_gpu_vram_gib_at_start"]:
        raise ValueError(f"gpu_free_below_floor:{free_gib:.3f}")
    model_config = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if type(model_config).__name__ != "Qwen3_5Config" or AutoModelForMultimodalLM.__name__ != "AutoModelForMultimodalLM":
        raise ValueError("Qwen3.5_auto_model_registration_missing")

    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    token_report = {}
    for filename, expected_count in (("train-candidate.jsonl", 25), ("dev-candidate.jsonl", 16)):
        rows = [json.loads(line) for line in (ROOT / "data/evidence-sft-v133-goal-c-private/frozen" / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != expected_count:
            raise ValueError(f"unexpected_row_count:{filename}:{len(rows)}")
        signatures = []
        for row in rows:
            encoded = encode_assistant_only(processor, row["messages"], max_length=2176,
                                            target_assistant_index=row.get("target_assistant_index"),
                                            thinking_enabled=False)
            signatures.append((len(encoded["input_ids"]), sum(token != -100 for token in encoded["labels"])))
        token_report[filename] = {
            "rows": len(signatures),
            "sequence_min": min(item[0] for item in signatures),
            "sequence_max": max(item[0] for item in signatures),
            "sequence_mean": round(statistics.mean(item[0] for item in signatures), 3),
            "target_min": min(item[1] for item in signatures),
            "target_max": max(item[1] for item in signatures),
        }

    result = {
        "status": "passed_cpu_only_preflight",
        "model_path": str(MODEL),
        "config_sha256": sha(CONFIG_PATH),
        "freeze_manifest_sha256": sha(FREEZE_PATH),
        "model_weights_loaded": False,
        "inference_started": False,
        "sft_started": False,
        "review_index_uploaded": False,
        "checked_asset_hashes": checked_hashes,
        "package_versions": package_versions,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "total_gib": round(total_gib, 3),
            "free_gib_at_check": round(free_gib, 3),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "model_architecture": type(model_config).__name__,
        "tokenization": token_report,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    target = RUN_DIR / "preflight.json"
    if target.exists():
        raise ValueError("preflight_report_exists; refusing_overwrite")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "package_versions": package_versions,
                      "gpu": result["gpu"], "tokenization": token_report,
                      "preflight_report": str(target)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
