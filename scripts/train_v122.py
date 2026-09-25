"""One smoke and one formal v122 SFT, independently copied from frozen v101."""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path

from scripts.preflight_v122 import SOURCE_SHA
from scripts.train_v115 import finite, rows, sha, token_preflight, token_signatures, tree


def check(args) -> tuple[dict, dict, dict]:
    preflight = json.loads(args.remote_preflight.read_text(encoding="utf-8"))
    token_audit = json.loads(args.token_audit.read_text(encoding="utf-8"))
    train_gate = json.loads(args.train_gate.read_text(encoding="utf-8"))
    internal_gate = json.loads(args.internal_gate.read_text(encoding="utf-8"))
    if (preflight.get("version") != "v122-remote-preflight-v1" or
            preflight.get("training_permitted") is not True or preflight.get("training_started") or
            preflight.get("independent100_accessed") or preflight.get("final_test_accessed") or
            not train_gate.get("local_train_ready") or not internal_gate.get("local_internal_ready")):
        raise ValueError("passing_data_and_remote_preflight_required")
    if preflight.get("training_config_sha256") != sha(args.config):
        raise ValueError("frozen_training_configuration_changed")
    if (preflight.get("train_gate_sha256") != sha(args.train_gate) or
            preflight.get("internal_gate_sha256") != sha(args.internal_gate) or
            preflight.get("train_sha256") != sha(args.train) or
            preflight.get("internal_sha256") != sha(args.internal) or
            token_audit.get("train_sha256") != sha(args.train) or
            token_audit.get("internal_sha256") != sha(args.internal) or
            token_audit.get("signature_sha256") != preflight.get("token_signatures_sha256")):
        raise ValueError("frozen_data_or_token_audit_changed")
    source_tree = tree(args.source_adapter)
    if source_tree.get("adapter_model.safetensors") != SOURCE_SHA or preflight.get("v101_adapter_sha256") != SOURCE_SHA:
        raise ValueError("v101_source_changed")
    if args.phase == "formal":
        if args.smoke_report is None or not args.smoke_report.is_file():
            raise ValueError("passing_smoke_required")
        smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
        if (smoke.get("version") != "v122-training-phase-v1" or smoke.get("phase") != "smoke" or
                smoke.get("global_step") != 5 or smoke.get("source_unchanged") is not True or
                smoke.get("reload_finite") is not True or smoke.get("source_adapter_sha256") != SOURCE_SHA or
                smoke.get("train_sha256") != sha(args.train)):
            raise ValueError("smoke_report_invalid")
    return preflight, token_audit, source_tree


def run(args) -> dict:
    from transformers import AutoProcessor, set_seed
    import torch
    from mini_agent_rl.training.sft import build_lora_model, train_sft

    if args.output_dir.exists() or args.output_dir == args.source_adapter or args.output_dir in args.source_adapter.parents:
        raise ValueError("unsafe_or_existing_output")
    preflight, audit, source_before = check(args)
    train, internal = rows(args.train), rows(args.internal)
    if len(train) != 395 or len(internal) != 100:
        raise ValueError("v122_frozen_row_counts_invalid")
    if not all(set(row) == {"messages", "target_assistant_index", "thinking_enabled"} for row in train + internal):
        raise ValueError("private_or_unexpected_training_fields")
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    if (token_signatures(processor, train) != audit["train_signatures"] or
            token_signatures(processor, internal) != audit["internal_signatures"]):
        raise ValueError("token_signatures_changed_before_training")
    train_limit = token_preflight(processor, train)
    steps = 5 if args.phase == "smoke" else 50
    args.output_dir.mkdir(parents=True)
    train_copy = args.output_dir / "train-only.jsonl"
    shutil.copyfile(args.train, train_copy)
    if sha(train_copy) != preflight["train_sha256"]:
        raise ValueError("train_copy_hash_mismatch")
    source_copy = args.output_dir / "fresh-source-v101"
    shutil.copytree(args.source_adapter, source_copy)
    if tree(source_copy) != source_before:
        raise ValueError("fresh_source_copy_mismatch")
    set_seed(42)
    destination = args.output_dir / "adapter"
    metrics = train_sft(model_path=str(args.model_path), train_path=str(train_copy),
        validation_path=str(train_copy), output_dir=str(destination), max_steps=steps,
        epochs=1, max_length=2176, seed=42, batch_size=1, gradient_accumulation_steps=8,
        gradient_checkpointing=False, adapter_path=str(source_copy), enable_thinking=False,
        learning_rate=5e-6, save_steps=steps, eval_steps=steps, warmup_ratio=0.05)
    finite(metrics)
    if metrics.get("global_step") != steps:
        raise ValueError("wrong_optimizer_update_count")
    logs = metrics.get("train_metrics", [])
    if not any("loss" in item for item in logs) or any(
            "grad_norm" in item and not math.isfinite(float(item["grad_norm"])) for item in logs):
        raise ValueError("nonfinite_or_missing_train_metrics")
    checkpoint = destination / f"checkpoint-{steps}"
    weight = checkpoint / "adapter_model.safetensors"
    if not weight.is_file() or sha(weight) == SOURCE_SHA:
        raise ValueError("last_checkpoint_missing_or_unchanged")
    gc.collect(); torch.cuda.empty_cache()
    model, _ = build_lora_model(str(args.model_path), gradient_checkpointing=False, adapter_path=str(checkpoint))
    reload_finite = all(torch.isfinite(param).all().item() for param in model.parameters() if param.requires_grad)
    del model; gc.collect(); torch.cuda.empty_cache()
    if not reload_finite or tree(args.source_adapter) != source_before:
        raise ValueError("reload_nonfinite_or_frozen_source_changed")
    report = {"version": "v122-training-phase-v1", "phase": args.phase, "steps": steps,
              "global_step": metrics["global_step"], "train_sha256": sha(args.train),
              "internal_sha256": sha(args.internal), "internal_used_for_training": False,
              "trainer_train_sha256": sha(train_copy), "trainer_validation_sha256": sha(train_copy),
              "source_adapter_sha256": SOURCE_SHA, "output_adapter_sha256": sha(weight),
              "checkpoint": str(checkpoint), "metrics": metrics,
              "reload_finite": reload_finite, "source_unchanged": True,
              "train_token_preflight": train_limit,
              "remote_preflight_sha256": sha(args.remote_preflight),
              "token_audit_sha256": sha(args.token_audit),
              "independent100_accessed": False, "final_test_accessed": False}
    path = args.output_dir / ("smoke-report.json" if args.phase == "smoke" else "formal-report.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    for name in ("source-adapter", "model-path", "train", "internal", "train-gate", "internal-gate",
                 "remote-preflight", "token-audit", "config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path)
    result = run(parser.parse_args())
    print(json.dumps({k: result[k] for k in ("version", "phase", "steps", "global_step", "output_adapter_sha256")}, indent=2))
