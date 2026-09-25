"""Fail-closed v118 SFT runner; each phase starts from frozen v101."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from train_v115 import ROOT, finite, rows, sha, token_preflight, token_signatures, tree
from verify_v118_freeze import EXPECTED, FROZEN, verify as verify_freeze

SOURCE_SHA = "3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768"
CONFIG = ROOT / "configs/v118-sft.json"


def check_report(report_path: Path, token_audit_path: Path, source_path: Path, phase: str, smoke_report_path: Path | None) -> dict:
    local = verify_freeze()
    if not local["passed"]:
        raise ValueError("v118_local_data_gate_failed")
    report, audit = json.loads(report_path.read_text(encoding="utf-8")), json.loads(token_audit_path.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if report.get("version") != "v118-remote-preflight-v1" or report.get("training_permitted") is not True:
        raise ValueError("remote_preflight_not_permitted")
    if report.get("independent100_accessed") or report.get("final_test_accessed") or report.get("training_started"):
        raise ValueError("unexpected_preflight_state")
    for key, filename in (("train_sha256", "train.jsonl"), ("internal_sha256", "internal.jsonl"), ("corpus_sha256", "internal-corpus.json")):
        if report.get(key) != EXPECTED[filename]:
            raise ValueError("remote_preflight_hash_mismatch:" + key)
    if audit.get("version") != "v118-token-audit-v1" or audit.get("train_sha256") != EXPECTED["train.jsonl"] or audit.get("internal_sha256") != EXPECTED["internal.jsonl"]:
        raise ValueError("token_audit_not_bound_to_frozen_exports")
    if report.get("token_signatures_sha256") != audit.get("signature_sha256") or report.get("local_remote_token_equal") is not True:
        raise ValueError("token_audit_remote_mismatch")
    source_hashes = tree(source_path)
    if source_hashes.get("adapter_model.safetensors") != SOURCE_SHA:
        raise ValueError("v101_source_hash_mismatch")
    if report.get("v101_adapter_sha256") != SOURCE_SHA:
        raise ValueError("remote_v101_hash_mismatch")
    if phase == "formal":
        if smoke_report_path is None or not smoke_report_path.is_file():
            raise ValueError("passing_smoke_report_required")
        smoke = json.loads(smoke_report_path.read_text(encoding="utf-8"))
        if (smoke.get("version") != "v118-training-phase-v1" or smoke.get("phase") != "smoke"
                or smoke.get("global_step") != 5 or smoke.get("source_unchanged") is not True
                or smoke.get("reload_finite") is not True or smoke.get("source_adapter_sha256") != SOURCE_SHA):
            raise ValueError("smoke_phase_not_passed")
    return {"local": local, "remote": report, "audit": audit, "config": config, "source_hashes": source_hashes}


def run(args) -> dict:
    from transformers import set_seed, AutoProcessor
    from mini_agent_rl.training.sft import train_sft, build_lora_model

    train_path, internal_path = FROZEN / "train.jsonl", FROZEN / "internal.jsonl"
    source_path, output = Path(args.source_adapter), Path(args.output_dir)
    if output.exists() or output == source_path or output in source_path.parents or source_path in output.parents:
        raise ValueError("unsafe_or_existing_output")
    smoke_path = Path(args.smoke_report) if args.smoke_report else None
    approved = check_report(Path(args.remote_preflight), Path(args.token_audit), source_path, args.phase, smoke_path)
    source_before = approved["source_hashes"]
    train, internal = rows(train_path), rows(internal_path)
    if len(train) != 345 or len(internal) != 100 or sha(train_path) != EXPECTED["train.jsonl"] or sha(internal_path) != EXPECTED["internal.jsonl"]:
        raise ValueError("frozen_v117_rows_or_hash_mismatch")
    if not all(set(row) == {"messages", "target_assistant_index", "thinking_enabled"} for row in train + internal):
        raise ValueError("private_or_unexpected_training_fields")
    if sha(train_path) == sha(internal_path):
        raise ValueError("internal_must_never_be_training_data")
    model_path = Path(args.model_path)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    train_sig, internal_sig = token_signatures(processor, train), token_signatures(processor, internal)
    audit = approved["audit"]
    if train_sig != audit["train_signatures"] or internal_sig != audit["internal_signatures"]:
        raise ValueError("remote_token_signature_changed_before_training")
    train_limit = token_preflight(processor, train)
    steps = 5 if args.phase == "smoke" else 50
    output.mkdir(parents=True)
    train_copy = output / "train-only.jsonl"
    shutil.copyfile(train_path, train_copy)
    if sha(train_copy) != EXPECTED["train.jsonl"]:
        raise ValueError("train_copy_hash_mismatch")
    source_copy = output / "fresh-source-v101"
    shutil.copytree(source_path, source_copy)
    if tree(source_copy) != source_before:
        raise ValueError("fresh_source_copy_mismatch")
    set_seed(42)
    destination = output / "adapter"
    metrics = train_sft(model_path=str(model_path), train_path=str(train_copy), validation_path=str(train_copy),
        output_dir=str(destination), max_steps=steps, epochs=1, max_length=2176, seed=42, batch_size=1,
        gradient_accumulation_steps=8, gradient_checkpointing=False, adapter_path=str(source_copy),
        enable_thinking=False, learning_rate=5e-6, save_steps=steps, eval_steps=steps, warmup_ratio=0.05)
    finite(metrics)
    if metrics.get("global_step") != steps:
        raise ValueError(f"wrong_optimizer_update_count:{metrics.get('global_step')}:{steps}")
    logs = metrics.get("train_metrics", [])
    if not any("loss" in item for item in logs):
        raise ValueError("training_loss_missing")
    if any("grad_norm" in item and not math.isfinite(float(item["grad_norm"])) for item in logs):
        raise ValueError("gradient_nonfinite")
    checkpoint = destination / f"checkpoint-{steps}"
    weight = checkpoint / "adapter_model.safetensors"
    if not weight.is_file() or sha(weight) == SOURCE_SHA:
        raise ValueError("last_checkpoint_missing_or_unchanged")
    import gc
    import torch
    gc.collect(); torch.cuda.empty_cache()
    model, _ = build_lora_model(str(model_path), gradient_checkpointing=False, adapter_path=str(checkpoint))
    reload_finite = all(torch.isfinite(param).all().item() for param in model.parameters() if param.requires_grad)
    del model; gc.collect(); torch.cuda.empty_cache()
    if not reload_finite:
        raise ValueError("checkpoint_reload_nonfinite")
    if tree(source_path) != source_before:
        raise ValueError("frozen_v101_changed")
    result = {"version": "v118-training-phase-v1", "phase": args.phase, "steps": steps,
        "global_step": metrics["global_step"], "train_sha256": EXPECTED["train.jsonl"],
        "internal_sha256": EXPECTED["internal.jsonl"], "internal_used_for_training": False,
        "trainer_train_sha256": sha(train_copy), "trainer_validation_sha256": sha(train_copy),
        "source_adapter_sha256": SOURCE_SHA, "output_adapter_sha256": sha(weight),
        "checkpoint": str(checkpoint), "metrics": metrics, "reload_finite": reload_finite,
        "source_unchanged": True, "train_token_preflight": train_limit,
        "remote_preflight_sha256": sha(Path(args.remote_preflight)),
        "token_audit_sha256": sha(Path(args.token_audit)), "config_sha256": sha(CONFIG),
        "independent100_accessed": False, "final_test_accessed": False}
    report_path = output / ("smoke-report.json" if args.phase == "smoke" else "formal-report.json")
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    for name in ("source-adapter", "model-path", "remote-preflight", "token-audit", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--smoke-report")
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("version", "phase", "steps", "global_step", "output_adapter_sha256", "reload_finite", "source_unchanged")}, ensure_ascii=False, indent=2))
