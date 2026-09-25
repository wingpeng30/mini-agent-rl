"""Fail-closed remote v122 resource, model, data, and token preflight."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.audit_v122_tokens import run as audit_tokens
from scripts.train_v115 import sha, tree

SOURCE_SHA = "3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768"
REFERENCE_SHA = "31b625a9e3934012f9ba9b2f2054006ab968caeabb5238c5934988edccf423fd"


def run(args) -> dict:
    import torch
    import transformers

    train_gate = json.loads(args.train_gate.read_text(encoding="utf-8"))
    internal_gate = json.loads(args.internal_gate.read_text(encoding="utf-8"))
    train_manifest = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    internal_manifest = json.loads(args.internal_manifest.read_text(encoding="utf-8"))
    local_audit = json.loads(args.local_token_audit.read_text(encoding="utf-8"))
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    if (configuration.get("version") != "v122-targeted-sft-freeze-v1" or
            configuration.get("source_adapter_sha256") != SOURCE_SHA or
            configuration.get("reference_adapter_sha256") != REFERENCE_SHA or
            configuration.get("smoke", {}).get("max_steps") != 5 or
            configuration.get("formal", {}).get("max_steps") != 50 or
            configuration.get("formal", {}).get("checkpoint_selection") != "last" or
            configuration.get("shared") != {
                "dtype": "bfloat16", "learning_rate": 5e-6, "batch_size": 1,
                "gradient_accumulation_steps": 8, "seed": 42, "thinking": False,
                "gradient_checkpointing": False, "max_input_tokens": 2048,
                "max_target_tokens": 128, "max_total_tokens": 2176}):
        raise ValueError("frozen_training_configuration_mismatch")
    if not train_gate.get("local_train_ready") or not internal_gate.get("local_internal_ready"):
        raise ValueError("local_double_review_and_data_gates_required")
    if (sha(args.train) != train_manifest.get("train_sha256") or
            sha(args.internal) != internal_manifest.get("files", {}).get("internal.jsonl") or
            sha(args.corpus) != internal_manifest.get("files", {}).get("internal-corpus.json")):
        raise ValueError("remote_frozen_data_hash_mismatch")
    if sha(args.source_adapter / "adapter_model.safetensors") != SOURCE_SHA:
        raise ValueError("v101_source_hash_mismatch")
    if sha(args.reference_adapter / "adapter_model.safetensors") != REFERENCE_SHA:
        raise ValueError("v112_reference_hash_mismatch")
    if not args.model_path.is_dir() or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("base_model_and_bf16_gpu_required")
    disk = shutil.disk_usage(args.workspace)
    if disk.free < 10 * 1024**3:
        raise RuntimeError("insufficient_workspace_disk")
    remote_audit_path = args.output.with_suffix(".tokens.json")
    remote_audit = audit_tokens(args.model_path, args.train, args.internal, remote_audit_path)
    if (local_audit.get("version") != "v122-token-audit-v1" or
            local_audit.get("train_sha256") != remote_audit["train_sha256"] or
            local_audit.get("internal_sha256") != remote_audit["internal_sha256"] or
            local_audit.get("train_signatures") != remote_audit["train_signatures"] or
            local_audit.get("internal_signatures") != remote_audit["internal_signatures"]):
        raise ValueError("local_remote_token_ids_or_masks_mismatch")
    device = torch.cuda.get_device_properties(0)
    report = {"version": "v122-remote-preflight-v1", "training_permitted": True,
              "training_started": False, "gpu": {"name": device.name, "memory_bytes": device.total_memory,
                                              "bf16_supported": True},
              "free_disk_bytes": disk.free, "torch_version": torch.__version__,
              "transformers_version": transformers.__version__,
              "model_files": tree(args.model_path),
              "v101_adapter_files": tree(args.source_adapter),
              "v112_adapter_files": tree(args.reference_adapter),
              "v101_adapter_sha256": SOURCE_SHA, "v112_adapter_sha256": REFERENCE_SHA,
              "train_sha256": sha(args.train), "internal_sha256": sha(args.internal),
              "corpus_sha256": sha(args.corpus), "token_signatures_sha256": remote_audit["signature_sha256"],
              "local_remote_token_equal": True,
              "train_gate_sha256": sha(args.train_gate), "internal_gate_sha256": sha(args.internal_gate),
              "training_config_sha256": sha(args.config),
              "independent100_accessed": False, "final_test_accessed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("model-path", "source-adapter", "reference-adapter", "train", "internal", "corpus",
                 "train-gate", "internal-gate", "train-manifest", "internal-manifest",
                 "local-token-audit", "config", "workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({k: result[k] for k in ("version", "gpu", "free_disk_bytes", "training_permitted", "token_signatures_sha256")}, indent=2))
