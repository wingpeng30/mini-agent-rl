"""Remote-side, fail-closed v118 GPU and data preflight; does not train."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from audit_v118_tokens import run as audit_tokens
from verify_v118_freeze import EXPECTED, FROZEN, sha

SOURCE_SHA = "3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768"
REFERENCE_SHA = "31b625a9e3934012f9ba9b2f2054006ab968caeabb5238c5934988edccf423fd"


def tree(path: Path) -> dict[str, str]:
    if path.is_symlink() or any(item.is_symlink() for item in path.rglob("*")):
        raise ValueError("symlink_in_adapter")
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def run(args) -> dict:
    import torch
    import transformers

    from verify_v118_freeze import verify

    local_audit = json.loads(Path(args.local_token_audit).read_text(encoding="utf-8"))
    local_gate = verify()
    model_path, source_path, reference_path = map(Path, (args.model_path, args.source_adapter, args.reference_adapter))
    if not model_path.is_dir() or not (source_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("model_or_v101_weights_missing")
    if not (reference_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("v112_weights_missing")
    source_tree, reference_tree = tree(source_path), tree(reference_path)
    if source_tree.get("adapter_model.safetensors") != SOURCE_SHA:
        raise ValueError("v101_source_hash_mismatch")
    if reference_tree.get("adapter_model.safetensors") != REFERENCE_SHA:
        raise ValueError("v112_reference_hash_mismatch")
    cuda = torch.cuda.is_available()
    if not cuda or not torch.cuda.is_bf16_supported():
        raise RuntimeError("cuda_with_bf16_required")
    device = torch.cuda.get_device_properties(0)
    disk = shutil.disk_usage(args.workspace)
    remote_audit_path = Path(args.output).with_suffix(".tokens.json")
    remote_audit = audit_tokens(str(model_path), str(remote_audit_path))
    if local_audit.get("version") != "v118-token-audit-v1":
        raise ValueError("invalid_local_token_audit")
    if local_audit.get("train_sha256") != EXPECTED["train.jsonl"] or local_audit.get("internal_sha256") != EXPECTED["internal.jsonl"]:
        raise ValueError("local_token_audit_data_hash_mismatch")
    if local_audit.get("train_signatures") != remote_audit.get("train_signatures") or local_audit.get("internal_signatures") != remote_audit.get("internal_signatures"):
        raise ValueError("local_remote_token_or_mask_mismatch")
    report = {"version": "v118-remote-preflight-v1", "gpu": {"name": device.name, "memory_bytes": device.total_memory,
                "bf16_supported": True}, "cuda_available": cuda,
        "python_version": __import__("sys").version, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "free_disk_bytes": disk.free,
        "workspace": str(Path(args.workspace).resolve()), "model_files": tree(model_path),
        "v101_adapter_files": source_tree, "v112_adapter_files": reference_tree,
        "v101_adapter_sha256": source_tree["adapter_model.safetensors"],
        "v112_adapter_sha256": reference_tree["adapter_model.safetensors"],
        "train_sha256": sha(FROZEN / "train.jsonl"), "internal_sha256": sha(FROZEN / "internal.jsonl"),
        "corpus_sha256": sha(FROZEN / "internal-corpus.json"),
        "token_signatures_sha256": remote_audit["signature_sha256"],
        "local_remote_token_equal": True, "local_data_gate": local_gate["passed"],
        "training_permitted": True, "training_started": False,
        "independent100_accessed": False, "final_test_accessed": False}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("model-path", "source-adapter", "reference-adapter", "local-token-audit", "workspace", "output"):
        parser.add_argument("--" + name, required=True)
    result = run(parser.parse_args())
    print(json.dumps({k: v for k, v in result.items() if k in {"version", "gpu", "free_disk_bytes", "training_permitted", "token_signatures_sha256"}}, ensure_ascii=False, indent=2))
