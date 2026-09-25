"""Replay frozen v117 training inputs with the selected model tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from train_v115 import ROOT, rows, sha, token_preflight, token_signatures


def run(model_path: str, output: str) -> dict:
    from transformers import AutoProcessor

    frozen = ROOT / "data/evidence-sft-v117/frozen"
    train_path, internal_path = frozen / "train.jsonl", frozen / "internal.jsonl"
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    train, internal = rows(train_path), rows(internal_path)
    report = {
        "version": "v118-token-audit-v1",
        "model_path": str(Path(model_path).resolve()),
        "train_sha256": sha(train_path),
        "internal_sha256": sha(internal_path),
        "train_preflight": token_preflight(processor, train),
        "internal_preflight": token_preflight(processor, internal),
        "train_signatures": token_signatures(processor, train),
        "internal_signatures": token_signatures(processor, internal),
    }
    report["signature_sha256"] = hashlib.sha256(json.dumps(
        {"train": report["train_signatures"], "internal": report["internal_signatures"]},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.model_path, args.output)
    print(json.dumps({"version": result["version"], "signature_sha256": result["signature_sha256"],
                      "train": result["train_preflight"], "internal": result["internal_preflight"]},
                     ensure_ascii=False, indent=2))
