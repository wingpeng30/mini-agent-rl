"""Record exact tokenizer input IDs and target masks for frozen v122 exports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.train_v115 import rows, sha, token_preflight, token_signatures


def run(model_path: Path, train_path: Path, internal_path: Path, output: Path) -> dict:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    train, internal = rows(train_path), rows(internal_path)
    if len(train) != 395 or len(internal) != 100:
        raise ValueError("frozen_v122_row_counts_required")
    report = {"version": "v122-token-audit-v1", "model_path": str(model_path.resolve()),
              "train_sha256": sha(train_path), "internal_sha256": sha(internal_path),
              "train_preflight": token_preflight(processor, train),
              "internal_preflight": token_preflight(processor, internal),
              "train_signatures": token_signatures(processor, train),
              "internal_signatures": token_signatures(processor, internal)}
    report["signature_sha256"] = hashlib.sha256(json.dumps(
        {"train": report["train_signatures"], "internal": report["internal_signatures"]},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("model-path", "train", "internal", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model_path, args.train, args.internal, args.output)
    print(json.dumps({k: result[k] for k in ("version", "train_sha256", "internal_sha256", "signature_sha256")}, indent=2))
