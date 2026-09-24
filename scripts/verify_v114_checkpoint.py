"""Verify a v114 progress snapshot before resuming or moving it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(checkpoint_path: Path) -> dict:
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    backup = ROOT / checkpoint["backup_path"]
    if not backup.is_dir():
        raise RuntimeError("backup_directory_missing")
    saved = json.loads((backup / "checkpoint.json").read_text(encoding="utf-8"))
    if saved != checkpoint:
        raise RuntimeError("checkpoint_manifest_mismatch")
    for name, metadata in checkpoint["files"].items():
        path = backup / name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != metadata["sha256"] or len(raw) != metadata["bytes"]:
            raise RuntimeError(f"backup_file_mismatch:{name}")
        if name.endswith(".jsonl"):
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if len(rows) != metadata["rows"]:
                raise RuntimeError(f"backup_row_count_mismatch:{name}")
    return {"verified_files": len(checkpoint["files"]), "backup": checkpoint["backup_path"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "reports/v114-review-checkpoint.json")
    args = parser.parse_args()
    print(json.dumps(verify(args.checkpoint), ensure_ascii=False))


if __name__ == "__main__":
    main()
