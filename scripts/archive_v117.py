"""Create and verify the local v117 dataset and audit backup."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "backups/v117-pre-sft.zip"
MANIFEST = ROOT / "backups/v117-pre-sft-manifest.json"
REQUIRED = [
    "reports/v117-local-preflight.json", "configs/v117-next-sft.json",
    "docs/v117-pre-sft-preparation.md", "data/evidence-sft-v116/staged-train/train-input-target.jsonl",
    "data/evidence-sft-v116/staged-train/manifest.json",
    "data/evidence-sft-v116/token-replay.json", "reports/v116-sft-preflight.json",
    "backups/v116-pre-sft-manifest.json",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    files = [ROOT / name for name in REQUIRED]
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("required_backup_input_missing")
    prior = json.loads((ROOT / "backups/v116-pre-sft-manifest.json").read_text(encoding="utf-8"))
    if sha((ROOT / "backups/v116-pre-sft.zip").read_bytes()) != prior["archive_sha256"]:
        raise ValueError("v116_lineage_backup_hash_mismatch")
    files.extend(path for path in sorted((ROOT / "data/evidence-sft-v117").rglob("*")) if path.is_file())
    files.extend(sorted((ROOT / "scripts").glob("*v117*.py")))
    files.append(ROOT / "tests/test_v117_data.py")
    entries = {}
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            if name in entries:
                raise ValueError("duplicate_backup_path")
            body = path.read_bytes()
            entries[name] = {"sha256": sha(body), "bytes": len(body)}
            output.writestr(name, body)
    with zipfile.ZipFile(ARCHIVE) as archive:
        if set(archive.namelist()) != set(entries):
            raise ValueError("backup_file_list_mismatch")
        if any(sha(archive.read(name)) != info["sha256"] for name, info in entries.items()):
            raise ValueError("backup_entry_hash_mismatch")
    result = {"version": "v117-local-backup-v1", "archive": str(ARCHIVE.relative_to(ROOT)),
              "archive_sha256": sha(ARCHIVE.read_bytes()), "files": entries,
              "all_entry_hashes_verified": True}
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(ARCHIVE), "files": len(entries),
                      "sha256": result["archive_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
