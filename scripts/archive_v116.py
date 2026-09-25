"""Create and verify a local, hash-indexed v116 pre-SFT backup."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "backups/v116-pre-sft.zip"
MANIFEST = ROOT / "backups/v116-pre-sft-manifest.json"
SINGLE = [
    "configs/v116-next-sft.json", "docs/v116-pre-sft-preparation.md",
    "reports/v116-v115-diagnosis.json", "reports/v116-v115-semantic-merge.json",
    "reports/v116-v115-score-vs-evidence.json",
    "reports/v116-sft-preflight.json", "reports/v116-single-review-gap.json",
    "tests/test_v116_review.py",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    paths = [ROOT / p for p in SINGLE]
    paths += sorted((ROOT / "data/evidence-sft-v116").rglob("*"))
    paths += sorted((ROOT / "scripts").glob("*v116*.py"))
    files = [p for p in paths if p.is_file()]
    if any(not (ROOT / p).exists() for p in SINGLE):
        raise FileNotFoundError("required_backup_input_missing")
    entries = {}
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            if name in entries:
                raise ValueError("duplicate_backup_path")
            data = path.read_bytes()
            entries[name] = {"sha256": sha(data), "bytes": len(data)}
            z.writestr(name, data)
    with zipfile.ZipFile(ARCHIVE) as z:
        if sorted(z.namelist()) != sorted(entries):
            raise ValueError("backup_file_list_mismatch")
        for name, expected in entries.items():
            if sha(z.read(name)) != expected["sha256"]:
                raise ValueError("backup_entry_hash_mismatch:" + name)
    result = {"version": "v116-local-backup-v1", "archive": str(ARCHIVE.relative_to(ROOT)),
              "archive_sha256": sha(ARCHIVE.read_bytes()), "files": entries,
              "all_entry_hashes_verified": True}
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(ARCHIVE), "files": len(entries),
                      "sha256": result["archive_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
