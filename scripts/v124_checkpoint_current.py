"""Archive the current v124 forensic state without replacing earlier checkpoints."""
from __future__ import annotations

import hashlib
import json
import argparse
import re
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.v123_cpu_core import ROOT

PRIVATE = ROOT / "data/evidence-sft-v124-forensic"
DOC = ROOT / "docs/v124-forensic-data-repair.md"
FORBIDDEN = ("final-test", "independent100", "independent-100", "holdout100")


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="", help="Optional lowercase checkpoint milestone")
    args = parser.parse_args()
    assert not args.label or re.fullmatch(r"[a-z][a-z0-9-]{0,31}", args.label)
    progress = json.loads((PRIVATE / "progress.json").read_text(encoding="utf-8"))
    reviewed = progress["primary_reviewed"]
    next_record = progress["next_record"]
    tag = f"-{args.label}" if args.label else ""
    today = date.today().strftime("%Y%m%d")
    dest = ROOT / f"backups/v124-forensic-checkpoint-{reviewed}{tag}-{today}.zip"
    assert not dest.exists(), "checkpoint_exists_refuse_overwrite"
    # Include nested response journals/artifacts, but never follow symlinks or
    # traverse beyond this explicitly scoped private audit directory.
    private_files = sorted(p for p in PRIVATE.rglob("*") if p.is_file() and not p.is_symlink())
    files = (private_files + sorted((ROOT / "scripts").glob("v124_*.py")) +
             sorted((ROOT / "scripts").glob("v125_*.py")) +
             sorted((ROOT / "scripts").glob("v126_*.py")) +
             sorted((ROOT / "scripts").glob("v127_*.py")) +
             [ROOT / "tests/test_v125_review_protocol.py",
              ROOT / "tests/test_v126_review_protocol.py", DOC])
    assert files and all(p.is_file() and p.resolve().is_relative_to(ROOT.resolve()) for p in files)
    names = [p.relative_to(ROOT).as_posix() for p in files]
    assert len(names) == len(set(names))
    assert all(not any(term in name.lower() for term in FORBIDDEN) for name in names)
    contents = {name: path.read_bytes() for name, path in zip(names, files)}
    manifest = {"checkpoint": "v124-primary-current" + tag, "reviewed": reviewed,
                "next_record": next_record, "training_allowed": False,
                "members": {name: {"size": len(blob), "sha256": digest(blob)}
                            for name, blob in contents.items()}}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(dest, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for name, blob in contents.items():
            archive.writestr(name, blob)
        archive.writestr("checkpoint-manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode())
    with ZipFile(dest) as archive:
        assert sorted(archive.namelist()) == sorted([*names, "checkpoint-manifest.json"])
        assert json.loads(archive.read("checkpoint-manifest.json")) == manifest
        for name, metadata in manifest["members"].items():
            blob = archive.read(name)
            assert len(blob) == metadata["size"] and digest(blob) == metadata["sha256"]
    result = {"path": str(dest), "archive_sha256": digest(dest.read_bytes()),
              "members": len(contents), "reviewed": reviewed, "next_record": next_record,
              "training_allowed": False}
    dest.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
