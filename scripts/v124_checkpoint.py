"""Create a local, verified checkpoint of the v124 review work only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.v123_cpu_core import ROOT

OUT = ROOT / "backups/v124-forensic-checkpoint-20260926.zip"
PRIVATE = ROOT / "data/evidence-sft-v124-forensic"
DOC = ROOT / "docs/v124-forensic-data-repair.md"
FORBIDDEN = ("final-test", "independent100", "independent-100", "holdout100")


def sha_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def main():
    files = sorted(PRIVATE.glob("*")) + sorted((ROOT / "scripts").glob("v124_*.py")) + [DOC]
    assert files and all(p.is_file() and p.resolve().is_relative_to(ROOT.resolve()) for p in files)
    names = [p.relative_to(ROOT).as_posix() for p in files]
    assert len(names) == len(set(names))
    assert all(not any(term in name.lower() for term in FORBIDDEN) for name in names)
    assert not OUT.exists(), "checkpoint_exists_refuse_overwrite"
    contents = {name: path.read_bytes() for name, path in zip(names, files)}
    manifest = {"checkpoint": "v124-primary-110", "next_record": "v117_train:0111",
                "training_allowed": False,
                "members": {name: {"size": len(blob), "sha256": sha_bytes(blob)}
                            for name, blob in contents.items()}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for name, blob in contents.items():
            archive.writestr(name, blob)
        archive.writestr("checkpoint-manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode())
    with ZipFile(OUT) as archive:
        assert sorted(archive.namelist()) == sorted([*names, "checkpoint-manifest.json"])
        checked = json.loads(archive.read("checkpoint-manifest.json"))
        assert checked == manifest
        for name, metadata in checked["members"].items():
            blob = archive.read(name)
            assert len(blob) == metadata["size"] and sha_bytes(blob) == metadata["sha256"]
    result = {"path": str(OUT), "archive_sha256": sha_bytes(OUT.read_bytes()),
              "members": len(contents), "reviewed": 110, "next_record": "v117_train:0111",
              "training_allowed": False}
    sidecar = OUT.with_suffix(".json")
    sidecar.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
