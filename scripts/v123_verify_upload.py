"""Verify a named v123 archive before extracting to a fresh workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

FORBIDDEN = ("final-test", "final_test", "independent", "holdout")


def hashed(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_extract(archive_path: Path, destination: Path, expected_sha256: str) -> dict:
    with ZipFile(archive_path) as archive:
        info = archive.infolist()
        names = [item.filename for item in info]
        if len(names) != len(set(name.casefold() for name in names)) or "v123-transfer-manifest.json" not in names:
            raise ValueError("archive_duplicate_or_missing_manifest")
        for item in info:
            name = item.filename.replace("\\", "/")
            parts = name.split("/")
            if (item.is_dir() or name != item.filename or name.startswith("/") or ":" in name or
                    any(part in {"", ".", ".."} for part in parts) or
                    any(any(word in part.casefold() for word in FORBIDDEN) for part in parts) or
                    stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError("forbidden_archive_member")
        with archive.open("v123-transfer-manifest.json") as stream:
            manifest = json.load(stream)
        if (manifest.get("version") != "v123-transfer-v1" or manifest.get("training_allowed") is not False or
                manifest.get("credentials_included") is not False or manifest.get("sealed_assets_included") is not False or
                set(manifest.get("files", {})) != set(names) - {"v123-transfer-manifest.json"}):
            raise ValueError("transfer_manifest_invalid")
        with archive_path.open("rb") as stream:
            if hashed(stream) != expected_sha256:
                raise ValueError("archive_transport_hash_mismatch")
        for name, expected in manifest["files"].items():
            with archive.open(name) as stream:
                if hashed(stream) != expected:
                    raise ValueError("archive_member_hash_mismatch")
        root = destination.resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        for name in manifest["files"]:
            target = (root / name).resolve(strict=False)
            if not target.is_relative_to(root) or target.exists():
                raise ValueError("upload_target_not_fresh_or_outside_root")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("xb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
        return {"status": "passed", "archive_sha256": expected_sha256,
                "files": len(manifest["files"]), "destination": str(root),
                "training_allowed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("expected_sha256")
    args = parser.parse_args()
    print(json.dumps(verify_extract(args.archive, args.destination, args.expected_sha256), indent=2))
