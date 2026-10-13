"""Synchronize only the BD-frozen code differences; preserve every replaced file."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliveries/v133-interview-final-web"
REMOTE = "root@connect.cqa1.seetacloud.com"
REMOTE_ROOT = "/root/autodl-tmp/mini-agent-rl"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT / "asset-sync.json"
    if receipt.exists():
        raise FileExistsError("Synchronization already recorded; inspect its receipt rather than overwrite")
    manifest = json.loads((ROOT / "runs/v133-goal-bd-final-regression-v1/manifest.json").read_text(encoding="utf-8"))
    changes = []
    checked = 0
    for key, expected in manifest["hashes"].items():
        if not key.startswith("code:"):
            continue
        relative = key[5:]
        destination = ROOT / relative
        if not destination.resolve().is_relative_to(ROOT):
            raise ValueError("Code destination escaped workspace")
        checked += 1
        before = sha(destination) if destination.is_file() else None
        if before == expected:
            continue
        incoming = OUT / "incoming" / relative
        incoming.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", "-P", "25320", f"{REMOTE}:{REMOTE_ROOT}/{relative}", str(incoming)], check=True)
        if sha(incoming) != expected:
            raise RuntimeError(f"Remote frozen source mismatch: {relative}")
        changes.append(dict(path=relative, old_sha256=before, new_sha256=expected))
    # No destination is changed until every downloaded source has passed hashing.
    for item in changes:
        relative = item["path"]
        destination = ROOT / relative
        backup = OUT / "local-before-sync" / relative
        if destination.is_file():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT / "incoming" / relative, destination)
    identities = {}
    for arm, folder in (("goal-z", "v133-goal-z-multihop-sft/seed42-trajectory"),
                        ("am-research", "v133-goal-am-action-sft-v2/seed42-A")):
        p = ROOT / "runs" / folder / "adapter-final/adapter_model.safetensors"
        expected = manifest["hashes"]["A:weights" if arm == "goal-z" else "B:weights"]
        if sha(p) != expected:
            raise RuntimeError(f"Model identity mismatch: {arm}")
        identities[arm] = dict(path=str(p), sha256=expected)
    for key, remote_path in manifest["paths"].items():
        if key.startswith("base:"):
            p = Path("D:/qwen_08b") / key[5:]
        elif key.endswith(":corpus"):
            p = ROOT / remote_path.removeprefix(REMOTE_ROOT + "/")
        else:
            continue
        if sha(p) != manifest["hashes"][key]:
            raise RuntimeError(f"Base/corpus identity mismatch: {key}")
        identities[key] = dict(path=str(p), sha256=manifest["hashes"][key])
    result = dict(passed=True, frozen_code_files_checked=checked, synchronized=changes,
        identities=identities, historical_BD_manifest_unchanged=True,
        source="Remote BD-frozen code; existing local model/corpus files verified",
        remote_model_loads=0, training_updates=0)
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=True, checked=checked, synchronized=len(changes))))

if __name__ == "__main__":
    main()
