"""Create immutable bounded blinded waves from the frozen reserve inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/evidence-sft-v114-review"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_once(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"immutable_wave_changed:{path.name}")
    if not path.exists():
        path.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--internal", type=int, required=True)
    parser.add_argument("--train", type=int, required=True)
    args = parser.parse_args()
    if args.wave < 1 or args.internal < 0 or args.train < 0:
        raise ValueError("invalid_wave_or_counts")
    manifest = json.loads((WORK / "reserve-inventory.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["files_sha256"].items():
        if sha((WORK / filename).read_bytes()) != expected:
            raise RuntimeError(f"reserve_inventory_changed:{filename}")
    keys = load(WORK / "reserve-private.jsonl")
    rows = load(WORK / "reserve-a-packets.jsonl")
    if rows != load(WORK / "reserve-b-packets.jsonl"):
        raise RuntimeError("reserve_reviewer_packets_differ")
    if len(rows) != len(keys) or any(r["case_id"] != k["case_id"] for r, k in zip(rows, keys)):
        raise RuntimeError("reserve_private_mapping_invalid")
    prior = set()
    for path in sorted(WORK.glob("reserve-wave-*-manifest.json")):
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["wave"] == args.wave:
            continue
        prior.update(old["case_ids"])
    picks = []
    for split, count in (("internal", args.internal), ("train", args.train)):
        available = [r for r, k in zip(rows, keys) if k["split"] == split and r["case_id"] not in prior]
        if len(available) < count:
            raise RuntimeError(f"insufficient_reserves:{split}:{len(available)}<{count}")
        picks.extend(available[:count])
    content = b"".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n" for r in picks)
    for reviewer in ("a", "b"):
        write_once(WORK / f"reserve-wave-{args.wave}-{reviewer}-packets.jsonl", content)
    wave_manifest = {"wave": args.wave, "internal": args.internal, "train": args.train,
                     "case_ids": [r["case_id"] for r in picks], "packet_sha256": sha(content),
                     "inventory_sha256": sha((WORK / "reserve-inventory.json").read_bytes())}
    write_once(WORK / f"reserve-wave-{args.wave}-manifest.json",
               json.dumps(wave_manifest, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"wave": args.wave, "states": len(picks), "internal": args.internal, "train": args.train}))


if __name__ == "__main__":
    main()
