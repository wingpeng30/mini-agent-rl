"""Hash and snapshot append-only v114 review progress for safe resumption."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/evidence-sft-v114-review"
REPORT = ROOT / "reports/v114-review-checkpoint.json"
BACKUPS = ROOT / "backups/v114-review-progress"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def snapshot(path: Path) -> tuple[bytes, int]:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise RuntimeError(f"incomplete_last_line:{path.name}")
    count = 0
    if path.suffix == ".jsonl":
        for line in data.splitlines():
            if line.strip():
                json.loads(line)
                count += 1
    return data, count


def main() -> None:
    wanted = [WORK / "preflight.json", WORK / "review-quality-quarantine.json",
              WORK / "supplement-gap-private.json", WORK / "reserve-inventory.json",
              WORK / "reserve-private.jsonl", ROOT / "reports/v114-review-gate.json",
              ROOT / "reports/v114-sft-export-manifest.json",
              ROOT / "data/evidence-sft-v114-clean/manifest.json",
              ROOT / "data/evidence-sft-v114-clean/internal-freeze.json",
              ROOT / "data/evidence-sft-v114-clean/training-question-freeze.json",
              WORK / "internal-freeze-v2.json", WORK / "internal-freeze-v3.json",
              WORK / "training-question-freeze-v2.json", WORK / "replacement-question.json",
              WORK / "internal-replacement-proposals-manifest.json",
              WORK / "measurement-supplement-manifest.json",
              WORK / "internal-replacements-v3-manifest.json",
              WORK / "measurement-risk-recheck-manifest.json",
              WORK / "accepted.jsonl", WORK / "disputed.jsonl", WORK / "rejected.jsonl",
              WORK / "invalid.jsonl", WORK / "pending.jsonl", WORK / "retired.jsonl",
              WORK / "sft-train.jsonl", WORK / "sft-internal.jsonl",
              WORK / "internal-question-quality-audit-b.jsonl",
              ROOT / "docs/v114-review-progress.md", ROOT / "tests/test_v114_review_gate.py"]
    wanted += [ROOT / "scripts" / name for name in (
        "prepare_v114_reviews.py", "build_v114_internal_gap.py", "prepare_v114_reserves.py",
        "dispatch_v114_reserve_wave.py", "finalize_v114_reviews.py", "checkpoint_v114_review.py",
        "verify_v114_checkpoint.py", "prepare_v114_internal_replacement_proposals.py",
        "build_v114_internal_replacements_v3.py", "prepare_v114_measurement_supplement.py",
        "verify_v114_sft_exports.py")]
    wanted += sorted(WORK.glob("*packets.jsonl"))
    wanted += sorted(WORK.glob("*decisions.jsonl"))
    wanted += sorted(WORK.glob("*-addenda.jsonl"))
    wanted += sorted(WORK.glob("reserve-wave-*-manifest.json"))
    wanted += sorted(WORK.glob("*-manifest.json"))
    captured = {}
    for path in wanted:
        if path.exists():
            captured[path.relative_to(ROOT).as_posix()] = snapshot(path)
    if REPORT.exists():
        prior = json.loads(REPORT.read_text(encoding="utf-8"))
        old_root = ROOT / prior["backup_path"]
        for name, (current, _) in captured.items():
            if not name.endswith("decisions.jsonl") or name not in prior["files"]:
                continue
            old = (old_root / name).read_bytes()
            if not current.startswith(old):
                raise RuntimeError(f"append_only_history_changed:{name}")
    digest = sha(b"".join(name.encode() + b"\0" + sha(content).encode() + b"\n"
                          for name, (content, _) in sorted(captured.items())))[:16]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = BACKUPS / f"{stamp}-{digest}"
    folder.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, (content, count) in captured.items():
        dest = folder / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        files[name] = {"sha256": sha(content), "rows": count, "bytes": len(content)}
    checkpoint = {"version": "v114-review-checkpoint-v1", "created_utc": stamp,
                  "backup_path": folder.relative_to(ROOT).as_posix(), "files": files,
                  "source_manifest_sha256": sha((ROOT / "data/evidence-sft-v114-clean/manifest.json").read_bytes()),
                  "training_permitted": False}
    manifest = json.dumps(checkpoint, ensure_ascii=False, indent=2).encode() + b"\n"
    (folder / "checkpoint.json").write_bytes(manifest)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_bytes(manifest)
    print(json.dumps({"backup": checkpoint["backup_path"],
                      "decision_rows": {name: meta["rows"] for name, meta in files.items() if name.endswith("decisions.jsonl")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
