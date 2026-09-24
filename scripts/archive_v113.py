"""Build and verify a self-contained, hash-sealed v113 diagnostic archive."""
import hashlib, io, json, sqlite3, tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

FILES = [
    "docs/v113-diagnostics-results.md",
    "scripts/build_v113_diagnostics.py", "scripts/compact_v113_p0.py",
    "scripts/merge_v113_p0_audit.py", "scripts/build_v113_cross_review.py",
    "scripts/run_v113_cross.py", "scripts/merge_v113_p1_audit.py",
    "scripts/analyze_v113_blind_reviews.py", "scripts/build_v113_results.py",
    "scripts/check_v113_stage_gate.py", "scripts/archive_v113.py",
    "tests/test_v113_cross_runner.py", "tests/test_v113_stage_gate.py",
    "reports/v113-freeze-manifest.json", "reports/v113-diagnostics-manifest.json",
    "reports/v113-summary.json", "reports/v113-p2-source-gate.json",
    "reports/v113-p2-source-gate-evidence.json",
    "reports/v113-ledger-reconciliation.json", "reports/v113-p0-external-api-block.json",
    "reports/v113-p0-blind-packet.jsonl", "reports/v113-p0-review-cases.jsonl",
    "reports/v113-p0-private-key.json", "reports/v113-p0-score-key.json",
    "reports/v113-p0-review-document-key.json", "reports/v113-p0-agent-audit-summary.json",
    "reports/v113-p0-adjudication-summary.json",
    "reports/v113-p0-agent-review-0.jsonl", "reports/v113-p0-agent-review-1.jsonl",
    "reports/v113-p0-agent-review-2.jsonl", "reports/v113-p0-agent-review-extra-0.jsonl",
    "reports/v113-p0-agent-review-extra-1.jsonl", "reports/v113-p0-agent-review-extra-2.jsonl",
    "reports/v113-p0-review-chunks/chunk-0.jsonl", "reports/v113-p0-review-chunks/chunk-1.jsonl",
    "reports/v113-p0-review-chunks/chunk-2.jsonl",
    "reports/v113-p1-blind-review.jsonl", "reports/v113-p1-private-key.json",
    "reports/v113-p1-review-document-key.json", "reports/v113-p1-agent-audit-summary.json",
    "reports/v113-p1-unblinded-audit-summary.json", "reports/v113-cross-inputs.jsonl",
    "reports/v113-cross-results-stable.jsonl",
    "reports/v113-p1-agent-review-0.jsonl", "reports/v113-p1-agent-review-1.jsonl",
    "reports/v113-p1-agent-review-2.jsonl",
    "reports/v113-p1-review-0.jsonl", "reports/v113-p1-review-1.jsonl",
    "reports/v113-p1-review-2.jsonl",
    "reports/v112-replay2-baseline-tasks.json", "reports/v112-replay2-baseline-states.json",
    "reports/v112-replay2-candidate-tasks.json", "reports/v112-replay2-candidate-states.json",
    "reports/v112-training-report.json", "reports/v112-final-verification.json",
    "reports/v112-internal-comparison.json", "reports/v112-data-gate.json",
    "reports/v112-remote-receipt.json", "configs/v112-budget-policy.json",
    "data/hotpot-agent-v041/train.jsonl", "data/hotpot-agent-v041/manifest.json",
    "data/evidence-sft-v112/train.jsonl", "data/evidence-sft-v112/validation.jsonl",
    "data/evidence-sft-v112/manifest.json",
    "data/answer-repair-v105/manifest.json", "docs/v105-answer-repair-pilot.md",
    "configs/v105-answer-repair-pilot.toml",
    "backups/v100-v101/checkpoints/v101-answer50-seed42/adapter_model.safetensors",
    "backups/v100-v101/checkpoints/v101-answer50-seed42/adapter_config.json",
    "backups/v112-formal-adapter/adapter_model.safetensors",
    "backups/v112-formal-adapter/adapter_config.json",
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("archive inputs missing: " + repr(missing))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = REPORTS / f"v113-backup-{stamp}.tar.gz"
    if archive.exists():
        raise FileExistsError(str(archive))
    members = {}
    for name in FILES:
        source = ROOT / name
        members[name] = {"sha256": digest(source), "bytes": source.stat().st_size}
    # Snapshot SQLite through its backup API so the WAL state is consistent.
    ledger_source = ROOT / "runs/v110-api.sqlite"
    ledger_target = REPORTS / f"v113-ledger-snapshot-{stamp}.sqlite"
    with sqlite3.connect(f"file:{ledger_source.as_posix()}?mode=ro", uri=True) as src:
        with sqlite3.connect(ledger_target) as dst:
            src.backup(dst)
    members["runs/v110-api.sqlite"] = {"sha256": digest(ledger_target), "bytes": ledger_target.stat().st_size}
    payload_bytes = (json.dumps({"version": "v113-archive-payload", "members": members}, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    members["payload-manifest.json"] = {"sha256": hashlib.sha256(payload_bytes).hexdigest(), "bytes": len(payload_bytes)}
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for name in FILES:
            tar.add(ROOT / name, arcname=name, recursive=False)
        tar.add(ledger_target, arcname="runs/v110-api.sqlite", recursive=False)
        info = tarfile.TarInfo("payload-manifest.json")
        info.size = len(payload_bytes)
        tar.addfile(info, io.BytesIO(payload_bytes))
    # Verify every tar member against the staged manifest and write an external seal.
    with tarfile.open(archive, "r:gz") as tar:
        extracted = {m.name: m for m in tar.getmembers() if m.isfile()}
        for name, info in members.items():
            member = extracted.get(name)
            if member is None:
                raise ValueError("archive_member_missing:" + name)
            payload = tar.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != info["sha256"] or len(payload) != info["bytes"]:
                raise ValueError("archive_member_hash_mismatch:" + name)
        if set(extracted) != set(members):
            raise ValueError("unexpected_archive_members")
    seal = {
        "version": "v113",
        "archive": str(archive.relative_to(ROOT)).replace("\\", "/"),
        "archive_sha256": digest(archive),
        "archive_bytes": archive.stat().st_size,
        "verified_payload_files": len(members),
        "members": members,
        "ledger_snapshot_contains_unknown_call": True,
        "independent100_accessed": False,
        "final_test_accessed": False,
        "autodl_shutdown_requested": False,
    }
    seal_path = REPORTS / "v113-backup-verification.json"
    if seal_path.exists():
        raise FileExistsError(str(seal_path))
    seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in seal.items() if k != "members"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
