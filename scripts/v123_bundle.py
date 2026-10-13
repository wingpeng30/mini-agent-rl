"""Create two explicit, hash-manifested AutoDL upload archives.

The review archive contains only v123 CPU code and approved inputs. Official
training shards are in a separate archive so review transport can start first.
Neither archive contains credentials, adapters, model weights, or sealed sets.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from scripts.v123_cpu_core import ROOT, OUT, digest, safe, sha, source_manifest, validate_archive_members, write

TRANSFER = OUT / "transfer"


def _review_paths() -> list[Path]:
    listed = [
        "scripts/run_v123_cpu_audit.py", "scripts/v123_cpu_core.py", "scripts/v123_review.py",
        "scripts/v123_api_protocol.py", "scripts/v123_api_review.py", "scripts/v123_train_assess.py",
        "scripts/v123_train_review.py",
        "scripts/v123_prepare_internal.py", "scripts/v123_query_gate.py", "scripts/v123_token_replay.py",
        "scripts/v123_verify_upload.py",
        "docs/v123-luna-blind-rubric.md",
        "src/mini_agent_rl/__init__.py", "src/mini_agent_rl/domain/__init__.py",
        "src/mini_agent_rl/domain/models.py",
        "src/mini_agent_rl/tools/__init__.py", "src/mini_agent_rl/tools/registry.py",
        "src/mini_agent_rl/tools/search.py",
        "data/evidence-sft-v117/frozen/train.jsonl",
        "reports/v117-local-preflight.json",
        "data/evidence-sft-v122-internal-candidates/history-private.json",
        "data/evidence-sft-v122-internal-candidates/candidate-corpus-private.json",
        "data/evidence-sft-v122-internal-candidates/review/internal-packets.jsonl",
        "runs/v122-download/v122-unified/fixed-summary.json",
        "runs/v122-download/dual-audit-report.json",
        "backups/v118/extracted/runs/v118/eval/v101-tasks.json",
        "backups/v118/extracted/runs/v118/eval/v112-tasks.json",
        "backups/v118/extracted/runs/v118/eval/v118-tasks.json",
        "data/evidence-sft-v122-train-candidates/history-private.json",
        "data/evidence-sft-v122-train-candidates/candidate-freeze.json",
        "data/evidence-sft-v122-train-candidates/candidate-corpus-private.json",
        "data/evidence-sft-v122-train-candidates/token-replay.json",
        "data/evidence-sft-v122-train-candidates/review/manifest.json",
        "data/evidence-sft-v122-train-candidates/review/eligible-private.jsonl",
        "data/evidence-sft-v122-train-candidates/review/internal-packets.jsonl",
        "data/evidence-sft-v123/internal/candidate-freeze.json",
        "data/evidence-sft-v123/internal/token-replay.json",
        "data/evidence-sft-v123/internal/candidate-corpus-private.json",
        "data/evidence-sft-v123/internal/review/internal-packets.jsonl",
        "data/evidence-sft-v123/internal/review/source-map-private.jsonl",
        "data/evidence-sft-v123/internal/review/old-vote-reuse-private.jsonl",
    ]
    roots = [ROOT / item for item in listed]
    for prefix, base in (("api", ROOT / "data/evidence-sft-v122-internal-candidates/review"),
                         ("luna", ROOT / "data/evidence-sft-v122-internal-candidates/review"),
                         ("api", ROOT / "data/evidence-sft-v122-train-candidates/review")):
        roots.extend(sorted(base.glob(prefix + "-internal-batch-*.jsonl")))
    return [safe(p) for p in roots]


def _write_archive(path: Path, files: dict[str, Path], *, compress: int) -> dict:
    if path.exists():
        raise ValueError("one_time_bundle_already_exists")
    names = list(files)
    if len(names) != len(set(names)) or any(not p.is_file() for p in files.values()):
        raise ValueError("bundle_missing_or_duplicate_source")
    def file_hash(source: Path) -> str:
        if not (source.resolve().is_relative_to(ROOT.resolve()) or
                source.resolve().is_relative_to(Path("D:/qwen_08b").resolve())):
            raise ValueError("bundle_source_outside_whitelist")
        return hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {"version": "v123-transfer-v1", "files": {name: file_hash(p) for name, p in files.items()},
                "training_allowed": False, "credentials_included": False, "sealed_assets_included": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=compress, allowZip64=True) as archive:
        for name, source in files.items():
            archive.write(source, arcname=name)
        archive.writestr("v123-transfer-manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    validate_archive_members(path)
    return {"archive": str(path), "sha256": sha(path), "files": len(files),
            "manifest_sha256": digest(manifest), "training_allowed": False}


def build_review() -> dict:
    files = {str(p.relative_to(ROOT)).replace("\\", "/"): p for p in _review_paths()}
    tokenizer = Path("D:/qwen_08b")
    for name in ("tokenizer.json", "tokenizer_config.json", "processor_config.json",
                 "special_tokens_map.json", "chat_template.jinja"):
        path = tokenizer / name
        if path.is_file():
            files["tokenizer/" + name] = path
    result = _write_archive(TRANSFER / "v123-review-upload-v5.zip", files, compress=ZIP_DEFLATED)
    write(TRANSFER / "review-upload-v5-manifest.json", result)
    return result


def build_sources() -> dict:
    source = source_manifest()
    files = {"data/hotpot-full-train-v114/" + p.name: p for p in source["files"]}
    provenance = ROOT / "data/hotpot-full-train-v114/download-provenance.json"
    files["data/hotpot-full-train-v114/download-provenance.json"] = provenance
    result = _write_archive(TRANSFER / "v123-official-source-upload.zip", files, compress=ZIP_STORED)
    write(TRANSFER / "source-upload-manifest.json", result)
    return result
