"""Create temporary, allowlisted AutoDL transfer archives; never include .env or sealed tests."""
from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V101 = ROOT / "backups/v115-final/unpacked/checkpoints/v101-answer50-seed42"
V112 = ROOT / "backups/v115-final/unpacked/checkpoints/v112-evidence-sft-seed42/formal/checkpoint-50"
MODEL = Path("D:/qwen_08b")
TEMP = Path(tempfile.gettempdir())


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add_tree(bundle: zipfile.ZipFile, base: Path, target: str, suffixes: set[str] | None = None):
    for path in sorted(base.rglob("*")):
        if not path.is_file(): continue
        if suffixes is not None and path.suffix.lower() not in suffixes: continue
        if path.name.lower().startswith(".env") or "final-test" in path.parts: raise ValueError("forbidden_file_in_transfer:"+str(path))
        bundle.write(path, (Path(target)/path.relative_to(base)).as_posix())


def package() -> dict:
    code_zip = TEMP / "v118-code-transfer.zip"
    model_zip = TEMP / "v118-model-transfer.zip"
    code_paths = [
        "pyproject.toml", "configs/v118-sft.json", "reports/v117-local-preflight.json",
        "reports/v118-token-audit-local.json", "backups/v117-pre-sft.zip",
        "backups/v117-pre-sft-manifest.json", "data/evidence-sft-v117/frozen",
        "docs/v117-pre-sft-preparation.md", "docs/v118-sft-experiment.md", "src/mini_agent_rl",
    ]
    script_names = ["train_v115.py", "train_v118.py", "audit_v118_tokens.py", "preflight_v118.py",
        "verify_v118_freeze.py", "evaluate_v118.py", "compare_v118_internal.py", "build_v118_blind_audit.py",
        "merge_v118_semantic_reviews.py", "review_v118_api.py", "package_v118_remote.py"]
    with zipfile.ZipFile(code_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        files = []
        for item in code_paths:
            path = ROOT/item
            if path.is_dir():
                for file in sorted(path.rglob("*")):
                    if file.is_file():
                        rel = file.relative_to(ROOT).as_posix()
                        if file.name.lower().startswith(".env") or "final-test" in file.parts: raise ValueError("forbidden_file_in_transfer:"+rel)
                        bundle.write(file, rel); files.append(rel)
            else:
                bundle.write(path, item); files.append(item)
        for name in script_names:
            item = "scripts/"+name; bundle.write(ROOT/item, item); files.append(item)
    with zipfile.ZipFile(model_zip, "w", compression=zipfile.ZIP_STORED) as bundle:
        for base, dest in ((MODEL, "models/qwen_08b"), (V101, "adapters/v101-answer50-seed42"),
                           (V112, "adapters/v112-evidence-sft-seed42")):
            if not base.is_dir(): raise FileNotFoundError(base)
            files = [p for p in sorted(base.rglob("*")) if p.is_file() and p.name in {"adapter_config.json", "adapter_model.safetensors"}]
            if base == MODEL: files = [p for p in sorted(base.rglob("*")) if p.is_file()]
            if not files: raise ValueError("empty_model_bundle:"+str(base))
            for file in files:
                if file.name.lower().startswith(".env"): raise ValueError("credential_file_in_model_bundle")
                bundle.write(file, (Path(dest)/file.relative_to(base)).as_posix())
    def describe(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""): digest.update(block)
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    result = {"version": "v118-remote-transfer-manifest-v1", "code_archive": describe(code_zip),
        "model_archive": describe(model_zip), "sealed_eval_included": False, "env_included": False,
        "adapter_sources": {"v101": "backups/v115-final/unpacked/checkpoints/v101-answer50-seed42",
                            "v112": "backups/v115-final/unpacked/checkpoints/v112-evidence-sft-seed42/formal/checkpoint-50"}}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__": package()
