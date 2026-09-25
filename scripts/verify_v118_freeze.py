"""Fail-closed integrity check for the local v117 SFT freeze used by v118."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data/evidence-sft-v117/frozen"
EXPECTED = {
    "train.jsonl": "26ced5bafb5210fefe9b007915598c8d58d2ebec156e938a48d86e1f1577ad8b",
    "internal.jsonl": "1a33ec08d5b23a365f0b679034806cfb89bdb48c59711db72b98dc253a4ab4f0",
    "internal-corpus.json": "989d7ea08187dfdbf2a07d69cbf6a8d7dd1127ac6d971c9f62b365475f46cea9",
    "internal-environment-manifest.json": "ccd334605167aaf928d84ecad0d9f53c698a106e2ac93f2d90b78a4c411f154a",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify() -> dict:
    manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    gate = json.loads((ROOT / "reports/v117-local-preflight.json").read_text(encoding="utf-8"))
    backup = json.loads((ROOT / "backups/v117-pre-sft-manifest.json").read_text(encoding="utf-8"))
    checks = {}
    archive = ROOT / backup["archive"].replace("\\", "/")
    checks["backup_archive_sha"] = archive.is_file() and sha(archive) == backup.get("archive_sha256")
    for name, expected in EXPECTED.items():
        actual = sha(FROZEN / name)
        checks[f"{name}_sha"] = actual == expected
    checks["freeze_manifest"] = all(
        manifest.get(field) == EXPECTED[name]
        for field, name in (("train_sha256", "train.jsonl"), ("internal_sha256", "internal.jsonl"),
                            ("corpus_sha256", "internal-corpus.json"),
                            ("environment_manifest_sha256", "internal-environment-manifest.json"))
    )
    checks["backup_hash_bindings"] = all(
        backup.get("files", {}).get(f"data/evidence-sft-v117/frozen/{name}", {}).get("sha256") == EXPECTED[name]
        for name in ("train.jsonl", "internal.jsonl", "internal-corpus.json", "internal-environment-manifest.json")
    )
    if archive.is_file():
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
            expected_names = set(backup.get("files", {}))
            checks["backup_contents"] = names == expected_names and all(
                hashlib.sha256(bundle.read(name)).hexdigest() == details.get("sha256")
                for name, details in backup.get("files", {}).items())
            checks["backup_has_no_env"] = not any(Path(name).name.lower().startswith(".env") for name in names)
    else:
        checks["backup_contents"] = False
        checks["backup_has_no_env"] = False
    checks["local_gate"] = gate.get("local_data_ready") is True and all(gate.get("checks", {}).values())
    checks["forbidden_eval_untouched"] = gate.get("independent100_accessed") is False and gate.get("final_test_accessed") is False
    train, internal = rows(FROZEN / "train.jsonl"), rows(FROZEN / "internal.jsonl")
    checks["row_counts"] = len(train) == 345 and len(internal) == 100
    checks["export_schema"] = all(set(row) == {"messages", "target_assistant_index", "thinking_enabled"} for row in train + internal)
    checks["internal_targets_not_training"] = sha(FROZEN / "train.jsonl") != sha(FROZEN / "internal.jsonl")
    env = json.loads((FROZEN / "internal-environment-manifest.json").read_text(encoding="utf-8"))
    corpus = json.loads((FROZEN / "internal-corpus.json").read_text(encoding="utf-8"))
    checks["internal_environment"] = (env.get("question_count") == 50 and env.get("state_count") == 100
        and env.get("document_count") == 1986 and len(corpus) == 1986
        and env.get("corpus_sha256") == EXPECTED["internal-corpus.json"])
    if not all(checks.values()):
        raise ValueError("v118_local_freeze_failed:" + ",".join(key for key, passed in checks.items() if not passed))
    return {"version": "v118-local-freeze-check-v1", "passed": True, "checks": checks,
            "hashes": EXPECTED, "train_states": len(train), "internal_states": len(internal),
            "internal_questions": 50, "corpus_documents": len(corpus), "training_permitted": False,
            "reason": "remote GPU, weights, environment, and token equivalence still require verification"}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
