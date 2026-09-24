import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_v115


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_args(tmp_path, monkeypatch, *, gate_passed=True):
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"frozen-v101")
    monkeypatch.setattr(train_v115, "EXPECTED_SOURCE", digest(source / "adapter_model.safetensors"))
    model = tmp_path / "model"
    model.mkdir()
    train = tmp_path / "train.jsonl"
    internal = tmp_path / "internal.jsonl"
    train.write_text('{"old":"v112-format"}\n', encoding="utf-8")
    internal.write_text('{"old":"v112-format"}\n', encoding="utf-8")
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"passed": gate_passed, "training_permitted": gate_passed,
                                "gate_errors": [], "final_test_accessed": False,
                                "independent100_evaluated": False}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": "v114-sft-export-manifest-v1",
                                    "gate_report_sha256": digest(gate),
                                    "exports": {"sft-train.jsonl": {"sha256": train_v115.EXPECTED_TRAIN},
                                                "sft-internal.jsonl": {"sha256": train_v115.EXPECTED_INTERNAL}}}), encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text("{}", encoding="utf-8")
    out = tmp_path / "output"
    return SimpleNamespace(train_data=str(train), internal_data=str(internal), gate_report=str(gate),
                           export_manifest=str(manifest), source_adapter=str(source), output_dir=str(out),
                           model_path=str(model), local_token_audit=str(audit), dry_run=True, execute=False), out


def test_v115_rejects_failed_v114_gate_before_creating_output(tmp_path, monkeypatch):
    args, output = setup_args(tmp_path, monkeypatch, gate_passed=False)
    with pytest.raises(ValueError, match="v114_gate_not_passed"):
        train_v115.run(args)
    assert not output.exists()


def test_v115_rejects_old_v112_export_format_before_tokenizer_or_training(tmp_path, monkeypatch):
    args, output = setup_args(tmp_path, monkeypatch, gate_passed=True)
    with pytest.raises(ValueError, match="v114_export_sha_mismatch"):
        train_v115.run(args)
    assert not output.exists()
