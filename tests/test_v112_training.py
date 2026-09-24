import json
from pathlib import Path

import pytest

from scripts import train_v112 as training


class Processor:
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(text.encode())}
    tokenizer = Tokenizer()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False
        return [token for message in messages for token in [1, *message["content"].encode(), 2]]


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'validate_dataset', lambda *args: {'passed': True})
    monkeypatch.setattr(training, 'check_budget', lambda *args: None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_model.safetensors").write_bytes(b"frozen")
    digest = training.sha256(source / "adapter_model.safetensors")
    monkeypatch.setattr(training, "SOURCE_SHA256", digest)
    row = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
           "target_assistant_index": 1, "thinking_enabled": False}
    files = {}
    for name in ("train", "validation"):
        files[name] = tmp_path / f"{name}.jsonl"
        files[name].write_text(json.dumps(row) + "\n", encoding="utf-8")
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"version": "v112", "approved": True,
                               "source_adapter_sha256": digest,
                               "artifacts": {name: {"sha256": training.sha256(path)} for name, path in files.items()}}))
    return dict(model_path=str(tmp_path / "model"), source_adapter=str(source),
                train_data=str(files["train"]), validation_data=str(files["validation"]),
                gate_report=str(gate), output_dir=str(tmp_path / "output"), processor=Processor())


def test_default_preflight_creates_nothing(setup_run):
    result = training.run(**setup_run)
    assert result["execute"] is False
    assert not Path(setup_run["output_dir"]).exists()


def test_overlap_rejected_before_changes(setup_run):
    setup_run["output_dir"] = str(Path(setup_run["source_adapter"]) / "child")
    with pytest.raises(ValueError, match="overlap"):
        training.run(**setup_run, execute=True)
    assert not Path(setup_run["output_dir"]).exists()


def test_data_hash_is_binding(setup_run):
    Path(setup_run["train_data"]).write_text("{}")
    with pytest.raises(ValueError, match="data hash mismatch"):
        training.run(**setup_run)


def test_smoke_and_formal_start_fresh(setup_run):
    calls = []
    def trainer(**kwargs):
        calls.append(kwargs)
        source = Path(kwargs["adapter_path"]) / "adapter_model.safetensors"
        assert source.read_bytes() == b"frozen"
        source.write_bytes(b"trained-copy")
        destination = Path(kwargs['output_dir'])
        destination.mkdir()
        (destination / 'adapter_model.safetensors').write_bytes(b'updated')
        return {"train_metrics": [{"loss": 1.0}]}
    report = training.run(**setup_run, execute=True, trainer=trainer,
                          reloader=lambda *args: {"reload_ok": True})
    assert [call["max_steps"] for call in calls] == [5, None]
    for call in calls:
        assert call["epochs"] == 1 and call["learning_rate"] == 5e-6
        assert call["gradient_accumulation_steps"] == 8
        assert call["gradient_checkpointing"] is False and call["enable_thinking"] is False
    assert report["source_before"] == report["source_after"]


def test_bad_smoke_prevents_formal(setup_run):
    calls = []
    def trainer(**kwargs):
        calls.append(kwargs)
        return {"train_metrics": [{"loss": float("nan")}]}
    with pytest.raises(ValueError, match="Nonfinite"):
        training.run(**setup_run, execute=True, trainer=trainer)
    assert len(calls) == 1
    assert not (Path(setup_run["output_dir"]) / "formal-source").exists()


def test_token_budget_and_last_target():
    row = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A" * 129}],
           "target_assistant_index": 1}
    with pytest.raises(ValueError, match="Token budget"):
        training.token_preflight(Processor(), [row])
    row["messages"][-1]["content"] = "A"
    row["target_assistant_index"] = 0
    with pytest.raises(ValueError, match="last-assistant"):
        training.token_preflight(Processor(), [row])


def test_output_cannot_be_inside_base_model(tmp_path):
    with pytest.raises(ValueError,match='protected'):
        training.check_paths(tmp_path/'adapter',tmp_path/'model'/'new',[tmp_path/'model'])


def test_missing_budget_gate_blocks_training(tmp_path):
    with pytest.raises(ValueError,match='budget'):
        training.check_budget({},tmp_path/'reports'/'gate.json')
