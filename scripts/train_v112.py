"""Fail-closed v112 SFT launcher. Default invocation only performs preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from mini_agent_rl.training.sft import encode_assistant_only, load_jsonl, train_sft
from mini_agent_rl.evidence_sft_v112 import validate_dataset, validate_source_path, load_budget_policy

SOURCE_SHA256 = "3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(directory: Path) -> dict[str, str]:
    files = list(directory.rglob("*"))
    if directory.is_symlink() or any(p.is_symlink() for p in files):
        raise ValueError("Adapter must be an independent directory without symlinks")
    return {p.relative_to(directory).as_posix(): sha256(p) for p in files if p.is_file()}


def check_paths(source: Path, output: Path, protected: list[Path]) -> None:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output overlap")
    if output.exists():
        raise ValueError("Output must not already exist")
    if any(output == p.resolve() or output in p.resolve().parents or p.resolve() in output.parents for p in protected):
        raise ValueError("Output overlaps protected input")


def check_budget(gate: dict, gate_path: Path) -> None:
    budget = gate.get('budget_gate',{})
    if budget.get('passed') is not True:
        raise ValueError('Approved budget/provenance gate required')
    ledger = gate_path.resolve().parent.parent / 'runs/v110-api.sqlite'
    if not ledger.is_file():
        snapshot=gate.get('ledger_snapshot',{})
        relative=Path(snapshot.get('path',''))
        if not relative.parts or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Authoritative ledger missing')
        ledger=gate_path.resolve().parent.parent/relative
        if not ledger.is_file() or sha256(ledger)!=snapshot.get('sha256'):
            raise ValueError('Settled ledger verification snapshot missing or changed')
    with sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True) as db:
        old = db.execute("select * from calls where stage not like 'v112-%' order by key").fetchall()
        rows = db.execute('select stage,status,charged from calls').fetchall()
    prior = hashlib.sha256(json.dumps(old, ensure_ascii=False, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    if prior != budget.get('prior_rows_sha256') or any(status!='complete' for _,status,_ in rows):
        raise ValueError('Ledger prior rows changed or calls unresolved')
    policy=load_budget_policy(gate_path.resolve().parent.parent)
    if policy!=budget.get('policy'):
        raise ValueError('Budget authorization changed after data gate')
    if (policy['cumulative_cap_cny'] is not None and sum(r[2] for r in rows)>policy['cumulative_cap_cny']+1e-8) or (policy['new_cap_cny'] is not None and sum(r[2] for r in rows if r[0].startswith('v112-'))>policy['new_cap_cny']+1e-8):
        raise ValueError('API budget exceeded')


def token_preflight(processor: Any, rows: list[dict]) -> dict:
    maxima = {"input": 0, "output": 0, "full": 0}
    for row in rows:
        messages = row["messages"]
        last = len(messages) - 1
        if last < 1 or messages[last]["role"] != "assistant":
            raise ValueError("Training target must be the last assistant message")
        if row.get("target_assistant_index") != last or row.get("thinking_enabled", False) is not False:
            raise ValueError("Explicit last-assistant target and no-thinking required")
        encoded = encode_assistant_only(processor, messages, 2176, last, False)
        prompt = processor.apply_chat_template(messages[:-1], tokenize=True,
                                              add_generation_prompt=True, enable_thinking=False)
        if isinstance(prompt, dict):
            prompt = prompt["input_ids"]
        if prompt and isinstance(prompt[0], list):
            prompt = prompt[0]
        counts = {"input": len(prompt), "output": sum(t != -100 for t in encoded["labels"]),
                  "full": len(encoded["input_ids"])}
        if any(counts[k] > limit for k, limit in (("input", 2048), ("output", 128), ("full", 2176))):
            raise ValueError(f"Token budget exceeded, no truncation allowed: {counts}")
        for key, count in counts.items():
            maxima[key] = max(maxima[key], count)
    return maxima


def finite_metrics(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite training metrics")
    if isinstance(value, dict):
        for item in value.values():
            finite_metrics(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_metrics(item)


def reload_check(model_path: str, adapter: Path) -> dict:
    import gc
    import torch
    from mini_agent_rl.training.sft import build_lora_model
    gc.collect()
    torch.cuda.empty_cache()
    model, _ = build_lora_model(model_path, gradient_checkpointing=False, adapter_path=str(adapter))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or not all(torch.isfinite(p).all().item() for p in trainable):
        raise ValueError("Reloaded adapter has missing or nonfinite trainable parameters")
    del model, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return {"reload_ok": True}


def run(*, model_path: str, source_adapter: str, train_data: str, validation_data: str,
        gate_report: str, output_dir: str, execute: bool = False, processor=None,
        trainer=None, reloader=None) -> dict:
    source, output = Path(source_adapter), Path(output_dir)
    train, validation, gate_path = map(Path, (train_data, validation_data, gate_report))
    for path in (source, output, train, validation, gate_path, Path(model_path)):
        validate_source_path(path)
    check_paths(source, output, [train, validation, gate_path, Path(model_path)])
    before = fingerprint(source)
    if before.get("adapter_model.safetensors") != SOURCE_SHA256:
        raise ValueError("Frozen v101 source adapter hash mismatch")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("version") != "v112" or gate.get("approved") is not True:
        raise ValueError("A v112 approved data gate report is required")
    if gate.get("source_adapter_sha256") != SOURCE_SHA256:
        raise ValueError("Gate source adapter hash mismatch")
    check_budget(gate,gate_path)
    for name, path in (("train", train), ("validation", validation)):
        if gate.get("artifacts", {}).get(name, {}).get("sha256") != sha256(path):
            raise ValueError(f"Gate {name} data hash mismatch")
    train_rows, validation_rows = load_jsonl(train), load_jsonl(validation)
    checked = validate_dataset(train_rows, validation_rows)
    if not checked['passed']:
        raise ValueError('Data gate recheck failed: '+str(checked['errors']))
    if processor is None:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    report = {"version": "v112", "execute": execute, "source_before": before,
              "train_tokens": token_preflight(processor, train_rows),
              "validation_tokens": token_preflight(processor, validation_rows),
              "formal_updates": math.ceil(len(train_rows) / 8), "phases": {}}
    if not execute:
        return report
    trainer, reloader = trainer or train_sft, reloader or reload_check
    from transformers import set_seed
    set_seed(42)
    output.mkdir(parents=True)
    # Snapshot the exact approved bytes; training never reads mutable external data.
    for name, path in (("train", train), ("validation", validation)):
        shutil.copyfile(path, output / f"{name}.jsonl")
        if sha256(output / f"{name}.jsonl") != gate["artifacts"][name]["sha256"]:
            raise ValueError("Data changed after gate validation")
    try:
        for phase, max_steps in (("smoke", 5), ("formal", None)):
            set_seed(42)
            copied = output / f"{phase}-source"
            shutil.copytree(source, copied)
            if fingerprint(copied) != before:
                raise ValueError("Source changed before phase copy")
            destination = output / phase
            metrics = trainer(model_path=model_path, train_path=str(output / "train.jsonl"),
                              validation_path=str(output / "validation.jsonl"), output_dir=str(destination),
                              adapter_path=str(copied), max_steps=max_steps, epochs=1,
                              max_length=2176, seed=42, batch_size=1, gradient_accumulation_steps=8,
                              gradient_checkpointing=False, enable_thinking=False, learning_rate=5e-6,
                              save_steps=50, eval_steps=5 if phase == "smoke" else math.ceil(len(train_rows)/8))
            finite_metrics(metrics)
            if metrics.get('peak_gpu_memory_gb',0)>21:
                raise ValueError('GPU memory exceeded 21 GiB gate')
            logs = metrics.get("train_metrics", [])
            if not any("loss" in entry for entry in logs):
                raise ValueError("Missing observed training loss")
            trained_weight = destination / 'adapter_model.safetensors'
            if not trained_weight.is_file() or sha256(trained_weight) == SOURCE_SHA256:
                raise ValueError('Output adapter was not updated')
            report["phases"][phase] = {"metrics": metrics, **reloader(model_path, destination)}
            if report["phases"][phase].get("reload_ok") is not True:
                raise ValueError("Reload verification failed")
            if fingerprint(source) != before:
                raise ValueError("Frozen source changed during training")
    finally:
        report["source_after"] = fingerprint(source)
        (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if report["source_after"] != before:
            raise ValueError("Frozen source changed during training")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "source-adapter", "train-data", "validation-data", "gate-report", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly run GPU smoke then formal training")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
