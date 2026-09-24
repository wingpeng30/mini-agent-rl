"""Fail-closed v115 SFT: frozen v101 source, v114 approved train-only export."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE = "3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768"
EXPECTED_TRAIN = "56c32bf103ec4fd51e800c50eec4a64efd9233160729bf00712e9ccec407872f"
EXPECTED_INTERNAL = "9e40e3cd996cabb99ebaad96bf5e033e5a922e360e44faac932df6ee6da59b8c"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree(path: Path) -> dict[str, str]:
    if path.is_symlink() or any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("symlink_in_adapter")
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def token_signatures(processor: Any, records: list[dict]) -> list[dict]:
    from mini_agent_rl.training.sft import encode_assistant_only
    result=[]
    for i,row in enumerate(records):
        messages=row["messages"]
        prompt=processor.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if isinstance(prompt,dict): prompt=prompt["input_ids"]
        if prompt and isinstance(prompt[0],list):
            if len(prompt)!=1: raise ValueError(f"multiple_prompt_batches:{i}")
            prompt=prompt[0]
        enc=encode_assistant_only(processor,messages,2176,len(messages)-1,False)
        target_ids=[x for x in enc["labels"] if x!=-100]
        if not target_ids or enc["input_ids"][-(len(target_ids)+1):-1] != target_ids:
            raise ValueError(f"target_mask_not_final_assistant_span:{i}")
        digest=lambda v:hashlib.sha256(json.dumps(v,separators=(",",":" )).encode()).hexdigest()
        result.append({"index":i,"prompt_tokens":len(prompt),"target_tokens":len(target_ids),
            "full_tokens":len(enc["input_ids"]),"prompt_token_ids_sha256":digest(prompt),
            "full_token_ids_sha256":digest(enc["input_ids"]),"label_mask_sha256":digest(enc["labels"])})
    return result


def token_preflight(processor: Any, records: list[dict]) -> dict:
    from mini_agent_rl.training.sft import encode_assistant_only
    maximum = {"input": 0, "target": 0, "full": 0}
    for i, row in enumerate(records):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"bad_target_position:{i}")
        if row.get("target_assistant_index") != len(messages) - 1 or row.get("thinking_enabled") is not False:
            raise ValueError(f"bad_target_metadata:{i}")
        prompt = processor.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False)
        if isinstance(prompt, dict): prompt = prompt["input_ids"]
        if prompt and isinstance(prompt[0], list):
            if len(prompt) != 1: raise ValueError(f"multiple_prompt_batches:{i}")
            prompt = prompt[0]
        encoded = encode_assistant_only(processor, messages, 2176, len(messages)-1, False)
        target = sum(token != -100 for token in encoded["labels"])
        full = len(encoded["input_ids"])
        counts = {"input": len(prompt), "target": target, "full": full}
        if counts["input"] > 2048 or target > 128 or full > 2176:
            raise ValueError(f"token_budget_exceeded:{i}:{counts}")
        for key, value in counts.items(): maximum[key] = max(maximum[key], value)
    return {"count": len(records), "maxima": maximum, "all_fit": True,
            "label_mask": "assistant_final_message_only; observations masked by encode_assistant_only"}


def finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value): raise ValueError("nonfinite_training_metric")
    if isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value: finite(v)


def run(args) -> dict:
    train_path, internal_path = Path(args.train_data), Path(args.internal_data)
    gate_path, export_path, source, output, audit_path = map(Path, (args.gate_report, args.export_manifest, args.source_adapter, args.output_dir,args.local_token_audit))
    for path in (train_path, internal_path, gate_path, export_path, source, Path(args.model_path),audit_path):
        if not path.exists(): raise FileNotFoundError(path)
    if output.exists() or output == source or output in source.parents or source in output.parents:
        raise ValueError("unsafe_or_existing_output")
    source_before = tree(source)
    if source_before.get("adapter_model.safetensors") != EXPECTED_SOURCE: raise ValueError("v101_source_hash_mismatch")
    gate, manifest = json.loads(gate_path.read_text()), json.loads(export_path.read_text())
    if gate.get("passed") is not True or gate.get("training_permitted") is not True or gate.get("gate_errors"):
        raise ValueError("v114_gate_not_passed")
    if gate.get("final_test_accessed") or gate.get("independent100_evaluated"):
        raise ValueError("forbidden_eval_already_recorded")
    if manifest.get("version") != "v114-sft-export-manifest-v1" or manifest.get("gate_report_sha256") != sha(gate_path):
        raise ValueError("v114_export_manifest_mismatch")
    if sha(train_path) != EXPECTED_TRAIN or sha(internal_path) != EXPECTED_INTERNAL:
        raise ValueError("v114_export_sha_mismatch")
    if manifest["exports"]["sft-train.jsonl"]["sha256"] != sha(train_path) or manifest["exports"]["sft-internal.jsonl"]["sha256"] != sha(internal_path):
        raise ValueError("export_manifest_content_mismatch")
    train, internal = rows(train_path), rows(internal_path)
    if len(train) != 300 or len(internal) != 100: raise ValueError("v114_export_count_mismatch")
    if args.dry_run:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        if audit.get('version')!='v115-token-audit-v1' or audit.get('train_sha256')!=sha(train_path) or audit.get('internal_sha256')!=sha(internal_path):
            raise ValueError('local_token_audit_export_mismatch')
        train_sig,internal_sig=token_signatures(processor,train),token_signatures(processor,internal)
        if audit.get('train_signatures')!=train_sig or audit.get('internal_signatures')!=internal_sig:
            raise ValueError('local_remote_token_mismatch')
        return {"version":"v115","dry_run":True,"source_sha256":EXPECTED_SOURCE,
                "train_sha256":sha(train_path),"internal_sha256":sha(internal_path),
                "train_tokens":token_preflight(processor, train),
                "internal_tokens":token_preflight(processor, internal),
                "token_equivalence":{"local_audit_sha256":sha(audit_path),"train_tokens_equal":True,"internal_tokens_equal":True},
                "internal_used_for_training":False,"formal_max_steps":50}
    if not args.execute: raise ValueError("pass --dry-run or --execute")
    from transformers import set_seed, AutoProcessor
    from mini_agent_rl.training.sft import train_sft, build_lora_model
    output.mkdir(parents=True)
    train_copy = output / "sft-train.jsonl"
    shutil.copyfile(train_path, train_copy)
    if sha(train_copy) != EXPECTED_TRAIN: raise ValueError("train_copy_hash_mismatch")
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    preflight = token_preflight(processor, train)
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    train_sig,internal_sig=token_signatures(processor,train),token_signatures(processor,internal)
    if audit.get('version')!='v115-token-audit-v1' or audit.get('train_sha256')!=sha(train_path) or audit.get('internal_sha256')!=sha(internal_path):
        raise ValueError('local_token_audit_export_mismatch')
    if audit.get('train_signatures')!=train_sig or audit.get('internal_signatures')!=internal_sig:
        raise ValueError('local_remote_token_mismatch')
    # Deliberately use the training split as Trainer's loss-only eval input.
    # The internal split never reaches the trainer, sampler, or gradients.
    result = {"version":"v115","config":{"dtype":"bfloat16","learning_rate":5e-6,"batch_size":1,
              "gradient_accumulation_steps":8,"seed":42,"gradient_checkpointing":False,
              "thinking":False,"smoke_steps":5,"formal_steps":50,"max_input":2048,"max_target":128,
              "max_full":2176},"source_sha256":EXPECTED_SOURCE,"source_before":source_before,
              "train_sha256":EXPECTED_TRAIN,"train_token_preflight":preflight,
              "token_equivalence":{"local_audit_sha256":sha(audit_path),"train_tokens_equal":True,"internal_tokens_equal":True},
              "internal_sha256":EXPECTED_INTERNAL,"internal_used_for_training":False,"phases":{}}
    try:
        for phase, steps in (("smoke",5),("formal",50)):
            set_seed(42)
            source_copy = output / f"{phase}-source-v101"
            shutil.copytree(source, source_copy)
            if tree(source_copy) != source_before: raise ValueError("source_copy_mismatch")
            destination = output / phase
            metrics = train_sft(model_path=args.model_path, train_path=str(train_copy),
                validation_path=str(train_copy), output_dir=str(destination), max_steps=steps, epochs=1,
                max_length=2176, seed=42, batch_size=1, gradient_accumulation_steps=8,
                gradient_checkpointing=False, adapter_path=str(source_copy), enable_thinking=False,
                learning_rate=5e-6, save_steps=steps, eval_steps=steps, warmup_ratio=0.05)
            finite(metrics)
            logs = metrics.get("train_metrics", [])
            if not any("loss" in item for item in logs): raise ValueError(f"{phase}_loss_missing")
            if any("grad_norm" in item and not math.isfinite(float(item["grad_norm"])) for item in logs):
                raise ValueError(f"{phase}_gradient_nonfinite")
            expected_checkpoint = destination / f"checkpoint-{steps}"
            if not (expected_checkpoint / "adapter_model.safetensors").is_file():
                raise ValueError(f"{phase}_last_checkpoint_missing")
            if sha(expected_checkpoint / "adapter_model.safetensors") == EXPECTED_SOURCE:
                raise ValueError(f"{phase}_did_not_update_weights")
            __import__("gc").collect(); __import__("torch").cuda.empty_cache()
            model, processor = build_lora_model(args.model_path, gradient_checkpointing=False,
                                               adapter_path=str(expected_checkpoint))
            finite_parameters = all(__import__("torch").isfinite(p).all().item() for p in model.parameters() if p.requires_grad)
            del model
            if not finite_parameters: raise ValueError(f"{phase}_reload_nonfinite")
            result["phases"][phase] = {"steps":steps,"metrics":metrics,
                "last_checkpoint":str(expected_checkpoint),"adapter_sha256":sha(expected_checkpoint / "adapter_model.safetensors"),
                "reload_ok":True}
            if tree(source) != source_before: raise ValueError("frozen_v101_changed")
        result["source_after"] = tree(source)
        if result["source_after"] != source_before: raise ValueError("frozen_v101_changed")
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        (output / "training_report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        raise
    (output / "training_report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ("model-path","source-adapter","train-data","internal-data","gate-report","export-manifest","local-token-audit","output-dir"):
        p.add_argument("--"+name,required=True)
    mode=p.add_mutually_exclusive_group(required=True); mode.add_argument("--dry-run",action="store_true"); mode.add_argument("--execute",action="store_true")
    print(json.dumps(run(p.parse_args()),ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
