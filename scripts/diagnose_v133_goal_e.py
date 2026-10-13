"""Frozen Goal E CPU preflight and bounded, inference-only adapter diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mini_agent_rl.training.sft import encode_assistant_only, load_jsonl
from evaluate_v133_goal_c_aligned_context import aligned_rows
from evaluate_v133_goal_c_sft import f1_score, normalize_answer, parse_action

TRAIN = ROOT / "data/evidence-sft-v133-goal-c-private/frozen/train-candidate.jsonl"
DEV = ROOT / "data/evidence-sft-v133-goal-c-private/frozen/dev-candidate.jsonl"
ADAPTERS = {
    "v118": (ROOT / "backups/v118/extracted/runs/v118/formal/adapter", "ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5"),
    "sft_c": (ROOT / "runs/v133-goal-c-sft-once/adapter", "0a80d82918e301cac05f0cc2892a7ccd007138996e121549b95dbcbf7f8919c6"),
    "sft_d": (ROOT / "runs/v133-goal-d-sft-once/adapter", "b8782a1a36ddaa44e2a2d7446763b7e50ae40865aae326366d3f0e50b9fc94af"),
}
EXPECTED = {TRAIN: "b4d38e1e99886e3a254a6e881a28b7a36194f7f185b686a64dc7dbdf449ee9df", DEV: "23cde9d8f25376d469c91128742ca647fc4d84afe04b2009e38e571198780e58"}
CAP = 7200


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def inputs(model_path: Path) -> dict:
    frozen = json.loads((ROOT / "data/evidence-sft-v133-goal-c-private/frozen/freeze-manifest-private.json").read_text(encoding="utf-8"))
    checks = {str(path.relative_to(ROOT)): sha(path) for path in EXPECTED}
    for path, expected in EXPECTED.items():
        if checks[str(path.relative_to(ROOT))] != expected:
            raise ValueError(f"data_hash_mismatch:{path}")
    for name, (directory, expected) in ADAPTERS.items():
        actual = sha(directory / "adapter_model.safetensors")
        if actual != expected:
            raise ValueError(f"adapter_hash_mismatch:{name}")
        checks[f"adapter:{name}"] = actual
        checks[f"adapter_config:{name}"] = sha(directory / "adapter_config.json")
    for group in ("tokenizer_sha256", "model_runtime_files_sha256", "model_weight_files_sha256"):
        for name, expected in frozen["baseline"][group].items():
            actual = sha(model_path / name)
            if actual != expected:
                raise ValueError(f"model_hash_mismatch:{name}")
            checks[f"model:{name}"] = actual
    checks["script:diagnostic"] = sha(Path(__file__))
    checks["script:encoder"] = sha(ROOT / "src/mini_agent_rl/training/sft.py")
    checks["script:aligned_context"] = sha(ROOT / "scripts/evaluate_v133_goal_c_aligned_context.py")
    return checks


def rows() -> list[tuple[str, int, dict]]:
    train = load_jsonl(TRAIN)
    dev = aligned_rows(load_jsonl(DEV))
    if len(train) != 25 or len(dev) != 16:
        raise ValueError("row_count_mismatch")
    return [("train", i, row) for i, row in enumerate(train)] + [("dev", i, row) for i, row in enumerate(dev)]


def prompt_ids(processor, messages):
    value = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if isinstance(value, dict):
        value = value["input_ids"]
    return list(value[0] if value and isinstance(value[0], list) else value)


def inspect_row(processor, split: str, index: int, row: dict) -> tuple[dict, dict]:
    messages = row["messages"]
    target = int(row["target_assistant_index"])
    if target != len(messages) - 1 or messages[target]["role"] != "assistant":
        raise ValueError(f"wrong_target:{split}:{index}")
    encoded = encode_assistant_only(processor, messages, 2176, target, False)
    full, labels = encoded["input_ids"], encoded["labels"]
    prompt = prompt_ids(processor, messages[:target])
    positions = [i for i, value in enumerate(labels) if value != -100]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError(f"non_contiguous_target:{split}:{index}")
    if full[:positions[0]] != prompt:
        raise ValueError(f"generation_training_prefix_mismatch:{split}:{index}:{len(prompt)}:{positions[0]}")
    if labels[positions[0]:positions[-1]+1] != full[positions[0]:positions[-1]+1]:
        raise ValueError(f"label_values_mismatch:{split}:{index}")
    if full[positions[-1]] != processor.tokenizer.convert_tokens_to_ids("<|im_end|>"):
        raise ValueError(f"terminator_not_supervised:{split}:{index}")
    content = messages[target]["content"]
    action = json.loads(content)
    if action.get("action") != "answer" or not isinstance(action.get("answer"), str):
        raise ValueError(f"invalid_target:{split}:{index}")
    # Offset mapping is checked against the actual content tokens. Tokens that
    # cross the answer-value boundary are explicitly marked ambiguous.
    tokenized = processor.tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
    content_ids = tokenized["input_ids"]
    content_positions = positions[:-1]
    if content_ids != [full[p] for p in content_positions]:
        raise ValueError(f"content_tokenization_mismatch:{split}:{index}")
    answer_start = content.index(json.dumps(action["answer"], ensure_ascii=False)[1:-1], content.index('"answer"'))
    answer_end = answer_start + len(action["answer"])
    groups = []
    for start, end in tokenized["offset_mapping"]:
        if start >= answer_start and end <= answer_end:
            groups.append("answer")
        elif end <= answer_start or start >= answer_end:
            groups.append("format")
        else:
            groups.append("ambiguous")
    groups.append("format")
    if "ambiguous" in groups:
        raise ValueError(f"answer_boundary_ambiguous:{split}:{index}")
    # All source observations must survive rendering; target answer must not.
    rendered = processor.apply_chat_template(messages[:target], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if messages[1]["content"].strip() not in rendered or any(m["content"].strip() not in rendered for m in messages[:target] if m["role"] == "tool"):
        raise ValueError(f"source_missing_from_prompt:{split}:{index}")
    if content in rendered:
        raise ValueError(f"target_leakage:{split}:{index}")
    summary = {"split": split, "index": index, "target_index": target, "prompt_tokens": len(prompt), "full_tokens": len(full), "target_tokens": len(positions), "answer_tokens": groups.count("answer"), "format_tokens": groups.count("format"), "question_sha256": hashlib.sha256(messages[1]["content"].encode()).hexdigest()}
    return summary, {"row": row, "prompt": prompt, "full": full, "labels": labels, "groups": groups, "positions": positions, "gold": action["answer"]}


def preflight(model_path: Path, output: Path) -> tuple[dict, list[dict]]:
    from transformers import AutoProcessor
    import importlib.metadata as metadata
    checks = inputs(model_path)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    prepared = [inspect_row(processor, *item) for item in rows()]
    versions = {name: metadata.version(name) for name in ("torch", "transformers", "peft", "tokenizers", "safetensors")}
    report = {"status": "passed", "checks": checks, "runtime_versions": versions, "rows": [x[0] for x in prepared], "answer_tokens": sum(x[0]["answer_tokens"] for x in prepared), "format_tokens": sum(x[0]["format_tokens"] for x in prepared)}
    save(output, report)
    return report, [x[1] for x in prepared]


def guard(start: float):
    if time.monotonic() - start >= CAP:
        raise TimeoutError("goal_e_gpu_cap_7200_seconds")


def score(raw: str, gold: str) -> dict:
    try:
        prediction = parse_action(raw)["answer"]
        valid = True
    except (ValueError, json.JSONDecodeError):
        prediction, valid = "", False
    return {"raw_output": raw, "prediction": prediction, "gold": gold, "protocol_valid": valid, "em": int(valid and normalize_answer(prediction) == normalize_answer(gold)), "f1": f1_score(prediction, gold) if valid else 0.0}


def teacher_metrics(model, processor, item: dict) -> dict:
    import torch
    import torch.nn.functional as F
    device = next(model.parameters()).device
    ids = torch.tensor([item["full"]], device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
        positions = torch.tensor(item["positions"], device=device)
        selected = logits[positions - 1].float()
        correct = ids[0, positions]
        nll = F.cross_entropy(selected, correct, reduction="none")
        hit = selected.argmax(-1) == correct
    output = {}
    for group in ("answer", "format"):
        mask = torch.tensor([g == group for g in item["groups"]], device=device)
        output[group] = {"count": int(mask.sum()), "nll_mean": float(nll[mask].mean()), "accuracy": float(hit[mask].float().mean()), "nll_sum": float(nll[mask].sum())}
    return output


def generate(model, processor, item: dict) -> tuple[dict, float]:
    import torch
    device = next(model.parameters()).device
    ids = torch.tensor([item["prompt"]], device=device)
    t0 = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=128, pad_token_id=processor.tokenizer.eos_token_id)
    elapsed = time.monotonic() - t0
    raw = processor.tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return score(raw, item["gold"]), elapsed


def anchor(model, item: dict) -> dict:
    import torch
    ids = torch.tensor([item["prompt"]], device=next(model.parameters()).device)
    with torch.inference_mode():
        enabled1 = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1].float().cpu()
        enabled2 = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1].float().cpu()
        with model.disable_adapter():
            disabled = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1].float().cpu()
    return {"enabled_disabled_max_abs": float((enabled1-disabled).abs().max()), "repeat_max_abs": float((enabled1-enabled2).abs().max()), "enabled_argmax": int(enabled1.argmax()), "disabled_argmax": int(disabled.argmax()), "enabled_top32": {str(int(i)): float(enabled1[i]) for i in torch.topk(enabled1, 32).indices}, "enabled_full_sha256": hashlib.sha256(enabled1.numpy().tobytes()).hexdigest()}


def run_inference(model_path: Path, output: Path, prepared: list[dict], report: dict) -> None:
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.sft import build_lora_model
    start = time.monotonic()
    state = {"status": "running", "cpu_preflight_sha256": sha(output.with_name("cpu-preflight.json")), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model_results": {}, "stop_reason": None}
    save(output, state)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda_unavailable")
        free, total = torch.cuda.mem_get_info()
        if total / 1024**3 < 23.5 or free / 1024**3 < 22:
            raise RuntimeError("gpu_memory_below_frozen_floor")
        for name, (directory, _) in ADAPTERS.items():
            guard(start)
            set_seed(42)
            model, processor = build_lora_model(str(model_path), dtype="bfloat16", gradient_checkpointing=False, adapter_path=str(directory))
            model.eval()
            status = {"adapter_path": str(directory), "active_adapters": list(model.active_adapters), "training": model.training, "matched_lora_modules": sum(1 for m in model.modules() if hasattr(m, "lora_A")), "adapter_dtypes": sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" in n}), "anchors": {}, "rows": {}}
            for index in (0, 8, 16, 24):
                guard(start)
                status["anchors"][str(index)] = anchor(model, prepared[index])
            durations = []
            for number, item in enumerate(prepared):
                guard(start)
                split = "train" if number < 25 else "dev"
                index = number if split == "train" else number - 25
                record = {"teacher": teacher_metrics(model, processor, item)}
                if split == "train":
                    record["generation"], duration = generate(model, processor, item)
                    durations.append(duration)
                status["rows"][f"{split}:{index:02d}"] = record
                state["model_results"][name] = status
                save(output, state)
                if name == "v118" and number == 1:
                    projected = (time.monotonic() - start) + (sum(durations) / 2) * 73 * 3 + 41 * 3 * max(durations) + 60
                    state["speed_projection_seconds"] = round(projected, 2)
                    if projected > CAP:
                        raise TimeoutError("first_two_projection_exceeds_7200_seconds")
            del model, processor
            torch.cuda.empty_cache()
        state["status"] = "complete"
    except Exception as exc:
        state["status"] = "stopped"
        state["stop_reason"] = f"{type(exc).__name__}:{exc}"
        raise
    finally:
        state["elapsed_seconds"] = round(time.monotonic() - start, 3)
        save(output, state)


def run_dev_inference(model_path: Path, output: Path, prepared: list[dict]) -> None:
    """Refresh all 48 dev outputs because old reports lack prompt-token hashes."""
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.sft import build_lora_model
    from copy import deepcopy
    start = time.monotonic()
    state = {"status": "running", "reason": "historical_reports_do_not_store_prompt_token_hashes", "results": {}, "stop_reason": None}
    save(output, state)
    try:
        for name, (directory, _) in ADAPTERS.items():
            guard(start)
            set_seed(42)
            model, processor = build_lora_model(str(model_path), dtype="bfloat16", gradient_checkpointing=False, adapter_path=str(directory))
            model.eval()
            for index in range(16):
                guard(start)
                item = prepared[25 + index]
                result, seconds = generate(model, processor, item)
                result["prompt_sha256"] = hashlib.sha256(bytes().join(int(x).to_bytes(4, "little") for x in item["prompt"])).hexdigest()
                result["generation_seconds"] = round(seconds, 3)
                if index in (0, 2, 3, 15) and result["protocol_valid"]:
                    comparison = {}
                    for label, answer in (("gold", item["gold"]), ("prediction", result["prediction"])):
                        modified = deepcopy(item["row"])
                        modified["messages"][-1]["content"] = json.dumps({"action": "answer", "answer": answer}, ensure_ascii=False, separators=(",", ":"))
                        _, candidate = inspect_row(processor, "dev", index, modified)
                        comparison[label] = teacher_metrics(model, processor, candidate)["answer"]
                        comparison[label]["token_ids"] = [candidate["full"][p] for p, group in zip(candidate["positions"], candidate["groups"]) if group == "answer"]
                    gold_ids, pred_ids = comparison["gold"]["token_ids"], comparison["prediction"]["token_ids"]
                    first = next((j for j, (a, b) in enumerate(zip(gold_ids, pred_ids)) if a != b), min(len(gold_ids), len(pred_ids)))
                    comparison["first_divergent_answer_token"] = first
                    result["candidate_likelihood"] = comparison
                state["results"][f"{name}:{index:02d}"] = result
                save(output, state)
                if name == "v118" and index == 1:
                    projected = (time.monotonic() - start) + (result["generation_seconds"] + state["results"]["v118:00"]["generation_seconds"]) / 2 * 46 + 120
                    state["speed_projection_seconds"] = round(projected, 2)
                    if projected > CAP:
                        raise TimeoutError("dev_projection_exceeds_7200_seconds")
            del model, processor
            torch.cuda.empty_cache()
        state["status"] = "complete"
    except Exception as exc:
        state["status"] = "stopped"
        state["stop_reason"] = f"{type(exc).__name__}:{exc}"
        raise
    finally:
        state["elapsed_seconds"] = round(time.monotonic() - start, 3)
        save(output, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "infer", "dev-infer"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("output_directory_not_empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, prepared = preflight(args.model_path, args.output_dir / "cpu-preflight.json")
    print(json.dumps({"cpu_preflight": report["status"], "rows": len(prepared), "answer_tokens": report["answer_tokens"], "format_tokens": report["format_tokens"]}), flush=True)
    if args.mode == "infer":
        run_inference(args.model_path, args.output_dir / "diagnostic.json", prepared, report)
    elif args.mode == "dev-infer":
        run_dev_inference(args.model_path, args.output_dir / "dev-generation.json", prepared)


if __name__ == "__main__":
    main()
