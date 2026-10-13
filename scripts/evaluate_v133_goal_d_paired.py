"""One-shot paired evaluation of v118 versus the single Goal D continuation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import time
from pathlib import Path

from evaluate_v133_goal_c_aligned_context import aligned_rows
from evaluate_v133_goal_c_sft import check_gpu_floor, evaluate_adapter, load_rows, sha

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/v133-goal-d-sft-incremental.json"


def verify_inputs(cfg: dict, model: Path, train: Path, dev: Path, v118: Path,
                  parent: Path, prior_eval: Path, frozen: dict) -> dict[str, str]:
    expected = {
        "train": (train, cfg["train_sha256"]),
        "dev": (dev, cfg["dev_sha256"]),
        "v118_adapter": (v118 / "adapter_model.safetensors", cfg["v118_adapter_sha256"]),
        "parent_adapter": (parent / "adapter_model.safetensors", cfg["initial_adapter_sha256"]),
        "prior_aligned_eval": (prior_eval, cfg["prior_aligned_eval_sha256"]),
    }
    actual: dict[str, str] = {}
    for name, (path, digest) in expected.items():
        actual[name] = sha(path)
        if actual[name] != digest:
            raise SystemExit(f"input_hash_mismatch:{name}")
    for group in ("model_weight_files_sha256", "tokenizer_sha256", "model_runtime_files_sha256"):
        for name, digest in frozen["baseline"][group].items():
            actual[f"{group}:{name}"] = sha(model / name)
            if actual[f"{group}:{name}"] != digest:
                raise SystemExit(f"model_asset_hash_mismatch:{name}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize-model-inference", action="store_true")
    parser.add_argument("--cpu-only-preflight", action="store_true")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-minutes", type=int, default=60)
    args = parser.parse_args()
    if not args.cpu_only_preflight and not args.authorize_model_inference:
        raise SystemExit("explicit_model_inference_authorization_required")
    if not 1 <= args.max_minutes <= 60:
        raise SystemExit("max_minutes_out_of_bounds")
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "data/evidence-sft-v133-goal-c-private/frozen/freeze-manifest-private.json").read_text(encoding="utf-8"))
    model = Path(args.model_path).resolve()
    dev = ROOT / cfg["dev_data"]
    train = ROOT / cfg["train_data"]
    v118 = ROOT / cfg["v118_adapter"]
    parent = ROOT / cfg["initial_adapter"]
    candidate = ROOT / "runs/v133-goal-d-sft-once/adapter"
    prior_eval = ROOT / cfg["prior_aligned_eval"]
    output = Path(args.output)
    if output.exists():
        raise SystemExit("output_exists_no_retry_or_overwrite")
    asset_hashes = verify_inputs(cfg, model, train, dev, v118, parent, prior_eval, frozen)
    versions = {name: metadata.version(name) for name in cfg["runtime_dependencies"]}
    if versions != cfg["runtime_dependencies"]:
        raise SystemExit("runtime_version_mismatch")
    rows = aligned_rows(load_rows(dev))
    if args.cpu_only_preflight:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        lengths = []
        for row in rows:
            prefix = row["messages"][:6]
            rendered = processor.apply_chat_template(prefix, tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
            if any(prefix[i]["content"] not in rendered for i in (3, 5)):
                raise ValueError("evidence_missing_from_prompt_render")
            encoded = processor.apply_chat_template(prefix, tokenize=True,
                                                    add_generation_prompt=True, enable_thinking=False)
            ids = encoded[0] if encoded and isinstance(encoded[0], list) else encoded
            lengths.append(len(ids))
        if max(lengths) > 2048:
            raise ValueError("prompt_token_limit_exceeded")
        print(json.dumps({"status": "cpu_preflight_passed", "n": len(rows),
                          "prompt_token_min": min(lengths), "prompt_token_max": max(lengths),
                          "asset_hashes": asset_hashes}, indent=2))
        return

    start_path = ROOT / "runs/v133-goal-d-sft-once/autodl-run-20260928/training-start.json"
    metrics_path = candidate / "training_metrics.json"
    if not candidate.is_dir() or not start_path.is_file() or not metrics_path.is_file():
        raise SystemExit("incremental_training_outputs_missing")
    training_start = json.loads(start_path.read_text(encoding="utf-8"))
    training_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (training_start.get("status") != "complete"
            or training_start.get("global_step") != cfg["training"]["max_steps"]
            or training_metrics.get("global_step") != cfg["training"]["max_steps"]
            or training_start.get("parent_adapter_sha256") != cfg["initial_adapter_sha256"]
            or sha(candidate / "adapter_model.safetensors") == cfg["initial_adapter_sha256"]):
        raise SystemExit("incremental_training_record_mismatch")
    candidate_hash = sha(candidate / "adapter_model.safetensors")
    check_gpu_floor(float(cfg["resource_limits"]["minimum_gpu_vram_gib_for_planning"]),
                    float(cfg["resource_limits"]["minimum_free_gpu_vram_gib_at_start"]))
    state = {
        "version": "v133-goal-d-context-aligned-paired-v1", "status": "running",
        "post_hoc_diagnostic": True, "results": {}, "stop_reason": None, "speed_check": None,
        "config_sha256": sha(CFG_PATH), "runner_sha256": sha(Path(__file__)),
        "asset_hashes": asset_hashes, "runtime_versions": versions,
        "parent_adapter_sha256": cfg["initial_adapter_sha256"],
        "candidate_adapter_sha256": candidate_hash,
        "training_start_sha256": sha(start_path), "training_metrics_sha256": sha(metrics_path),
        "dev_sha256": sha(dev), "changed_answer_support_review": "pending_gate",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cap = args.max_minutes * 60
    evaluate_adapter(str(model), str(v118), rows, "v118", state, output, started, cap)
    if state["stop_reason"] is None:
        evaluate_adapter(str(model), str(candidate), rows, "goal_d", state, output, started, cap)
    base = [state["results"].get(f"v118:{i:02d}") for i in range(16)]
    tuned = [state["results"].get(f"goal_d:{i:02d}") for i in range(16)]
    if all(x is not None for x in base + tuned):
        state["paired_summary"] = {
            "n": 16, "v118_em": sum(x["em"] for x in base), "goal_d_em": sum(x["em"] for x in tuned),
            "v118_mean_f1": sum(x["f1"] for x in base) / 16,
            "goal_d_mean_f1": sum(x["f1"] for x in tuned) / 16,
            "new_correct": [i for i, (a, b) in enumerate(zip(base, tuned)) if not a["em"] and b["em"]],
            "lost_correct": [i for i, (a, b) in enumerate(zip(base, tuned)) if a["em"] and not b["em"]],
            "retained_correct": [i for i, (a, b) in enumerate(zip(base, tuned)) if a["em"] and b["em"]],
            "changed_answer_indices": [i for i, (a, b) in enumerate(zip(base, tuned)) if a["raw_output"] != b["raw_output"]],
            "protocol_valid_v118": sum(x["protocol_valid"] for x in base),
            "protocol_valid_goal_d": sum(x["protocol_valid"] for x in tuned),
        }
        state["changed_answer_support_review"] = (
            "required_before_promotion" if state["paired_summary"]["goal_d_em"] - state["paired_summary"]["v118_em"] >= 2
            else "not_triggered")
    else:
        state["paired_summary"] = None
        state["changed_answer_support_review"] = "not_triggered_incomplete_run"
    state["elapsed_seconds"] = round(time.monotonic() - started, 3)
    state["status"] = "complete" if state["paired_summary"] else "incomplete"
    output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": state["status"], "elapsed_seconds": state["elapsed_seconds"],
                      "paired_summary": state["paired_summary"],
                      "changed_answer_support_review": state["changed_answer_support_review"]}, indent=2))


if __name__ == "__main__":
    main()
