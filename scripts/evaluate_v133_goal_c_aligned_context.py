"""One-shot paired diagnosis with dev prompts aligned to the SFT turn structure.

The original frozen run remains untouched. This script only constructs a
deterministic, answer-blind search/tool history from each existing dev row.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import time
from pathlib import Path

from evaluate_v133_goal_c_sft import check_gpu_floor, evaluate_adapter, load_rows, sha


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v133-goal-c-small-sft.json"
FREEZE = ROOT / "data/evidence-sft-v133-goal-c-private/frozen/freeze-manifest-private.json"
ORIGINAL_EVAL_SHA256 = "86cc52318ccf07afb1839971585e1efbc7a497e6caf0995b115814b3b5fc120c"
SFT_ADAPTER_SHA256 = "0a80d82918e301cac05f0cc2892a7ccd007138996e121549b95dbcbf7f8919c6"


def aligned_row(row: dict) -> dict:
    messages = row["messages"]
    if ([m["role"] for m in messages] != ["system", "user", "tool", "assistant"]
            or row["target_assistant_index"] != 3):
        raise ValueError("unexpected_original_dev_roles")
    observation = json.loads(messages[2]["content"])
    if observation.get("ok") is not True or len(observation.get("result", [])) != 2:
        raise ValueError("expected_two_existing_evidence_items")
    first, second = observation["result"]
    if not isinstance(first.get("text"), str) or not isinstance(second.get("text"), str):
        raise ValueError("missing_evidence_text")
    if json.loads(messages[3]["content"]).get("action") != "answer":
        raise ValueError("missing_original_answer_target")

    # The first query sees only the question. The second sees the question and
    # first tool result. Neither query reads the target answer or second result.
    search1 = json.dumps({"action": "search", "query": messages[1]["content"]}, ensure_ascii=False)
    search2 = json.dumps({"action": "search", "query": first["text"]}, ensure_ascii=False)
    tool1 = json.dumps({"ok": True, "result": [first]}, ensure_ascii=False, separators=(",", ":"))
    tool2 = json.dumps({"ok": True, "result": [second]}, ensure_ascii=False, separators=(",", ":"))
    converted = dict(row)
    converted["messages"] = [messages[0], messages[1],
                             {"role": "assistant", "content": search1},
                             {"role": "tool", "content": tool1},
                             {"role": "assistant", "content": search2},
                             {"role": "tool", "content": tool2},
                             messages[3]]
    converted["target_assistant_index"] = 6
    return converted


def aligned_rows(original_rows: list[dict]) -> list[dict]:
    converted = [aligned_row(row) for row in original_rows]
    if len(converted) != 16:
        raise ValueError("expected_16_aligned_rows")
    for old, new in zip(original_rows, converted):
        if (new["messages"][0:2] != old["messages"][0:2]
                or new["messages"][6] != old["messages"][3]):
            raise ValueError("question_or_target_changed")
        old_evidence = json.loads(old["messages"][2]["content"])["result"]
        new_evidence = [json.loads(new["messages"][j]["content"])["result"][0] for j in (3, 5)]
        if old_evidence != new_evidence:
            raise ValueError("evidence_changed")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize-model-inference", action="store_true")
    parser.add_argument("--cpu-only-preflight", action="store_true")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-adapter", required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--dev-data", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--prior-eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-minutes", type=int, default=60)
    args = parser.parse_args()
    if not args.cpu_only_preflight and not args.authorize_model_inference:
        raise SystemExit("explicit_model_inference_authorization_required")
    if not 1 <= args.max_minutes <= 60:
        raise SystemExit("max_minutes_out_of_bounds")
    output = Path(args.output)
    if output.exists():
        raise SystemExit("output_exists_no_retry_or_overwrite")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    dev_path = Path(args.dev_data)
    base_path = Path(args.base_adapter)
    sft_path = Path(args.sft_adapter)
    model_path = Path(args.model_path)
    report_path = Path(args.training_report)
    if (sha(CONFIG) != frozen["baseline"]["config_sha256"]
            or sha(dev_path) != frozen["splits"]["dev_sha256"]
            or sha(Path(args.prior_eval)) != ORIGINAL_EVAL_SHA256
            or sha(base_path / "adapter_model.safetensors") != frozen["baseline"]["v118_adapter_sha256"]
            or sha(sft_path / "adapter_model.safetensors") != SFT_ADAPTER_SHA256):
        raise SystemExit("frozen_asset_hash_mismatch")
    for name, expected in frozen["baseline"]["model_weight_files_sha256"].items():
        if sha(model_path / name) != expected:
            raise SystemExit("model_weight_hash_mismatch")
    for name, expected in frozen["baseline"]["tokenizer_sha256"].items():
        if sha(model_path / name) != expected:
            raise SystemExit("tokenizer_hash_mismatch")
    actual_versions = {name: metadata.version(name) for name in config["runtime_dependencies"]}
    if actual_versions != config["runtime_dependencies"]:
        raise SystemExit("runtime_version_mismatch")
    training_report = json.loads(report_path.read_text(encoding="utf-8"))
    if training_report.get("global_step") != 7 or training_report.get("train_samples") != 25:
        raise SystemExit("training_report_mismatch")
    original = load_rows(dev_path)
    rows = aligned_rows(original)

    if args.cpu_only_preflight:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        tokens = []
        for row in rows:
            prefix = row["messages"][:6]
            rendered = processor.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
            for evidence_message in (prefix[3], prefix[5]):
                if evidence_message["content"] not in rendered:
                    raise ValueError("evidence_missing_from_rendered_prompt")
            encoded = processor.apply_chat_template(prefix, tokenize=True,
                                                    add_generation_prompt=True, enable_thinking=False)
            ids = encoded[0] if encoded and isinstance(encoded[0], list) else encoded
            tokens.append(len(ids))
        if max(tokens) > 2048:
            raise ValueError("aligned_prompt_exceeds_frozen_inference_limit")
        print(json.dumps({"status": "cpu_preflight_passed", "n": 16,
                          "prompt_token_min": min(tokens), "prompt_token_max": max(tokens),
                          "dev_sha256": sha(dev_path), "sft_adapter_sha256": SFT_ADAPTER_SHA256}, indent=2))
        return

    check_gpu_floor(float(config["resource_limits"]["minimum_gpu_vram_gib_for_planning"]),
                    float(config["resource_limits"]["minimum_free_gpu_vram_gib_at_start"]))
    state = {"version": "v133-goal-c-aligned-context-diagnostic-v1",
             "status": "running", "results": {}, "stop_reason": None, "speed_check": None,
             "post_hoc_diagnostic": True, "original_eval_sha256": ORIGINAL_EVAL_SHA256,
             "config_sha256": sha(CONFIG), "freeze_manifest_sha256": sha(FREEZE),
             "dev_sha256": sha(dev_path), "base_adapter_sha256": sha(base_path / "adapter_model.safetensors"),
             "sft_adapter_sha256": SFT_ADAPTER_SHA256, "training_report_sha256": sha(report_path),
             "transformation": "question -> search(question) -> first existing evidence -> search(first evidence text) -> second existing evidence -> answer",
             "changed_answer_support_review": "not performed"}
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cap = args.max_minutes * 60
    evaluate_adapter(str(model_path), str(base_path), rows, "v118", state, output, started, cap)
    if state["stop_reason"] is None:
        evaluate_adapter(str(model_path), str(sft_path), rows, "sft", state, output, started, cap)
    base = [state["results"].get(f"v118:{i:02d}") for i in range(16)]
    tuned = [state["results"].get(f"sft:{i:02d}") for i in range(16)]
    if all(x is not None for x in base + tuned):
        state["paired_summary"] = {
            "n": 16, "v118_em": sum(x["em"] for x in base), "sft_em": sum(x["em"] for x in tuned),
            "v118_mean_f1": sum(x["f1"] for x in base) / 16,
            "sft_mean_f1": sum(x["f1"] for x in tuned) / 16,
            "new_correct": [i for i, (a, b) in enumerate(zip(base, tuned)) if not a["em"] and b["em"]],
            "lost_correct": [i for i, (a, b) in enumerate(zip(base, tuned)) if a["em"] and not b["em"]],
            "retained_correct": [i for i, (a, b) in enumerate(zip(base, tuned)) if a["em"] and b["em"]],
            "changed_answer_indices": [i for i, (a, b) in enumerate(zip(base, tuned)) if a["raw_output"] != b["raw_output"]],
            "protocol_valid_v118": sum(x["protocol_valid"] for x in base),
            "protocol_valid_sft": sum(x["protocol_valid"] for x in tuned),
        }
    else:
        state["paired_summary"] = None
    state["elapsed_seconds"] = round(time.monotonic() - started, 3)
    state["status"] = "complete" if state["paired_summary"] else "incomplete"
    output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": state["status"], "elapsed_seconds": state["elapsed_seconds"],
                      "paired_summary": state["paired_summary"], "stop_reason": state["stop_reason"]}, indent=2))


if __name__ == "__main__":
    main()
