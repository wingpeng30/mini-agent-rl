"""Summarize the frozen Goal E diagnosis without further model inference."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "runs/v133-goal-e-diagnostic"
WEIGHTS = {
    "v118": ROOT / "backups/v118/extracted/runs/v118/formal/adapter/adapter_model.safetensors",
    "sft_c": ROOT / "runs/v133-goal-c-sft-once/adapter/adapter_model.safetensors",
    "sft_d": ROOT / "runs/v133-goal-d-sft-once/adapter/adapter_model.safetensors",
}


def effective_norms() -> dict:
    files = {name: safe_open(str(path), framework="pt", device="cpu") for name, path in WEIGHTS.items()}
    keys = list(files["v118"].keys())
    if any(list(f.keys()) != keys for f in files.values()):
        raise ValueError("adapter_tensor_keys_differ")
    scale = 16 / 8
    baseline_sq = 0.0
    diff_sq = {"sft_c": 0.0, "sft_d": 0.0}
    module_count = 0
    for key in keys:
        if not key.endswith(".lora_A.weight"):
            continue
        bkey = key.replace(".lora_A.weight", ".lora_B.weight")
        a0 = files["v118"].get_tensor(key).double()
        b0 = files["v118"].get_tensor(bkey).double()
        # ||BA||_F² = tr((A Aᵀ)(Bᵀ B)); avoids building large dense deltas.
        n0 = float(((a0 @ a0.T) * (b0.T @ b0).T).sum()) * scale**2
        baseline_sq += n0
        for name in diff_sq:
            a = files[name].get_tensor(key).double()
            b = files[name].get_tensor(bkey).double()
            n = float(((a @ a.T) * (b.T @ b).T).sum()) * scale**2
            cross = float(((a0 @ a.T) * (b.T @ b0).T).sum()) * scale**2
            diff_sq[name] += max(0.0, n + n0 - 2 * cross)
        module_count += 1
    return {"lora_modules": module_count, "v118_effective_frobenius": math.sqrt(baseline_sq), "relative_effective_delta_from_v118": {name: math.sqrt(value / baseline_sq) for name, value in diff_sq.items()}}


def metrics(records: list[dict]) -> dict:
    return {"n": len(records), "valid": sum(x["protocol_valid"] for x in records), "em": sum(x["em"] for x in records), "mean_f1": sum(x["f1"] for x in records) / len(records)}


def main():
    diagnostic = json.loads((DIR / "diagnostic.json").read_text(encoding="utf-8"))
    dev = json.loads((DIR / "dev-generation.json").read_text(encoding="utf-8"))
    if diagnostic["status"] != "complete" or dev["status"] != "complete":
        raise ValueError("incomplete_diagnosis")
    result = {"status": "complete", "runtime_seconds": diagnostic["elapsed_seconds"] + dev["elapsed_seconds"], "effective_lora": effective_norms(), "models": {}, "paired": {}, "dev_error_candidates": {}}
    for name, state in diagnostic["model_results"].items():
        train = [state["rows"][f"train:{i:02d}"] for i in range(25)]
        development = [state["rows"][f"dev:{i:02d}"] for i in range(16)]
        dev_outputs = [dev["results"][f"{name}:{i:02d}"] for i in range(16)]
        group = {}
        for split, records in (("train", train), ("dev", development)):
            for kind in ("answer", "format"):
                parts = [row["teacher"][kind] for row in records]
                total = sum(x["count"] for x in parts)
                group[f"{split}_{kind}"] = {"tokens": total, "token_weighted_nll": sum(x["nll_sum"] for x in parts) / total, "row_mean_token_accuracy": sum(x["accuracy"] for x in parts) / len(parts)}
        result["models"][name] = {"train": metrics([x["generation"] for x in train]), "dev": metrics(dev_outputs), "token_metrics": group, "anchor_active_vs_disabled": {key: {k: value[k] for k in ("enabled_disabled_max_abs", "repeat_max_abs")} for key, value in state["anchors"].items()}, "active_adapters": state["active_adapters"], "matched_lora_modules": state["matched_lora_modules"], "adapter_dtypes": state["adapter_dtypes"]}
    for name in ("sft_c", "sft_d"):
        paired = {}
        for split, length in (("train", 25), ("dev", 16)):
            def row(arm, index):
                if split == "train":
                    return diagnostic["model_results"][arm]["rows"][f"train:{index:02d}"]["generation"]
                return dev["results"][f"{arm}:{index:02d}"]
            paired[split] = {"new_correct": [], "lost_correct": [], "retained_correct": [], "changed_outputs": []}
            for index in range(length):
                base, tuned = row("v118", index), row(name, index)
                if not base["em"] and tuned["em"]:
                    paired[split]["new_correct"].append(index)
                if base["em"] and not tuned["em"]:
                    paired[split]["lost_correct"].append(index)
                if base["em"] and tuned["em"]:
                    paired[split]["retained_correct"].append(index)
                if base["raw_output"] != tuned["raw_output"]:
                    paired[split]["changed_outputs"].append(index)
        result["paired"][name] = paired
    for index in (0, 2, 3, 15):
        result["dev_error_candidates"][str(index)] = {name: {"prediction": dev["results"][f"{name}:{index:02d}"]["prediction"], "gold": dev["results"][f"{name}:{index:02d}"]["gold"], "candidate_likelihood": dev["results"][f"{name}:{index:02d}"]["candidate_likelihood"]} for name in WEIGHTS}
    output = DIR / "summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "runtime_seconds": result["runtime_seconds"], "effective_lora": result["effective_lora"], "models": {k: {"train": v["train"], "dev": v["dev"]} for k, v in result["models"].items()}, "paired": result["paired"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
