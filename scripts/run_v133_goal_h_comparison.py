"""Goal H: frozen 16-case transfer comparison and four exposed fit probes.

`prepare` is CPU only. `run` launches one bounded GPU worker in a subprocess.
Existing Goal C/E data paths are deliberately absent from the inference path.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.run_v133_goal_g_cpu import cpu_check, norm, qhash
from evaluate_v133_goal_c_sft import f1_score, normalize_answer, parse_action

OUT = ROOT / "runs/v133-goal-h-comparison"
G_EVAL = ROOT / "runs/v133-goal-g-cpu/evaluation-frozen.jsonl"
G_MANIFEST = ROOT / "runs/v133-goal-g-cpu/freeze-manifest.json"
CORE = ROOT / "data/evidence-sft-v124-forensic/train-core-candidate-private.jsonl"
V118_TRAIN = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
MODEL = Path("/root/autodl-tmp/qwen_08b")
ADAPTERS = {
    "v118": (ROOT / "backups/v118/extracted/runs/v118/formal/adapter", "ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5"),
    "sft_c": (ROOT / "runs/v133-goal-c-sft-once/adapter", "0a80d82918e301cac05f0cc2892a7ccd007138996e121549b95dbcbf7f8919c6"),
    "sft_d": (ROOT / "runs/v133-goal-d-sft-once/adapter", "b8782a1a36ddaa44e2a2d7446763b7e50ae40865aae326366d3f0e50b9fc94af"),
}
G_SHA = "f689f94b598a1c0d89f1b87f0b008c8a03158a7ab479028368285564244aad69"
G_MANIFEST_SHA = "76605092f8582111093138ba14c31204b3303ffe8025d87a3dce2dd3fa661c36"
CORE_SHA = "465bbec180c89a5a3088a19d7c953adf502f2a03dc0869e19ab20dc0fa5a1e79"
V118_TRAIN_SHA = "26ced5bafb5210fefe9b007915598c8d58d2ebec156e938a48d86e1f1577ad8b"
SELECTED = {
    46: ("South Pennines", "Wardle, Greater Manchester"),
    80: ("The Academy Is...", "Strangers in Paradise (band)"),
    129: ("Stresa Front", "Locarno Treaties"),
    217: ("Izu Ojukwu", "Fred Savage"),
}
SYSTEM = "You are an evidence-driven retrieval agent. Output one JSON search or answer action, using visible evidence only."
CAP = 7200


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def readl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def writel(path: Path, data) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in data:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def expected_assets() -> dict[str, str]:
    baseline = json.loads((ROOT / "data/evidence-sft-v133-goal-c-private/frozen/freeze-manifest-private.json").read_text())["baseline"]
    required = {G_EVAL: G_SHA, G_MANIFEST: G_MANIFEST_SHA, CORE: CORE_SHA, V118_TRAIN: V118_TRAIN_SHA}
    for directory, expected in ADAPTERS.values():
        required[directory / "adapter_model.safetensors"] = expected
        if not (directory / "adapter_config.json").is_file():
            raise FileNotFoundError(directory / "adapter_config.json")
    for group in ("tokenizer_sha256", "model_runtime_files_sha256", "model_weight_files_sha256"):
        required.update({MODEL / name: digest for name, digest in baseline[group].items()})
    for path, expected in required.items():
        if sha(path) != expected:
            raise ValueError(f"frozen_input_hash_mismatch:{path}")
    return {str(path): digest for path, digest in required.items()}


def probe_from_core(index: int, row: dict) -> dict:
    if [m["role"] for m in row["messages"]] != ["system", "user", "assistant", "tool", "assistant"]:
        raise ValueError(f"unexpected_core_roles:{index}")
    target = row["messages"][row["target_assistant_index"]]
    gold = json.loads(target["content"])
    if gold.get("action") != "answer":
        raise ValueError(f"not_answer_state:{index}")
    source_docs = json.loads(row["messages"][3]["content"])["result"]
    by_title = {doc["title"]: doc for doc in source_docs}
    first, second = [by_title[t] for t in SELECTED[index]]
    question = row["messages"][1]["content"]
    tool = lambda doc: json.dumps({"ok": True, "result": [doc]}, ensure_ascii=False, separators=(",", ":"))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": json.dumps({"action": "search", "query": question}, ensure_ascii=False)},
        {"role": "tool", "content": tool(first)},
        {"role": "assistant", "content": json.dumps({"action": "search", "query": first["text"]}, ensure_ascii=False)},
        {"role": "tool", "content": tool(second)},
        {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)},
    ]
    return {"case_id": f"core:{index}", "source_id": f"core:{index}", "question_sha256": qhash(question),
            "messages": messages, "target_assistant_index": 6, "thinking_enabled": False,
            "evidence_titles": list(SELECTED[index]), "evidence_documents": [first, second],
            "official_answer": gold["answer"], "role": "training_exposed_fit_probe_only",
            "old_context_changed": True, "qualified_for_new_training": False}


def prepare() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoProcessor

    if torch.cuda.is_available():
        raise RuntimeError("prepare_must_be_cpu_only")
    if (OUT / "h0-manifest.json").exists():
        raise FileExistsError("H0_already_frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    assets = expected_assets()
    versions = {name: md.version(name) for name in ("torch", "transformers", "peft", "tokenizers", "safetensors")}
    if versions != {"torch": "2.5.1+cu124", "transformers": "5.17.0", "peft": "0.20.0",
                    "tokenizers": "0.23.2", "safetensors": "0.8.0"}:
        raise ValueError(f"runtime_version_mismatch:{versions}")
    evaluation = readl(G_EVAL)
    core = readl(CORE)
    actual_train = readl(V118_TRAIN)
    if len(evaluation) != 16 or len({x["case_id"] for x in evaluation}) != 16 or len(core) != 228:
        raise ValueError("frozen_dataset_count_mismatch")
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    eval_checks = []
    for row in evaluation:
        check = cpu_check(processor, row)
        if check["prompt_sha256"] != row["cpu_check"]["prompt_sha256"]:
            raise ValueError(f"G_prompt_changed:{row['case_id']}")
        messages = row["messages"]
        if json.loads(messages[2]["content"])["query"] != messages[1]["content"]:
            raise ValueError("first_search_uses_other_text")
        if json.loads(messages[4]["content"])["query"] != json.loads(messages[3]["content"])["result"][0]["text"]:
            raise ValueError("second_search_uses_future_or_target_text")
        eval_checks.append({"case_id": row["case_id"], **check})
    probes = []
    eval_questions = {norm(row["messages"][1]["content"]) for row in evaluation}
    eval_titles = {norm(doc["title"]) for row in evaluation for doc in row["evidence_documents"]}
    eval_source_ids = {row["source_id"] for row in evaluation}
    for index in SELECTED:
        if core[index] not in actual_train:
            raise ValueError(f"probe_not_actually_exposed_to_v118:{index}")
        probe = probe_from_core(index, core[index])
        question = norm(probe["messages"][1]["content"])
        titles = {norm(t) for t in probe["evidence_titles"]}
        if question in eval_questions or titles & eval_titles or probe["source_id"] in eval_source_ids:
            raise ValueError(f"probe_overlaps_G_directly:{index}")
        probe["cpu_check"] = cpu_check(processor, probe)
        probes.append(probe)
    if len(probes) != 4:
        raise ValueError("probe_count_mismatch")
    writel(OUT / "fit-probes-frozen.jsonl", probes)
    manifest = {"version": "v133-goal-h-h0-v1", "input_verified": True,
                "prompt_hashes_verified": True, "environment_aligned": True, "unmet_conditions": [],
                "inference_started": False, "training_started": False,
                "evaluation_path": str(G_EVAL), "evaluation_sha256": G_SHA,
                "fit_probe_path": str(OUT / "fit-probes-frozen.jsonl"),
                "fit_probe_sha256": sha(OUT / "fit-probes-frozen.jsonl"),
                "actual_input_sha256": assets, "runtime_versions": versions,
                "eval_checks": eval_checks,
                "fit_probe_checks": [{"case_id": row["case_id"], **row["cpu_check"]} for row in probes],
                "review_boundary": "four probe answers were previously reviewed in original one-tool context; transformed context is diagnostic only, not training authorized",
                "isolation_scope": "exact question, source ID, and evidence title non-overlap with Goal G; entity aliases require separate review before any independent-evaluation claim"}
    write(OUT / "h0-manifest.json", manifest)
    print(json.dumps({"status": "H0_cpu_ready", "main": 16, "fit_probes": 4,
                      "manifest_sha256": sha(OUT / "h0-manifest.json")}, ensure_ascii=False))


def assert_h0() -> tuple[list[dict], list[dict], dict]:
    manifest = json.loads((OUT / "h0-manifest.json").read_text(encoding="utf-8"))
    if not (manifest["input_verified"] and manifest["prompt_hashes_verified"] and manifest["environment_aligned"]):
        raise ValueError("H0_gates_not_satisfied")
    if sha(G_EVAL) != G_SHA or sha(OUT / "fit-probes-frozen.jsonl") != manifest["fit_probe_sha256"]:
        raise ValueError("frozen_data_changed")
    for name, (path, expected) in ADAPTERS.items():
        if sha(path / "adapter_model.safetensors") != expected:
            raise ValueError(f"adapter_changed:{name}")
    return readl(G_EVAL), readl(OUT / "fit-probes-frozen.jsonl"), manifest


def run() -> None:
    assert_h0()
    if (OUT / "run-state.json").exists() or (OUT / "trajectories.jsonl").exists():
        raise FileExistsError("H1_one_shot_run_already_attempted")
    command = [sys.executable, str(Path(__file__).resolve()), "worker"]
    try:
        completed = subprocess.run(command, cwd=ROOT, timeout=CAP, check=False)
    except subprocess.TimeoutExpired:
        previous = json.loads((OUT / "run-state.json").read_text(encoding="utf-8")) if (OUT / "run-state.json").exists() else {}
        write(OUT / "controller-stop.json", {"status": "hard_timeout", "seconds": CAP,
             "last_worker_status": previous.get("status"), "last_completed": previous.get("completed")})
        raise SystemExit("H1_hard_7200_second_timeout") from None
    if completed.returncode:
        raise SystemExit(f"H1_worker_failed_exit:{completed.returncode}")


def _append_trajectory(row: dict) -> None:
    with (OUT / "trajectories.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def worker() -> None:
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.sft import build_lora_model

    evaluation, probes, h0 = assert_h0()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_unavailable")
    free, total = torch.cuda.mem_get_info()
    if total / 1024**3 < 23.5 or free / 1024**3 < 22:
        raise RuntimeError("GPU_memory_below_frozen_floor")
    if len(evaluation) != 16 or len(probes) != 4:
        raise ValueError("wrong_run_size")
    start = time.monotonic()
    state = {"status": "running", "h0_manifest_sha256": sha(OUT / "h0-manifest.json"),
             "gpu": torch.cuda.get_device_name(0), "budget_seconds": CAP, "completed": 0,
             "planned": 52, "model_loads": {}, "adapter_states": {}, "speed_check": None,
             "stop_reason": None}
    write(OUT / "run-state.json", state)
    try:
        for arm, (adapter_path, expected) in ADAPTERS.items():
            if time.monotonic() - start >= CAP:
                state["stop_reason"] = "time_cap_before_model_load"
                break
            set_seed(42)
            loaded_at = time.monotonic()
            model, processor = build_lora_model(str(MODEL), dtype="bfloat16",
                            gradient_checkpointing=False, adapter_path=str(adapter_path))
            model.eval()
            load_seconds = time.monotonic() - loaded_at
            state["model_loads"][arm] = round(load_seconds, 3)
            active = model.active_adapters() if callable(getattr(model, "active_adapters", None)) else [model.active_adapter]
            if "default" not in active:
                raise ValueError(f"adapter_not_active:{arm}")
            modules = [n for n, m in model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A]
            if len(modules) != 96:
                raise ValueError(f"unexpected_LoRA_module_count:{arm}:{len(modules)}")
            state["adapter_states"][arm] = {"path": str(adapter_path),
                "weight_sha256": expected, "active": active,
                "lora_modules": len(modules), "parameter_dtype": str(next(model.parameters()).dtype)}
            for split, rows in (("main", evaluation), ("fit_probe", probes if arm == "v118" else [])):
                for case in rows:
                    if time.monotonic() - start >= CAP:
                        state["stop_reason"] = "time_cap_between_cases"
                        break
                    target = case["target_assistant_index"]
                    inputs = processor.apply_chat_template(case["messages"][:target], tokenize=True,
                        add_generation_prompt=True, return_tensors="pt", return_dict=True,
                        enable_thinking=False)
                    ids = inputs["input_ids"][0].tolist()
                    ids_sha = hashlib.sha256(bytes().join(int(i).to_bytes(4, "little") for i in ids)).hexdigest()
                    expected_sha = case["cpu_check"]["prompt_sha256"]
                    if ids_sha != expected_sha:
                        raise ValueError(f"runtime_input_token_mismatch:{arm}:{case['case_id']}")
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
                    t0 = time.monotonic()
                    with torch.inference_mode():
                        generated = model.generate(**inputs, do_sample=False, max_new_tokens=128,
                            pad_token_id=processor.tokenizer.eos_token_id)
                    generation_seconds = time.monotonic() - t0
                    raw = processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True).strip()
                    gold = str(json.loads(case["messages"][target]["content"])["answer"])
                    valid = True
                    try:
                        prediction = parse_action(raw)["answer"]
                    except (ValueError, json.JSONDecodeError, KeyError, TypeError):
                        valid, prediction = False, ""
                    record = {"arm": arm, "split": split, "case_id": case["case_id"],
                        "prompt_sha256": ids_sha, "prompt_tokens": len(ids),
                        "raw_output": raw, "prediction": prediction, "gold": gold,
                        "protocol_valid": valid,
                        "em": int(valid and normalize_answer(prediction) == normalize_answer(gold)),
                        "f1": f1_score(prediction, gold) if valid else 0.0,
                        "generation_seconds": round(generation_seconds, 3)}
                    _append_trajectory(record)
                    state["completed"] += 1
                    if arm == "v118" and split == "main" and state["completed"] == 2:
                        first = readl(OUT / "trajectories.jsonl")
                        mean_generation = sum(x["generation_seconds"] for x in first[:2]) / 2
                        projected = time.monotonic() - start + mean_generation * 50 + load_seconds * 2
                        state["speed_check"] = {"after_full_cases": 2,
                            "mean_generation_seconds": round(mean_generation, 3),
                            "projected_total_seconds": round(projected, 3),
                            "decision": "continue" if projected <= CAP else "stop"}
                        if projected > CAP:
                            state["stop_reason"] = "two_case_projection_exceeds_7200_seconds"
                    write(OUT / "run-state.json", state)
                    if state["stop_reason"]:
                        break
                if state["stop_reason"]:
                    break
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
            if state["stop_reason"]:
                break
        state["status"] = "complete" if state["completed"] == 52 else "stopped"
        state["elapsed_seconds"] = round(time.monotonic() - start, 3)
        write(OUT / "run-state.json", state)
    except BaseException as exc:
        state["status"] = "failed"
        state["elapsed_seconds"] = round(time.monotonic() - start, 3)
        state["stop_reason"] = type(exc).__name__ + ":" + str(exc)[:300]
        write(OUT / "run-state.json", state)
        raise


def summarize() -> None:
    if (OUT / "summary.json").exists():
        raise FileExistsError("summary_already_written")
    state = json.loads((OUT / "run-state.json").read_text(encoding="utf-8"))
    rows = readl(OUT / "trajectories.jsonl")
    if state["status"] != "complete" or len(rows) != 52:
        write(OUT / "partial-summary.json", {"state": state, "observed": len(rows),
            "reason": "incomplete_run_not_scored_as_full_comparison"})
        return
    matrix = {arm: {x["case_id"]: x for x in rows if x["arm"] == arm and x["split"] == "main"}
              for arm in ADAPTERS}
    if any(len(v) != 16 for v in matrix.values()):
        raise ValueError("incomplete_arm_matrix")
    expected_cases = set(matrix["v118"])
    if any(set(v) != expected_cases for v in matrix.values()):
        raise ValueError("nonpaired_cases")
    if any(len({matrix[a][c]["prompt_sha256"] for a in matrix}) != 1 for c in expected_cases):
        raise ValueError("nonidentical_arm_prompt_tokens")
    metrics = {arm: {"n": 16, "em": sum(x["em"] for x in data.values()),
              "f1": sum(x["f1"] for x in data.values()) / 16,
              "protocol_valid": sum(x["protocol_valid"] for x in data.values())}
              for arm, data in matrix.items()}
    changes = {}
    for arm in ("sft_c", "sft_d"):
        changes[arm] = {
            "new_correct": sorted(c for c in expected_cases if not matrix["v118"][c]["em"] and matrix[arm][c]["em"]),
            "lost_correct": sorted(c for c in expected_cases if matrix["v118"][c]["em"] and not matrix[arm][c]["em"]),
            "retained_correct": sorted(c for c in expected_cases if matrix["v118"][c]["em"] and matrix[arm][c]["em"]),
            "changed_raw": sorted(c for c in expected_cases if matrix["v118"][c]["raw_output"] != matrix[arm][c]["raw_output"]),
        }
    fit = {x["case_id"]: x for x in rows if x["split"] == "fit_probe"}
    if len(fit) != 4:
        raise ValueError("incomplete_fit_probe")
    write(OUT / "summary.json", {"status": "complete_exploratory_synthetic_history_comparison",
          "runtime_seconds": state["elapsed_seconds"], "speed_check": state["speed_check"],
          "metrics": metrics, "changes_vs_v118": changes,
          "v118_fit_probe": {"n": 4, "em": sum(x["em"] for x in fit.values()),
              "case_results": {k: {"prediction": v["prediction"], "gold": v["gold"],
                      "em": v["em"], "protocol_valid": v["protocol_valid"]} for k, v in fit.items()}},
          "input_sha256": sha(OUT / "h0-manifest.json"),
          "trajectories_sha256": sha(OUT / "trajectories.jsonl"),
          "run_state_sha256": sha(OUT / "run-state.json"),
          "not_end_to_end_retrieval": True,
          "evaluation_16_now_exposed_to_research_decisions": True})
    print(json.dumps({"metrics": metrics, "changes": changes,
                     "fit_em": sum(x["em"] for x in fit.values())}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run", "worker", "summarize"))
    phase = parser.parse_args().phase
    {"prepare": prepare, "run": run, "worker": worker, "summarize": summarize}[phase]()
