"""Goal I: one-shot, frozen microfit diagnosis for four previously exposed cases."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.run_v133_goal_g_cpu import cpu_check
from scripts.evaluate_v133_goal_c_sft import f1_score, normalize_answer, parse_action

DEFAULT_OUT = ROOT / "runs/v133-goal-i-microfit"
OUT = Path(os.environ.get("V133_GOAL_I_OUTPUT_DIR", str(DEFAULT_OUT))).resolve()
H_PROBES = ROOT / "runs/v133-goal-h-comparison/fit-probes-frozen.jsonl"
G_EVAL = ROOT / "runs/v133-goal-g-cpu/evaluation-frozen.jsonl"
V118 = ROOT / "backups/v118/extracted/runs/v118/formal/adapter"
MODEL = Path("/root/autodl-tmp/qwen_08b")
CAP = 7200
H_PROBES_SHA = "c59b330eb1172b93ee92afdf97e612ddccb526e9a8c697e3138d14d57d641228"
G_EVAL_SHA = "f689f94b598a1c0d89f1b87f0b008c8a03158a7ab479028368285564244aad69"
V118_SHA = "ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5"
H_MANIFEST_SHA = "b160fc622362632f60f1a9b5103dc0b37ea8e63ffee4a56095330c08845232d8"
EXPECTED = {
    "torch": "2.5.1+cu124", "transformers": "5.17.0", "peft": "0.20.0",
    "tokenizers": "0.23.2", "safetensors": "0.8.0", "datasets": "5.0.1",
    "accelerate": "1.15.0",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def readl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def writel(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def probe_rows() -> list[dict]:
    rows = readl(H_PROBES)
    if len(rows) != 4 or [r["case_id"] for r in rows] != ["core:46", "core:80", "core:129", "core:217"]:
        raise ValueError("frozen_probe_set_mismatch")
    return rows


def evaluation_rows() -> list[dict]:
    rows = readl(G_EVAL)
    if len(rows) != 16:
        raise ValueError("frozen_G_evaluation_count_mismatch")
    return rows


def probe_packet(row: dict, alias: str) -> dict:
    docs = []
    for i, doc in enumerate(row["evidence_documents"], 1):
        docs.append({"id": doc["id"], "title": doc["title"], "text": doc["text"],
                     "source": f"visible_evidence_{i}"})
    action = json.loads(row["messages"][row["target_assistant_index"]]["content"])
    visible = row["messages"][:row["target_assistant_index"]]
    return {
        "case_id": alias,
        "question": row["messages"][1]["content"],
        "proposed_action": action,
        "before_action": docs,
        "retrieved_after_search": [],
        "visible_messages": visible,
        "review_scope": "Check that the open-ended question asks for a shared category/property, each of the two visible evidence passages supports the proposed answer at compatible granularity, ambiguity is acceptable, and visible search history does not constructively leak the answer.",
        "protocol": {"version": "v130-evidence-review-citations-v1"},
    }


def expected_assets() -> dict:
    h_manifest = ROOT / "runs/v133-goal-h-comparison/h0-manifest.json"
    if sha(h_manifest) != H_MANIFEST_SHA:
        raise ValueError("goal_H_manifest_hash_changed")
    if sha(H_PROBES) != H_PROBES_SHA or sha(G_EVAL) != G_EVAL_SHA:
        raise ValueError("frozen_data_hash_changed")
    if sha(V118 / "adapter_model.safetensors") != V118_SHA:
        raise ValueError("v118_adapter_hash_changed")
    return {str(H_PROBES): H_PROBES_SHA, str(G_EVAL): G_EVAL_SHA,
            str(V118 / "adapter_model.safetensors"): V118_SHA,
            str(h_manifest): H_MANIFEST_SHA}


def verify_review_gate(manifest: dict) -> None:
    if not (manifest.get("cpu_ready") is True and manifest.get("dual_review_complete") is True
            and manifest.get("eligible_count", 0) >= 2):
        raise ValueError("Goal_I_review_or_CPU_gate_not_satisfied")
    packets = {r["case_id"]: r for r in readl(OUT / "review-packets.jsonl")}
    luna_path = OUT / "luna-votes.json"
    deepseek_path = OUT / "deepseek/i-00-03/validated-review-private.json"
    disposition_path = OUT / "review-disposition.json"
    train_path = OUT / "train-frozen.jsonl"
    for key, path in (("review_packets_sha256", OUT / "review-packets.jsonl"),
                      ("luna_votes_sha256", luna_path), ("deepseek_votes_sha256", deepseek_path),
                      ("review_disposition_sha256", disposition_path),
                      ("train_frozen_sha256", train_path)):
        if sha(path) != manifest.get(key):
            raise ValueError(f"review_gate_file_hash_mismatch:{key}")
    lv = {r["case_id"]: r for r in json.loads(luna_path.read_text(encoding="utf-8"))}
    dv_blob = json.loads(deepseek_path.read_text(encoding="utf-8"))
    dv = {r["case_id"]: r for r in dv_blob["reviews"]}
    if set(lv) != set(packets) or set(dv) != set(packets):
        raise ValueError("review_vote_coverage_incomplete")
    from scripts.v130_review_protocol import valid_vote
    approved = set()
    for alias, packet in packets.items():
        for vote in (lv[alias], dv[alias]):
            valid, reason = valid_vote(packet, vote)
            if not valid:
                raise ValueError(f"review_vote_invalid:{alias}:{reason}")
        if lv[alias]["approved"] and dv[alias]["approved"]:
            approved.add(alias)
    alias_to_original = {alias: original for original, alias in manifest["review_case_map_private"].items()}
    expected_ids = {alias_to_original[a] for a in approved}
    training = readl(train_path)
    if {r["case_id"] for r in training} != expected_ids or len(training) != len(expected_ids):
        raise ValueError("training_rows_do_not_match_dual_approval_set")
    if len(training) < 2 or len(training) > 4:
        raise ValueError("eligible_training_count_out_of_range")


def verify_model_files(h_manifest: dict) -> dict[str, str]:
    model_hashes = {name: digest for name, digest in h_manifest["actual_input_sha256"].items()
                    if name.startswith(str(MODEL) + "/")}
    if not model_hashes:
        raise ValueError("Goal_H_manifest_has_no_model_assets")
    for name, digest in model_hashes.items():
        if sha(Path(name)) != digest:
            raise ValueError(f"base_model_or_tokenizer_hash_mismatch:{name}")
    config = V118 / "adapter_config.json"
    return {**model_hashes, str(config): sha(config),
            str(V118 / "adapter_model.safetensors"): sha(V118 / "adapter_model.safetensors")}


def preflight() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoProcessor

    if torch.cuda.is_available():
        raise RuntimeError("preflight_must_be_CPU_only")
    manifest = json.loads((OUT / "freeze-manifest.json").read_text(encoding="utf-8"))
    verify_review_gate(manifest)
    base_manifest = json.loads((ROOT / "data/evidence-sft-v133-goal-c-private/frozen/freeze-manifest-private.json").read_text(encoding="utf-8"))
    hashes = verify_model_files({"actual_input_sha256": {str(MODEL / name): digest
        for group in ("tokenizer_sha256", "model_runtime_files_sha256", "model_weight_files_sha256")
        for name, digest in base_manifest["baseline"][group].items()}})
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    train = readl(OUT / "train-frozen.jsonl")
    g = evaluation_rows()
    for row in train + g:
        current = cpu_check(processor, row)
        if current["prompt_sha256"] != row["cpu_check"]["prompt_sha256"]:
            raise ValueError(f"preflight_prompt_mismatch:{row['case_id']}")
    code = {str(path): sha(path) for path in (
        Path(__file__).resolve(), ROOT / "src/mini_agent_rl/training/sft.py",
        ROOT / "scripts/run_v133_goal_g_cpu.py", ROOT / "scripts/evaluate_v133_goal_c_sft.py",
        ROOT / "scripts/v130_review_protocol.py")}
    versions = {name: metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise ValueError(f"runtime_version_mismatch:{versions}")
    receipt = {"status":"preflight_pass", "freeze_manifest_sha256":sha(OUT/"freeze-manifest.json"),
        "train_sha256":sha(OUT/"train-frozen.jsonl"), "reviews_verified":True,
        "verified_model_and_tokenizer_sha256":hashes, "code_sha256":code,
        "runtime_versions":versions, "cpu_checks_repeated":len(train)+len(g),
        "gpu_check_deferred_until_run":True,
        "training_started":False,"inference_started":False}
    write(OUT/"execution-preflight.json",receipt)
    print(json.dumps({"status":receipt["status"],"eligible":len(train),
                      "model_files":len(hashes),"cpu_checks":len(train)+len(g)},ensure_ascii=False))


def prepare() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoProcessor

    if torch.cuda.is_available():
        raise RuntimeError("CPU_prepare_saw_CUDA")
    if OUT.exists():
        raise FileExistsError("Goal_I_output_directory_already_exists")
    assets = expected_assets()
    versions = {name: metadata.version(name) for name in EXPECTED}
    probes, evaluations = probe_rows(), evaluation_rows()
    h_manifest = json.loads((ROOT / "runs/v133-goal-h-comparison/h0-manifest.json").read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    checks = []
    for row in probes:
        check = cpu_check(processor, row)
        if check["prompt_sha256"] != row["cpu_check"]["prompt_sha256"]:
            raise ValueError(f"probe_prompt_hash_mismatch:{row['case_id']}")
        checks.append({"case_id": row["case_id"], **check})
    eval_checks = []
    for row in evaluations:
        check = cpu_check(processor, row)
        if check["prompt_sha256"] != row["cpu_check"]["prompt_sha256"]:
            raise ValueError(f"G_prompt_hash_mismatch:{row['case_id']}")
        eval_checks.append({"case_id": row["case_id"], **check})
    aliases = {"core:46": "I-A", "core:80": "I-B", "core:129": "I-C", "core:217": "I-D"}
    packets = [probe_packet(row, aliases[row["case_id"]]) for row in probes]
    writel(OUT / "review-packets.jsonl", packets)
    manifest = {
        "version": "v133-goal-i-freeze-v1", "assets_sha256": assets,
        "runtime_versions": versions, "expected_runtime_versions": EXPECTED,
        "environment_aligned": versions == EXPECTED,
        "source_probe_manifest_sha256": H_MANIFEST_SHA,
        "probe_source_role": "four v118-training-exposed cases; fit diagnostic only",
        "probe_ids_private": list(aliases), "review_case_map_private": aliases,
        "review_packets_sha256": sha(OUT / "review-packets.jsonl"),
        "goal_G_evaluation_sha256": G_EVAL_SHA,
        "goal_G_role": "exploratory regression only; previously exposed to research decisions",
        "probe_cpu_checks": checks, "goal_G_cpu_checks": eval_checks,
        "review_protocol": "v130-evidence-review-citations-v1",
        "input_verified": True, "prompt_hashes_verified": True,
        "cpu_ready": True, "dual_review_complete": False,
        "eligible_count": 0, "training_started": False,
        "unmet_conditions": ["DeepSeek and isolated Luna approvals for at least two probes"],
        "train_config": {"starting_adapter_sha256": V118_SHA, "seed": 42, "dtype": "bfloat16",
            "thinking": False, "optimizer": "AdamW", "learning_rate": 1e-4,
            "weight_decay": 0.0, "max_grad_norm": 1.0, "microbatch": 1,
            "gradient_accumulation": "eligible probe count", "max_steps": 20,
            "scheduler": "linear", "warmup_steps": 1, "final_checkpoint_only": True,
            "max_length": 2176, "wall_clock_cap_seconds": CAP},
        "output_layout": "runs/v133-goal-i-microfit/",
    }
    write(OUT / "freeze-manifest.json", manifest)
    print(json.dumps({"status": "cpu_frozen_review_pending", "probes": len(probes),
                      "goal_G_eval": len(evaluations), "manifest_sha256": sha(OUT / "freeze-manifest.json")}))


def install_votes() -> None:
    manifest_path = OUT / "freeze-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha(OUT / "review-packets.jsonl") != manifest["review_packets_sha256"]:
        raise ValueError("review_packet_hash_changed")
    packets = {r["case_id"]: r for r in readl(OUT / "review-packets.jsonl")}
    luna_file, deepseek_file = OUT / "luna-votes.json", OUT / "deepseek/i-00-03/validated-review-private.json"
    luna = {r["case_id"]: r for r in json.loads(luna_file.read_text(encoding="utf-8"))}
    deepseek_blob = json.loads(deepseek_file.read_text(encoding="utf-8"))
    deepseek = {r["case_id"]: r for r in deepseek_blob["reviews"]}
    if set(luna) != set(packets) or set(deepseek) != set(packets):
        raise ValueError("review_coverage_mismatch")
    from scripts.v130_review_protocol import valid_vote
    dispositions = []
    for alias, packet in packets.items():
        lv, dv = luna[alias], deepseek[alias]
        for vote in (lv, dv):
            valid, reason = valid_vote(packet, vote)
            if not valid:
                raise ValueError(f"invalid_review_ticket:{alias}:{reason}")
        approved = bool(lv["approved"] and dv["approved"])
        dispositions.append({"case_id": alias, "luna_approved": bool(lv["approved"]),
            "deepseek_approved": bool(dv["approved"]), "disposition": "eligible" if approved else "excluded",
            "reason": "both valid approvals" if approved else "valid rejection or disagreement"})
    eligible_aliases = {r["case_id"] for r in dispositions if r["disposition"] == "eligible"}
    probe_map = manifest["review_case_map_private"]
    source_rows = {r["case_id"]: r for r in probe_rows()}
    selected = [source_rows[original] for original, alias in probe_map.items() if alias in eligible_aliases]
    if len(selected) < 2:
        manifest.update({"dual_review_complete": True, "eligible_count": len(selected),
                         "training_started": False, "unmet_conditions": ["fewer than two doubly approved probes"]})
        write(OUT / "review-disposition.json", dispositions)
        write(manifest_path, manifest)
        raise SystemExit("STOP: fewer than two doubly approved probes")
    writel(OUT / "train-frozen.jsonl", selected)
    write(OUT / "review-disposition.json", dispositions)
    manifest.update({"dual_review_complete": True, "eligible_count": len(selected),
        "eligible_case_ids_private": [r["case_id"] for r in selected],
        "luna_votes_sha256": sha(luna_file), "deepseek_votes_sha256": sha(deepseek_file),
        "review_disposition_sha256": sha(OUT / "review-disposition.json"),
        "train_frozen_sha256": sha(OUT / "train-frozen.jsonl"),
        "unmet_conditions": []})
    write(manifest_path, manifest)
    write(OUT / "review-freeze.json", {"packets_sha256": manifest["review_packets_sha256"],
        "luna_votes_sha256": manifest["luna_votes_sha256"],
        "deepseek_votes_sha256": manifest["deepseek_votes_sha256"],
        "eligible_count": len(selected), "train_frozen_sha256": manifest["train_frozen_sha256"]})
    print(json.dumps({"status": "review_and_cpu_ready", "eligible_count": len(selected),
                      "train_sha256": manifest["train_frozen_sha256"]}))


def assert_ready() -> tuple[list[dict], list[dict], dict]:
    manifest_path = OUT / "freeze-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_review_gate(manifest)
    expected_assets()
    preflight_path = OUT / "execution-preflight.json"
    preflight_record = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight_record.get("status") != "preflight_pass" or preflight_record.get("training_started") is not False:
        raise ValueError("execution_preflight_missing_or_invalid")
    if preflight_record.get("freeze_manifest_sha256") != sha(manifest_path):
        raise ValueError("preflight_freeze_manifest_binding_mismatch")
    for raw_path, digest in preflight_record["code_sha256"].items():
        if sha(Path(raw_path)) != digest:
            raise ValueError(f"execution_code_changed:{raw_path}")
    if sha(OUT / "train-frozen.jsonl") != manifest["train_frozen_sha256"]:
        raise ValueError("train_freeze_changed")
    if sha(OUT / "review-packets.jsonl") != manifest["review_packets_sha256"]:
        raise ValueError("review_packet_changed")
    return readl(OUT / "train-frozen.jsonl"), evaluation_rows(), manifest


def run() -> None:
    train, evaluation, manifest = assert_ready()
    if (OUT / "run-state.json").exists() or (OUT / "trajectories.jsonl").exists():
        raise FileExistsError("Goal_I_is_one_shot_no_retry")
    command = [sys.executable, str(Path(__file__).resolve()), "worker"]
    try:
        result = subprocess.run(command, cwd=ROOT, timeout=CAP, check=False)
    except subprocess.TimeoutExpired:
        old = json.loads((OUT / "run-state.json").read_text(encoding="utf-8")) if (OUT / "run-state.json").exists() else {}
        write(OUT / "controller-stop.json", {"status": "hard_timeout", "seconds": CAP,
            "completed": old.get("completed"), "last_status": old.get("status")})
        raise SystemExit("Goal_I_hard_timeout_7200_seconds") from None
    if result.returncode:
        raise SystemExit(f"Goal_I_worker_failed:{result.returncode}")


def append_trajectory(row: dict) -> None:
    with (OUT / "trajectories.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def target_slices(processor, row: dict, encoded: dict) -> tuple[list[int], list[int]]:
    messages = row["messages"]
    target_index = row["target_assistant_index"]
    content = messages[target_index]["content"]
    answer = json.dumps(row["official_answer"], ensure_ascii=False)
    start = content.index(answer, content.index('"answer"')) + 1
    end = start + len(answer) - 2
    offsets = processor.tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    target_positions = [i for i, label in enumerate(encoded["labels"]) if label != -100]
    content_positions = target_positions[:-1]
    input_ids = encoded["input_ids"]
    token_ids = processor.tokenizer(content, add_special_tokens=False)["input_ids"]
    if len(content_positions) != len(offsets) or [input_ids[i] for i in content_positions] != token_ids:
        raise ValueError(f"teacher_forcing_offset_mismatch:{row['case_id']}")
    answer_positions, format_positions = [], []
    for position, (left, right) in zip(content_positions, offsets):
        if left >= start and right <= end:
            answer_positions.append(position)
        elif right <= start or left >= end:
            format_positions.append(position)
        else:
            raise ValueError(f"ambiguous_answer_token_boundary:{row['case_id']}:{left}:{right}")
    format_positions.append(target_positions[-1])
    return answer_positions, format_positions


def generate(model, processor, row: dict, arm: str, split: str, device, started: float) -> None:
    import torch
    target = row["target_assistant_index"]
    prompt = processor.apply_chat_template(row["messages"][:target], tokenize=True,
        add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False)
    ids = prompt["input_ids"][0].tolist()
    ids_sha = hashlib.sha256(b"".join(int(i).to_bytes(4, "little") for i in ids)).hexdigest()
    if ids_sha != row["cpu_check"]["prompt_sha256"]:
        raise ValueError(f"runtime_prompt_hash_mismatch:{arm}:{row['case_id']}")
    prompt = {k: v.to(device) for k, v in prompt.items() if hasattr(v, "to")}
    t0 = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**prompt, do_sample=False, max_new_tokens=128,
                                pad_token_id=processor.tokenizer.eos_token_id)
    generation_s = time.monotonic() - t0
    raw = processor.tokenizer.decode(output[0, prompt["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    gold = row["official_answer"]
    try:
        prediction = parse_action(raw)["answer"]
        valid = True
    except (ValueError, json.JSONDecodeError, KeyError, TypeError):
        prediction, valid = "", False
    append_trajectory({"arm": arm, "split": split, "case_id": row["case_id"],
        "prompt_sha256": ids_sha, "prompt_tokens": len(ids), "raw_output": raw,
        "prediction": prediction, "gold": gold, "protocol_valid": valid,
        "em": int(valid and normalize_answer(prediction) == normalize_answer(gold)),
        "f1": f1_score(prediction, gold) if valid else 0.0,
        "generation_seconds": round(generation_s, 3), "elapsed_seconds": round(time.monotonic()-started, 3)})


def score_targets(model, processor, rows: list[dict], arm: str, device, started: float) -> list[dict]:
    import torch
    from mini_agent_rl.training.sft import encode_assistant_only
    output = []
    for row in rows:
        encoded = encode_assistant_only(processor, row["messages"], 2176,
            row["target_assistant_index"], False)
        answer_positions, format_positions = target_slices(processor, row, encoded)
        ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
        mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits[0]
        labels = torch.tensor(encoded["labels"], dtype=torch.long, device=device)
        positions = [p for p, label in enumerate(encoded["labels"]) if label != -100]
        pred_ids = logits[torch.tensor([p-1 for p in positions], device=device)].argmax(dim=-1)
        observed = ids[0, torch.tensor(positions, device=device)]
        per_token = torch.nn.functional.cross_entropy(
            logits[torch.tensor([p-1 for p in positions], device=device)].float(), observed, reduction="none")
        by_position = {p: (float(loss), int(pred == gold)) for p, loss, pred, gold in
            zip(positions, per_token.tolist(), pred_ids.tolist(), observed.tolist())}
        def aggregate(selected):
            values = [by_position[p] for p in selected]
            return {"tokens": len(values), "nll": sum(x[0] for x in values)/len(values) if values else None,
                    "token_accuracy": sum(x[1] for x in values)/len(values) if values else None}
        output.append({"arm": arm, "case_id": row["case_id"],
            "answer": aggregate(answer_positions), "format": aggregate(format_positions),
            "total_target_tokens": len(positions), "elapsed_seconds": round(time.monotonic()-started, 3)})
    return output


def worker() -> None:
    import torch
    from transformers import AutoProcessor, set_seed
    from peft import PeftModel
    from mini_agent_rl.training.sft import build_lora_model, train_sft

    train, evaluation, manifest = assert_ready()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_required_for_authorized_training")
    free, total = torch.cuda.mem_get_info()
    if total / 1024**3 < 23.5 or free / 1024**3 < 22:
        raise RuntimeError("GPU_memory_below_frozen_floor")
    if len(train) < 2 or len(train) > 4:
        raise ValueError("eligible_training_count_out_of_range")
    train_file = OUT / "train-frozen.jsonl"
    state = {"status":"running", "planned": {"baseline_generations":len(train)+16,
        "trained_generations":len(train)+16, "training_steps":20}, "completed":0,
        "started_unix":time.time(), "started_monotonic":time.monotonic(), "cap_seconds":CAP,
        "gpu":torch.cuda.get_device_name(0), "training_count":len(train),
        "train_sha256":sha(train_file), "model_loads":{}, "speed_check":None,
        "adapter_active":None, "stop_reason":None}
    write(OUT / "run-state.json", state)
    started = time.monotonic()
    try:
        set_seed(42)
        load_at = time.monotonic()
        model, processor = build_lora_model(str(MODEL), dtype="bfloat16", gradient_checkpointing=False,
                                            adapter_path=str(V118))
        model.eval()
        state["model_loads"]["v118"] = round(time.monotonic()-load_at,3)
        active = model.active_adapters() if callable(getattr(model,"active_adapters",None)) else [model.active_adapter]
        if "default" not in active:
            raise ValueError("v118_adapter_not_active")
        state["adapter_active"] = {"adapter":"v118", "active":active,
            "parameter_dtype":str(next(model.parameters()).dtype),
            "lora_modules":sum(1 for n,m in model.named_modules() if hasattr(m,"lora_A") and "default" in m.lora_A)}
        if state["adapter_active"]["lora_modules"] != 96:
            raise ValueError("unexpected_v118_lora_module_count")
        device = next(model.parameters()).device
        rows = [("train", train), ("exploratory_eval", evaluation)]
        for split, group in rows:
            for row in group:
                if time.monotonic()-started >= CAP:
                    state["stop_reason"]="cap_before_baseline_complete"
                    break
                generate(model,processor,row,"v118",split,device,started)
                state["completed"] += 1
                if state["completed"] == 2:
                    elapsed=time.monotonic()-started
                    mean=sum(r["generation_seconds"] for r in readl(OUT/"trajectories.jsonl")[:2])/2
                    projected=elapsed+mean*(2*(len(train)+16))+2*len(train)*0.3+60
                    state["speed_check"]={"after_two_official_generations":True,
                        "mean_generation_seconds":round(mean,3), "projected_nontraining_seconds":round(projected,3),
                        "decision":"continue" if projected <= CAP else "stop"}
                    if projected > CAP: state["stop_reason"]="initial_speed_projection_over_cap"
                write(OUT/"run-state.json",state)
                if state["stop_reason"]: break
            if state["stop_reason"]: break
        if state["stop_reason"]: raise RuntimeError(state["stop_reason"])
        baseline_scores=score_targets(model,processor,train,"v118",device,started)
        write(OUT/"target-token-scores-v118.json",baseline_scores)
        del model,processor; gc.collect(); torch.cuda.empty_cache()
        if time.monotonic()-started >= CAP: raise TimeoutError("cap_before_training")
        set_seed(42)
        mean_generation = state["speed_check"]["mean_generation_seconds"]
        post_reserve = max(120, math.ceil(2 * (len(train) + 16) * mean_generation + 2 * len(train) * 0.3 + 60))
        training_budget = CAP - int(time.monotonic()-started) - post_reserve
        if training_budget <= 0:
            raise TimeoutError("baseline_projection_leaves_no_training_budget")
        train_config={"model_path":str(MODEL),"train_path":str(train_file),"validation_path":str(train_file),
            "output_dir":str(OUT/"adapter"),"max_steps":20,"epochs":20,"max_length":2176,
            "seed":42,"batch_size":1,"gradient_accumulation_steps":len(train),
            "gradient_checkpointing":False,"adapter_path":str(V118),"enable_thinking":False,
            "learning_rate":1e-4,"save_steps":20,"eval_steps":21,"warmup_ratio":0.05,
            "wall_clock_cap_seconds":training_budget}
        write(OUT/"training-config-effective.json",{"trainer_kwargs":train_config,
            "optimizer":"AdamW","weight_decay":0.0,"scheduler":"linear",
            "warmup_steps":1,"validation_not_run":True,
            "post_eval_reserve_seconds":post_reserve})
        metrics=train_sft(**train_config)
        write(OUT/"training-metrics.json",metrics)
        if metrics["global_step"] != 20 or metrics["speed_check"]["status"] != "within_cap":
            raise RuntimeError("training_did_not_complete_frozen_20_steps")
        after_delta={"files":{p.name:sha(p) for p in (OUT/"adapter").iterdir() if p.is_file()},
                     "adapter_model_sha256":sha(OUT/"adapter/adapter_model.safetensors")}
        from safetensors.torch import load_file
        original_weights=load_file(str(V118/"adapter_model.safetensors"),device="cpu")
        trained_weights=load_file(str(OUT/"adapter/adapter_model.safetensors"),device="cpu")
        if set(original_weights)!=set(trained_weights):
            raise ValueError("trained_adapter_tensor_key_mismatch")
        deltas=[(trained_weights[k].float()-original_weights[k].float()) for k in original_weights]
        delta_sq=sum(float(torch.sum(value*value)) for value in deltas)
        original_sq=sum(float(torch.sum(value.float()*value.float())) for value in original_weights.values())
        after_delta.update({"tensor_keys":len(original_weights),"changed_tensor_count":sum(bool(torch.any(x!=0)) for x in deltas),
            "delta_l2_norm":math.sqrt(delta_sq),"starting_weight_l2_norm":math.sqrt(original_sq),
            "relative_delta_l2":math.sqrt(delta_sq)/max(math.sqrt(original_sq),1e-12)})
        write(OUT/"trained-adapter-hashes.json",after_delta)
        del metrics; gc.collect(); torch.cuda.empty_cache()
        if time.monotonic()-started >= CAP: raise TimeoutError("cap_before_post_eval")
        set_seed(42)
        load_at=time.monotonic()
        model,processor=build_lora_model(str(MODEL),dtype="bfloat16",gradient_checkpointing=False,
                                         adapter_path=str(OUT/"adapter"))
        model.eval(); state["model_loads"]["microfit"] = round(time.monotonic()-load_at,3)
        active=model.active_adapters() if callable(getattr(model,"active_adapters",None)) else [model.active_adapter]
        if "default" not in active: raise ValueError("trained_adapter_not_active")
        device=next(model.parameters()).device
        for split,group in rows:
            for row in group:
                if time.monotonic()-started >= CAP:
                    state["stop_reason"]="cap_during_post_eval"
                    break
                generate(model,processor,row,"microfit",split,device,started)
                state["completed"] += 1
                write(OUT/"run-state.json",state)
            if state["stop_reason"]: break
        if state["stop_reason"]: raise RuntimeError(state["stop_reason"])
        post_scores=score_targets(model,processor,train,"microfit",device,started)
        write(OUT/"target-token-scores-microfit.json",post_scores)
        state["status"]="complete"; state["elapsed_seconds"]=round(time.monotonic()-started,3)
        state["completed"]=2*(len(train)+16)
        write(OUT/"run-state.json",state)
    except BaseException as exc:
        state["status"]="failed"; state["stop_reason"]=type(exc).__name__+":"+str(exc)[:300]
        state["elapsed_seconds"]=round(time.monotonic()-started,3)
        write(OUT/"run-state.json",state)
        raise


def summarize() -> None:
    if (OUT/"summary.json").exists(): raise FileExistsError("summary_already_exists")
    state=json.loads((OUT/"run-state.json").read_text(encoding="utf-8"))
    if state["status"] != "complete":
        write(OUT/"partial-summary.json",{"state":state,"reason":"incomplete_run_not_scored_as_full"})
        return
    rows=readl(OUT/"trajectories.jsonl")
    train=readl(OUT/"train-frozen.jsonl")
    matrix={(r["arm"],r["case_id"]):r for r in rows}
    baseline={r["case_id"]:r for r in rows if r["arm"]=="v118" and r["split"]=="exploratory_eval"}
    trained={r["case_id"]:r for r in rows if r["arm"]=="microfit" and r["split"]=="exploratory_eval"}
    if len(baseline)!=16 or len(trained)!=16: raise ValueError("incomplete_G_eval_pairing")
    if any(baseline[c]["prompt_sha256"]!=trained[c]["prompt_sha256"] for c in baseline):
        raise ValueError("G_eval_prompt_mismatch")
    if len({r["case_id"] for r in rows if r["split"]=="train" and r["arm"]=="v118"})!=len(train):
        raise ValueError("incomplete_train_generation")
    def metrics(group):
        return {"n":len(group),"em":sum(r["em"] for r in group),
            "mean_f1":sum(r["f1"] for r in group)/len(group),
            "protocol_valid":sum(r["protocol_valid"] for r in group)}
    btrain=[r for r in rows if r["arm"]=="v118" and r["split"]=="train"]
    ttrain=[r for r in rows if r["arm"]=="microfit" and r["split"]=="train"]
    beval=list(baseline.values()); teval=list(trained.values())
    comparisons={"train":{"v118":metrics(btrain),"microfit":metrics(ttrain)},
        "goal_G_exploratory":{"v118":metrics(beval),"microfit":metrics(teval),
        "new_correct":sorted(c for c in baseline if not baseline[c]["em"] and trained[c]["em"]),
        "lost_correct":sorted(c for c in baseline if baseline[c]["em"] and not trained[c]["em"]),
        "retained_correct":sorted(c for c in baseline if baseline[c]["em"] and trained[c]["em"]),
        "changed_raw":sorted(c for c in baseline if baseline[c]["raw_output"]!=trained[c]["raw_output"])} }
    scores_before={r["case_id"]:r for r in read_json(OUT/"target-token-scores-v118.json")}
    scores_after={r["case_id"]:r for r in read_json(OUT/"target-token-scores-microfit.json")}
    token={cid:{"v118":scores_before[cid],"microfit":scores_after[cid],
        "answer_nll_delta":scores_after[cid]["answer"]["nll"]-scores_before[cid]["answer"]["nll"],
        "format_nll_delta":scores_after[cid]["format"]["nll"]-scores_before[cid]["format"]["nll"]}
        for cid in scores_before}
    fit_success=(comparisons["train"]["microfit"]["em"]==len(train) and
                 comparisons["train"]["microfit"]["protocol_valid"]==len(train))
    decision={"fit_success":fit_success,"generalization_claim":False,
        "exploratory_eval_now_used_for_research_decision":True,
        "recommendation":("fitting succeeded; broaden ability coverage before considering more training" if fit_success and not comparisons["goal_G_exploratory"]["new_correct"] else
                          "microfit learned selected cases; interpret Goal G shifts as exploratory only" if fit_success else
                          "fit incomplete; inspect per-token probabilities and optimization signals; do not extend this run")}
    report={"status":"complete_microfit_diagnostic","elapsed_seconds":state["elapsed_seconds"],
        "speed_check":state["speed_check"],"comparisons":comparisons,"training_token_scores":token,
        "fit_decision":decision,"gradient_and_optimizer_history":read_json(OUT/"training-metrics.json")["train_metrics"],
        "adapter_sha256":sha(OUT/"adapter/adapter_model.safetensors"),
        "trajectories_sha256":sha(OUT/"trajectories.jsonl"),
        "freeze_manifest_sha256":sha(OUT/"freeze-manifest.json"),
        "final_test_accessed":False,"sealed_independent_100_accessed":False,"additional_training":False}
    write(OUT/"summary.json",report)
    print(json.dumps({"status":report["status"],"elapsed_seconds":report["elapsed_seconds"],
        "train":comparisons["train"],"exploratory_eval":comparisons["goal_G_exploratory"],"fit_success":fit_success},ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("prepare","install-votes","preflight","run","worker","summarize"))
    phase=parser.parse_args().phase
    {"prepare":prepare,"install-votes":install_votes,"preflight":preflight,
     "run":run,"worker":worker,"summarize":summarize}[phase]()
