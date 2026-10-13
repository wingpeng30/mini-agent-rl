"""Goal AG: audit released-model failures and train only if repair evidence passes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import run_v133_goal_aa_audit_confirmation as aa
import run_v133_goal_ab_evidence_decision as ab
import run_v133_goal_ac_failure_state_repair as ac
import run_v133_goal_af_retrieval_validation as af
import run_v133_goal_q_search as q
import run_v133_goal_z_multihop_sft as z

ROOT = z.ROOT
OUT = ROOT / "runs/v133-goal-ag-failure-repair"
SUPP = OUT / "supplemental"
CFG = OUT / "config.json"
STATE = OUT / "state.json"
PACKET = af.OUT / "anonymous-semantic-review-packet.jsonl"
REVIEW_FILES = (OUT / "review-first-half.jsonl", OUT / "review-second-half.jsonl")
CATEGORIES = {"correct_sufficient", "missing_required_information", "factual_error",
              "unsupported_by_visible_evidence", "indeterminate"}
EXPECTED_SPLITS = {"goal_z_dev": 64, "old_main": 16, "old_regression": 16,
                   "goal_y_regression": 48, "aa_regression": 64}
SOURCE = z.OUT / "seed42-trajectory" / "adapter-final"
MIN_REPAIRS, TARGET_REPAIRS = 16, 32
AG_REPAIRS = OUT / "repair-material-final-ag.jsonl"
AG_REPAIR_GATE = OUT / "repair-material-gate-ag.json"
TRAINING_FREEZE = OUT / "training-freeze.json"
TRAINING_LABELS = ("42-A", "42-B", "43-A", "43-B")
AG_SPLITS = ("dev", "old", "y", "aa_regression")
FINAL_LEDGER_V2 = OUT / "correction-ledger-final-v2.jsonl"
GOAL_Z_EXPECTED_CODE_DRIFT = {
    "model_code": {
        "frozen": "062afb7c5d0b40d627612cfaaef5924e32abb29faf3412afcbd369b2d41a8f56",
        "current": "609b97144b4ece53b6805f8d18fe291b93f53b99e09c32bf39a87c6c7f84a2f6",
        "scope": "optional inference timeout support; model scoring and adapter loading unchanged",
    },
    "search_code": {
        "frozen": "57b8002e6d7a2746a59babfcb6b694ebc3992cd70048cbc862363e2aec42b23c",
        "current": "24aed5225b40b178112edfaf2631cdb214112b5e795e43c8c29389b00d556935",
        "scope": "additional v3/v4 retrievers; BM25-v2 implementation retained unchanged",
    },
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected JSON list: {path}")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_once(path: Path, value) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def write(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                          encoding="utf-8")


def append(path: Path, value) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def visible_tool_documents(actual_prompt: str) -> list[dict]:
    """Parse only tool JSON serialized in the model's actual final prompt."""
    documents: list[dict] = []
    for match in re.finditer(r"<tool_response>\s*(\{.*?\})\s*</tool_response>",
                             actual_prompt, re.DOTALL):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed tool-response JSON in actual prompt") from exc
        result = payload.get("result", [])
        if not isinstance(result, list):
            continue
        for doc in result:
            if isinstance(doc, dict):
                documents.append({"document_id": doc.get("id"), "title": doc.get("title"),
                    "text": doc.get("text", ""), "prompt_start": match.start(),
                    "prompt_end": match.end(), "tool_response": len(documents)})
    return documents


def merge_review_packet_item(packet_by_id: dict[str, dict], review_id: str, item: dict) -> None:
    """Deduplicate equal final answer contexts while retaining distinct traces."""
    if review_id not in packet_by_id:
        packet_by_id[review_id] = item
        return
    prior = packet_by_id[review_id]
    prior_core = {k: v for k, v in prior.items() if k not in {"steps", "trace_variants"}}
    item_core = {k: v for k, v in item.items() if k != "steps"}
    if prior_core != item_core:
        raise RuntimeError("Review ID collision with different final evidence packet")
    variants = prior.get("trace_variants", [prior.pop("steps", [])])
    trace = item.get("steps", [])
    if trace not in variants:
        variants.append(trace)
    prior["trace_variants"] = variants


def protected_successes(current_start_rows: list[dict], semantic_by_case: dict[str, str]) -> set[str]:
    """Only currently successful cases can be protected against regression."""
    return {str(row["case_id"]) for row in current_start_rows
            if semantic_by_case.get(str(row["case_id"])) == "correct_sufficient" and
            row.get("required_evidence_first_found_after_first_search", False)}


def material_gate(material: list[dict], primary_cause: str) -> dict:
    focus = sum(x.get("primary_cause_category", x.get("primary_error")) == primary_cause
                for x in material)
    n = len(material)
    return {"passed": MIN_REPAIRS <= n <= TARGET_REPAIRS and focus * 2 >= n,
            "n": n, "focus_count": focus, "required_minimum": MIN_REPAIRS,
            "required_focus_fraction": 0.5}


def development_gate(candidate: dict, baseline: dict, old_splits: dict[str, int],
                     candidate_old: dict[str, int], baseline_protected: set[str],
                     candidate_successes: set[str]) -> dict:
    failures = []
    if candidate.get("dev_success", 0) < baseline.get("dev_success", 0) + 4:
        failures.append("dev_gain")
    if candidate.get("dev_bad", 0) > baseline.get("dev_bad", 0):
        failures.append("dev_bad_total")
    if candidate.get("dev_invalid", 0) > baseline.get("dev_invalid", 0):
        failures.append("dev_invalid")
    for split, base_count in old_splits.items():
        if candidate_old.get(split, 0) < base_count - 1:
            failures.append(f"old_drop:{split}")
    lost = baseline_protected - candidate_successes
    if len(lost) > 1:
        failures.append("protected_continuous_search")
    return {"passed": not failures, "failures": failures, "protected_lost": sorted(lost)}


def confirmation_gate(start_success: int, candidate_success: int, start_bad: int,
                      candidate_bad: int, start_invalid: int, candidate_invalid: int,
                      protected_lost: int) -> bool:
    return (candidate_success >= start_success + 4 and candidate_bad <= start_bad and
            candidate_invalid <= start_invalid and protected_lost <= 1)


def state(status: str, **updates) -> None:
    current = json.loads(STATE.read_text(encoding="utf-8"))
    current.update(status=status, updated_at=time.time(), **updates)
    write(STATE, current)


def source_identity() -> dict:
    """Check the actual published Goal Z artifact against AA's release decision."""
    release = json.loads((aa.OUT / "product-decision.json").read_text(encoding="utf-8"))
    expected_adapter = json.loads((z.OUT / "seed42-trajectory" / "summary.json")
                                  .read_text(encoding="utf-8"))["adapter_sha256"]
    adapter_file = SOURCE / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)
    actual_adapter = sha(adapter_file)
    if expected_adapter != actual_adapter:
        raise RuntimeError(f"Published Goal Z adapter mismatch: AA={expected_adapter}, actual={actual_adapter}")
    if release.get("published") is not True or release.get("selected_route") != "trajectory" or \
       release.get("selected_seed") != 42:
        raise RuntimeError("AA release record does not publish Goal Z trajectory seed 42")
    if Path(release.get("adapter", "")).as_posix().rstrip("/") != \
       "/root/autodl-tmp/mini-agent-rl/runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final":
        raise RuntimeError("AA release points at a different adapter path")
    if af.sha(af.PUBLISHED_ADAPTER / "adapter_model.safetensors") != actual_adapter:
        raise RuntimeError("AF product adapter is not the released Goal Z adapter")
    expected_base = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
    base_files = sorted(q.p.BASE.glob("*.safetensors"))
    if not base_files:
        raise RuntimeError(f"Base weight files unavailable under {q.p.BASE}")
    base_hashes = {p.name: sha(p) for p in base_files}
    if len(base_hashes) != 1 or next(iter(base_hashes.values())) != expected_base:
        raise RuntimeError(f"Base model weight mismatch: {base_hashes}")
    return {"release": release, "adapter_path": str(SOURCE), "adapter_sha256": actual_adapter,
            "base_weights": base_hashes, "tokenizer_sha256": sha(q.p.BASE / "tokenizer.json")}


def goal_z_dependency_compatibility() -> dict:
    """Allow only the two known post-Goal-Z source extensions used by later goals."""
    frozen = z.read(z.CFG)
    current = {key: z.sha(path) for key, path in z.frozen_files().items()}
    differences = {key: {"frozen": frozen["files"].get(key), "current": current.get(key)}
                   for key in set(frozen["files"]) | set(current)
                   if frozen["files"].get(key) != current.get(key)}
    expected = {key: {"frozen": item["frozen"], "current": item["current"]}
                for key, item in GOAL_Z_EXPECTED_CODE_DRIFT.items()}
    if differences != expected:
        raise RuntimeError(f"Unreviewed Goal Z frozen dependency changes: {differences}")
    if z.software() != frozen["software"]:
        raise RuntimeError("Goal Z frozen runtime/software changed")
    return {key: {**item, "verified_frozen_sha256": differences[key]["frozen"],
                  "verified_current_sha256": differences[key]["current"]}
            for key, item in GOAL_Z_EXPECTED_CODE_DRIFT.items()}


def record_compatibility_addendum() -> dict:
    path = OUT / "training-compatibility-addendum.json"
    value = {"goal_z_dependency_compatibility": goal_z_dependency_compatibility(),
        "frozen_training_manifest_sha256": sha(TRAINING_FREEZE),
        "preflight_failure": {
            "error": "Goal Z config() rejected known later code extensions before model loading",
            "resolution": "Goal AG now loads with explicit frozen Goal Z product parameters after a narrow code-drift audit",
            "no_model_weights_loaded": True, "optimizer_steps": 0,
            "empty_seed42_A_directory_archived": True},
        "note": "The model-scoring and BM25-v2 paths are unchanged; runtime/software and all non-code Goal Z frozen assets match."}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError("Goal AG compatibility addendum already exists with different contents")
        return existing
    write_once(path, value)
    return value


def prepare() -> dict:
    if CFG.exists() or STATE.exists():
        raise FileExistsError(f"Goal AG was already prepared: {OUT}")
    identity = source_identity()
    if not PACKET.is_file():
        raise FileNotFoundError(PACKET)
    packet = rows(PACKET)
    if len(packet) != 624 or len({x.get("review_id") for x in packet}) != 624:
        raise RuntimeError("AF stored audit packet must contain 624 unique review rows")
    baseline = rows(af.OUT / "eval-A.jsonl")
    split_counts = Counter(x.get("split") for x in baseline)
    # AF uses these exact four split names; the old cohort retains its two subsplits.
    if len(baseline) != 208:
        raise RuntimeError(f"AF A baseline has {len(baseline)}/208 rows")
    config = {
        "goal": "AG", "start": str(SOURCE), "identity": identity,
        "packet_sha256": sha(PACKET), "baseline_sha256": sha(af.OUT / "eval-A.jsonl"),
        "prior_review_sha256": sha(af.OUT / "semantic-review-map.jsonl"),
        "evaluation_counts": EXPECTED_SPLITS, "evaluation_rows_by_source_split": dict(split_counts),
        "model": {"base": str(q.p.BASE), "dtype": "bfloat16", "quantization": False,
                  "eval_mode": True, "generation": "greedy", "temperature": 0,
                  "do_sample": False},
        "search": {"runner": "Goal Q", "prompt": "direct-v090", "thinking": False,
                   "encoding": "legacy", "retriever": "BM25-v2", "top_k": 3,
                   "max_model_calls": 4, "max_searches": 3,
                   "max_input_tokens": 2048, "max_new_tokens": 128},
        "optimization": {"learning_rate": 5e-6, "betas": [0.9, 0.999], "eps": 1e-8,
                         "weight_decay": 0, "gradient_clip": 1},
        "training": {"seeds": [42, 43], "updates_per_run": 128,
                     "arms": ["ordinary_continuation", "failure_repair"],
                     "repair_target": TARGET_REPAIRS, "repair_minimum": MIN_REPAIRS,
                     "repair_blend": [0.5, 0.5]},
        "gate": {"dev_gain": 4, "old_set_drop_max_each": 1,
                 "confirmation_outputs_max": 192},
        "confirmation_opened": False,
        "created_at": time.time(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_once(CFG, config)
    write_once(OUT / "baseline-file-facts.json", {"rows": len(baseline),
        "split_counts": dict(split_counts), "status": "semantic_audit_pending"})
    write_once(STATE, {"status": "awaiting_blind_review", "stage": "AG0",
        "confirmation_accessed": False, "created_at": time.time()})
    return config


def merge_reviews() -> dict:
    packet = rows(PACKET)
    packet_by_id = {x["review_id"]: x for x in packet}
    if len(packet_by_id) != 624:
        raise RuntimeError("Packet review_id collision")
    reviews = []
    for path in REVIEW_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Independent review not ready: {path}")
        reviews.extend(rows(path))
    by_id = {x.get("review_id"): x for x in reviews}
    if len(by_id) != 624 or set(by_id) != set(packet_by_id):
        raise RuntimeError(f"Review coverage mismatch: {len(by_id)}/624")
    validation = af.validate_semantic_reviews(packet_by_id, reviews)
    counts = Counter(x["category"] for x in reviews)
    dest = OUT / "review-merged-provisional.jsonl"
    if dest.exists():
        raise FileExistsError(dest)
    with dest.open("w", encoding="utf-8") as f:
        for packet_row in packet:
            row = {**by_id[packet_row["review_id"]], "case_id": packet_row["case_id"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"rows": len(by_id), "categories": dict(counts),
            "validator_complete": validation["complete"],
            "status": "provisional_requires_lead_adjudication"}


def summarize() -> dict:
    if not STATE.exists():
        raise FileNotFoundError("Run prepare first")
    s = json.loads(STATE.read_text(encoding="utf-8"))
    result = {"state": s, "config": json.loads(CFG.read_text(encoding="utf-8"))}
    if (OUT / "review-merged-provisional.jsonl").exists():
        merged = rows(OUT / "review-merged-provisional.jsonl")
        result["provisional_review_counts"] = dict(Counter(x["category"] for x in merged))
    if (OUT / "diagnosis-final.jsonl").exists():
        diagnosis = rows(OUT / "diagnosis-final.jsonl")
        result["diagnosis"] = {"rows": len(diagnosis),
            "causes": dict(Counter(x.get("primary_cause", x.get("primary_cause_category", "unclassified"))
                                   for x in diagnosis))}
    if (OUT / "repair-material-final.jsonl").exists():
        material = rows(OUT / "repair-material-final.jsonl")
        result["repair_material"] = {"rows": len(material),
            "focus": dict(Counter(x["primary_error"] for x in material))}
    if AG_REPAIRS.exists():
        material = rows(AG_REPAIRS)
        result["repair_material_final"] = {"rows": len(material),
            "focus": dict(Counter(x.get("primary_cause_category", x.get("primary_error"))
                                  for x in material))}
    if AG_REPAIR_GATE.exists():
        result["repair_gate_final"] = json.loads(AG_REPAIR_GATE.read_text(encoding="utf-8"))
    if TRAINING_FREEZE.exists():
        result["training_freeze"] = json.loads(TRAINING_FREEZE.read_text(encoding="utf-8"))
    if (OUT / "correction-ledger.jsonl").exists():
        ledger = rows(OUT / "correction-ledger.jsonl")
        result["corrected_af_reviews"] = {
            "rows": len(ledger),
            "categories": dict(Counter(x["category"] for x in ledger)),
            "by_arm_and_split": {
                f"{arm}:{split}": dict(Counter(x["category"] for x in ledger
                    if x.get("arm") == arm and x.get("split") == split))
                for arm, split in sorted({(x.get("arm"), x.get("split")) for x in ledger})}}
    if FINAL_LEDGER_V2.exists():
        ledger = rows(FINAL_LEDGER_V2)
        result["corrected_af_reviews_v2"] = {
            "rows": len(ledger),
            "categories": dict(Counter(x["category"] for x in ledger)),
            "by_arm_and_split": {
                f"{arm}:{split}": dict(Counter(x["category"] for x in ledger
                    if x.get("arm") == arm and x.get("split") == split))
                for arm, split in sorted({(x.get("arm"), x.get("split")) for x in ledger})}}
    if (OUT / "repair-material-gate.json").exists():
        result["repair_gate"] = json.loads((OUT / "repair-material-gate.json").read_text(encoding="utf-8"))
    if (OUT / "ag1-diagnosis-report.json").exists():
        result["ag1_diagnosis"] = json.loads((OUT / "ag1-diagnosis-report.json").read_text(encoding="utf-8"))
    write(OUT / "summary.json", result)
    return result


def render_report() -> str:
    """Preserve the detailed lead-authored report, or render a minimal decision."""
    report = OUT / "REPORT.md"
    if report.is_file():
        return report.read_text(encoding="utf-8")
    decision_path = OUT / "confirmation-decision.json"
    if not decision_path.is_file():
        decision_path = OUT / "development-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    return ("# Goal AG results\n\n"
            f"Status: {decision['status']}\n\n"
            "Frozen decision and scores:\n\n```json\n"
            + json.dumps(decision, ensure_ascii=False, indent=2) + "\n```\n")


def run() -> dict:
    if not CFG.is_file() or not STATE.is_file():
        raise RuntimeError("Run prepare first")
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    current = source_identity()
    if current["adapter_sha256"] != cfg["identity"]["adapter_sha256"]:
        raise RuntimeError("Published Goal Z model identity changed after prepare")
    if sha(PACKET) != cfg["packet_sha256"] or sha(af.OUT / "eval-A.jsonl") != cfg["baseline_sha256"]:
        raise RuntimeError("Frozen AF inputs changed after prepare")
    st = json.loads(STATE.read_text(encoding="utf-8"))
    if st.get("status") in {"complete", "stopped"}:
        return summarize()
    if st.get("status") == "awaiting_blind_review":
        if FINAL_LEDGER_V2.is_file():
            state("awaiting_lead_adjudication", review_rows=624,
                  review_rows_validated=True,
                  next_action="使用版本化最终输入纠错账本及冻结训练材料继续")
            return run()
        merged = merge_reviews()
        write_once(OUT / "review-merge-summary.json", merged)
        state("awaiting_lead_adjudication", review_rows=merged["rows"],
              next_action="主 agent 复核全部成功答案、引用及审核改判，冻结纠错账本后进入 AG1")
        return summarize()
    if st.get("status") == "awaiting_lead_adjudication":
        needed = (FINAL_LEDGER_V2, OUT / "diagnosis-final.jsonl",
                  OUT / "repair-adjudication-ledger.jsonl", AG_REPAIR_GATE, AG_REPAIRS)
        missing = [str(p.name) for p in needed if not p.is_file()]
        if missing:
            return {**summarize(), "awaiting": missing}
        gate = json.loads(AG_REPAIR_GATE.read_text(encoding="utf-8"))
        if not gate.get("passed"):
            state("stopped", stage="AG2", stop_reason="repair_material_gate_failed",
                  training_started=False, confirmation_accessed=False,
                  eligible_repairs=gate.get("selected_repairs"),
                  required_repairs=gate.get("minimum_repairs"))
        else:
            state("awaiting_training_preparation", stage="AG2",
                  eligible_repairs=gate.get("selected_repairs"),
                  next_action="冻结 A/B 两 seed 的题序、绑定关系与训练资产")
    elif st.get("status") == "awaiting_training_preparation":
        prepared = prepare_training()
        return {**prepared, "run_result": run()}
    elif st.get("status") == "training":
        for seed in (42, 43):
            for arm in ("A", "B"):
                folder = OUT / f"seed{seed}-{arm}"
                if not (folder / "summary.json").exists():
                    if folder.exists():
                        finalize_saved_checkpoint(seed, arm)
                    else:
                        train(seed, arm)
        state("evaluating", stage="AG3", training_finished=True)
        return run()
    elif st.get("status") == "evaluating":
        evaluate_development()
        return summarize()
    elif st.get("status") == "awaiting_development_review":
        needed = (OUT / "development-review-a.jsonl", OUT / "development-review-b.jsonl")
        missing = [p.name for p in needed if not p.exists()]
        if missing:
            return {**summarize(), "awaiting": missing}
        decision = calculate_development_gate()
        if decision["status"] == "confirmation_authorized":
            return run()
        write(OUT / "REPORT.md", render_report())
        return decision
    elif st.get("status") == "confirmation_authorized":
        evaluate_confirmation()
        state("awaiting_confirmation_review", stage="AG5", confirmation_accessed=True)
        return summarize()
    elif st.get("status") == "awaiting_confirmation_review":
        if not (OUT / "confirmation-review-final.jsonl").exists() and not all(
            (OUT / f"confirmation-review-{suffix}.jsonl").exists() for suffix in ("a", "b")):
            return {**summarize(), "awaiting": ["confirmation-review-final.jsonl"]}
        result = calculate_confirmation_gate()
        write(OUT / "REPORT.md", render_report())
        return result
    return summarize()


def prepare_training() -> dict:
    """Freeze Goal Z update order and bind the reviewed repairs before training."""
    current = source_identity()
    dependency_compatibility = goal_z_dependency_compatibility()
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    if current["adapter_sha256"] != cfg["identity"]["adapter_sha256"]:
        raise RuntimeError("Goal Z published adapter changed after Goal AG prepare")
    gate = json.loads(AG_REPAIR_GATE.read_text(encoding="utf-8"))
    repairs = rows(AG_REPAIRS)
    inputs = rows(z.OUT / "training-inputs.jsonl")
    if not gate.get("passed") or len(repairs) < 16 or len(repairs) > 32:
        raise RuntimeError("Reviewed repair material does not pass its frozen gate")
    input_by_id = {str(x["id"]): x for x in inputs}
    if len(input_by_id) != 128:
        raise RuntimeError("Goal Z must have exactly 128 original training trajectories")
    repair_by_id = {str(x["case_id"]): x for x in repairs}
    if len(repair_by_id) != len(repairs) or any(not x.get("eligible") for x in repairs):
        raise RuntimeError("Repair material has duplicate or unapproved rows")
    if not all(x.get("repair_actions") for x in repairs):
        raise RuntimeError("Every selected repair must include a supervised action")
    zcfg = json.loads(z.CFG.read_text(encoding="utf-8"))
    frozen_orders = zcfg["order"]
    binding = {}
    extra_ids = sorted((case_id for case_id in repair_by_id if case_id not in input_by_id),
                       key=lambda case_id: hashlib.sha256(case_id.encode()).hexdigest())
    original_ids = set(input_by_id)
    if not extra_ids:
        raise RuntimeError("Expected reviewed supplement examples were not present")
    for seed in (42, 43):
        order = [str(x) for x in frozen_orders[str(seed)]]
        if len(order) != 128 or set(order) != original_ids:
            raise RuntimeError(f"Goal Z frozen seed {seed} order mismatch")
        position = {case_id: i + 1 for i, case_id in enumerate(order)}
        bind = {position[case_id]: case_id for case_id in repair_by_id if case_id in position}
        open_positions = [i for i in range(1, 129) if i not in bind]
        if len(open_positions) < len(extra_ids):
            raise RuntimeError("Not enough update positions for the frozen supplement")
        for update, repair_case_id in zip(open_positions, extra_ids):
            bind[update] = repair_case_id
        if len(bind) != len(repair_by_id) or set(bind.values()) != set(repair_by_id):
            raise RuntimeError("Repair binding is not one-to-one")
        binding[str(seed)] = [{"update": update, "training_case_id": order[update - 1],
            "repair_case_id": repair_id,
            "repair_source": "original_train_case" if repair_id in original_ids else "frozen_supplement"}
            for update, repair_id in sorted(bind.items())]
    freeze = {"start_adapter": str(SOURCE), "start_adapter_sha256": current["adapter_sha256"],
        "base_weights": current["base_weights"], "tokenizer_sha256": current["tokenizer_sha256"],
        "original_training_input_sha256": sha(z.OUT / "training-inputs.jsonl"),
        "original_train_corpus_sha256": sha(z.OUT / "corpus-train.json"),
        "repair_material_sha256": sha(AG_REPAIRS), "repair_gate_sha256": sha(AG_REPAIR_GATE),
        "supplement_trace_sha256": sha(SUPP / "attempt-02/start-model-traces.jsonl"),
        "seeds": [42, 43], "arms": ["A", "B"], "updates_per_run": 128,
        "order": {str(seed): [str(x) for x in frozen_orders[str(seed)]] for seed in (42, 43)},
        "repair_binding": binding, "repair_count": len(repairs),
        "primary_query_bridge_repairs": gate["primary_cause_repairs"],
        "post_goal_z_code_compatibility": dependency_compatibility,
        "optimization": cfg["optimization"], "product": cfg["search"],
        "evaluation_after_all_training": True, "confirmation_accessed": False}
    write_once(TRAINING_FREEZE, freeze)
    state("training", stage="AG3", training_started=True,
          training_freeze_sha256=sha(TRAINING_FREEZE), repair_count=len(repairs))
    return {"status": "training", "training_freeze": freeze}


def train(seed: int, arm: str) -> dict:
    """Run one matched 128-update continuation from the published Goal Z adapter."""
    import gc
    import math
    import time
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as o

    if seed not in (42, 43) or arm not in ("A", "B") or not TRAINING_FREEZE.is_file():
        raise RuntimeError("AG training inputs are not frozen")
    if z.read(STATE)["status"] != "training":
        raise RuntimeError("Goal AG training stage is not active")
    goal_z_dependency_compatibility()
    record_compatibility_addendum()
    folder = OUT / f"seed{seed}-{arm}"
    if folder.exists():
        raise FileExistsError(f"Refusing to resume or overwrite an interrupted run: {folder}")
    folder.mkdir()
    freeze = json.loads(TRAINING_FREEZE.read_text(encoding="utf-8"))
    if sha(AG_REPAIRS) != freeze["repair_material_sha256"] or \
       sha(z.OUT / "training-inputs.jsonl") != freeze["original_training_input_sha256"] or \
       source_identity()["adapter_sha256"] != freeze["start_adapter_sha256"]:
        raise RuntimeError("Frozen AG training input or Goal Z start adapter changed")
    inputs = {str(x["id"]): x for x in rows(z.OUT / "training-inputs.jsonl")}
    repairs = {str(x["case_id"]): x for x in rows(AG_REPAIRS)}
    bind = {int(x["update"]): repairs[str(x["repair_case_id"])]
            for x in freeze["repair_binding"][str(seed)]}
    set_seed(seed)
    client = make_ag_client(SOURCE)
    params = [p for _, p in client.prepare_for_training()]
    base_before = o.parameter_hashes(client._model, lora=False)
    lora_before = o.parameter_hashes(client._model, lora=True)
    opt = freeze["optimization"]
    optimizer = torch.optim.AdamW(params, lr=opt["learning_rate"],
        betas=tuple(opt["betas"]), eps=opt["eps"], weight_decay=opt["weight_decay"])
    order = freeze["order"][str(seed)]
    started = time.monotonic()
    last_action = None
    try:
        for update, case_id in enumerate(order, 1):
            if update > 128:
                raise RuntimeError("AG update limit exceeded")
            actions = inputs[str(case_id)]["actions"]
            original_tokens = sum(len(x["action_token_ids"]) for x in actions)
            if original_tokens <= 0:
                raise RuntimeError("Empty original trajectory supervision")
            repair = bind.get(update) if arm == "B" else None
            optimizer.zero_grad(set_to_none=True)
            client.set_training_forward_mode()
            scalar = 0.
            original_weight = .5 if repair else 1.
            for action in actions:
                values = client.score_action_tensor([], action["action_token_ids"], True,
                    action["prompt_token_ids"], action["prompt_attention_mask"]).float()
                loss = -original_weight * values.sum() / original_tokens
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite original trajectory loss")
                loss.backward()
                scalar += float(loss.detach())
            repair_tokens = 0
            if repair:
                repair_actions = repair["repair_actions"]
                repair_tokens = sum(len(x["action_token_ids"]) for x in repair_actions)
                if repair_tokens <= 0:
                    raise RuntimeError("Empty correction-action supervision")
                for action in repair_actions:
                    values = client.score_action_tensor([], action["action_token_ids"], True,
                        action["prompt_token_ids"], action["prompt_attention_mask"]).float()
                    loss = -.5 * values.sum() / repair_tokens
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite correction trajectory loss")
                    loss.backward()
                    scalar += float(loss.detach())
            grad = float(torch.nn.utils.clip_grad_norm_(params, opt["gradient_clip"]))
            if not math.isfinite(grad) or grad <= 0:
                raise RuntimeError("Invalid gradient")
            guarded_optimizer_step(optimizer, params, client.gpu_memory_stats(), 21.)
            z.append(folder / "updates.jsonl", {"step": update, "case_id": str(case_id),
                "arm": arm, "loss": scalar, "gradient": grad,
                "original_tokens": original_tokens, "repair_tokens": repair_tokens,
                "repair_case_id": repair.get("case_id") if repair else None,
                "gpu_memory": client.gpu_memory_stats(),
                "elapsed_seconds": time.monotonic() - started})
            last_action = actions[-1]
            if update % 8 == 0:
                print(json.dumps({"seed": seed, "arm": arm, "updates": update,
                    "repair": bool(repair), "seconds": round(time.monotonic() - started, 2)}), flush=True)
        if len(rows(folder / "updates.jsonl")) != 128:
            raise RuntimeError("AG update count mismatch")
        if o.parameter_hashes(client._model, lora=False) != base_before:
            raise RuntimeError("Base weights changed during training")
        lora_after = o.parameter_hashes(client._model, lora=True)
        changed = sum(lora_after[k] != v for k, v in lora_before.items())
        if changed <= 0:
            raise RuntimeError("LoRA weights did not change")
        adapter = folder / "adapter-final"
        client._model.save_pretrained(adapter)
        expected = client.score_action_tensor([], last_action["action_token_ids"], False,
            last_action["prompt_token_ids"], last_action["prompt_attention_mask"]).detach().float().cpu()
        del client, optimizer, params
        gc.collect(); torch.cuda.empty_cache()
        loaded = make_ag_client(adapter)
        actual = loaded.score_action_tensor([], last_action["action_token_ids"], False,
            last_action["prompt_token_ids"], last_action["prompt_attention_mask"]).detach().float().cpu()
        difference = float((expected - actual).abs().max())
        if difference > .001:
            raise RuntimeError(f"Independent reload mismatch: {difference}")
        result = {"seed": seed, "arm": arm, "updates": 128,
            "base_unchanged": True, "changed_lora_tensors": changed,
            "reload_max_abs": difference,
            "adapter_sha256": sha(adapter / "adapter_model.safetensors"),
            "repair_update_count": len(bind) if arm == "B" else 0,
            "elapsed_seconds": time.monotonic() - started}
        write(folder / "summary.json", result)
        del loaded
        gc.collect(); torch.cuda.empty_cache()
        return result
    except BaseException as exc:
        write(folder / "failure.json", {"seed": seed, "arm": arm,
            "error": str(exc), "completed_updates": len(rows(folder / "updates.jsonl"))
                if (folder / "updates.jsonl").exists() else 0,
            "elapsed_seconds": time.monotonic() - started})
        raise


def make_ag_client(adapter: Path):
    """Instantiate the fixed Goal Z model settings without rechecking unrelated code hashes."""
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    product = z.read(z.CFG)["product"]
    return TransformersModelClient(LocalModelConfig(
        model_id=str(q.p.BASE), adapter_path=str(adapter), device="cuda", dtype="bfloat16",
        load_in_4bit=False, gradient_checkpointing=True,
        max_input_tokens=product["max_input_tokens"], max_new_tokens=product["max_new_tokens"],
        enable_thinking=False, deterministic=True,
        observation_encoding=product["observation_encoding"], recompute_old_logprobs=False))


def finalize_saved_checkpoint(seed: int, arm: str) -> dict:
    """Validate a completed checkpoint after a post-save preflight-only interruption."""
    import gc
    import math
    import torch
    from safetensors.torch import load_file
    import run_v133_goal_o_budget as o

    folder = OUT / f"seed{seed}-{arm}"
    failure_path = folder / "failure.json"
    updates_path = folder / "updates.jsonl"
    adapter = folder / "adapter-final"
    failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else {}
    if not folder.is_dir() or not adapter.is_dir() or not (adapter / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"Interrupted Goal AG run has no complete adapter checkpoint: {folder}")
    updates = rows(updates_path)
    if len(updates) != 128 or failure.get("completed_updates") != 128 or \
       "Frozen asset changed" not in failure.get("error", ""):
        raise RuntimeError("Only a 128-update run interrupted by the known post-save Goal Z guard is eligible for checkpoint finalization")
    goal_z_dependency_compatibility()
    record_compatibility_addendum()
    source_adapter_file = SOURCE / "adapter_model.safetensors"
    source_tensors = load_file(str(source_adapter_file), device="cpu")
    final_tensors = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    common = set(source_tensors) & set(final_tensors)
    changed = sum(not torch.equal(source_tensors[key], final_tensors[key]) for key in common)
    if not common or changed <= 0:
        raise RuntimeError("Saved adapter does not contain a verified LoRA update")
    frozen = json.loads(TRAINING_FREEZE.read_text(encoding="utf-8"))
    if source_identity()["base_weights"] != frozen["base_weights"]:
        raise RuntimeError("Base model weight files changed during Goal AG training")
    last_case = frozen["order"][str(seed)][-1]
    inputs = {str(x["id"]): x for x in rows(z.OUT / "training-inputs.jsonl")}
    action = inputs[str(last_case)]["actions"][-1]
    logits = []
    for _ in range(2):
        client = make_ag_client(adapter)
        values = client.score_action_tensor([], action["action_token_ids"], False,
            action["prompt_token_ids"], action["prompt_attention_mask"]).detach().float().cpu()
        if not torch.isfinite(values).all():
            raise RuntimeError("Independent checkpoint reload produced nonfinite action scores")
        logits.append(values)
        del client
        gc.collect(); torch.cuda.empty_cache()
    difference = float((logits[0] - logits[1]).abs().max())
    if difference > .001:
        raise RuntimeError(f"Independent reload consistency failed: {difference}")
    result = {"seed": seed, "arm": arm, "updates": 128, "base_unchanged": True,
        "changed_lora_tensors": changed, "reload_max_abs": difference,
        "reload_method": "two independent loads of the saved checkpoint scored on the same frozen action",
        "adapter_sha256": sha(adapter / "adapter_model.safetensors"),
        "repair_update_count": sum(bool(row.get("repair_case_id")) for row in updates),
        "elapsed_seconds": failure.get("elapsed_seconds"),
        "preflight_failure_recovered_without_retraining": True}
    write(folder / "summary.json", result)
    return result


def _evaluation_specs():
    return ac.evaluation_specs()


def evaluate_candidate(label: str, adapter: Path, split: str, cases: list[dict], corpus: Path) -> None:
    import asyncio
    import gc
    import torch
    path = OUT / f"eval-{split}-{label}.jsonl"
    if path.exists():
        raise FileExistsError(path)
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    pc = q.load_config()["product"]
    client = TransformersModelClient(LocalModelConfig(model_id=str(q.p.BASE),
        adapter_path=str(adapter), max_input_tokens=pc["max_input_tokens"],
        max_new_tokens=pc["max_new_tokens"], enable_thinking=False,
        deterministic=True, observation_encoding=pc["observation_encoding"],
        recompute_old_logprobs=False, timeout=None))
    registry = ToolRegistry()
    registry.register(SearchTool(rows(corpus), pc["retriever"]))
    wrapped = q.SearchTextAliasClient(client)
    for index, case in enumerate(cases, 1):
        answer = case.get("answer", case.get("official_answer"))
        value = asyncio.run(q.rollout(case["question"], wrapped, registry,
                                      sample=False, seed=42))
        value.update(case_id=case.get("id", case.get("case_id")), split=split,
            model=label, official_answer=answer, **q.score(value["answer"], answer))
        append(path, value)
        if index % 8 == 0:
            print(json.dumps({"eval": split, "model": label, "completed": index,
                "total": len(cases)}), flush=True)
    del client, wrapped, registry
    gc.collect(); torch.cuda.empty_cache()


def evaluate_development() -> dict:
    if z.read(STATE)["status"] != "evaluating":
        raise RuntimeError("All four training runs must finish before evaluation")
    outputs = 0
    for seed in (42, 43):
        for arm in ("A", "B"):
            label = f"{seed}-{arm}"
            summary = json.loads((OUT / f"seed{seed}-{arm}/summary.json").read_text(encoding="utf-8"))
            adapter = OUT / f"seed{seed}-{arm}/adapter-final"
            if summary["updates"] != 128 or not summary["base_unchanged"] or \
               summary["reload_max_abs"] > .001 or sha(adapter / "adapter_model.safetensors") != summary["adapter_sha256"]:
                raise RuntimeError(f"Candidate failed checkpoint validation: {label}")
            for split, cases, corpus in _evaluation_specs():
                evaluate_candidate(label, adapter, split, cases, corpus)
                outputs += len(cases)
    if outputs != 4 * 208:
        raise RuntimeError(f"Goal AG candidate evaluation row count mismatch: {outputs}/832")
    packet = build_review_packet(TRAINING_LABELS, "development")
    state("awaiting_development_review", stage="AG4", evaluation_outputs=outputs,
          review_packet=packet)
    return packet


def build_review_packet(labels: tuple[str, ...] | list[str], stage: str) -> dict:
    """Create anonymous, deduplicated reviews from the final actual model inputs."""
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(q.p.BASE, local_files_only=True)
    tok = processor.tokenizer
    packet_by_id: dict[str, dict] = {}
    mapping: list[dict] = []
    for split in AG_SPLITS if stage == "development" else ("confirmation",):
        for label in labels:
            raw_path = OUT / f"eval-{split}-{label}.jsonl"
            for row in rows(raw_path):
                steps = row.get("trace", [])
                final_step = steps[-1] if steps else {}
                actual = tok.decode(final_step.get("prompt_token_ids", []), skip_special_tokens=False)
                visible = visible_tool_documents(actual)
                visible_ids = sorted({str(x["document_id"]) for x in visible if x.get("document_id")})
                trace_view = [{"step": x.get("step"), "action": x.get("action"),
                    "query": x.get("query"), "answer": x.get("answer"),
                    "result_titles": [d.get("title") for d in x.get("observation", {}).get("result", [])],
                    "invalid": bool(x.get("parse_error")),
                    "usage": {k: x.get("usage", {}).get(k) for k in
                        ("actual_prompt_tokens", "history_dropped", "lossy_body_trimmed",
                         "documents_deleted", "context_budget_error", "output_truncated")}}
                    for x in steps]
                combo = {"question": row["question"], "final_prompt": actual,
                    "answer": row.get("answer"), "reason": row.get("reason")}
                review_id = hashlib.sha256(json.dumps(combo, ensure_ascii=False,
                    sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                item = {"review_id": review_id, "question": row["question"],
                    "official_answer": row.get("official_answer"), "answer": row.get("answer"),
                    "reason": row.get("reason"), "actual_final_prompt": actual,
                    "visible_documents": visible, "visible_document_ids": visible_ids,
                    "steps": trace_view, "prompt_token_sha256": hashlib.sha256(
                        json.dumps(final_step.get("prompt_token_ids", []), separators=(",", ":")).encode()).hexdigest()}
                merge_review_packet_item(packet_by_id, review_id, item)
                mapping.append({"review_id": review_id, "split": split,
                    "case_id": row["case_id"], "model": label})
    packet_path = OUT / f"{stage}-review-packet.jsonl"
    map_path = OUT / f"{stage}-review-map.jsonl"
    if packet_path.exists() or map_path.exists():
        raise FileExistsError("Anonymous review packet already exists")
    ordered = [packet_by_id[k] for k in sorted(packet_by_id)]
    write_jsonl(packet_path, ordered)
    write_jsonl(map_path, mapping)
    # Each item is reviewed once. Stable ID parity creates disjoint reviewer batches.
    batches = {"a": [], "b": []}
    for item in ordered:
        batches["a" if int(item["review_id"][-1], 16) % 2 == 0 else "b"].append(item)
    for suffix, batch in batches.items():
        write_jsonl(OUT / f"{stage}-review-{suffix}-packet.jsonl", batch)
    return {"unique_combinations": len(ordered), "candidate_rows": len(mapping),
            "reviewer_a_rows": len(batches["a"]), "reviewer_b_rows": len(batches["b"]) }


def write_jsonl(path: Path, values: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def candidate_scores(review_files: tuple[Path, Path] | None = None) -> tuple[dict, dict]:
    packet = rows(OUT / "development-review-packet.jsonl")
    mapping = rows(OUT / "development-review-map.jsonl")
    final_reviews = OUT / "development-review-final.jsonl"
    if review_files is None and final_reviews.is_file():
        votes = rows(final_reviews)
    else:
        paths = review_files or (OUT / "development-review-a.jsonl", OUT / "development-review-b.jsonl")
        votes = [row for path in paths for row in rows(path)]
    by_vote = {str(x.get("review_id")): x for x in votes}
    if len(by_vote) != len(votes) or set(by_vote) != {x["review_id"] for x in packet}:
        raise RuntimeError("Candidate semantic review coverage is incomplete or duplicated")
    packet_by_id = {x["review_id"]: x for x in packet}
    if any(x.get("category") not in CATEGORIES or not x.get("reason") or
           not set(x.get("evidence_ids", [])).issubset(packet_by_id[x["review_id"]]["visible_document_ids"])
           for x in votes):
        raise RuntimeError("Candidate semantic review has invalid category or evidence reference")
    for vote in votes:
        item = packet_by_id[vote["review_id"]]
        if vote["category"] == "correct_sufficient" and (item["reason"] != "answer" or
                                                           not vote.get("evidence_ids")):
            raise RuntimeError("A semantic success requires an answer and visible evidence citation")
    vote_by_id = by_vote
    case_votes = {(x["split"], x["model"], x["case_id"]): vote_by_id[x["review_id"]]
                  for x in mapping}
    results: dict[str, dict] = {}
    raw_by_key = {}
    for label in TRAINING_LABELS:
        for split in AG_SPLITS:
            raw = rows(OUT / f"eval-{split}-{label}.jsonl")
            expected = {"dev": 64, "old": 32, "y": 48, "aa_regression": 64}[split]
            if len(raw) != expected:
                raise RuntimeError(f"Incomplete product evaluation {split}/{label}: {len(raw)}/{expected}")
            cats = Counter(case_votes[split, label, x["case_id"]]["category"] for x in raw)
            cats["invalid"] = sum(x.get("reason") == "invalid_action" for x in raw)
            cats["em"] = sum(x.get("em", 0) for x in raw)
            cats["f1_sum"] = sum(x.get("f1", 0.) for x in raw)
            cats["searches_sum"] = sum(x.get("searches", 0) for x in raw)
            cats["latency_sum"] = sum(x.get("elapsed_seconds", 0.) for x in raw)
            cats["bad_total"] = cats["factual_error"] + cats["unsupported_by_visible_evidence"]
            if split == "old":
                groups = {x["case_id"]: x["split"] for x in q.dev_cases()}
                cats["main"] = sum(case_votes[split, label, x["case_id"]]["category"] == "correct_sufficient"
                                    for x in raw if groups[x["case_id"]] == "new_eval")
                cats["regression"] = sum(case_votes[split, label, x["case_id"]]["category"] == "correct_sufficient"
                                         for x in raw if groups[x["case_id"]] == "goal_G_regression")
            results[f"{split}-{label}"] = dict(cats)
            for row in raw:
                raw_by_key[split, label, row["case_id"]] = row
    for key, vote in case_votes.items():
        if key[1] in TRAINING_LABELS:
            raw_by_key[key]["semantic_category"] = vote["category"]
            raw_by_key[key]["semantic_reason"] = vote["reason"]
            raw_by_key[key]["evidence_ids"] = vote.get("evidence_ids", [])
    write(OUT / "candidate-scores.json", results)
    return results, {"cases": raw_by_key, "votes": case_votes}


def baseline_semantic_scores() -> dict[str, dict]:
    ledger = rows(FINAL_LEDGER_V2)
    by_case = {str(x["case_id"]): x for x in ledger if x.get("arm") == "A"}
    if len(by_case) != 208:
        raise RuntimeError(f"Corrected Goal AG start ledger has {len(by_case)}/208 rows")
    cases = {x["case_id"]: x["split"] for x in q.dev_cases()}
    # The already exposed AA regression-set EM is taken from its published
    # decision record. Do not open Goal Z confirmation questions/answers here:
    # this function runs before the development gate authorizes confirmation.
    all_case_files = (z.OUT / "selected-dev.jsonl", q.p.CASES,
                      Path(af.y.OUT / "dev-cases.jsonl"))
    official = {str(case.get("case_id", case.get("id"))):
                case.get("official_answer", case.get("answer"))
                for path in all_case_files for case in rows(path)
                if case.get("official_answer", case.get("answer")) is not None}
    result: dict[str, dict] = {}
    for split in AG_SPLITS:
        keys = [case_id for case_id, row in by_case.items()
                if {"goal_z_dev": "dev", "old_regression": "old",
                    "goal_y_regression": "y", "aa_regression": "aa_regression"}.get(row["split"]) == split]
        cats = Counter(by_case[x]["category"] for x in keys)
        raw = rows(af.OUT / "eval-A.jsonl")
        raw_rows = {str(x["case_id"]): x for x in raw}
        cats["invalid"] = sum(raw_rows[x].get("reason") == "invalid_action" for x in keys)
        cats["em"] = sum(q.score(raw_rows[x].get("answer"), official[x])["em"]
                          for x in keys if x in official)
        if split == "aa_regression":
            published = json.loads((aa.OUT / "product-decision.json").read_text(encoding="utf-8"))
            cats["em"] = int(published["counts"]["42-trajectory"]["em"])
        cats["bad_total"] = cats["factual_error"] + cats["unsupported_by_visible_evidence"]
        if split == "old":
            cats["main"] = sum(by_case[x]["category"] == "correct_sufficient" for x in keys
                               if cases.get(x) == "new_eval")
            cats["regression"] = sum(by_case[x]["category"] == "correct_sufficient" for x in keys
                                     if cases.get(x) == "goal_G_regression")
        result[split] = dict(cats)
    expected = {"dev": (34, 23, 0), "old": (21, 11, 0), "y": (34, 14, 0), "aa_regression": (27, 19, 0)}
    for split, triple in expected.items():
        got = result[split]
        if (got.get("correct_sufficient", 0), got["bad_total"], got["invalid"]) != triple:
            raise RuntimeError(f"Corrected Goal AG baseline changed for {split}: {got}")
    write(OUT / "baseline-semantic-scores.json", result)
    return result


def protected_aa_cases(baseline_vote_by_case: dict[str, dict]) -> set[str]:
    """Protect only same-rubric current successes with later-acquired evidence."""
    raw_path = aa.OUT / "confirmation-revised-results.jsonl"
    if not raw_path.is_file():
        return set()
    historical = rows(raw_path)
    later = {str(x["case_id"]) for x in historical if x.get("model") == "42-trajectory" and
             x.get("category") == "correct_sufficient" and x.get("second_side_first_seen_later")}
    current = {case_id for case_id, vote in baseline_vote_by_case.items()
               if vote["split"] == "aa_regression" and vote["category"] == "correct_sufficient"}
    protected = current & later
    write_once(OUT / "protection-set-ag.json", {"case_ids": sorted(protected),
        "current_same_rubric_successes": len(current), "historical_later_evidence_successes": len(later),
        "intersection": len(protected)})
    return protected


def later_support_title_cases(baseline_rows: list[dict], cases_by_id: dict[str, dict],
                              successful_case_ids: set[str]) -> set[str]:
    """Find same-rubric successes whose two support pages arrive in separate searches."""
    protected: set[str] = set()
    for row in baseline_rows:
        case_id = str(row["case_id"])
        if case_id not in successful_case_ids:
            continue
        titles = cases_by_id[case_id].get("support_titles", [])
        if len(titles) != 2:
            continue
        title_steps: dict[str, int] = {}
        for index, step in enumerate(row.get("trace", [])):
            for doc in step.get("observation", {}).get("result", []):
                title_steps.setdefault(str(doc.get("title")), index)
        if all(title in title_steps for title in titles) and \
           max(title_steps[titles[0]], title_steps[titles[1]]) > \
           min(title_steps[titles[0]], title_steps[titles[1]]):
            protected.add(case_id)
    return protected


def calculate_development_gate() -> dict:
    scores, details = candidate_scores()
    baseline = baseline_semantic_scores()
    baseline_votes = {str(x["case_id"]): x for x in rows(FINAL_LEDGER_V2)
                      if x.get("arm") == "A"}
    protected = protected_aa_cases(baseline_votes)
    gates = {}
    for arm in ("A", "B"):
        arm_gates = {}
        for seed in (42, 43):
            label = f"{seed}-{arm}"
            dev = scores[f"dev-{label}"]
            old, y, aa_reg = (scores[f"{split}-{label}"] for split in ("old", "y", "aa_regression"))
            lost = sum(details["cases"]["aa_regression", label, case_id]["semantic_category"] != "correct_sufficient"
                       for case_id in protected)
            start_dev = baseline["dev"]
            reasons = []
            if dev.get("correct_sufficient", 0) < start_dev["correct_sufficient"] + 4: reasons.append("dev_gain")
            if dev["bad_total"] > start_dev["bad_total"]: reasons.append("dev_bad_total")
            if dev["invalid"] > start_dev["invalid"]: reasons.append("dev_invalid")
            if old.get("main", 0) < baseline["old"].get("main", 0) - 1: reasons.append("old_main_drop")
            if old.get("regression", 0) < baseline["old"].get("regression", 0) - 1: reasons.append("old_regression_drop")
            if y.get("correct_sufficient", 0) < baseline["y"]["correct_sufficient"] - 1: reasons.append("goal_y_drop")
            if aa_reg.get("correct_sufficient", 0) < baseline["aa_regression"]["correct_sufficient"] - 1:
                reasons.append("aa_drop")
            if lost > 1: reasons.append("protected_multihop_lost")
            summary = json.loads((OUT / f"seed{seed}-{arm}/summary.json").read_text(encoding="utf-8"))
            if not summary.get("base_unchanged") or summary.get("reload_max_abs", 99) > .001:
                reasons.append("checkpoint_integrity")
            arm_gates[str(seed)] = {"passed": not reasons, "failures": reasons,
                "dev_success": dev.get("correct_sufficient", 0), "dev_start_success": start_dev["correct_sufficient"],
                "dev_bad": dev["bad_total"], "dev_bad_start": start_dev["bad_total"],
                "invalid": dev["invalid"], "old_main": old.get("main", 0),
                "old_regression": old.get("regression", 0), "goal_y": y["correct_sufficient"],
                "aa_regression": aa_reg["correct_sufficient"], "protected_lost": lost}
        gates[arm] = arm_gates
    eligible = [arm for arm in ("A", "B") if all(gates[arm][str(seed)]["passed"] for seed in (42, 43))]
    selected = None
    if eligible:
        def rank(arm: str):
            values = [scores[f"dev-{seed}-{arm}"] for seed in (42, 43)]
            return (min(x["correct_sufficient"] for x in values),
                    sum(x["correct_sufficient"] for x in values) / 2,
                    -sum(x["bad_total"] for x in values),
                    int(arm == "A"))
        selected = max(eligible, key=rank)
    mechanism = all(
        scores[f"dev-{seed}-B"]["correct_sufficient"] >= scores[f"dev-{seed}-A"]["correct_sufficient"] and
        scores[f"dev-{seed}-B"]["bad_total"] < scores[f"dev-{seed}-A"]["bad_total"]
        for seed in (42, 43))
    status = "confirmation_authorized" if selected else "complete_development_gate_failed"
    result = {"status": status, "route_gates": gates, "eligible_routes": eligible,
        "selected_route": selected, "mechanism_additional_value_supported": mechanism,
        "scores": scores, "baseline": baseline,
        "protection_set": json.loads((OUT / "protection-set-ag.json").read_text(encoding="utf-8"))}
    write(OUT / "development-decision.json", result)
    state(status, stage="AG4", development_decision=result)
    return result


def evaluate_confirmation() -> dict:
    """Open the AB confirmation set only after a passing development gate."""
    current = json.loads(STATE.read_text(encoding="utf-8"))
    decision = json.loads((OUT / "development-decision.json").read_text(encoding="utf-8"))
    if current.get("status") != "confirmation_authorized" or not decision.get("selected_route"):
        raise RuntimeError("New confirmation access is not authorized by the development gate")
    manifest = json.loads((ab.OUT / "fresh-confirm-manifest.json").read_text(encoding="utf-8"))
    case_path = ab.OUT / "fresh-confirm-cases.jsonl"
    corpus_path = ab.OUT / "fresh-confirm-corpus.json"
    if manifest.get("confirmation_accessed") or sha(case_path) != manifest.get("case_sha256") or \
       sha(corpus_path) != manifest.get("corpus_sha256"):
        raise RuntimeError("Frozen confirmation integrity/status check failed")
    route = decision["selected_route"]
    labels = ("baseline", f"42-{route}", f"43-{route}")
    output_paths = [OUT / f"eval-confirmation-{label}.jsonl" for label in labels]
    existing = [path.exists() for path in output_paths]
    if any(existing):
        if not all(existing) or any(len(rows(path)) != 64 for path in output_paths):
            raise RuntimeError("Partial confirmation outputs exist; refusing to rerun or overwrite")
        outputs = 192
        recovery = "reused_existing_complete_outputs"
    else:
        cases = rows(case_path)
        if len(cases) != 64:
            raise RuntimeError("Frozen confirmation set must contain 64 questions")
        outputs = 0
        for label, adapter in (("baseline", SOURCE), (f"42-{route}", OUT / f"seed42-{route}/adapter-final"),
                               (f"43-{route}", OUT / f"seed43-{route}/adapter-final")):
            evaluate_candidate(label, adapter, "confirmation", cases, corpus_path)
            outputs += 64
        recovery = "generated_after_development_gate"
    if outputs != 192:
        raise RuntimeError("Goal AG confirmation output limit exceeded")
    write_once(OUT / "confirmation-access-record.json", {
        "reason": "development decision authorized the single frozen confirmation evaluation",
        "selected_route": route, "outputs": outputs, "case_sha256": sha(case_path),
        "corpus_sha256": sha(corpus_path), "recovery": recovery})
    state("confirmation_authorized", stage="AG5", confirmation_accessed=True,
          confirmation_outputs=outputs)
    packet = build_review_packet(("baseline", f"42-{route}", f"43-{route}"), "confirmation")
    state("awaiting_confirmation_review", stage="AG5", confirmation_accessed=True,
          confirmation_outputs=outputs, confirmation_review=packet)
    return {"outputs": outputs, "route": route, "review_packet": packet,
        "confirmation_accessed": True, "case_sha256": sha(case_path), "corpus_sha256": sha(corpus_path)}


def calculate_confirmation_gate() -> dict:
    decision = json.loads((OUT / "development-decision.json").read_text(encoding="utf-8"))
    route = decision["selected_route"]
    packet = rows(OUT / "confirmation-review-packet.jsonl")
    mapping = rows(OUT / "confirmation-review-map.jsonl")
    if (OUT / "confirmation-review-final.jsonl").exists():
        votes = rows(OUT / "confirmation-review-final.jsonl")
    else:
        votes = rows(OUT / "confirmation-review-a.jsonl") + rows(OUT / "confirmation-review-b.jsonl")
    by_vote = {str(x["review_id"]): x for x in votes}
    if set(by_vote) != {x["review_id"] for x in packet}:
        raise RuntimeError("Confirmation semantic review is incomplete")
    by_case = {(x["split"], x["model"], x["case_id"]): by_vote[x["review_id"]]
               for x in mapping}
    counts = {}
    for label in ("baseline", f"42-{route}", f"43-{route}"):
        raw = rows(OUT / f"eval-confirmation-{label}.jsonl")
        c = Counter(by_case["confirmation", label, x["case_id"]]["category"] for x in raw)
        c["invalid"] = sum(x.get("reason") == "invalid_action" for x in raw)
        c["em"] = sum(x.get("em", 0) for x in raw)
        c["bad_total"] = c["factual_error"] + c["unsupported_by_visible_evidence"]
        counts[label] = dict(c)
    # Build the confirmation protection set only now that the development gate
    # has opened this set. It must use confirmation case IDs and trajectories,
    # not unrelated AA regression IDs used by the development gate.
    confirmation_cases = {str(x["id"]): x for x in rows(ab.OUT / "fresh-confirm-cases.jsonl")}
    baseline_rows = rows(OUT / "eval-confirmation-baseline.jsonl")
    successful_baseline_cases = {str(row["case_id"]) for row in baseline_rows
        if by_case["confirmation", "baseline", str(row["case_id"])]["category"] == "correct_sufficient"}
    protected = later_support_title_cases(baseline_rows, confirmation_cases, successful_baseline_cases)
    write(OUT / "confirmation-protection-set-ag.json", {
        "case_ids": sorted(protected), "count": len(protected),
        "rule": "baseline same-rubric successes whose two support titles first appear at different search steps"})
    outcomes = {}
    for seed in (42, 43):
        label = f"{seed}-{route}"
        lost = sum(by_case["confirmation", label, case_id]["category"] != "correct_sufficient"
                   for case_id in protected)
        start = counts["baseline"]; cand = counts[label]
        passed = (cand.get("correct_sufficient", 0) >= start.get("correct_sufficient", 0) + 4 and
            cand["bad_total"] <= start["bad_total"] and cand["invalid"] <= start["invalid"] and lost <= 1)
        outcomes[str(seed)] = {"passed": passed, "candidate": cand, "baseline": start,
                               "protected_lost": lost}
    passed = all(x["passed"] for x in outcomes.values())
    result = {"status": "complete_published" if passed else "complete_confirmation_failed",
        "selected_route": route, "seeds": outcomes, "scores": counts,
        "published_adapter": str(OUT / f"seed42-{route}/adapter-final") if passed else str(SOURCE),
        "published": passed}
    write(OUT / "confirmation-decision.json", result)
    state(result["status"], stage="AG5", confirmation_accessed=True,
          product_decision=result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        run()
    else:
        summarize()


if __name__ == "__main__":
    main()
