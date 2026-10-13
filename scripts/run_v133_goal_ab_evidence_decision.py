"""Goal AB: staged evidence-decision audit and matched trajectory SFT experiment."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import run_v133_goal_aa_audit_confirmation as aa
import run_v133_goal_z_multihop_sft as z

ROOT = z.ROOT
OUT = ROOT / "runs/v133-goal-ab-evidence-decision"
CFG, STATE = OUT / "config.json", OUT / "state.json"
SOURCE = z.OUT / "seed42-trajectory/adapter-final"
SPLITS = {"dev": (z.OUT / "eval-dev-42-trajectory.jsonl", 64),
          "old": (z.OUT / "eval-old-42-trajectory.jsonl", 32),
          "y": (z.OUT / "eval-y-42-trajectory.jsonl", 48),
          "aa_regression": (aa.OUT / "eval-confirm-42-trajectory.jsonl", 64)}
EVENTS = ("premature_answer", "unnecessary_search", "wrong_bridge_query",
          "sufficient_evidence_wrong_answer", "retrieval_failure", "truncation")
STATES = ("sufficient", "insufficient", "unclear")


def digest(path: Path) -> str:
    return z.sha(path)


def state(status: str, **updates) -> None:
    value = z.read(STATE)
    value.update(status=status, **updates)
    STATE.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def frozen_assets() -> dict[str, Path]:
    assets = {"source_adapter": SOURCE / "adapter_model.safetensors",
              "base_tokenizer": z.q.p.BASE / "tokenizer.json",
              "goal_z_config": z.CFG, "goal_aa_decision": aa.OUT / "product-decision.json",
              "goal_z_training_inputs": z.OUT / "training-inputs.jsonl",
              "goal_z_train_cases": z.OUT / "selected-train.jsonl",
              "goal_z_dev_cases": z.OUT / "selected-dev.jsonl",
              "goal_z_dev_corpus": z.OUT / "corpus-dev.json",
              "goal_z_train_corpus": z.OUT / "corpus-train.json",
              "goal_y_corpus": z.y.OUT / "dev-corpus.json",
              "old_corpus": z.q.p.CORPUS,
              "search_runner": Path(z.q.__file__),
              "goal_z_runner": Path(z.__file__),
              "model_code": ROOT / "src/mini_agent_rl/model/local.py",
              "search_code": ROOT / "src/mini_agent_rl/tools/search.py"}
    for split, (path, _) in SPLITS.items():
        assets[f"archived_{split}"] = path
    assets.update({f"base_weights_{i}": p for i, p in enumerate(sorted(z.q.p.BASE.glob("*.safetensors")))})
    return assets


def rubric() -> dict:
    return {"success": "The answer is correct, complete and supported by facts present in the actual prompt tokens at answer time.",
            "fact_error": "Requires an explicit wrong relation or contradiction; absent support alone does not establish factual error.",
            "unsupported": "The answer's necessary relation is missing from visible evidence, without explicit contradiction.",
            "indeterminate": "Counts as failure and is reported separately.",
            "bad_total": "factual_error + unsupported_by_visible_evidence",
            "sufficient": "Every entity, relation and required qualifier needed to answer is visible in this step's actual prompt.",
            "review": "Blind model identity; cite actual visible passages. Agent reviewers are diagnostic, not independent human gold."}


def archived() -> list[dict]:
    result = []
    for split, (path, count) in SPLITS.items():
        values = z.rows(path)
        if len(values) != count or len({x["case_id"] for x in values}) != count:
            raise RuntimeError(f"Archived split incomplete: {split}")
        result.extend({**x, "ab_split": split} for x in values)
    if len(result) != 208:
        raise RuntimeError("Expected 208 archived product trajectories")
    return result


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    decision = z.read(aa.OUT / "product-decision.json")
    if not decision.get("published") or decision.get("selected_seed") != 42 or \
       decision.get("selected_route") != "trajectory" or Path(decision["adapter"]) != SOURCE:
        raise RuntimeError("Published Goal AA adapter differs from Goal AB source")
    if z.read(aa.STATE)["status"] != "complete_confirmed_and_published":
        raise RuntimeError("Goal AA product state incomplete")
    assets = frozen_assets()
    if not any(k.startswith("base_weights_") for k in assets):
        raise RuntimeError("Actual base weights absent")
    if any(not p.is_file() for p in assets.values()):
        raise RuntimeError("Frozen asset absent")
    archived()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(z.q.p.BASE, local_files_only=True)
    order = {}
    ids = [x["id"] for x in z.rows(z.OUT / "selected-train.jsonl")]
    if len(ids) != 128 or len(set(ids)) != 128:
        raise RuntimeError("Goal Z training IDs incomplete")
    for seed in (42, 43):
        values = ids.copy()
        random.Random(seed).shuffle(values)
        order[str(seed)] = values
    OUT.mkdir(parents=True)
    z.write(CFG, {"version": "goal-ab-v1", "assets": {k: digest(p) for k, p in assets.items()},
                  "software": z.software(), "source": str(SOURCE),
                  "product": z.q.load_config()["product"], "rubric": rubric(),
                  "order": order, "seeds": [42, 43], "arms": ["A", "B"],
                  "max_updates_per_run": 128, "max_train_rollouts": 128,
                  "max_decision_pairs": 64, "min_decision_pairs": 32,
                  "min_train_errors": 16, "min_dev_stop_errors": 8,
                  "max_new_confirmation_outputs": 192,
                  "optimization": {"learning_rate": 5e-6, "betas": [.9, .999],
                                    "eps": 1e-8, "weight_decay": 0., "gradient_clip": 1.},
                  "start_time": time.time()})
    z.write(OUT / "evidence-rubric.json", rubric())
    for row in archived():
        steps = []
        for item in row["trace"]:
            ids = item.get("prompt_token_ids")
            if not ids or len(ids) > 2048:
                raise RuntimeError(f"Missing or overlong actual prompt: {row['case_id']}")
            actual = tok.decode(ids, skip_special_tokens=False)
            visible = [d for x in row["trace"] for d in x.get("observation", {}).get("result", [])
                       if d["id"] in actual]
            steps.append({"step": item["step"], "action": item.get("action"),
                          "query": item.get("query"), "answer": item.get("answer"),
                          "actual_prompt": actual, "prompt_sha256": hashlib.sha256(
                              json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                          "visible_document_ids": sorted({d["id"] for d in visible})})
        z.append(OUT / "diagnostic-packet.jsonl", {"split": row["ab_split"],
                 "case_id": row["case_id"], "question": row["question"],
                 "answer": row.get("answer"), "reason": row["reason"],
                 "steps": steps})
    z.write(STATE, {"status": "awaiting_diagnostic_review", "stages": [
            {"name": "AB0_prepared", "seconds": time.time() - z.read(CFG)["start_time"]}],
            "confirmation_accessed": False, "gpu_seconds": 0.})


def verify() -> dict:
    cfg = z.read(CFG)
    if cfg["assets"] != {k: digest(p) for k, p in frozen_assets().items()}:
        raise RuntimeError("Frozen asset hash changed")
    if cfg["software"] != z.software():
        raise RuntimeError("GPU/software changed")
    return cfg


def diagnostic_gate() -> dict:
    cfg = verify()
    packet = z.rows(OUT / "diagnostic-packet.jsonl")
    reviews = z.rows(OUT / "diagnostic-review.jsonl")
    keys = {(x["split"], x["case_id"]) for x in packet}
    mapped = {(x["split"], x["case_id"]): x for x in reviews}
    if len(packet) != 208 or len(keys) != 208 or len(reviews) != len(mapped) or set(mapped) != keys:
        raise RuntimeError("Diagnostic review incomplete")
    for item in packet:
        review = mapped[item["split"], item["case_id"]]
        if len(review.get("steps", [])) != len(item["steps"]):
            raise RuntimeError("Every actual decision point requires review")
        for step, vote in zip(item["steps"], review["steps"]):
            if vote.get("evidence_state") not in STATES or \
               not set(vote.get("evidence_ids", [])).issubset(step["visible_document_ids"]) or \
               not vote.get("reason"):
                raise RuntimeError("Invalid step evidence review")
        if not set(review.get("events", [])).issubset(EVENTS) or \
           not review.get("primary_failure") or not review.get("reason"):
            raise RuntimeError("Invalid case review")
    count = sum(bool({"premature_answer", "unnecessary_search"} & set(mapped["dev", x["case_id"]]["events"]))
                for x in packet if x["split"] == "dev")
    by_event = {split: dict(Counter(e for x in reviews if x["split"] == split for e in x["events"]))
                for split in SPLITS}
    result = {"dev_stop_error_cases": count, "threshold": cfg["min_dev_stop_errors"],
              "passed": count >= cfg["min_dev_stop_errors"], "events": by_event,
              "review_sha256": digest(OUT / "diagnostic-review.jsonl")}
    z.write(OUT / "diagnostic-decision.json", result)
    state("awaiting_train_material" if result["passed"] else "complete_diagnostic_gate_failed",
          diagnostic_gate=result)
    return result


def collect_train() -> None:
    """One frozen greedy rollout per original training question."""
    import torch
    cfg = verify()
    if z.read(STATE)["status"] != "collecting_train":
        raise RuntimeError("Diagnostic gate not passed")
    from prepare_v133_goal_ab_confirmation import prepare as freeze_confirmation
    manifest_path = OUT / "fresh-confirm-manifest.json"
    if manifest_path.exists():
        manifest = z.read(manifest_path)
        if manifest["counts"] != {"musique": 32, "2wiki": 32} or \
           digest(OUT / "fresh-confirm-cases.jsonl") != manifest["case_sha256"] or \
           digest(OUT / "fresh-confirm-corpus.json") != manifest["corpus_sha256"] or \
           manifest["confirmation_accessed"]:
            raise RuntimeError("Frozen confirmation material mismatch")
    else:
        freeze_confirmation()
    output = OUT / "train-on-policy.jsonl"
    if output.exists():
        raise FileExistsError(output)
    client, wrapped, registry = z.make_client(SOURCE, z.OUT / "corpus-train.json")
    cases = z.rows(z.OUT / "selected-train.jsonl")
    for case in cases:
        result = asyncio.run(z.q.rollout(case["question"], wrapped, registry, sample=False, seed=42))
        result.update(case_id=case["id"], source=case["source"],
                      **z.q.score(result["answer"], case["answer"]))
        z.append(output, result)
        if len(z.rows(output)) % 8 == 0:
            print(json.dumps({"train_rollouts": len(z.rows(output))}), flush=True)
    if len(z.rows(output)) != cfg["max_train_rollouts"]:
        raise RuntimeError("Incomplete training rollouts")
    del client, wrapped, registry
    gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(z.q.p.BASE, local_files_only=True)
    for case, result in zip(cases, z.rows(output)):
        if case["id"] != result["case_id"]:
            raise RuntimeError("Training rollout order mismatch")
        steps = []
        for item in result["trace"]:
            prompt = tok.decode(item["prompt_token_ids"], skip_special_tokens=False)
            visible = {d["id"] for earlier in result["trace"]
                       for d in earlier.get("observation", {}).get("result", []) if d["id"] in prompt}
            steps.append({"step": item["step"], "action": item.get("action"),
                          "query": item.get("query"), "answer": item.get("answer"),
                          "actual_prompt": prompt, "visible_document_ids": sorted(visible),
                          "prompt_token_sha256": hashlib.sha256(json.dumps(
                              item["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest()})
        z.append(OUT / "train-decision-packet.jsonl", {
            "case_id": case["id"], "question": case["question"], "official_answer": case["answer"],
            "on_policy_reason": result["reason"], "steps": steps})
    state("awaiting_pair_review", train_rollout_sha256=digest(output),
          fresh_confirmation_sha256=digest(OUT / "fresh-confirm-manifest.json"))
    propose_pairs()


def validate_pairs() -> dict:
    """Validate reviewed decision pairs without loading model or confirmation data."""
    cfg = verify()
    path = OUT / "decision-pairs.jsonl"
    if not path.exists():
        raise RuntimeError("Reviewed decision pairs absent")
    pairs = z.rows(path)
    if not (cfg["min_decision_pairs"] <= len(pairs) <= cfg["max_decision_pairs"]):
        raise RuntimeError("Decision pair count outside frozen bounds")
    original = {x["id"]: x for x in z.rows(z.OUT / "training-inputs.jsonl")}
    policy = {x["case_id"]: x for x in z.rows(OUT / "train-on-policy.jsonl")}
    proposals = {x["case_id"]: x for x in z.rows(OUT / "pair-proposals.jsonl")}
    if len(policy) != 128 or len(original) != 128:
        raise RuntimeError("Training materials incomplete")
    seen = set()
    for row in pairs:
        key = row.get("case_id")
        if key in seen or key not in original or key not in policy or key not in proposals:
            raise RuntimeError("Duplicate or non-training decision pair")
        seen.add(key)
        for label, action in (("insufficient", "search"), ("sufficient", "answer")):
            step = row[label]
            reference = proposals[key]["teacher_steps"][label]
            if any(step.get(field) != reference[field] for field in
                   ("prompt_token_ids", "prompt_attention_mask", "action_token_ids", "action")):
                raise RuntimeError("Decision tokens differ from frozen real trajectory")
            if step.get("action") != action or not step.get("prompt_token_ids") or \
               not step.get("action_token_ids") or \
               len(step["prompt_token_ids"]) > cfg["product"]["max_input_tokens"] or \
               len(step["action_token_ids"]) > cfg["product"]["max_new_tokens"] or \
               len(step["prompt_token_ids"]) != len(step["prompt_attention_mask"]):
                raise RuntimeError("Invalid actual decision tokens")
            if not step.get("evidence_reason") or \
               not set(step.get("evidence_ids", [])).issubset(step.get("visible_document_ids", [])):
                raise RuntimeError("Decision evidence audit incomplete")
        if not row.get("actual_search_results_verified") or \
           row.get("product_model_calls") != proposals[key]["teacher_model_calls"] or \
           row.get("product_searches") != proposals[key]["teacher_searches"] or \
           row.get("product_model_calls", 5) > 4 or row.get("product_searches", 4) > 3 or \
           not row.get("review_reason"):
            raise RuntimeError("Decision path outside real product boundary")
    train_reviews = z.rows(OUT / "train-decision-review.jsonl")
    reviewed_ids = {x["case_id"] for x in train_reviews}
    if len(train_reviews) != 128 or reviewed_ids != set(policy) or \
       any(not isinstance(x.get("on_policy_stop_error"), bool) or not x.get("reason")
           for x in train_reviews):
        raise RuntimeError("All 128 on-policy stopping decisions require review")
    error_count = sum(x["on_policy_stop_error"] for x in train_reviews)
    if error_count < cfg["min_train_errors"]:
        raise RuntimeError(f"On-policy stopping errors {error_count} < {cfg['min_train_errors']}")
    result = {"pairs": len(pairs), "on_policy_stop_error_cases": error_count,
              "pair_sha256": digest(path), "passed": True}
    z.write(OUT / "pair-decision.json", result)
    z.write(OUT / "execution-code.json", {"runner": digest(Path(__file__)),
        "confirmation_prep": digest(ROOT / "scripts/prepare_v133_goal_ab_confirmation.py"),
        "search_runner": digest(Path(z.q.__file__)),
        "goal_z_runner": digest(Path(z.__file__))})
    state("training", pair_gate=result)
    return result


def propose_pairs() -> None:
    """Replay the approved Goal Z tool trajectory as a bounded correction path."""
    if z.read(STATE)["status"] != "awaiting_pair_review":
        raise RuntimeError("On-policy material not ready")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(z.q.p.BASE, local_files_only=True)
    cases = {x["id"]: x for x in z.rows(z.OUT / "selected-train.jsonl")}
    originals = {x["id"]: x for x in z.rows(z.OUT / "training-inputs.jsonl")}
    policy = {x["case_id"]: x for x in z.rows(OUT / "train-on-policy.jsonl")}
    if len(cases) != 128 or len(originals) != 128 or len(policy) != 128:
        raise RuntimeError("Frozen training material incomplete")
    eligible = [key for key, case in cases.items() if case["source"] in ("musique", "2wiki")
                and len(originals[key]["actions"]) >= 3]
    eligible.sort(key=lambda key: hashlib.sha256(("goal-ab-v1\0" + key).encode()).hexdigest())
    proposals, rejections = [], Counter()
    for key in eligible:
        if len(proposals) == 64:
            break
        teacher = originals[key]["actions"]
        case = cases[key]
        if teacher[-2]["action"] != "search" or teacher[-1]["action"] != "answer":
            raise RuntimeError("Frozen teacher lacks two decisions")
        support = set(case["teacher"]["support_titles"])
        if len(support) != 2 or len(teacher) > 4:
            raise RuntimeError("Teacher path outside product boundary")
        steps = {}
        rejected = None
        for name, action in (("insufficient", teacher[-2]), ("sufficient", teacher[-1])):
            prompt = tok.decode(action["prompt_token_ids"], skip_special_tokens=False)
            visible = [d for step in teacher for d in step.get("observation", {}).get("result", [])
                       if d["id"] in prompt]
            observed = {d["title"] for d in visible}
            if name == "insufficient" and support.issubset(observed):
                rejected = "insufficient_prompt_already_has_both_supports"
                break
            if name == "sufficient" and not support.issubset(observed):
                rejected = "answer_prompt_missing_support_side"
                break
            steps[name] = {"action": action["action"], "query": action.get("query"),
                           "answer": action.get("answer"),
                           "prompt_token_ids": action["prompt_token_ids"],
                           "prompt_attention_mask": action["prompt_attention_mask"],
                           "action_token_ids": action["action_token_ids"],
                           "actual_prompt": prompt,
                           "visible_document_ids": sorted({d["id"] for d in visible}),
                           "support_titles_visible": sorted(support & observed)}
        if rejected:
            rejections[rejected] += 1
            continue
        current = policy[key]
        policy_steps = [{"action": item.get("action"), "query": item.get("query"),
                         "answer": item.get("answer"), "actual_prompt": tok.decode(
                             item["prompt_token_ids"], skip_special_tokens=False)}
                        for item in current["trace"]]
        proposals.append({"case_id": key, "source": case["source"],
            "question": case["question"], "answer": case["answer"],
            "teacher_support_titles": sorted(support), "teacher_steps": steps,
            "on_policy_reason": current["reason"], "on_policy_steps": policy_steps,
            "teacher_model_calls": len(teacher),
            "teacher_searches": sum(x["action"] == "search" for x in teacher)})
    z.write(OUT / "pair-proposal-audit.json", {"eligible": len(eligible),
        "qualified": len(proposals), "rejections": dict(rejections)})
    if len(proposals) < 32:
        state("complete_pair_gate_failed", pair_shortfall={"qualified": len(proposals),
              "required": 32, "rejections": dict(rejections)})
        return
    for row in proposals:
        z.append(OUT / "pair-proposals.jsonl", row)


def finalize_pair_review() -> dict:
    proposals = z.rows(OUT / "pair-proposals.jsonl")
    training_cases = {x["id"]: x for x in z.rows(z.OUT / "selected-train.jsonl")}
    votes = z.rows(OUT / "pair-review-a.jsonl") + z.rows(OUT / "pair-review-b.jsonl")
    mapping = {x["case_id"]: x for x in votes}
    override_path = OUT / "pair-main-overrides.jsonl"
    overrides = z.rows(override_path) if override_path.exists() else []
    if len({x["case_id"] for x in overrides}) != len(overrides) or \
       not {x["case_id"] for x in overrides}.issubset(mapping):
        raise RuntimeError("Invalid main pair overrides")
    for item in overrides:
        prior = mapping[item["case_id"]]
        if prior["approved"] != item["old_approved"] or not item.get("reason"):
            raise RuntimeError("Main pair override lacks traceable prior decision")
        mapping[item["case_id"]] = {**prior, "approved": item["approved"],
            "reason": item["reason"],
            "insufficient_evidence_ids": item.get("insufficient_evidence_ids",
                                                   prior["insufficient_evidence_ids"]),
            "sufficient_evidence_ids": item.get("sufficient_evidence_ids",
                                                 prior["sufficient_evidence_ids"])}
    if len(proposals) != 64 or len(votes) != len(mapping) or \
       set(mapping) != {x["case_id"] for x in proposals}:
        raise RuntimeError("Pair evidence review incomplete")
    train_review_path = OUT / "train-decision-review.jsonl"
    if not train_review_path.exists():
        raise RuntimeError("Full 128-case on-policy review absent")
    train_reviews = z.rows(train_review_path)
    reviewed = {x["case_id"]: x for x in train_reviews}
    policy_ids = {x["case_id"] for x in z.rows(OUT / "train-on-policy.jsonl")}
    if len(train_reviews) != len(reviewed) or len(reviewed) != 128 or \
       set(reviewed) != policy_ids or \
       any(not isinstance(x.get("on_policy_stop_error"), bool) or not x.get("reason")
           for x in train_reviews):
        raise RuntimeError("Full 128-case on-policy review incomplete")
    for key, vote in mapping.items():
        if vote["on_policy_stop_error"] != reviewed[key]["on_policy_stop_error"]:
            raise RuntimeError("Pair and full training stop audits disagree")
    approved, rejected = [], Counter()
    for proposal in proposals:
        vote = mapping[proposal["case_id"]]
        if not vote.get("reason") or not isinstance(vote.get("on_policy_stop_error"), bool):
            raise RuntimeError("Pair vote lacks reason or error decision")
        if not vote["approved"]:
            rejected[vote["reason"]] += 1
            continue
        steps = {}
        case = training_cases[proposal["case_id"]]
        support_ids = {doc["title"]: doc["id"] for batch in case["teacher"]["retrieved"]
                       for doc in batch if doc["title"] in case["teacher"]["support_titles"]}
        if set(support_ids) != set(case["teacher"]["support_titles"]):
            raise RuntimeError("Teacher support mapping incomplete")
        for name in ("insufficient", "sufficient"):
            original = proposal["teacher_steps"][name]
            ids = vote[f"{name}_evidence_ids"]
            if not isinstance(ids, list) or not ids or \
               not set(ids).issubset(original["visible_document_ids"]):
                raise RuntimeError("Approved pair lacks visible evidence citation")
            if name == "sufficient":
                necessary = set(support_ids.values())
                if not necessary.issubset(original["visible_document_ids"]):
                    raise RuntimeError("Both teacher support passages not actually visible")
                ids = sorted(set(ids) | necessary)
            steps[name] = {k: original[k] for k in ("action", "query", "answer",
                "prompt_token_ids", "prompt_attention_mask", "action_token_ids",
                "visible_document_ids")}
            steps[name].update(evidence_ids=ids, evidence_reason=vote["reason"])
        approved.append({"case_id": proposal["case_id"], **steps,
            "on_policy_stop_error": vote["on_policy_stop_error"],
            "actual_search_results_verified": True,
            "product_model_calls": proposal["teacher_model_calls"],
            "product_searches": proposal["teacher_searches"],
            "review_reason": vote["reason"], "on_policy_error_reason": vote.get("error_reason")})
    report = {"proposed": len(proposals), "approved": len(approved),
        "on_policy_stop_error_cases": sum(x["on_policy_stop_error"] for x in train_reviews),
        "approved_pair_stop_error_cases": sum(x["on_policy_stop_error"] for x in approved),
        "rejected": dict(rejected), "main_overrides": len(overrides),
        "min_pairs": 32, "min_errors": 16}
    report["passed"] = report["approved"] >= 32 and report["on_policy_stop_error_cases"] >= 16
    z.write(OUT / "pair-review-summary.json", report)
    for row in approved:
        z.append(OUT / "reviewed-decision-pairs.jsonl", row)
    if report["passed"]:
        for row in approved:
            z.append(OUT / "decision-pairs.jsonl", row)
    else:
        state("complete_pair_gate_failed", pair_shortfall=report)
    return report


def train(seed: int, arm: str) -> None:
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as o
    cfg = verify()
    if seed not in (42, 43) or arm not in ("A", "B"):
        raise ValueError((seed, arm))
    inputs = {x["id"]: x for x in z.rows(z.OUT / "training-inputs.jsonl")}
    pairs = {x["case_id"]: x for x in z.rows(OUT / "decision-pairs.jsonl")}
    folder = OUT / f"seed{seed}-{arm}"
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir()
    set_seed(seed)
    client, _, _ = z.make_client(SOURCE, z.OUT / "corpus-train.json")
    params = [p for _, p in client.prepare_for_training()]
    base_before = o.parameter_hashes(client._model, lora=False)
    lora_before = o.parameter_hashes(client._model, lora=True)
    opt = cfg["optimization"]
    optimizer = torch.optim.AdamW(params, lr=opt["learning_rate"], betas=tuple(opt["betas"]),
                                  eps=opt["eps"], weight_decay=opt["weight_decay"])
    last = None
    try:
        for index, case_id in enumerate(cfg["order"][str(seed)], 1):
            if index > cfg["max_updates_per_run"]:
                raise RuntimeError("Update bound exceeded")
            actions = inputs[case_id]["actions"]
            original_tokens = sum(len(a["action_token_ids"]) for a in actions)
            optimizer.zero_grad(set_to_none=True)
            client.set_training_forward_mode()
            scalar = 0.
            original_weight = .5 if arm == "B" and case_id in pairs else 1.
            for action in actions:
                values = client.score_action_tensor([], action["action_token_ids"], True,
                    action["prompt_token_ids"], action["prompt_attention_mask"]).float()
                loss = -original_weight * values.sum() / original_tokens
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite original loss")
                loss.backward()
                scalar += float(loss.detach())
            decision_tokens = 0
            if original_weight == .5:
                for label in ("insufficient", "sufficient"):
                    action = pairs[case_id][label]
                    values = client.score_action_tensor([], action["action_token_ids"], True,
                        action["prompt_token_ids"], action["prompt_attention_mask"]).float()
                    loss = -.25 * values.mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite decision loss")
                    loss.backward()
                    scalar += float(loss.detach())
                    decision_tokens += len(action["action_token_ids"])
            grad = float(torch.nn.utils.clip_grad_norm_(params, opt["gradient_clip"]))
            if not math.isfinite(grad) or grad <= 0:
                raise RuntimeError("Invalid gradient")
            guarded_optimizer_step(optimizer, params, client.gpu_memory_stats(), 21.)
            z.append(folder / "updates.jsonl", {"step": index, "case_id": case_id,
                "loss": scalar, "gradient": grad, "original_tokens": original_tokens,
                "decision_tokens": decision_tokens, "gpu_memory": client.gpu_memory_stats()})
            last = actions[-1]
            if index % 8 == 0:
                print(json.dumps({"seed": seed, "arm": arm, "updates": index}), flush=True)
        if o.parameter_hashes(client._model, lora=False) != base_before:
            raise RuntimeError("Base weights changed")
        changed = sum(o.parameter_hashes(client._model, lora=True)[k] != v for k, v in lora_before.items())
        if not changed:
            raise RuntimeError("LoRA unchanged")
        adapter = folder / "adapter-final"
        client._model.save_pretrained(adapter)
        expected = client.score_action_tensor([], last["action_token_ids"], False,
            last["prompt_token_ids"], last["prompt_attention_mask"]).detach().float().cpu()
        del client, optimizer, params
        gc.collect(); torch.cuda.empty_cache()
        loaded, _, _ = z.make_client(adapter, z.OUT / "corpus-train.json")
        actual = loaded.score_action_tensor([], last["action_token_ids"], False,
            last["prompt_token_ids"], last["prompt_attention_mask"]).detach().float().cpu()
        difference = float((expected - actual).abs().max())
        if difference > .001:
            raise RuntimeError(f"Independent reload mismatch: {difference}")
        z.write(folder / "summary.json", {"updates": 128, "base_unchanged": True,
            "changed_lora_tensors": changed, "reload_max_abs": difference,
            "adapter_sha256": digest(adapter / "adapter_model.safetensors")})
        del loaded
        gc.collect(); torch.cuda.empty_cache()
    except BaseException as exc:
        z.write(folder / "failure.json", {"error": str(exc),
            "completed_updates": len(z.rows(folder / "updates.jsonl"))
            if (folder / "updates.jsonl").exists() else 0})
        raise


def evaluation_specs() -> list[tuple[str, list[dict], Path]]:
    return [("dev", z.rows(z.OUT / "selected-dev.jsonl"), z.OUT / "corpus-dev.json"),
            ("old", z.q.dev_cases(), z.q.p.CORPUS),
            ("y", z.y.dev_cases(), z.y.OUT / "dev-corpus.json"),
            ("aa_regression", z.rows(z.OUT / "selected-confirm.jsonl"),
             z.OUT / "corpus-confirm.json")]


def evaluate(label: str, adapter: Path, split: str, cases: list[dict], corpus: Path) -> None:
    import torch
    path = OUT / f"eval-{split}-{label}.jsonl"
    if path.exists():
        raise FileExistsError(path)
    client, wrapped, registry = z.make_client(adapter, corpus)
    for case in cases:
        question = case["question"]
        answer = case.get("answer", case.get("official_answer"))
        if answer is None:
            raise RuntimeError("Evaluation answer absent")
        result = asyncio.run(z.q.rollout(question, wrapped, registry, sample=False, seed=42))
        result.update(case_id=case.get("id", case.get("case_id")), split=split,
                      model=label, official_answer=answer,
                      **z.q.score(result["answer"], answer))
        z.append(path, result)
        if len(z.rows(path)) % 8 == 0:
            print(json.dumps({"eval": split, "model": label,
                              "completed": len(z.rows(path))}), flush=True)
    del client, wrapped, registry
    gc.collect(); torch.cuda.empty_cache()


def copy_archived_baseline() -> None:
    for split, (path, count) in SPLITS.items():
        target = OUT / f"eval-{split}-baseline.jsonl"
        if target.exists():
            raise FileExistsError(target)
        values = z.rows(path)
        if len(values) != count:
            raise RuntimeError("Archived baseline incomplete")
        for row in values:
            z.append(target, {**row, "split": split, "model": "baseline"})


def review_packet(stage: str, splits: tuple[str, ...], labels: tuple[str, ...]) -> dict:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(z.q.p.BASE, local_files_only=True)
    packet, mapping = {}, []
    for split in splits:
        for label in labels:
            path = OUT / f"eval-{split}-{label}.jsonl"
            for row in z.rows(path):
                final = row["trace"][-1]
                actual = tokenizer.decode(final["prompt_token_ids"], skip_special_tokens=False)
                if row["question"].strip().casefold() not in actual.casefold():
                    raise RuntimeError("Question absent from actual final prompt")
                visible = sorted({doc["id"] for item in row["trace"]
                                  for doc in item.get("observation", {}).get("result", [])
                                  if doc["id"] in actual})
                steps = [{"action": item.get("action"), "query": item.get("query"),
                          "answer": item.get("answer"), "actual_prompt": tokenizer.decode(
                              item["prompt_token_ids"], skip_special_tokens=False)}
                         for item in row["trace"]]
                key = hashlib.sha256(json.dumps([row["question"], steps,
                    row.get("answer"), row["reason"]], ensure_ascii=False,
                    separators=(",", ":")).encode()).hexdigest()
                packet[key] = {"review_id": key, "question": row["question"],
                    "answer": row.get("answer"), "reason": row["reason"],
                    "actual_prompt": actual, "visible_document_ids": visible,
                    "steps": steps,
                    "prompt_token_sha256": hashlib.sha256(json.dumps(
                        final["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest()}
                mapping.append({"split": split, "case_id": row["case_id"],
                                "model": label, "review_id": key})
    if not packet or len(mapping) != len({(x["split"], x["case_id"], x["model"]) for x in mapping}):
        raise RuntimeError("Review mapping incomplete")
    for row in sorted(packet.values(), key=lambda x: x["review_id"]):
        z.append(OUT / f"{stage}-review-packet.jsonl", row)
    z.write(OUT / f"{stage}-review-map.json", mapping)
    return {"outputs": len(mapping), "unique": len(packet)}


def train_and_evaluate() -> None:
    cfg = verify()
    if z.read(OUT / "execution-code.json") != {"runner": digest(Path(__file__)),
        "confirmation_prep": digest(ROOT / "scripts/prepare_v133_goal_ab_confirmation.py"),
        "search_runner": digest(Path(z.q.__file__)),
        "goal_z_runner": digest(Path(z.__file__))}:
        raise RuntimeError("Execution code changed after pair gate")
    if z.read(STATE)["status"] != "training":
        raise RuntimeError("Pair gate not passed")
    if z.read(OUT / "pair-decision.json")["pair_sha256"] != digest(OUT / "decision-pairs.jsonl"):
        raise RuntimeError("Decision pairs changed")
    start = time.monotonic()
    for seed in (42, 43):
        for arm in ("A", "B"):
            train(seed, arm)
            value = z.read(STATE)
            value["stages"].append({"name": f"train-{seed}-{arm}",
                "updates": 128, "elapsed_from_stage_start": time.monotonic() - start})
            STATE.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    copy_archived_baseline()
    for seed in (42, 43):
        for arm in ("A", "B"):
            label = f"{seed}-{arm}"
            adapter = OUT / f"seed{label}/adapter-final"
            summary = z.read(OUT / f"seed{label}/summary.json")
            if summary["updates"] != cfg["max_updates_per_run"] or \
               digest(adapter / "adapter_model.safetensors") != summary["adapter_sha256"]:
                raise RuntimeError("Candidate checkpoint incomplete")
            for split, cases, corpus in evaluation_specs():
                evaluate(label, adapter, split, cases, corpus)
    info = review_packet("development", tuple(SPLITS),
        ("baseline", "42-A", "42-B", "43-A", "43-B"))
    state("awaiting_development_review", development_review=info,
          training_and_evaluation_seconds=time.monotonic() - start)


def reviewed(stage: str) -> dict[str, dict]:
    packet = z.rows(OUT / f"{stage}-review-packet.jsonl")
    path = OUT / f"{stage}-review-final.jsonl"
    if not path.exists():
        raise RuntimeError(f"{stage} semantic review absent")
    votes = z.rows(path)
    ids = {x["review_id"] for x in packet}
    found = {x["review_id"]: x for x in votes}
    if len(ids) != len(packet) or len(found) != len(votes) or set(found) != ids:
        raise RuntimeError(f"{stage} semantic review incomplete")
    for item in packet:
        vote = found[item["review_id"]]
        if vote.get("category") not in aa.CATEGORIES or \
           not set(vote.get("evidence_ids", [])).issubset(item["visible_document_ids"]) or \
           not set(vote.get("stop_events", [])).issubset(EVENTS) or \
           not vote.get("reason"):
            raise RuntimeError("Invalid frozen semantic review")
        if vote["category"] == "correct_sufficient" and \
           (item["reason"] != "answer" or not vote["evidence_ids"]):
            raise RuntimeError("Success without a cited visible answer")
    return found


def scored(stage: str) -> tuple[dict, dict]:
    votes = reviewed(stage)
    mapping = z.read(OUT / f"{stage}-review-map.json")
    by_case = {(x["split"], x["model"], x["case_id"]): votes[x["review_id"]]
               for x in mapping}
    if len(by_case) != len(mapping):
        raise RuntimeError("Duplicate scored case")
    totals = {}
    splits = ("fresh_confirm",) if stage == "confirmation" else tuple(SPLITS)
    labels = ("baseline", "42-A", "43-A", "42-B", "43-B") if stage == "development" \
        else ("baseline", z.read(OUT / "development-decision.json")["selected_route"] and
              f"42-{z.read(OUT / 'development-decision.json')['selected_route']}",
              f"43-{z.read(OUT / 'development-decision.json')['selected_route']}")
    old_groups = {x["case_id"]: x["split"] for x in z.q.dev_cases()}
    for split in splits:
        for label in labels:
            path = OUT / f"eval-{split}-{label}.jsonl"
            raw = z.rows(path)
            expected = 64 if split in ("dev", "aa_regression", "fresh_confirm") else 32 if split == "old" else 48
            if len(raw) != expected:
                raise RuntimeError(f"Incomplete scored split {split}-{label}")
            cats = Counter(by_case[split, label, x["case_id"]]["category"] for x in raw)
            cats["invalid"] = sum(x["reason"] == "invalid_action" for x in raw)
            cats["em"] = sum(x["em"] for x in raw)
            cats["f1_sum"] = sum(x["f1"] for x in raw)
            cats["searches_sum"] = sum(x["searches"] for x in raw)
            cats["latency_sum"] = sum(x["elapsed_seconds"] for x in raw)
            cats["bad_total"] = cats["factual_error"] + cats["unsupported_by_visible_evidence"]
            cats["stop_errors"] = sum(bool({"premature_answer", "unnecessary_search"} &
                set(by_case[split, label, x["case_id"]]["stop_events"])) for x in raw)
            if split == "old":
                for group in ("new_eval", "goal_G_regression"):
                    cats["main" if group == "new_eval" else "regression"] = sum(
                        by_case[split, label, x["case_id"]]["category"] == "correct_sufficient"
                        for x in raw if old_groups[x["case_id"]] == group)
            totals[f"{split}-{label}"] = dict(cats)
    return totals, by_case


def development_decision() -> dict:
    totals, by_case = scored("development")
    base = totals["dev-baseline"]
    protected = {x["case_id"] for x in z.rows(aa.OUT / "confirmation-revised-results.jsonl")
                 if x["model"] == "42-trajectory" and x["category"] == "correct_sufficient"
                 and x["second_side_first_seen_later"]}
    gates = {}
    for arm in ("A", "B"):
        gates[arm] = []
        for seed in (42, 43):
            label = f"{seed}-{arm}"
            dev = totals[f"dev-{label}"]
            old, y, aar = (totals[f"{split}-{label}"] for split in
                           ("old", "y", "aa_regression"))
            lost = sum(by_case["aa_regression", label, key]["category"] != "correct_sufficient"
                       for key in protected)
            passed = (dev.get("correct_sufficient", 0) >= base.get("correct_sufficient", 0) + 4
                and dev["bad_total"] <= base["bad_total"]
                and dev["invalid"] <= base["invalid"]
                and all(candidate >= baseline - 1 for candidate, baseline in (
                    (old["main"], totals["old-baseline"]["main"]),
                    (old["regression"], totals["old-baseline"]["regression"]),
                    (y.get("correct_sufficient", 0), totals["y-baseline"].get("correct_sufficient", 0)),
                    (aar.get("correct_sufficient", 0), totals["aa_regression-baseline"].get("correct_sufficient", 0))))
                and lost <= 1 and z.read(OUT / f"seed{label}/summary.json")["base_unchanged"])
            gates[arm].append({"seed": seed, "passed": passed,
                "dev_success": dev.get("correct_sufficient", 0), "dev_bad": dev["bad_total"],
                "old_main": old["main"], "old_regression": old["regression"],
                "goal_y": y.get("correct_sufficient", 0),
                "aa_regression": aar.get("correct_sufficient", 0),
                "protected_multihop_losses": lost})
    eligible = [arm for arm in gates if all(x["passed"] for x in gates[arm])]
    eligible.sort(key=lambda arm: (-min(x["dev_success"] for x in gates[arm]),
        -sum(x["dev_success"] for x in gates[arm]),
        sum(x["dev_bad"] for x in gates[arm]), 0 if arm == "A" else 1))
    mechanism = all(totals[f"dev-{seed}-B"]["stop_errors"] < totals[f"dev-{seed}-A"]["stop_errors"]
                    and totals[f"dev-{seed}-B"].get("correct_sufficient", 0) >=
                    totals[f"dev-{seed}-A"].get("correct_sufficient", 0) for seed in (42, 43))
    result = {"selected_route": eligible[0] if eligible else None,
        "eligible_routes": eligible, "gates": gates,
        "mechanism_supported": mechanism, "counts": totals,
        "status": "confirmation_authorized" if eligible else "complete_development_gate_failed"}
    z.write(OUT / "development-decision.json", result)
    state(result["status"], selected_route=result["selected_route"])
    return result


def evaluate_confirmation() -> None:
    if z.read(STATE)["status"] != "confirmation_authorized":
        raise RuntimeError("Fresh confirmation is still sealed")
    manifest = z.read(OUT / "fresh-confirm-manifest.json")
    cases_path = OUT / "fresh-confirm-cases.jsonl"
    corpus_path = OUT / "fresh-confirm-corpus.json"
    if manifest["confirmation_accessed"] or digest(cases_path) != manifest["case_sha256"] or \
       digest(corpus_path) != manifest["corpus_sha256"]:
        raise RuntimeError("Frozen confirmation data changed")
    cases = z.rows(cases_path)
    corpus = z.read(corpus_path)
    if len(cases) != 64 or len(corpus) != 2048 or \
       Counter(x["source"] for x in cases) != {"musique": 32, "2wiki": 32}:
        raise RuntimeError("Fresh confirmation size changed")
    if list(OUT.glob("eval-fresh_confirm-*.jsonl")):
        raise RuntimeError("Confirmation outputs already started")
    route = z.read(OUT / "development-decision.json")["selected_route"]
    if route not in ("A", "B"):
        raise RuntimeError("No unique eligible route")
    state("evaluating_confirmation", confirmation_accessed=True)
    start = time.monotonic()
    for label, adapter in (("baseline", SOURCE),
                           (f"42-{route}", OUT / f"seed42-{route}/adapter-final"),
                           (f"43-{route}", OUT / f"seed43-{route}/adapter-final")):
        evaluate(label, adapter, "fresh_confirm", cases, corpus_path)
    info = review_packet("confirmation", ("fresh_confirm",),
                         ("baseline", f"42-{route}", f"43-{route}"))
    if info["outputs"] != 192:
        raise RuntimeError("Confirmation output bound mismatch")
    state("awaiting_confirmation_review", confirmation_review=info,
          confirmation_seconds=time.monotonic() - start)


def confirmation_decision() -> dict:
    if z.read(STATE)["status"] != "awaiting_confirmation_review":
        raise RuntimeError("Confirmation outputs not ready")
    totals, votes = scored("confirmation")
    route = z.read(OUT / "development-decision.json")["selected_route"]
    baseline = totals["fresh_confirm-baseline"]
    cases = {x["id"]: x for x in z.rows(OUT / "fresh-confirm-cases.jsonl")}
    protected = set()
    for row in z.rows(OUT / "eval-fresh_confirm-baseline.jsonl"):
        if votes["fresh_confirm", "baseline", row["case_id"]]["category"] != "correct_sufficient":
            continue
        title_steps = {}
        for i, step in enumerate(row["trace"]):
            for doc in step.get("observation", {}).get("result", []):
                title_steps.setdefault(doc["title"], i)
        titles = cases[row["case_id"]]["support_titles"]
        if len(titles) == 2 and all(title in title_steps for title in titles) and \
           max(title_steps[titles[0]], title_steps[titles[1]]) > \
           min(title_steps[titles[0]], title_steps[titles[1]]):
            protected.add(row["case_id"])
    checks = {}
    for seed in (42, 43):
        label = f"{seed}-{route}"
        result = totals[f"fresh_confirm-{label}"]
        lost = sum(votes["fresh_confirm", label, case_id]["category"] != "correct_sufficient"
                   for case_id in protected)
        checks[str(seed)] = {"success": result.get("correct_sufficient", 0),
            "gain": result.get("correct_sufficient", 0) - baseline.get("correct_sufficient", 0),
            "bad_total": result["bad_total"], "invalid": result["invalid"],
            "protected_multihop_losses": lost,
            "passed": result.get("correct_sufficient", 0) >= baseline.get("correct_sufficient", 0) + 4
                      and result["bad_total"] <= baseline["bad_total"]
                      and result["invalid"] <= baseline["invalid"] and lost <= 1}
    passed = all(x["passed"] for x in checks.values())
    adapter = OUT / f"seed42-{route}/adapter-final" if passed else SOURCE
    result = {"published": passed, "selected_route": route if passed else None,
              "selected_seed": 42 if passed else None, "adapter": str(adapter),
              "status": "complete_confirmed_and_published" if passed else "complete_confirmation_failed",
              "baseline": baseline, "seeds": checks, "counts": totals,
              "protected_multihop_count": len(protected)}
    z.write(OUT / "product-decision.json", result)
    state(result["status"], product_decision=result)
    return result


def summarize() -> dict:
    value = z.read(STATE)
    result = {"status": value["status"], "stages": value["stages"],
              "diagnostic_gate": z.read(OUT / "diagnostic-decision.json")
              if (OUT / "diagnostic-decision.json").exists() else None}
    for name in ("fresh-confirm-manifest", "pair-proposal-audit", "pair-review-summary",
                 "pair-decision", "development-decision"):
        path = OUT / f"{name}.json"
        if path.exists():
            result[name.replace("-", "_")] = z.read(path)
    if (OUT / "development-review-final.jsonl").exists():
        result["recomputed_development_counts"] = scored("development")[0]
    if (OUT / "confirmation-review-final.jsonl").exists():
        result["recomputed_confirmation_counts"] = scored("confirmation")[0]
    if (OUT / "product-decision.json").exists():
        result["product_decision"] = z.read(OUT / "product-decision.json")
    if value["status"].startswith("complete") and not (OUT / "REPORT.md").exists():
        gate = result.get("diagnostic_gate") or {}
        pairs = result.get("pair_review_summary") or {}
        decision = result.get("product_decision") or {}
        lines = ["# Goal AB：证据充分性决策监督", "",
                 f"状态：`{value['status']}`。", "",
                 "## 门槛与结果", "",
                 f"Goal Z 开发题中停止决策错误：{gate.get('dev_stop_error_cases', '未完成')}/64；预设门槛 8。", "",
                 f"合格决策对：{pairs.get('approved', '未完成')}/64；预设门槛 32。", "",
                 f"128 条训练轨迹的停止决策错误：{pairs.get('on_policy_stop_error_cases', '未完成')}；预设门槛 16。", "",
                 f"产品发布：{decision.get('published', False)}；推荐 adapter：`{decision.get('adapter', SOURCE)}`。", "",
                 "本轮语义审核由代理辅助、主 agent 裁定；不是独立人工金标准。新确认题只有开发门槛通过后才运行。", ""]
        if "recomputed_development_counts" in result:
            lines += ["## 开发与回归分数", "",
                      "| 集合 / 模型 | 语义成功 | EM | 事实错＋无证据 | 无效动作 | 停止错误 |",
                      "|---|---:|---:|---:|---:|---:|"]
            for key, counts in sorted(result["recomputed_development_counts"].items()):
                lines.append(f"| {key} | {counts.get('correct_sufficient', 0)} | {counts['em']} | "
                             f"{counts['bad_total']} | {counts['invalid']} | {counts['stop_errors']} |")
            lines.append("")
        if "recomputed_confirmation_counts" in result:
            lines += ["## 新确认分数", "",
                      "| 模型 | 语义成功 /64 | EM /64 | 事实错＋无证据 | 无效动作 |",
                      "|---|---:|---:|---:|---:|"]
            for key, counts in sorted(result["recomputed_confirmation_counts"].items()):
                lines.append(f"| {key} | {counts.get('correct_sufficient', 0)} | {counts['em']} | "
                             f"{counts['bad_total']} | {counts['invalid']} |")
            lines.append("")
        z.write(OUT / "summary.json", result)
        (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        status = z.read(STATE)["status"]
        if status == "awaiting_diagnostic_review":
            print(json.dumps(diagnostic_gate(), ensure_ascii=False))
        elif status == "awaiting_train_material":
            started = time.monotonic()
            state("collecting_train", active_stage="train_rollouts")
            try:
                collect_train()
                value = z.read(STATE)
                seconds = time.monotonic() - started
                value["gpu_seconds"] += seconds
                value["stages"].append({"name": "train_rollouts", "seconds": seconds,
                    "trajectories": 128})
                STATE.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            except BaseException as exc:
                state("stopped", failure={"stage": "train_rollouts", "error": str(exc),
                                           "seconds": time.monotonic() - started})
                raise
        elif status == "awaiting_pair_review":
            if not (OUT / "pair-proposals.jsonl").exists():
                propose_pairs()
            elif all((OUT / f"pair-review-{name}.jsonl").exists() for name in ("a", "b")) and \
                 not (OUT / "pair-review-summary.json").exists():
                result = finalize_pair_review()
                if result["passed"]:
                    print(json.dumps(validate_pairs(), ensure_ascii=False))
                else:
                    print(json.dumps(result, ensure_ascii=False))
            elif (OUT / "decision-pairs.jsonl").exists():
                print(json.dumps(validate_pairs(), ensure_ascii=False))
            else:
                print(json.dumps({"status": status, "proposals": 64,
                                  "next": "review decision pairs"}, ensure_ascii=False))
        elif status == "training":
            try:
                train_and_evaluate()
            except BaseException as exc:
                state("stopped", failure={"stage": "training_or_development_evaluation",
                                           "error": str(exc)})
                raise
        elif status == "awaiting_development_review":
            print(json.dumps(development_decision(), ensure_ascii=False))
        elif status == "confirmation_authorized":
            try:
                evaluate_confirmation()
            except BaseException as exc:
                state("stopped", failure={"stage": "fresh_confirmation", "error": str(exc)})
                raise
        elif status == "awaiting_confirmation_review":
            print(json.dumps(confirmation_decision(), ensure_ascii=False))
        else:
            raise RuntimeError(f"No automatic action from state {status}")
    else:
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
