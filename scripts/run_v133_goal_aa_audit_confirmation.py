"""Goal AA: correct Goal Z evidence review before a single sealed confirmation."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_v133_goal_z_multihop_sft as z

OUT = z.ROOT / "runs/v133-goal-aa-audit-confirmation"
CFG, STATE = OUT / "config.json", OUT / "state.json"
LABELS = ("incumbent", "42-answer", "42-trajectory", "43-answer", "43-trajectory")
CATEGORIES = ("correct_sufficient", "missing_required_information", "factual_error",
              "unsupported_by_visible_evidence", "indeterminate")
SPLITS = ("dev", "old", "y")


def update_state(status: str, **extra) -> None:
    state = z.read(STATE)
    state.update(status=status, **extra)
    STATE.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def archived_rows() -> list[dict]:
    result = []
    for split, count in (("dev", 64), ("old", 32), ("y", 48)):
        for label in LABELS:
            path = z.OUT / f"eval-{split}-{label}.jsonl"
            values = z.rows(path)
            if len(values) != count or len({x["case_id"] for x in values}) != count:
                raise RuntimeError(f"Archived evaluation incomplete: {path}")
            result.extend(value | {"model": label, "split": split} for value in values)
    if len(result) != 720:
        raise RuntimeError("Expected 720 archived outputs")
    return result


def verify_assets() -> dict:
    frozen = z.read(z.CFG)
    for name, path in z.frozen_files().items():
        if z.sha(path) != frozen["files"][name]:
            raise RuntimeError(f"Goal Z frozen asset changed: {name}")
    if z.read(z.STATE)["status"] != "complete_development_gate_failed":
        raise RuntimeError("Expected the original Goal Z decision to be frozen")
    if list(z.OUT.glob("eval-confirm-*.jsonl")):
        raise RuntimeError("Goal Z confirmation has already been opened")
    for label in LABELS[1:]:
        summary = z.read(z.OUT / f"seed{label}/summary.json")
        if summary["updates"] != 128 or not summary["base_unchanged"] or \
           summary["changed_lora_tensors"] <= 0 or summary["reload_max_abs"] > .001 or \
           z.sha(z.OUT / f"seed{label}/adapter-final/adapter_model.safetensors") != summary["adapter_sha256"]:
            raise RuntimeError(f"Checkpoint engineering verification failed: {label}")
    old = z.rows(z.OUT / "eval-old-incumbent.jsonl")
    if (sum(x["em"] for x in old),
        sum(x["em"] for x in old if x["split"] == "goal_G_regression"),
        sum(x["reason"] == "invalid_action" for x in old)) != (20, 7, 0):
        raise RuntimeError("Historical product baseline changed")
    for row in archived_rows():
        if row["model"] not in LABELS or row["model_calls"] > 4 or row["searches"] > 3:
            raise RuntimeError("Archived output identity or call budget mismatch")
    return {"goal_z_config": z.sha(z.CFG), "goal_z_state": z.sha(z.STATE),
            "goal_z_rubric": z.sha(z.OUT / "evidence-rubric.json"),
            "goal_z_decision": z.sha(z.OUT / "development-decision.json"),
            "goal_z_report": z.sha(z.OUT / "REPORT.md"),
            "eval_outputs": {f"{split}-{label}": z.sha(z.OUT / f"eval-{split}-{label}.jsonl")
                             for split in SPLITS for label in LABELS},
            "checkpoints": {label: z.sha(z.OUT / f"seed{label}/adapter-final/adapter_model.safetensors")
                            for label in LABELS[1:]}}


def make_packet(rows: list[dict], tokenizer, stage: str) -> tuple[list[dict], list[dict]]:
    packet_by_id = {}
    mapping = []
    for row in rows:
        final = row["trace"][-1]
        ids = final["prompt_token_ids"]
        if not ids:
            raise RuntimeError("Missing actual prompt tokens")
        actual = tokenizer.decode(ids, skip_special_tokens=False)
        if row["question"].strip().casefold() not in actual.casefold():
            raise RuntimeError(f"Question absent from actual model prompt: {row['case_id']} "
                               f"question={row['question']!r} prompt={actual[:240]!r}")
        observed = []
        for step in row["trace"]:
            for doc in step.get("observation", {}).get("result", []):
                if doc["id"] in actual and doc["id"] not in observed:
                    observed.append(doc["id"])
        token = digest([row["question"], row["answer"], actual, row["reason"]])
        mapping.append({"split": stage if stage == "confirm" else row["split"],
                        "case_id": row["case_id"], "model": row["model"], "review_id": token})
        packet_by_id[token] = {"review_id": token, "question": row["question"],
                               "answer": row["answer"], "reason": row["reason"],
                               "actual_prompt": actual, "visible_document_ids": observed,
                               "prompt_token_sha256": digest(ids)}
    return sorted(packet_by_id.values(), key=lambda r: r["review_id"]), mapping


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    assets = verify_assets()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(z.q.p.BASE, local_files_only=True)
    packet, mapping = make_packet(archived_rows(), tokenizer, "development")
    if len(mapping) != 720:
        raise RuntimeError("Audit map incomplete")
    OUT.mkdir(parents=True)
    z.write(CFG, {"version": "goal-aa-v1", "assets": assets,
                  "rubric": z.read(z.OUT / "evidence-rubric.json"),
                  "gate": z.read(z.CFG)["dev_gate"],
                  "confirmation_gate": z.read(z.CFG)["confirm_gate"],
                  "labels": LABELS, "max_confirmation_trajectories": 192,
                  "runner_sha256": z.sha(Path(__file__))})
    z.write(OUT / "development-map.json", mapping)
    for row in packet:
        z.append(OUT / "development-packet.jsonl", row)
    z.write(STATE, {"status": "awaiting_development_review", "stages": [],
                    "development_unique": len(packet), "confirmation_accessed": False})
    print(json.dumps({"archived_outputs": len(mapping), "unique_actual_prompts": len(packet)}))


def reviewed(stage: str) -> tuple[dict, list[dict]]:
    packet = z.rows(OUT / f"{stage}-packet.jsonl")
    ids = {x["review_id"] for x in packet}
    a = z.rows(OUT / f"{stage}-review-a.jsonl")
    b = z.rows(OUT / f"{stage}-review-b.jsonl")
    if {x["review_id"] for x in a} != ids or {x["review_id"] for x in b} != ids or \
       len(a) != len(ids) or len(b) != len(ids):
        raise RuntimeError(f"{stage} independent reviews incomplete")
    aa, bb = ({x["review_id"]: x for x in values} for values in (a, b))
    adjudications = z.rows(OUT / f"{stage}-adjudication.jsonl")
    adj = {x["review_id"]: x for x in adjudications}
    disputed = {key for key in ids if aa[key]["category"] != bb[key]["category"]}
    if set(adj) != disputed or len(adjudications) != len(disputed):
        raise RuntimeError("All and only reviewer disputes must be adjudicated")
    overrides_path = OUT / f"{stage}-main-overrides.jsonl"
    overrides = z.rows(overrides_path) if overrides_path.exists() else []
    override_map = {x["review_id"]: x for x in overrides}
    if len(overrides) != len(override_map) or not set(override_map).issubset(ids):
        raise RuntimeError("Invalid main override IDs")
    accepted = {}
    for item in packet:
        key = item["review_id"]
        vote = adj[key] if key in adj else aa[key]
        if key in override_map:
            override = override_map[key]
            if override.get("old_category") != vote["category"]:
                raise RuntimeError(f"Main override does not match prior decision: {key}")
            vote = override
        if vote["category"] not in CATEGORIES or \
           not set(vote.get("evidence_ids", [])).issubset(item["visible_document_ids"]):
            raise RuntimeError(f"Invalid evidence review: {key}")
        if item["reason"] != "answer" and vote["category"] == "correct_sufficient":
            raise RuntimeError("Unanswered case marked success")
        if vote["category"] == "correct_sufficient" and not vote.get("evidence_ids"):
            raise RuntimeError("Success lacks cited visible evidence")
        accepted[key] = vote
    return accepted, packet


def development_decision() -> dict:
    votes, packet = reviewed("development")
    mapping = z.read(OUT / "development-map.json")
    if len(mapping) != 720:
        raise RuntimeError("Audit map incomplete")
    by_case = {(r["split"], r["model"], r["case_id"]): votes[r["review_id"]]["category"]
               for r in mapping}
    if len(by_case) != 720:
        raise RuntimeError("Duplicate audit mapping")
    groups = {x["case_id"]: x["split"] for x in z.q.dev_cases()}
    counts = {}
    for split, total in (("dev", 64), ("old", 32), ("y", 48)):
        for label in LABELS:
            rows = z.rows(z.OUT / f"eval-{split}-{label}.jsonl")
            if len(rows) != total:
                raise RuntimeError("Output count changed")
            c = Counter(by_case[split, label, x["case_id"]] for x in rows)
            c["invalid"] = sum(x["reason"] == "invalid_action" for x in rows)
            c["em"] = sum(x["em"] for x in rows)
            if split == "old":
                c["main"] = sum(by_case[split, label, x["case_id"]] == "correct_sufficient"
                                for x in rows if groups[x["case_id"]] == "new_eval")
                c["regression"] = sum(by_case[split, label, x["case_id"]] == "correct_sufficient"
                                      for x in rows if groups[x["case_id"]] == "goal_G_regression")
            counts[f"{split}-{label}"] = dict(c)
    base = counts["dev-incumbent"]
    gates = {}
    for arm in ("answer", "trajectory"):
        gates[arm] = []
        for seed in (42, 43):
            label = f"{seed}-{arm}"
            dev, old, yy = (counts[f"{split}-{label}"] for split in SPLITS)
            passed = (dev.get("correct_sufficient", 0) >= base.get("correct_sufficient", 0) + 4 and
                      all(dev.get(key, 0) <= base.get(key, 0) for key in
                          ("factual_error", "unsupported_by_visible_evidence", "invalid")) and
                      old["main"] >= counts["old-incumbent"]["main"] and
                      old["regression"] >= counts["old-incumbent"]["regression"] and
                      yy.get("correct_sufficient", 0) >= counts["y-incumbent"].get("correct_sufficient", 0))
            gates[arm].append({"seed": seed, "passed": passed,
                               "dev_success": dev.get("correct_sufficient", 0),
                               "old_main": old["main"], "old_regression": old["regression"],
                               "goal_y": yy.get("correct_sufficient", 0)})
    eligible = [arm for arm in gates if all(x["passed"] for x in gates[arm])]
    eligible.sort(key=lambda arm: (-min(x["dev_success"] for x in gates[arm]),
                                   -sum(x["dev_success"] for x in gates[arm]),
                                   sum(counts[f"dev-{seed}-{arm}"].get("factual_error", 0)
                                       for seed in (42, 43)), 0 if arm == "answer" else 1))
    old_map = {(x["split"], x["model"], x["case_id"]): x["review_id"]
               for x in z.read(z.OUT / "blind-review-map.json")}
    old_votes = {x["review_id"]: x for x in z.read(z.OUT / "blind-review-inherited.json") +
                 z.rows(z.OUT / "blind-review-votes-0.jsonl") + z.rows(z.OUT / "blind-review-votes-1.jsonl")}
    changes = []
    for row in mapping:
        key = row["split"], row["model"], row["case_id"]
        previous = old_votes[old_map[key]]["category"]
        current = votes[row["review_id"]]["category"]
        if previous != current:
            changes.append({"split": row["split"], "model": row["model"], "case_id": row["case_id"],
                            "old": previous, "new": current, "review_id": row["review_id"],
                            "reason": votes[row["review_id"]]["reason"]})
    if not (OUT / "development-corrections.jsonl").exists():
        for row in changes:
            z.append(OUT / "development-corrections.jsonl", row)
    if not (OUT / "development-corrections.jsonl").exists():
        (OUT / "development-corrections.jsonl").touch()
    decision = {"status": "confirmation_authorized" if eligible else "complete_development_gate_failed",
                "selected_route": eligible[0] if eligible else None,
                "eligible_routes": eligible, "seed_gates": gates, "counts": counts,
                "correction_count": len(changes), "reviewed_unique": len(packet)}
    z.write(OUT / "development-decision.json", decision)
    update_state(decision["status"], development_corrections=len(changes))
    return decision


def evaluate_confirmation() -> None:
    state = z.read(STATE)
    if state["status"] != "confirmation_authorized":
        raise RuntimeError("Confirmation access gate closed")
    arm = z.read(OUT / "development-decision.json")["selected_route"]
    if arm not in ("answer", "trajectory"):
        raise RuntimeError("No selected route")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    labels = ("incumbent", f"42-{arm}", f"43-{arm}")
    update_state("evaluating_confirmation", confirmation_accessed=True)
    start = time.monotonic()
    try:
        for label in labels:
            path = OUT / f"eval-confirm-{label}.jsonl"
            if path.exists():
                raise FileExistsError(path)
            adapter = z.SOURCE if label == "incumbent" else z.OUT / f"seed{label}/adapter-final"
            client, wrapped, registry = z.make_client(adapter, z.OUT / "corpus-confirm.json")
            for case in z.cases("confirm"):
                result = asyncio.run(z.q.rollout(case["question"], wrapped, registry, sample=False, seed=42))
                found = {doc["title"] for step in result["trace"]
                         for doc in step.get("observation", {}).get("result", [])}
                result.update(case_id=case["id"], split="confirm", source=case["source"],
                              model=label, support_retrieved=set(case["support_titles"]).issubset(found),
                              **z.q.score(result["answer"], case["answer"]))
                z.append(path, result)
                print(json.dumps({"confirm": label, "completed": len(z.rows(path))}), flush=True)
            del client, wrapped, registry
            gc.collect(); torch.cuda.empty_cache()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(z.q.p.BASE, local_files_only=True)
        packet, mapping = make_packet([row for label in labels
                                       for row in z.rows(OUT / f"eval-confirm-{label}.jsonl")],
                                      tokenizer, "confirm")
        if len(mapping) != 192:
            raise RuntimeError("Confirmation output count mismatch")
        z.write(OUT / "confirmation-map.json", mapping)
        for row in packet:
            z.append(OUT / "confirmation-packet.jsonl", row)
        update_state("awaiting_confirmation_review", confirmation_seconds=time.monotonic() - start,
                     confirmation_unique=len(packet))
    except BaseException as exc:
        update_state("stopped", failure=str(exc), confirmation_seconds=time.monotonic() - start)
        raise


def confirmation_decision() -> dict:
    votes, packet = reviewed("confirmation")
    mapping = z.read(OUT / "confirmation-map.json")
    if len(mapping) != 192:
        raise RuntimeError("Confirmation map incomplete")
    category = {(x["model"], x["case_id"]): votes[x["review_id"]]["category"]
                for x in mapping}
    if len(category) != 192:
        raise RuntimeError("Confirmation duplicate mapping")
    arm = z.read(OUT / "development-decision.json")["selected_route"]
    labels = ("incumbent", f"42-{arm}", f"43-{arm}")
    counts = {}
    for label in labels:
        values = z.rows(OUT / f"eval-confirm-{label}.jsonl")
        if len(values) != 64:
            raise RuntimeError("Confirmation output incomplete")
        c = Counter(category[label, v["case_id"]] for v in values)
        c["invalid"] = sum(v["reason"] == "invalid_action" for v in values)
        c["em"] = sum(v["em"] for v in values)
        c["searches"] = sum(v["searches"] for v in values)
        counts[label] = dict(c)
    base = counts["incumbent"]
    passed = all(counts[label].get("correct_sufficient", 0) >= base.get("correct_sufficient", 0) + 4 and
                 all(counts[label].get(key, 0) <= base.get(key, 0) for key in
                     ("factual_error", "unsupported_by_visible_evidence", "invalid"))
                 for label in labels[1:])
    decision = {"published": passed, "selected_route": arm if passed else None,
                "selected_seed": 42 if passed else None,
                "adapter": str(z.OUT / f"seed42-{arm}/adapter-final") if passed else str(z.SOURCE),
                "counts": counts, "reviewed_unique": len(packet),
                "status": "complete_confirmed_and_published" if passed else "complete_confirmation_failed"}
    z.write(OUT / "product-decision.json", decision)
    update_state(decision["status"])
    return decision


def run() -> None:
    status = z.read(STATE)["status"]
    if status == "awaiting_development_review":
        development_decision()
        status = z.read(STATE)["status"]
    if status == "confirmation_authorized":
        evaluate_confirmation()
        return
    if status == "awaiting_confirmation_review":
        confirmation_decision()
        return
    if status == "complete_development_gate_failed":
        return
    raise RuntimeError(f"No automatic action from {status}")


def resume_confirmation() -> None:
    """One authorized recovery from the pre-model interpreter mismatch."""
    state = z.read(STATE)
    if state.get("status") != "stopped" or state.get("failure") != "GPU/software changed":
        raise RuntimeError("Recovery is limited to the recorded interpreter mismatch")
    if z.read(OUT / "development-decision.json").get("selected_route") != "trajectory":
        raise RuntimeError("Frozen selected route changed")
    if list(OUT.glob("eval-confirm-*.jsonl")) or (OUT / "confirmation-map.json").exists() or \
       (OUT / "confirmation-packet.jsonl").exists():
        raise RuntimeError("Confirmation outputs already exist; recovery refused")
    if verify_assets() != z.read(CFG)["assets"]:
        raise RuntimeError("Frozen Goal Z assets changed")
    if z.software() != z.read(z.CFG)["software"]:
        raise RuntimeError("Frozen software environment not restored")
    z.write(OUT / "confirmation-recovery.json", {
        "reason": "Wrong system Python was used before model loading; no confirmation case was read",
        "previous_state": state, "software": z.software(),
        "selected_route": "trajectory", "max_trajectories": 192})
    update_state("confirmation_authorized", failure=None, recovery_authorized=True)
    evaluate_confirmation()


def summarize() -> dict:
    state = z.read(STATE)
    result = {"state": state, "development": z.read(OUT / "development-decision.json")
              if (OUT / "development-decision.json").exists() else None,
              "product": z.read(OUT / "product-decision.json")
              if (OUT / "product-decision.json").exists() else None}
    if (OUT / "confirmation-map.json").exists():
        if state["status"] not in ("awaiting_confirmation_review", "complete_confirmed_and_published",
                                   "complete_confirmation_failed"):
            raise RuntimeError("Confirmation data exists outside its gate")
        if len(z.read(OUT / "confirmation-map.json")) != 192:
            raise RuntimeError("Incomplete confirmation")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "summarize", "resume-confirmation"))
    args = parser.parse_args()
    if args.command == "prepare": prepare()
    elif args.command == "run": run()
    elif args.command == "resume-confirmation": resume_confirmation()
    else: print(json.dumps(summarize(), ensure_ascii=False))
