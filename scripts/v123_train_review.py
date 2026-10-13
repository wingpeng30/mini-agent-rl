"""Conditional, original-order Luna review and fail-closed train export."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from scripts.v123_api_protocol import valid_vote
from scripts.v123_cpu_core import (ROOT, OUT, digest, export_packet, norm, read, readl, sha,
                                   validate_training, write, writel)
from scripts.v123_train_assess import SOURCE, REVIEW, assess

TARGET = OUT / "train"


def inputs():
    assessment = assess()
    if assessment["status"] != "ready_for_luna_only":
        return assessment, None, None, None
    manifest = read(REVIEW / "manifest.json")
    packets = readl(REVIEW / "internal-packets.jsonl")
    if sha(REVIEW / "internal-packets.jsonl") != manifest["packet_sha256"]:
        raise ValueError("train_review_input_changed")
    by_case = {p["case_id"]: p for p in packets}
    source = {case: row["source_id"] for row in readl(REVIEW / "eligible-private.jsonl") for case in row["case_ids"]}
    if len(by_case) != len(packets) or set(source) != set(by_case):
        raise ValueError("train_review_source_binding_invalid")
    return assessment, manifest, by_case, source


def saved_votes(by_case: dict) -> dict:
    path = TARGET / "luna-v123.jsonl"
    rows = readl(path) if path.exists() else []
    result = {}
    for row in rows:
        case = row.get("case_id")
        if case not in by_case or case in result or row.get("packet_sha256") != digest(by_case[case]):
            raise ValueError("train_luna_duplicate_or_unbound_vote")
        result[case] = row
    return result


def append_luna(case: str, decision: dict, model: str) -> dict:
    _, _, by_case, _ = inputs()
    if by_case is None or case not in by_case:
        raise ValueError("train_luna_review_not_allowed")
    if case in saved_votes(by_case):
        raise ValueError("train_luna_duplicate_vote")
    valid, reason = valid_vote(by_case[case], decision)
    if not valid:
        raise ValueError("invalid_train_luna_vote:" + reason)
    path = TARGET / "luna-v123.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"case_id": case, "packet_sha256": digest(by_case[case]),
              "model": model, "decision": decision}
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
    return {"status": "recorded", "case_id": case, "training_allowed": False}


def review_status() -> dict:
    assessment, manifest, by_case, source = inputs()
    if by_case is None:
        return {"status": "blocked", "reason": assessment["reason"], "training_allowed": False}
    votes = saved_votes(by_case)
    api = {r["case_id"]: r for name in manifest["batches"] for r in readl(REVIEW / ("api-" + name))}
    if len(api) != len(by_case):
        raise ValueError("train_api_coverage_changed")
    approved = defaultdict(list)
    counts = Counter()
    for case, p in by_case.items():
        if case not in votes:
            continue
        counts["luna_records"] += 1
        lv, _ = valid_vote(p, votes[case]["decision"])
        av, _ = valid_vote(p, api[case].get("decision"))
        if lv:
            counts["luna_valid"] += 1
        else:
            counts["luna_invalid"] += 1
        if lv and av and votes[case]["decision"]["approved"] and api[case]["decision"]["approved"]:
            counts["double_approved_states"] += 1
            labels = sorted(set(votes[case]["decision"]["risk_labels"]) &
                            set(api[case]["decision"]["risk_labels"]))
            approved[source[case]].append((p, labels))
    order = read(SOURCE / "candidate-freeze.json")["ids"]
    pairs = [(qid, approved[qid]) for qid in order if len(approved[qid]) == 2 and
             {p["proposed_action"]["action"] for p, _ in approved[qid]} == {"search", "answer"}]
    risks = Counter(label for _, records in pairs for _, labels in records for label in labels)
    reached = len(pairs) >= 25 and risks["relation"] >= 20
    return {"status": "ready_to_finalize" if reached else "review_pending",
            "candidate_states": len(by_case), "review_counts": dict(counts),
            "complete_double_approved_pairs": len(pairs), "pair_risks": dict(risks),
            "review_complete": reached, "training_allowed": False}


def next_luna_batch() -> dict:
    current = review_status()
    if current["status"] != "review_pending":
        return {"status": "stopped", "reason": current.get("reason", current["status"]),
                "training_allowed": False}
    _, manifest, by_case, _ = inputs()
    voted = saved_votes(by_case)
    for name, spec in manifest["batches"].items():
        batch = readl(REVIEW / name)
        if sha(REVIEW / name) != spec["sha256"] or len(batch) != spec["states"] or len(batch) > 20:
            raise ValueError("old_train_batch_changed")
        pending = [p for p in batch if p["case_id"] not in voted]
        if not pending:
            continue
        public = [{key: p[key] for key in ("case_id", "question", "proposed_action",
                                          "before_action", "retrieved_after_search")} for p in pending]
        path = TARGET / ("work-luna-" + name)
        if path.exists() and readl(path) != public:
            raise ValueError("train_luna_work_batch_changed")
        if not path.exists():
            writel(path, public)
        return {"status": "ready", "batch": path.name, "cases": len(public),
                "sha256": sha(path), "training_allowed": False}
    return {"status": "blocked", "reason": "all_train_candidates_reviewed_without_gate",
            "review": current, "training_allowed": False}


def _choose(pairs: list, count: int = 25, relation_min: int = 20) -> list | None:
    selected = []
    suffix = [0] * (len(pairs) + 1)
    for i in range(len(pairs) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sum("relation" in tags for _, tags in pairs[i][1])
    failed = set()

    def visit(i, need):
        if len(selected) == count:
            return need == 0
        if len(pairs) - i < count - len(selected) or suffix[i] < need:
            return False
        key = (i, count - len(selected), need)
        if key in failed:
            return False
        selected.append(pairs[i])
        gain = sum("relation" in tags for _, tags in pairs[i][1])
        if visit(i + 1, max(0, need - gain)):
            return True
        selected.pop()
        if visit(i + 1, need):
            return True
        failed.add(key)
        return False

    return list(selected) if visit(0, relation_min) else None


def finalize() -> dict:
    current = review_status()
    if not current.get("review_complete"):
        return {"status": "blocked", "reason": current.get("reason", "train_review_gate_not_met"),
                "review": current, "training_allowed": False}
    _, manifest, by_case, source = inputs()
    votes = saved_votes(by_case)
    api = {r["case_id"]: r for name in manifest["batches"] for r in readl(REVIEW / ("api-" + name))}
    grouped = defaultdict(list)
    for case, p in by_case.items():
        if case not in votes:
            continue
        lv, _ = valid_vote(p, votes[case]["decision"])
        av, _ = valid_vote(p, api[case].get("decision"))
        if lv and av and votes[case]["decision"]["approved"] and api[case]["decision"]["approved"]:
            labels = sorted(set(votes[case]["decision"]["risk_labels"]) &
                            set(api[case]["decision"]["risk_labels"]))
            grouped[source[case]].append((p, labels))
    order = read(SOURCE / "candidate-freeze.json")["ids"]
    pairs = [(qid, grouped[qid]) for qid in order if len(grouped[qid]) == 2]
    selected = _choose(pairs)
    if not selected:
        return {"status": "failed", "reason": "no_feasible_25_relation_selection",
                "training_allowed": False}
    base = readl(ROOT / "data/evidence-sft-v117/frozen/train.jsonl")
    supplement = [export_packet(p) for _, records in selected for p, _ in records]
    internal_packets = readl(OUT / "internal/review/internal-packets.jsonl")
    internal_freeze = read(OUT / "internal/candidate-freeze.json")
    internal_ids = set(internal_freeze["old_candidate_ids"] + internal_freeze["new_candidate_ids"])
    internal_text = {norm(p["question"]) for p in internal_packets}
    history = read(SOURCE / "history-private.json")
    gate = validate_training(base, supplement, [qid for qid, _ in selected], internal_ids,
                             internal_text, set(history["ids"]), set(history["normalized_questions"]))
    gate["checks"]["new_relation_risk_at_least_20"] = sum(
        "relation" in tags for _, records in selected for _, tags in records) >= 20
    gate["checks"]["original_training_gate_risks"] = read(ROOT / "reports/v117-local-preflight.json")["checks"]["train_actions_and_risks"]
    gate["passed"] = all(gate["checks"].values())
    write(TARGET / "gate.json", gate)
    if not gate["passed"]:
        return {"status": "failed", "gate": gate, "training_allowed": False}
    path = TARGET / "frozen/train.jsonl"
    if path.exists():
        raise ValueError("v123_train_export_already_exists")
    writel(path, base + supplement)
    write(TARGET / "frozen/manifest.json", {"version": "v123-train-data-v1",
          "train_sha256": sha(path), "states": 395, "questions": 211, "complete_pairs": 100,
          "selected_ids_sha256": digest([qid for qid, _ in selected]),
          "training_allowed": False})
    return {"status": "data_ready_training_not_authorized", "gate": gate,
            "training_allowed": False}
