"""Paired, read-only analysis of the v105 answer-repair pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from mini_agent_rl.answer_scoring import normalize_answer, token_f1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path, table: str) -> list[dict]:
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        return [json.loads(row[0]) for row in conn.execute(f"SELECT data FROM {table}")]
    finally:
        conn.close()


def load(path: Path) -> tuple[dict[str, dict], dict[str, list[dict]], dict[str, dict]]:
    rollouts = {row["task_id"]: row for row in rows(path, "rollouts")}
    transitions: dict[str, list[dict]] = defaultdict(list)
    for row in rows(path, "transitions"):
        transitions[row["rollout_id"]].append(row)
    for values in transitions.values():
        values.sort(key=lambda row: row["step"])
    tasks = {row["id"]: row for row in rows(path, "tasks")}
    return rollouts, transitions, tasks


def visible_evidence(transitions: list[dict]) -> tuple[list[dict], str]:
    if not transitions:
        return [], ""
    docs: list[dict] = []
    for message in transitions[-1].get("messages", []):
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        docs.extend(item for item in payload.get("result", []) if isinstance(item, dict))
    text = " ".join(f"{doc.get('title', '')} {doc.get('text', '')}" for doc in docs)
    return docs, text


def classify(rollout: dict, transitions: list[dict], task: dict) -> dict:
    answer = rollout.get("final_answer") or ""
    gold = task.get("answer") or ""
    answer_strict = " ".join(answer.strip().casefold().split())
    gold_strict = " ".join(gold.strip().casefold().split())
    answer_norm, gold_norm = normalize_answer(answer), normalize_answer(gold)
    exact = answer_strict == gold_strict
    normalized_exact = answer_norm == gold_norm
    f1 = token_f1(answer, gold)
    docs, evidence = visible_evidence(transitions)
    evidence_norm = normalize_answer(evidence)
    titles = {str(doc.get("title", "")).casefold() for doc in docs}
    wanted = {str(x).casefold() for x in task.get("metadata", {}).get("support_titles", [])}
    if exact:
        category = "correct"
    elif rollout.get("termination_reason") != "answer":
        category = "protocol_or_termination"
    elif normalized_exact or f1 >= 0.8 or (answer_norm and gold_norm and (answer_norm in gold_norm or gold_norm in answer_norm)):
        category = "answer_boundary_or_variant"
    elif gold_norm and gold_norm in evidence_norm:
        category = "entity_or_relation_selection"
    elif wanted and wanted <= titles:
        category = "support_complete_inference"
    elif wanted & titles:
        category = "partial_support_then_answer"
    elif not docs:
        category = "ungrounded_direct_answer"
    else:
        category = "retrieval_failure_or_wrong_query"
    return {
        "task_id": rollout["task_id"], "question": task.get("question", ""),
        "gold": gold, "answer": answer, "exact": exact,
        "normalized_exact": normalized_exact, "token_f1": f1,
        "termination_reason": rollout.get("termination_reason"), "category": category,
        "gold_string_visible": bool(gold_norm and gold_norm in evidence_norm),
        "visible_titles": sorted(titles), "visible_evidence_excerpt": evidence[:1200],
    }


def bootstrap(deltas: list[float], repeats: int = 10_000) -> dict:
    rng = random.Random(42)
    count = len(deltas)
    samples = sorted(sum(deltas[rng.randrange(count)] for _ in range(count)) / count for _ in range(repeats))
    return {
        "n": count, "delta": sum(deltas) / count,
        "low": samples[int(0.025 * repeats)], "high": samples[int(0.975 * repeats)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = {"baseline": sha256(args.baseline), "candidate": sha256(args.candidate)}
    baseline, candidate = load(args.baseline), load(args.candidate)
    task_ids = sorted(set(baseline[0]) & set(candidate[0]))
    details = []
    for task_id in task_ids:
        b_rollout, c_rollout = baseline[0][task_id], candidate[0][task_id]
        b = classify(b_rollout, baseline[1][b_rollout["id"]], baseline[2][task_id])
        c = classify(c_rollout, candidate[1][c_rollout["id"]], candidate[2][task_id])
        if not b["exact"] and c["exact"]:
            migration = "wrong_to_correct"
        elif b["exact"] and not c["exact"]:
            migration = "correct_to_wrong"
        elif b["exact"] and c["exact"]:
            migration = "correct_to_correct"
        else:
            migration = "wrong_to_wrong"
        details.append({"task_id": task_id, "migration": migration, "baseline": b, "candidate": c})
    em_delta = [float(row["candidate"]["exact"]) - float(row["baseline"]["exact"]) for row in details]
    f1_delta = [row["candidate"]["token_f1"] - row["baseline"]["token_f1"] for row in details]
    report = {
        "version": "v105", "task_count": len(task_ids),
        "source_hashes_before": before,
        "source_hashes_unchanged": before == {"baseline": sha256(args.baseline), "candidate": sha256(args.candidate)},
        "migration_counts": dict(Counter(row["migration"] for row in details)),
        "baseline_error_categories": dict(Counter(row["baseline"]["category"] for row in details if not row["baseline"]["exact"])),
        "candidate_error_categories": dict(Counter(row["candidate"]["category"] for row in details if not row["candidate"]["exact"])),
        "paired_strict_em": bootstrap(em_delta), "paired_token_f1": bootstrap(f1_delta),
        "decision": "stop_answer_repair_branch" if sum(em_delta) <= 0 and sum(f1_delta) <= 0 else "consider_replication",
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
