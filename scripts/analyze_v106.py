"""Combine v106 semantic audit, few-shot screen, and oracle-evidence probe."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def bootstrap(values: list[float], repeats: int = 10_000) -> dict:
    rng = random.Random(42)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(repeats))
    return {"n": n, "delta": sum(values) / n, "low": draws[int(.025 * repeats)], "high": draws[int(.975 * repeats)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autonomous-analysis", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--baseline20", type=Path, required=True)
    parser.add_argument("--fewshot20", type=Path, required=True)
    parser.add_argument("--oracle-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    autonomous = json.loads(args.autonomous_analysis.read_text(encoding="utf-8"))
    audit = json.loads(args.semantic_audit.read_text(encoding="utf-8"))
    baseline20 = json.loads(args.baseline20.read_text(encoding="utf-8"))
    fewshot20 = json.loads(args.fewshot20.read_text(encoding="utf-8"))
    auto_rows = {row["task_id"]: row["baseline"] for row in autonomous["details"]}
    oracle_rows = {row["task_id"]: row for row in (json.loads(line) for line in args.oracle_decisions.read_text(encoding="utf-8").splitlines() if line)}
    task_ids = sorted(set(auto_rows) & set(oracle_rows))
    migrations = Counter()
    em_deltas, f1_deltas, details = [], [], []
    for task_id in task_ids:
        a, o = auto_rows[task_id], oracle_rows[task_id]
        a_ok, o_ok = bool(a["exact"]), bool(o["strict_exact_match"])
        migration = ("correct" if a_ok else "wrong") + "_to_" + ("correct" if o_ok else "wrong")
        migrations[migration] += 1
        em_deltas.append(float(o_ok) - float(a_ok))
        f1_deltas.append(float(o["token_f1"]) - float(a["token_f1"]))
        details.append({"task_id": task_id, "migration": migration,
                        "autonomous_answer": a["answer"], "oracle_answer": o["response"].get("content", ""),
                        "gold": a["gold"], "autonomous_f1": a["token_f1"], "oracle_f1": o["token_f1"]})
    accepted = {row["task_id"] for row in audit["results"]
                if row["review"].get("verdict") in {"equivalent", "acceptable_short_form"}}
    autonomous_semantic = sum(bool(row["exact"]) or task_id in accepted for task_id, row in auto_rows.items()) / len(auto_rows)
    report = {
        "version": "v106", "task_count": len(task_ids),
        "semantic_audit": {"counts": audit["counts"], "accepted_task_count": len(accepted),
                           "autonomous_audit_adjusted_accuracy": autonomous_semantic,
                           "scope": "adjusts only the 17 preselected boundary candidates"},
        "fewshot20": {
            "baseline_strict_em": baseline20["strict_exact_match_rate"], "candidate_strict_em": fewshot20["strict_exact_match_rate"],
            "baseline_f1": baseline20["token_f1"], "candidate_f1": fewshot20["token_f1"],
            "strict_em_delta": fewshot20["strict_exact_match_rate"] - baseline20["strict_exact_match_rate"],
            "f1_delta": fewshot20["token_f1"] - baseline20["token_f1"], "decision": "reject_fewshot_v106"},
        "oracle_evidence": {
            "autonomous_strict_em": sum(bool(x["exact"]) for x in auto_rows.values()) / len(auto_rows),
            "oracle_strict_em": sum(float(x["strict_exact_match"]) for x in oracle_rows.values()) / len(oracle_rows),
            "autonomous_f1": sum(float(x["token_f1"]) for x in auto_rows.values()) / len(auto_rows),
            "oracle_f1": sum(float(x["token_f1"]) for x in oracle_rows.values()) / len(oracle_rows),
            "paired_strict_em": bootstrap(em_deltas), "paired_f1": bootstrap(f1_deltas),
            "migration_counts": dict(migrations),
            "interpretation": "Complete evidence helps, but most remaining failures persist after all support documents are visible."},
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
