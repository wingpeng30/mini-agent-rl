"""Paired analysis for v094 autonomous evaluation and clustered stop probe."""
from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path("reports")
SEEDS = (42, 43, 44)


def load_details(name: str) -> dict[str, dict]:
    rows = json.loads((ROOT / f"v094-full100-{name}-details.json").read_text())["details"]
    return {row["task_id"]: row for row in rows}


def load_probe(name: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    path = ROOT / f"v094-probe120-{name}" / "decisions.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line); grouped.setdefault(row["task_id"], []).append(row)
    if len(grouped) != 30 or any(len(rows) != 4 for rows in grouped.values()):
        raise RuntimeError(f"invalid clustered probe for {name}")
    return grouped


def interval(values: list[float]) -> dict[str, float]:
    values.sort(); n = len(values)
    return {"low": values[int(n * .025)], "high": values[min(n - 1, int(n * .975))]}


def paired(base: dict, candidate: dict, metric: str, *, seed: int = 42, repeats: int = 10_000) -> dict:
    ids = sorted(base)
    if ids != sorted(candidate): raise RuntimeError("paired ids differ")
    deltas = [float(candidate[key][metric]) - float(base[key][metric]) for key in ids]
    rng = random.Random(seed); draws = []
    for _ in range(repeats):
        draws.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    return {"n": len(ids), "delta": sum(deltas) / len(deltas), **interval(draws)}


def clustered_rows(grouped: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    return {task: {metric: sum(float(row[metric]) for row in rows) / len(rows)
                   for metric in ("strict_exact_match", "token_f1")}
            for task, rows in grouped.items()}


def cross_seed(base: dict, candidates: list[dict], metric: str, *, seed: int = 42, repeats: int = 10_000) -> dict:
    ids = sorted(base)
    if any(sorted(value) != ids for value in candidates): raise RuntimeError("cross-seed paired ids differ")
    deltas = [sum(float(candidate[key][metric]) for candidate in candidates) / len(candidates) - float(base[key][metric]) for key in ids]
    rng = random.Random(seed); draws=[]
    for _ in range(repeats):
        draws.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    return {"n": len(ids), "delta": sum(deltas) / len(deltas), **interval(draws)}


def main() -> None:
    base = load_details("sft"); candidates = [load_details(f"seed{seed}") for seed in SEEDS]
    autonomous = {f"seed{seed}": {metric: paired(base, candidate, metric)
                                  for metric in ("strict_exact_match", "token_f1")}
                  for seed, candidate in zip(SEEDS, candidates)}
    autonomous["cross_seed_mean"] = {metric: cross_seed(base, candidates, metric)
                                      for metric in ("strict_exact_match", "token_f1")}
    summaries={}
    for name in ("sft", *(f"seed{seed}" for seed in SEEDS)):
        report=json.loads((ROOT / f"v094-full100-{name}.json").read_text())
        summaries[name]={key: report[key] for key in ("strict_exact_match_rate", "token_f1", "invalid_action_rate", "valid_action_rate", "repeated_search_rate", "output_budget_exhaustion_rate", "average_steps", "mean_search_count", "peak_gpu_memory_gb")}
    probe_raw={name: load_probe(name) for name in ("sft", *(f"seed{seed}" for seed in SEEDS))}
    probe={name: clustered_rows(rows) for name, rows in probe_raw.items()}
    probe_stats={f"seed{seed}": {metric: paired(probe["sft"], probe[f"seed{seed}"], metric)
                                 for metric in ("strict_exact_match", "token_f1")} for seed in SEEDS}
    probe_stats["cross_seed_mean"]={metric: cross_seed(probe["sft"], [probe[f"seed{seed}"] for seed in SEEDS], metric)
                                    for metric in ("strict_exact_match", "token_f1")}
    probe_conditions={}
    for name in probe_raw:
        probe_conditions[name]=json.loads((ROOT / f"v094-probe120-{name}" / "report.json").read_text())["by_condition"]
    recommendation = {
        "three_seed_mean_em_not_below_sft": sum(summaries[f"seed{s}"]["strict_exact_match_rate"] for s in SEEDS) / 3 >= summaries["sft"]["strict_exact_match_rate"],
        "at_least_two_seeds_non_degraded": sum(summaries[f"seed{s}"]["strict_exact_match_rate"] >= summaries["sft"]["strict_exact_match_rate"] for s in SEEDS) >= 2,
        "mean_invalid_not_above_sft": sum(summaries[f"seed{s}"]["invalid_action_rate"] for s in SEEDS) / 3 <= summaries["sft"]["invalid_action_rate"],
        "cross_seed_em_ci_lower_above_zero": autonomous["cross_seed_mean"]["strict_exact_match"]["low"] > 0,
    }
    recommendation["recommend_replace_sft"] = all(recommendation.values())
    output={"version":"v0.9.4", "bootstrap_seed":42, "bootstrap_repeats":10_000,
            "autonomous_summary":summaries, "autonomous_paired":autonomous,
            "probe_clustered_paired":probe_stats, "probe_by_condition":probe_conditions,
            "recommendation":recommendation}
    path=ROOT / "v094-final-analysis.json"; path.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
