"""Paired 100-task comparison of v096 and v098-answer-v1 GRPO pilots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_grpo_effect_v095 import SEEDS, bootstrap, load_run, load_tasks


def mean(rows: dict[str, dict], key: str) -> float:
    return sum(float(row[key]) for row in rows.values()) / len(rows)


def paired(left: dict[str, dict], right: dict[str, dict], key: str) -> dict:
    ids = sorted(left)
    if ids != sorted(right):
        raise RuntimeError("paired task ids differ")
    return bootstrap([float(right[item][key]) - float(left[item][key]) for item in ids])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/hotpot-agent-v041/validation.jsonl")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tasks = load_tasks(Path(args.data))
    root = Path(args.runs_dir)
    sft, sft_audit = load_run(root / "v099-eval100-sft.db", tasks)
    old, answer, audits = {}, {}, {"sft": sft_audit}
    for seed in SEEDS:
        old[seed], audits[f"v096-{seed}"] = load_run(root / f"v099-eval100-v096-seed{seed}.db", tasks)
        answer[seed], audits[f"answer-{seed}"] = load_run(root / f"v099-eval100-answer-seed{seed}.db", tasks)

    keys = ("strict_exact_match", "token_f1", "invalid_action", "repeated_action",
            "partial_coverage_continue_search", "oversearch_after_complete",
            "empty_result_direct_answer_wrong")
    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {
            "v096": {key: mean(old[seed], key) for key in keys},
            "answer": {key: mean(answer[seed], key) for key in keys},
            "answer_minus_v096": {key: paired(old[seed], answer[seed], key) for key in keys},
            "answer_minus_sft": {key: paired(sft, answer[seed], key) for key in keys},
        }

    ids = sorted(sft)
    cross = {}
    for key in keys:
        cross[key] = bootstrap([
            sum(float(answer[seed][item][key]) - float(old[seed][item][key]) for seed in SEEDS) / len(SEEDS)
            for item in ids
        ])
    old_em = sum(mean(old[seed], "strict_exact_match") for seed in SEEDS) / len(SEEDS)
    answer_em = sum(mean(answer[seed], "strict_exact_match") for seed in SEEDS) / len(SEEDS)
    old_invalid = sum(mean(old[seed], "invalid_action") for seed in SEEDS) / len(SEEDS)
    answer_invalid = sum(mean(answer[seed], "invalid_action") for seed in SEEDS) / len(SEEDS)
    nondegraded = sum(mean(answer[seed], "strict_exact_match") >= mean(old[seed], "strict_exact_match") for seed in SEEDS)
    retain = answer_em >= old_em and nondegraded >= 2 and answer_invalid <= old_invalid
    statistically_supported = cross["strict_exact_match"]["low"] > 0
    payload = {
        "version": "v0.9.9",
        "bootstrap_seed": 42,
        "bootstrap_repeats": 10_000,
        "audits": audits,
        "all_direct_system": all(item["all_direct_system"] for item in audits.values()),
        "per_seed": per_seed,
        "cross_seed_answer_minus_v096": cross,
        "decision": {
            "v096_mean_em": old_em,
            "answer_mean_em": answer_em,
            "v096_mean_invalid": old_invalid,
            "answer_mean_invalid": answer_invalid,
            "answer_nondegraded_seed_count": nondegraded,
            "retain_candidate": retain,
            "statistically_supported_em_improvement": statistically_supported,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    em = cross["strict_exact_match"]
    f1 = cross["token_f1"]
    output.with_suffix(".md").write_text(
        "# v099：答案奖励三 seed 配对结果\n\n"
        f"- v096 平均 EM：{old_em:.4f}\n"
        f"- v098 平均 EM：{answer_em:.4f}\n"
        f"- v098-v096 配对 EM：{em['delta']:.4f}，95% CI [{em['low']:.4f}, {em['high']:.4f}]\n"
        f"- v098-v096 配对 F1：{f1['delta']:.4f}，95% CI [{f1['low']:.4f}, {f1['high']:.4f}]\n"
        f"- 候选不退化 seed 数：{nondegraded}/3\n"
        f"- 保留候选：{retain}\n"
        f"- EM 改善具有统计支持：{statistically_supported}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
