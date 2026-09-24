"""Paired comparison of v100 30-task and v099 10-task GRPO models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_grpo_effect_v095 import SEEDS, bootstrap, load_run, load_tasks


KEYS = ("strict_exact_match", "token_f1", "invalid_action", "repeated_action",
        "partial_coverage_continue_search", "oversearch_after_complete",
        "empty_result_direct_answer_wrong")


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
    small, large, audits = {}, {}, {"sft": sft_audit}
    for seed in SEEDS:
        small[seed], audits[f"v099-seed{seed}"] = load_run(root / f"v099-eval100-answer-seed{seed}.db", tasks)
        large[seed], audits[f"v100-seed{seed}"] = load_run(root / f"v100-eval100-seed{seed}.db", tasks)

    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {
            "v099": {key: mean(small[seed], key) for key in KEYS},
            "v100": {key: mean(large[seed], key) for key in KEYS},
            "v100_minus_v099": {key: paired(small[seed], large[seed], key) for key in KEYS},
            "v100_minus_sft": {key: paired(sft, large[seed], key) for key in KEYS},
        }
    ids = sorted(sft)
    cross = {
        key: bootstrap([
            sum(float(large[seed][item][key]) - float(small[seed][item][key]) for seed in SEEDS) / len(SEEDS)
            for item in ids
        ]) for key in KEYS
    }
    small_em = sum(mean(small[seed], "strict_exact_match") for seed in SEEDS) / len(SEEDS)
    large_em = sum(mean(large[seed], "strict_exact_match") for seed in SEEDS) / len(SEEDS)
    small_invalid = sum(mean(small[seed], "invalid_action") for seed in SEEDS) / len(SEEDS)
    large_invalid = sum(mean(large[seed], "invalid_action") for seed in SEEDS) / len(SEEDS)
    nondegraded = sum(mean(large[seed], "strict_exact_match") >= mean(small[seed], "strict_exact_match") for seed in SEEDS)
    retain = large_em >= small_em and nondegraded >= 2 and large_invalid <= small_invalid
    payload = {
        "version": "v0.10.0",
        "audits": audits,
        "all_direct_system": all(item["all_direct_system"] for item in audits.values()),
        "per_seed": per_seed,
        "cross_seed_v100_minus_v099": cross,
        "decision": {
            "v099_mean_em": small_em,
            "v100_mean_em": large_em,
            "v099_mean_invalid": small_invalid,
            "v100_mean_invalid": large_invalid,
            "v100_nondegraded_seed_count": nondegraded,
            "retain_v100": retain,
            "statistically_supported_scaling_improvement": cross["strict_exact_match"]["low"] > 0,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    em, f1 = cross["strict_exact_match"], cross["token_f1"]
    output.with_suffix(".md").write_text(
        "# v100：30题相对10题的三 seed 配对结果\n\n"
        f"- 10题平均 EM：{small_em:.4f}\n"
        f"- 30题平均 EM：{large_em:.4f}\n"
        f"- 配对 EM 差：{em['delta']:.4f}，95% CI [{em['low']:.4f}, {em['high']:.4f}]\n"
        f"- 配对 F1 差：{f1['delta']:.4f}，95% CI [{f1['low']:.4f}, {f1['high']:.4f}]\n"
        f"- 不退化 seed：{nondegraded}/3\n"
        f"- 保留30题配置：{retain}\n"
        f"- 扩量改善具有统计支持：{cross['strict_exact_match']['low'] > 0}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
