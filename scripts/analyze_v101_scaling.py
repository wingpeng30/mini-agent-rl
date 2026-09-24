"""Paired comparison of v101 50-task and v100 30-task GRPO models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_grpo_effect_v095 import SEEDS, bootstrap, load_run, load_tasks
from analyze_v100_scaling import KEYS, mean, paired


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
        small[seed], audits[f"v100-seed{seed}"] = load_run(root / f"v100-eval100-seed{seed}.db", tasks)
        large[seed], audits[f"v101-seed{seed}"] = load_run(root / f"v101-eval100-seed{seed}.db", tasks)
    per_seed = {
        str(seed): {
            "v100": {key: mean(small[seed], key) for key in KEYS},
            "v101": {key: mean(large[seed], key) for key in KEYS},
            "v101_minus_v100": {key: paired(small[seed], large[seed], key) for key in KEYS},
        } for seed in SEEDS
    }
    ids = sorted(small[SEEDS[0]])
    cross = {
        key: bootstrap([
            sum(float(large[seed][item][key]) - float(small[seed][item][key]) for seed in SEEDS) / len(SEEDS)
            for item in ids
        ]) for key in KEYS
    }
    cross_sft = {
        key: bootstrap([
            sum(float(large[seed][item][key]) - float(sft[item][key]) for seed in SEEDS) / len(SEEDS)
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
        "version": "v0.10.1", "audits": audits,
        "all_direct_system": all(item["all_direct_system"] for item in audits.values()),
        "per_seed": per_seed, "cross_seed_v101_minus_v100": cross,
        "cross_seed_v101_minus_sft": cross_sft,
        "decision": {"v100_mean_em": small_em, "v101_mean_em": large_em,
                     "v100_mean_invalid": small_invalid, "v101_mean_invalid": large_invalid,
                     "v101_nondegraded_seed_count": nondegraded, "retain_v101": retain,
                     "statistically_supported_scaling_improvement": cross["strict_exact_match"]["low"] > 0},
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    em, f1 = cross["strict_exact_match"], cross["token_f1"]
    output.with_suffix(".md").write_text(
        "# v101：50题相对30题的三 seed 配对结果\n\n"
        f"- 30题平均 EM：{small_em:.4f}\n- 50题平均 EM：{large_em:.4f}\n"
        f"- 配对 EM 差：{em['delta']:.4f}，95% CI [{em['low']:.4f}, {em['high']:.4f}]\n"
        f"- 配对 F1 差：{f1['delta']:.4f}，95% CI [{f1['low']:.4f}, {f1['high']:.4f}]\n"
        f"- 不退化 seed：{nondegraded}/3\n- 保留50题配置：{retain}\n"
        f"- 扩量改善具有统计支持：{cross['strict_exact_match']['low'] > 0}\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
