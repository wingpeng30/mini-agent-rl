"""Clustered paired probe analysis for v101 50-task versus v100 30-task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_grpo_effect_v095 import SEEDS, bootstrap
from analyze_v100_probe import METRICS, load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reports")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); root = Path(args.root)
    small, large, hashes = {}, {}, []
    for seed in SEEDS:
        small[seed], h0 = load(root / f"v100-probe120-seed{seed}")
        large[seed], h1 = load(root / f"v101-probe120-seed{seed}")
        hashes.extend((h0, h1))
    conditions = sorted(next(iter(next(iter(small.values())).values())).keys())
    cross = {}
    for condition in conditions:
        cross[condition] = {
            metric: bootstrap([
                sum(float(large[seed][task][condition][metric]) - float(small[seed][task][condition][metric]) for seed in SEEDS) / len(SEEDS)
                for task in sorted(small[SEEDS[0]])
            ]) for metric in METRICS
        }
    overall = {
        metric: bootstrap([
            sum(float(large[seed][task][condition][metric]) - float(small[seed][task][condition][metric])
                for seed in SEEDS for condition in conditions) / (len(SEEDS) * len(conditions))
            for task in sorted(small[SEEDS[0]])
        ]) for metric in METRICS
    }
    payload = {"version": "v0.10.1", "cluster_unit": "30 original questions",
               "input_non_system_messages_identical": len(set(hashes)) == 1,
               "cross_seed_by_condition": cross, "cross_seed_overall": overall}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
