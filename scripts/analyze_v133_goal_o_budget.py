"""CPU-only audit and plots for the completed, frozen Goal O experiment."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-o-budget"
sys.path.insert(0, str(ROOT / "scripts"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def paired(left, right):
    before = {item["case_id"]: item for item in left}
    after = {item["case_id"]: item for item in right}
    if before.keys() != after.keys() or len(before) != 16:
        raise RuntimeError("paired case IDs differ")
    return {
        "gained": [{"case_id": case, "before_action": before[case]["action"],
                    "before_prediction": before[case]["prediction"],
                    "after_action": after[case]["action"], "after_prediction": after[case]["prediction"]}
                   for case in sorted(before) if not before[case]["em"] and after[case]["em"]],
        "lost": [{"case_id": case, "before_action": before[case]["action"],
                  "before_prediction": before[case]["prediction"],
                  "after_action": after[case]["action"], "after_prediction": after[case]["prediction"]}
                 for case in sorted(before) if before[case]["em"] and not after[case]["em"]],
    }


def main():
    from run_v133_goal_n_v118_grpo import classify, score, PATHS
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = read_json(OUT / "config-frozen.json")
    status = read_json(OUT / "status.json")
    if status["status"] != "EXPERIMENT_COMPLETE":
        raise RuntimeError("experiment did not complete")
    evaluated = read_jsonl(OUT / "evaluation.jsonl")
    summary = read_json(OUT / "evaluation-summary.json")
    if len(evaluated) != 384 or summary["status"] != "PASSED":
        raise RuntimeError("evaluation incomplete")
    gold = {split: {row["case_id"]: row["official_answer"] for row in read_jsonl(path)}
            for split, path in ((key, PATHS[key]) for key in ("train", "new_eval", "goal_G_regression"))}
    by_arm_split = {}
    for item in evaluated:
        arm, split = item["arm"], item["split"]
        kind, answer = classify(item["raw"])
        em, f1, _ = score(kind, answer, gold[split][item["case_id"]], cfg)
        if (kind, answer, em) != (item["action"], item["prediction"], item["em"]) or abs(f1-item["f1"]) > 1e-9:
            raise RuntimeError(f"scoring mismatch: {arm}/{split}/{item['case_id']}")
        by_arm_split.setdefault((arm, split), []).append(item)
    if len(by_arm_split) != 24 or any(len(items) != 16 for items in by_arm_split.values()):
        raise RuntimeError("arm/split shape mismatch")
    for (arm, split), items in by_arm_split.items():
        cell = summary["sets"][arm][split]
        if cell["em"] != sum(item["em"] for item in items) or abs(cell["f1"]-statistics.mean(item["f1"] for item in items)) > 1e-9:
            raise RuntimeError("evaluation summary mismatch")

    seeds = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for seed in cfg["seeds"]:
        root = OUT / f"seed-{seed}"
        run = read_json(root / "summary.json")
        passes = read_jsonl(root / "passes.jsonl")
        groups = read_jsonl(root / "groups.jsonl")
        rollouts = read_jsonl(root / "rollouts.jsonl")
        if (run["status"] != "PASSED" or len(passes) != 4 or len(groups) != 64 or len(rollouts) != 256
                or run["groups"] != 64 or run["updates"] != sum(p["updates"] for p in passes)
                or run["updates"] != sum(g["updated"] for g in groups)
                or run["updates"] > 64):
            raise RuntimeError(f"seed {seed} training cap or shape mismatch")
        if any(g["updated"] == (not g["reward_variable"]) for g in groups):
            raise RuntimeError(f"seed {seed} zero-variance rule mismatch")
        if run["reload_max_abs"] != {"1": 0.0, "4": 0.0} or not run["base_unchanged"] or not run["reference_unchanged"]:
            raise RuntimeError(f"seed {seed} engineering gate mismatch")
        for p in (1, 4):
            checkpoint = read_json(root / f"checkpoint-pass-{p}.json")
            if checkpoint["changed_lora_parameters"] <= 0 or not checkpoint["base_unchanged"] or not checkpoint["reference_unchanged"]:
                raise RuntimeError(f"seed {seed} checkpoint gate mismatch")
        curve = []
        for p in range(1, 5):
            part = [g for g in groups if g["pass_index"] == p]
            if len(part) != 16:
                raise RuntimeError(f"seed {seed} pass incomplete")
            curve.append({"pass": p, "mean_sampled_reward": statistics.mean(g["mean_reward"] for g in part),
                          "effective_updates": sum(g["updated"] for g in part),
                          "reward_variable_groups": sum(g["reward_variable"] for g in part),
                          "mean_kl_updated": statistics.mean(g["kl"] for g in part if g["updated"]),
                          "max_grad_norm": max(g["grad_norm"] for g in part if g["updated"]),
                          "max_reserved_gib": max(g["gpu_memory"]["peak_reserved_gb"] for g in part if g["updated"]),
                          "mean_action_tokens": statistics.mean(g["action_tokens_mean"] for g in part)})
        seeds[str(seed)] = {"training": run, "passes": passes, "curve": curve,
                           "paired": {split: paired(by_arm_split[(f"seed-{seed}-pass-1", split)],
                                                    by_arm_split[(f"seed-{seed}-pass-4", split)])
                                      for split in gold},
                           "vs_v118": {split: paired(by_arm_split[("v118", split)],
                                                      by_arm_split[(f"seed-{seed}-pass-4", split)])
                                        for split in gold},
                           "vs_goal_J": {split: paired(by_arm_split[("goal_J_sft", split)],
                                                        by_arm_split[(f"seed-{seed}-pass-4", split)])
                                         for split in gold}}
        for ax, key, label in zip(axes,
                                  ("mean_sampled_reward", "effective_updates", "mean_kl_updated"),
                                  ("Mean sampled reward", "Effective updates", "Mean KL on updates")):
            ax.plot(range(1, 5), [point[key] for point in curve], marker="o", label=f"seed {seed}")
            ax.set_xticks(range(1, 5)); ax.set_xlabel("Training pass"); ax.set_ylabel(label)
            ax.grid(alpha=.25)
    axes[0].legend()
    fig.savefig(OUT / "training-curves.png", dpi=160)
    plt.close(fig)

    paired_main = [summary["sets"][f"seed-{seed}-pass-4"]["new_eval"]["em"]-
                   summary["sets"][f"seed-{seed}-pass-1"]["new_eval"]["em"] for seed in cfg["seeds"]]
    regression_short = [summary["sets"][f"seed-{seed}-pass-1"]["goal_G_regression"]["em"] for seed in cfg["seeds"]]
    regression_long = [summary["sets"][f"seed-{seed}-pass-4"]["goal_G_regression"]["em"] for seed in cfg["seeds"]]
    main_long = [summary["sets"][f"seed-{seed}-pass-4"]["new_eval"]["em"] for seed in cfg["seeds"]]
    gates = {"all_seeds_engineering_pass": True,
             "main_dev_non_decrease_every_seed": all(delta >= 0 for delta in paired_main),
             "main_dev_increase_at_least_two": sum(delta > 0 for delta in paired_main) >= 2,
             "regression_mean_non_decrease": statistics.mean(regression_long) >= statistics.mean(regression_short),
             "long_main_mean_above_v118": statistics.mean(main_long) > summary["sets"]["v118"]["new_eval"]["em"],
             "long_regression_mean_above_v118": statistics.mean(regression_long) > summary["sets"]["v118"]["goal_G_regression"]["em"]}
    report = {"status": "AUDITED", "rows_rescored": len(evaluated), "gpu_seconds": status["gpu_seconds_used"],
              "evaluation": summary["sets"], "seeds": seeds, "gates": gates,
              "recommend_expand_training_data": all(gates.values()),
              "main_dev_paired_deltas": paired_main,
              "regression_short_mean": statistics.mean(regression_short),
              "regression_long_mean": statistics.mean(regression_long),
              "main_long_mean": statistics.mean(main_long)}
    (OUT / "offline-summary.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"rows_rescored": len(evaluated), "gates": gates,
                      "recommend_expand_training_data": report["recommend_expand_training_data"]}))


if __name__ == "__main__":
    main()
