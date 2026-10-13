"""Offline paired summary of the frozen Goal N outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-n-v118-grpo"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    evaluations = rows(OUT / "evaluation.jsonl")
    assert len(evaluations) == 144
    lookup = {(row["arm"], row["split"], row["case_id"]): row for row in evaluations}
    paired = {}
    for split in ("train", "new_eval", "goal_G_regression"):
        ids = [row["case_id"] for row in evaluations if row["arm"] == "v118" and row["split"] == split]
        assert len(ids) == len(set(ids)) == 16
        paired[split] = {}
        for baseline in ("v118", "goal_J_sft"):
            gained = [case_id for case_id in ids if lookup[(baseline, split, case_id)]["em"] == 0 and lookup[("goal_N_grpo", split, case_id)]["em"] == 1]
            lost = [case_id for case_id in ids if lookup[(baseline, split, case_id)]["em"] == 1 and lookup[("goal_N_grpo", split, case_id)]["em"] == 0]
            retained = [case_id for case_id in ids if lookup[(baseline, split, case_id)]["em"] == 1 and lookup[("goal_N_grpo", split, case_id)]["em"] == 1]
            paired[split][baseline] = {"gained": gained, "lost": lost, "retained": retained}
    groups = rows(OUT / "train-groups.jsonl")
    assert len(groups) == 16
    updated = [row for row in groups if row["updated"]]
    probe = json.loads((OUT / "probe-summary.json").read_text(encoding="utf-8"))
    summary = {"paired": paired,
               "training": {"updated_groups": len(updated), "zero_variance_groups": len(groups) - len(updated),
                            "peak_reserved_gb": max(row["gpu_memory"]["peak_reserved_gb"] for row in updated),
                            "peak_allocated_gb": max(row["gpu_memory"]["peak_allocated_gb"] for row in updated),
                            "peak_device_process_gb": max(row["gpu_memory"]["device_process_gb"] for row in updated if row["gpu_memory"]["device_process_gb"] is not None),
                            "max_grad_norm_before_clipping": max(row["grad_norm"] for row in updated),
                            "max_first_backward_ratio": max(row["first_backward_ratio"]["max"] for row in updated),
                            "max_initial_ratio": max(row["initial_ratio"]["max"] for row in updated),
                            "mean_group_loss": sum(row["loss"] for row in updated)/len(updated),
                            "mean_group_kl": sum(row["kl"] for row in updated)/len(updated)},
               "probe": {"reward_variable_groups": probe["reward_variable_groups"],
                         "answer_reward_variable_groups": probe["answer_reward_variable_groups"],
                         "answer_correctness_variable_groups": probe["answer_correctness_variable_groups"],
                         "action_counts": probe["action_counts"]}}
    (OUT / "offline-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
