"""CPU-only Goal Q audit and training curve; never loads a model."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-q-search-success"
CASES = ROOT / "runs/v133-goal-p-system/eval-only-cases.jsonl"


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    from evaluate_v133_goal_c_sft import normalize_answer, f1_score
    gold = {x["case_id"]: x for x in records(CASES) if x["split"] != "train"}
    names = ("goal_j", "old_rl", "direct", "sft")
    values = {}
    for name in names:
        data = records(OUT / f"eval-{name}.jsonl")
        if len(data) != 32 or set(x["case_id"] for x in data) != set(gold):
            raise RuntimeError(f"{name} evaluation incomplete")
        table = {x["case_id"]: x for x in data}
        for case_id, row in table.items():
            answer = row["answer"] or ""
            actual = gold[case_id]["official_answer"]
            em = int(bool(answer) and normalize_answer(answer) == normalize_answer(actual))
            f1 = f1_score(answer, actual) if answer else 0.
            observed = {doc["id"] for step in row["trace"]
                        for doc in step.get("observation", {}).get("result", [])}
            if row["em"] != em or abs(row["f1"] - f1) > 1e-9:
                raise RuntimeError(f"score mismatch {name} {case_id}")
            if row["both_support_retrieved"] != set(gold[case_id]["support_ids"]).issubset(observed):
                raise RuntimeError(f"support mismatch {name} {case_id}")
            if row["model_calls"] > 4 or row["searches"] > 3:
                raise RuntimeError(f"budget exceeded {name} {case_id}")
            for step in row["trace"]:
                if not step["prompt_token_ids"] or not step["action_token_ids"]:
                    raise RuntimeError(f"missing raw tokens {name} {case_id}")
        values[name] = table
    j = values["goal_j"]
    paired = {}
    for name in names[1:]:
        current = values[name]
        paired[name] = {
            "gained": [key for key in gold if current[key]["em"] and not j[key]["em"]],
            "lost": [key for key in gold if j[key]["em"] and not current[key]["em"]],
            "mean_searches": statistics.mean(x["searches"] for x in current.values()),
        }
    result = {"audited_models": list(names), "audited_cases_per_model": 32,
              "all_scores_recomputed": True, "all_support_counts_recomputed": True,
              "paired_vs_goal_j": paired}
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sft = records(OUT / "sft/updates.jsonl")
        rl = records(OUT / "direct/groups.jsonl")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        axes[0].plot(range(1, len(sft)+1), [x["loss"] for x in sft], color="#1565c0")
        axes[0].set(xlabel="SFT update", ylabel="assistant token CE", title="Goal J → trajectory SFT")
        axes[1].plot(range(1, len(rl)+1), [statistics.mean(x["rewards"]) for x in rl],
                     color="#2e7d32", marker=".", linewidth=1)
        axes[1].set(xlabel="RL group", ylabel="mean final EM reward", title="Goal J → online RL")
        fig.tight_layout()
        fig.savefig(OUT / "training-curves.png", dpi=160)
        plt.close(fig)
    except ImportError:
        result["plot"] = "matplotlib_unavailable"
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
