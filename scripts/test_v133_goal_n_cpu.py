"""Check Goal N scoring against all frozen Goal J paired trajectories."""
from pathlib import Path

from run_v133_goal_n_v118_grpo import J, OUT, classify, read_rows, score, write_json


def main():
    traces = read_rows(J / "trajectories.jsonl")
    assert len(traces) == 96
    config = {"reward_em": 1.0, "reward_f1": 0.25}
    actions = {"answer": 0, "search": 0, "invalid": 0}
    for row in traces:
        kind, answer = classify(row["raw_output"])
        em, f1, _ = score(kind, answer, row["gold"], config)
        assert em == row["em"], (row["case_id"], row["arm"], "EM")
        assert abs(f1 - row["f1"]) <= 1e-9, (row["case_id"], row["arm"], "F1")
        actions[kind] += 1
    write_json(OUT / "scoring-regression.json", {"status": "pass", "rows": len(traces), "actions": actions})
    print({"status": "pass", "rows": len(traces), "actions": actions})


if __name__ == "__main__":
    main()
