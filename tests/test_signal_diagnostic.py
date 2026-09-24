import json

from scripts.analyze_signal_diagnostic_v094 import analyze


def test_signal_diagnostic_separates_semantic_and_protocol_failure_variance(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    rows = []
    def add(task, reward, exact, reason="answer"):
        rows.append({"rollout": {"task_id": task, "reward": reward,
                                  "reward_components": {"exact_match": exact},
                                  "termination_reason": reason}})
    for _ in range(4): add("all-correct", 1.0, 1.0)
    for _ in range(4): add("all-wrong", 0.0, 0.0)
    add("legal-var", 1.0, 1.0); add("legal-var", 0.2, 0.0)
    add("invalid-var", 1.0, 1.0); add("invalid-var", -0.5, 0.0, "invalid_action")
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = analyze(path)
    assert report["classifications"] == {
        "all_correct": 1, "all_wrong": 1,
        "legal_reward_variance": 1, "variance_from_protocol_failure": 1,
    }
