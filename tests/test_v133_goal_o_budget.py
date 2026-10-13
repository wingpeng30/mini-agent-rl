"""Focused checks of Goal O's frozen training boundaries and stop behavior."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("goal_o", ROOT / "scripts/run_v133_goal_o_budget.py")
GOAL_O = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOAL_O)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class GoalOBudgetTests(unittest.TestCase):
    def test_frozen_seed_pass_caps_and_zero_variance(self):
        output = ROOT / "runs/v133-goal-o-budget"
        config = json.loads((output / "config-frozen.json").read_text(encoding="utf-8"))
        self.assertEqual(config["seeds"], [42, 43, 44])
        self.assertEqual(config["passes"], 4)
        self.assertEqual(config["checkpoints_after_passes"], [1, 4])
        for seed in config["seeds"]:
            group = read_jsonl(output / f"seed-{seed}/groups.jsonl")
            rollout = read_jsonl(output / f"seed-{seed}/rollouts.jsonl")
            self.assertEqual(len(group), 64)
            self.assertEqual(len(rollout), 256)
            self.assertEqual([sum(row["pass_index"] == p for row in group) for p in range(1, 5)], [16] * 4)
            self.assertEqual({row["seed"] for row in group + rollout}, {seed})
            self.assertTrue(all(row["updated"] == row["reward_variable"] for row in group))
            self.assertLessEqual(sum(row["updated"] for row in group), 64)

    def test_reference_and_checkpoint_reload(self):
        output = ROOT / "runs/v133-goal-o-budget"
        for seed in (42, 43, 44):
            summary = json.loads((output / f"seed-{seed}/summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["reference_unchanged"] and summary["base_unchanged"])
            self.assertLessEqual(max(summary["reload_max_abs"].values()), 0.001)
            for index in (1, 4):
                checkpoint = json.loads((output / f"seed-{seed}/checkpoint-pass-{index}.json").read_text(encoding="utf-8"))
                self.assertTrue(checkpoint["reference_unchanged"] and checkpoint["base_unchanged"])
                self.assertGreater(checkpoint["changed_lora_parameters"], 0)
                self.assertEqual(GOAL_O.sha(output / f"seed-{seed}/checkpoint-pass-{index}/adapter_model.safetensors"),
                                 checkpoint["adapter_sha256"])

    def test_process_timeout_stops_next_seed_and_preserves_state(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temp:
            temp = Path(temp)
            state_file = temp / "status.json"
            GOAL_O.write(state_file, {"status": "CPU_READY"})
            config = {"seeds": [42, 43, 44], "max_seconds_per_seed": 1800,
                      "max_evaluation_seconds": 1800, "total_gpu_cap_seconds": 7200}
            with (patch.object(GOAL_O, "OUT", temp), patch.object(GOAL_O, "STATE", state_file),
                  patch.object(GOAL_O, "parse_cfg", return_value=config),
                  patch.object(GOAL_O.subprocess, "run", side_effect=subprocess.TimeoutExpired("seed-42", 1800)) as runner):
                with self.assertRaisesRegex(RuntimeError, "seed-42 hard timeout"):
                    GOAL_O.run()
            self.assertEqual(runner.call_count, 1)
            state = json.loads(state_file.read_text(encoding="utf-8"))
            self.assertEqual((state["status"], state["stage"]), ("TECHNICAL_BLOCKED", "seed-42"))


if __name__ == "__main__":
    unittest.main()
