"""Exercise Goal I's failure-closed CPU gates without running inference/training."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_v133_goal_g_cpu import cpu_check
from transformers import AutoProcessor

script = Path(__file__).with_name("run_v133_goal_i_microfit.py")
spec = importlib.util.spec_from_file_location("goal_i_runner", script)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

training, evaluation, manifest = runner.assert_ready()
assert len(training) == 4 and len(evaluation) == 16
assert {r["case_id"] for r in training}.isdisjoint({r["case_id"] for r in evaluation})
assert {r["question_sha256"] for r in training}.isdisjoint({r["question_sha256"] for r in evaluation})

def must_reject(fn, label):
    try:
        fn()
    except (ValueError, FileNotFoundError, KeyError):
        return
    raise AssertionError(label)

old = runner.H_PROBES_SHA
runner.H_PROBES_SHA = "0" * 64
must_reject(runner.expected_assets, "wrong frozen input hash accepted")
runner.H_PROBES_SHA = old

old = runner.V118_SHA
runner.V118_SHA = "0" * 64
must_reject(runner.expected_assets, "wrong adapter hash accepted")
runner.V118_SHA = old

bad_manifest = copy.deepcopy(manifest)
bad_manifest["eligible_count"] = 1
must_reject(lambda: runner.verify_review_gate(bad_manifest), "incomplete review count accepted")

processor = AutoProcessor.from_pretrained(runner.MODEL, local_files_only=True)
row = copy.deepcopy(training[0])
row["messages"][1]["content"] += "\n" + row["messages"][row["target_assistant_index"]]["content"]
must_reject(lambda: cpu_check(processor, row), "target answer leakage accepted")

row = copy.deepcopy(training[0])
row["cpu_check"]["prompt_sha256"] = "0" * 64
computed = cpu_check(processor, row)
assert computed["prompt_sha256"] != row["cpu_check"]["prompt_sha256"]

source = script.read_text(encoding="utf-8")
assert "subprocess.run(command, cwd=ROOT, timeout=CAP" in source
assert "hard_timeout" in source and "time.monotonic()-started >= CAP" in source
assert "wrong_LoRA_module_count" in source or "lora_modules\"] != 96" in source
print(json.dumps({"status": "preflight_regressions_pass", "review_gate": True,
    "wrong_input_hash_rejected": True, "wrong_adapter_hash_rejected": True,
    "answer_leak_rejected": True, "prompt_hash_mismatch_detected": True,
    "hard_timeout_guard_present": True, "training_cases": len(training),
    "goal_G_eval_cases": len(evaluation)}))
