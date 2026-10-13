"""CPU gate for Goal H's frozen, one-shot inference command."""
import importlib.util
import json
from pathlib import Path

path = Path(__file__).with_name("run_v133_goal_h_comparison.py")
spec = importlib.util.spec_from_file_location("goal_h", path)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

evaluation, probes, manifest = h.assert_h0()
assert len(evaluation) == 16 and len(probes) == 4
assert {r["case_id"] for r in evaluation}.isdisjoint({r["case_id"] for r in probes})
assert set(h.ADAPTERS) == {"v118", "sft_c", "sft_d"}
assert h.CAP == 7200
assert all(r["cpu_check"]["prompt_sha256"] for r in evaluation + probes)

original_g_sha = h.G_SHA
h.G_SHA = "0" * 64
try:
    h.assert_h0()
except ValueError as exc:
    assert "frozen_data_changed" in str(exc)
else:
    raise AssertionError("input hash change was not rejected")
h.G_SHA = original_g_sha

original_adapters = h.ADAPTERS
h.ADAPTERS = {**original_adapters, "v118": (original_adapters["v118"][0], "0" * 64)}
try:
    h.assert_h0()
except ValueError as exc:
    assert "adapter_changed:v118" in str(exc)
else:
    raise AssertionError("wrong adapter was not rejected")
h.ADAPTERS = original_adapters

for row in evaluation + probes:
    target = row["target_assistant_index"]
    prompt = row["messages"][:target]
    assert len(prompt) == 6
    assert all(m["role"] != "assistant" or json.loads(m["content"])["action"] == "search"
               for m in prompt if m["role"] == "assistant")
    assert row["messages"][target]["content"] not in [m["content"] for m in prompt]

source = path.read_text(encoding="utf-8")
assert "subprocess.run(command, cwd=ROOT, timeout=CAP" in source
assert "H1_one_shot_run_already_attempted" in source
assert "runtime_input_token_mismatch" in source
assert "runs/v133-goal-c-sft-once/adapter" in source
assert "data/evidence-sft-v133-goal-c-private/frozen/train" not in source
assert "data/evidence-sft-v133-goal-c-private/frozen/dev" not in source
print(json.dumps({"status": "preflight_pass", "main": len(evaluation), "fit_probes": len(probes),
                  "input_hash_rejection": True, "adapter_hash_rejection": True,
                  "answer_target_excluded": True, "hard_timeout_present": True}))
