"""Read-only diagnostic for differences from the frozen Goal Z file manifest."""
import json
import run_v133_goal_z_multihop_sft as z

config = json.loads(z.CFG.read_text(encoding="utf-8"))
diffs = []
for name, path in z.frozen_files().items():
    actual = z.sha(path)
    expected = config["files"].get(name)
    if actual != expected:
        diffs.append({"name": name, "expected": expected, "actual": actual,
                      "path": str(path)})
print(json.dumps(diffs, ensure_ascii=False, indent=2))
