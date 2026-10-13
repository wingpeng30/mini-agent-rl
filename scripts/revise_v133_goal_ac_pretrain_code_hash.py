"""Record a narrow AC runner correction made before any adapter updates."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import run_v133_goal_ac_failure_state_repair as ac

cfg = ac.z.read(ac.CFG)
state = ac.z.read(ac.STATE)
if state["status"] not in ("awaiting_extra_audit", "awaiting_repair_evidence_review", "ready_to_train"):
    raise RuntimeError(f"Code hash may only be revised before training: {state['status']}")
for seed in (42, 43):
    if (ac.OUT / f"seed{seed}-C").exists():
        raise RuntimeError("A C training directory already exists")
old = cfg["asset_hashes"]["ac_runner"]
new = ac.digest(Path(ac.__file__))
if old == new:
    raise RuntimeError("Runner hash did not change")
path = ac.OUT / "code-revisions.json"
history = ac.rows(path) if path.exists() else []
if history:
    raise RuntimeError("Only the pretraining loss aggregation correction is authorized")
record = {"revision": 1, "old_sha256": old, "new_sha256": new,
    "reason": "Aggregate all corrective search-action token logprobs before the 50/50 repair search-answer blend; previously multiple search actions each received a separate quarter weight.",
    "stage": "before training; no C checkpoint or optimizer update exists",
    "other_frozen_assets_unchanged": True,
    "utility_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
ac.write_jsonl(path, [record])
cfg["asset_hashes"]["ac_runner"] = new
cfg["code_revisions"] = [record]
ac.write(ac.CFG, cfg)
ac.write(ac.OUT / "code-revision-checkpoint.json", {"old_sha256": old, "new_sha256": new,
    "record_sha256": ac.digest(path), "updated_config_sha256": ac.digest(ac.CFG)})
print(json.dumps(record, indent=2))
