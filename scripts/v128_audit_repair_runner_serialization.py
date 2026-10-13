"""Compare the one structural repair link with available runner serializers.

This bounded audit reads only the named replay record and named source files;
it never reconstructs or exports a model-visible answer context.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "data/evidence-sft-v124-forensic/repair-draft-states-cpu-v2-private.jsonl"
CURRENT_RUNNER = ROOT / "src/mini_agent_rl/runtime/runner.py"
HISTORIC_COMMIT = "11260f2896bf1be8e747f3b0eb73fb468fa5b050"
OUTPUT = ROOT / "data/evidence-sft-v124-forensic/goal-v128/g2-runner-serialization-audit-private.json"


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def audit() -> dict:
    raw = REPLAY.read_bytes()
    found = []
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        group = json.loads(line)
        for state in group.get("states", []):
            if state.get("record_id") == "v117_train:0139":
                found.append(state)
    if len(found) != 1:
        raise ValueError("named_answer_record_not_unique")
    answer = found[0]
    messages = answer["visible_messages"]
    tool_shapes = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        payload = json.loads(message["content"])
        result = payload.get("result")
        tool_shapes.append({
            "top_level_keys": sorted(payload),
            "result_is_list": isinstance(result, list),
            "result_item_keys": sorted({tuple(sorted(item)) for item in result
                                         if isinstance(item, dict)}) if isinstance(result, list) else [],
        })
    current_bytes = CURRENT_RUNNER.read_bytes()
    current_text = current_bytes.decode("utf-8")
    current_has_searched_queries = 'observation["searched_queries"]' in current_text
    old = subprocess.run(
        ["git", "show", f"{HISTORIC_COMMIT}:src/mini_agent_rl/runtime/runner.py"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    old_text = old.decode("utf-8")
    historical_has_searched_queries = 'observation["searched_queries"]' in old_text
    return {
        "version": "v128-repair-runner-serialization-audit-v1",
        "scope": ["v117_train:0139", "current tracked runner.py",
                  f"{HISTORIC_COMMIT}:src/mini_agent_rl/runtime/runner.py"],
        "replay_input_sha256": sha(raw),
        "answer_record_visible_messages_sha256": sha(json.dumps(
            messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")),
        "historical_answer_message_count": len(messages),
        "historical_tool_message_shapes": tool_shapes,
        "current_runner_sha256": sha(current_bytes),
        "current_runner_adds_searched_queries": current_has_searched_queries,
        "comparison_commit": HISTORIC_COMMIT,
        "comparison_runner_sha256": sha(old),
        "comparison_runner_adds_searched_queries": historical_has_searched_queries,
        "v117_runtime_commit_bound": False,
        "answer_context_materialized": False,
        "semantic_review_performed": False,
        "training_allowed": False,
        "conclusion": (
            "The saved answer context uses an observed tool-payload shape without searched_queries, "
            "while both available runner implementations add searched_queries. The comparison commit "
            "is not bound to the v117 artifact; therefore neither serializer can be asserted as the "
            "historical v117 contract. Do not synthesize the changed-query observation from this evidence."
        ),
    }


if __name__ == "__main__":
    result = audit()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "historical_answer_message_count", "historical_tool_message_shapes",
        "current_runner_adds_searched_queries", "comparison_runner_adds_searched_queries",
        "v117_runtime_commit_bound", "answer_context_materialized", "training_allowed",
    )}, ensure_ascii=False))
