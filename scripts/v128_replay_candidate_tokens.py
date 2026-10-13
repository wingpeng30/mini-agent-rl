"""CPU-only tokenizer replay for the current non-final train candidates.

Loads only the frozen processor/tokenizer with local_files_only=True. This
module never imports a model class, trainer, inference runner, or remote client.
It does not authorize training and does not inspect sealed evaluation files.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoProcessor

from mini_agent_rl.training.sft import encode_assistant_only
from scripts.prepare_v116_review import PROCESSOR, render

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/evidence-sft-v124-forensic"
INPUTS = (
    OUT / "train-core-candidate-private.jsonl",
    OUT / "double-reviewed-v122-supplement-candidate-private.jsonl",
)
REPORT = OUT / "goal-v128" / "candidate-token-replay-private.json"


def digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def readl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def replay_row(row: dict, processor) -> dict:
    messages = row.get("messages")
    index = row.get("target_assistant_index")
    if (not isinstance(messages, list) or type(index) is not int or
            not 2 <= index < len(messages) or messages[index].get("role") != "assistant"):
        raise ValueError("candidate_target_position_invalid")
    try:
        target = json.loads(messages[index]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("candidate_target_json_invalid") from error
    if not isinstance(target, dict) or target.get("action") not in {"search", "answer"}:
        raise ValueError("candidate_target_action_invalid")
    value = target.get("query" if target["action"] == "search" else "answer")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("candidate_target_value_empty")

    prompt = messages[:index]
    rendered = render(processor, [SimpleNamespace(**message) for message in prompt])
    target_message = {"role": "assistant", "content": messages[index]["content"]}
    encoded = encode_assistant_only(processor, prompt + [target_message], 2176,
                                    len(prompt), False)
    target_tokens = sum(token != -100 for token in encoded["labels"])
    result = {
        "state_sha256": digest(row),
        "action": target["action"],
        "prompt_tokens": len(rendered),
        "target_tokens": target_tokens,
        "sequence_tokens": len(encoded["input_ids"]),
        "prompt_ids_sha256": digest(rendered),
        "target_mask_sha256": digest(encoded["labels"]),
    }
    result["within_limits"] = (result["prompt_tokens"] <= 2048 and
                               result["target_tokens"] <= 128 and
                               result["sequence_tokens"] <= 2176)
    return result


def main() -> dict:
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    inputs = {}
    rows = []
    for path in INPUTS:
        raw = path.read_bytes()
        chunk = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
        rows.extend((path.name, row) for row in chunk)
    if len({digest(row) for _, row in rows}) != len(rows):
        raise ValueError("duplicate_candidate_state")
    replayed = [{"candidate_file": name, **replay_row(row, processor)} for name, row in rows]
    report = {
        "version": "v128-candidate-token-replay-v1",
        "inputs": inputs,
        "processor_local_files_only": True,
        "model_loaded": False,
        "states": len(replayed),
        "actions": {action: sum(row["action"] == action for row in replayed)
                    for action in ("answer", "search")},
        "all_prompts_under_2048": all(row["prompt_tokens"] <= 2048 for row in replayed),
        "all_targets_under_128": all(row["target_tokens"] <= 128 for row in replayed),
        "all_sequences_under_2176": all(row["sequence_tokens"] <= 2176 for row in replayed),
        "row_results": replayed,
        "training_allowed": False,
        "data_ready_for_experiment": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in report.items() if key != "row_results"}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False))
