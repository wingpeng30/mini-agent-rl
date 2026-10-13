"""Block an SFT/evaluation run when targets and runtime actions use different protocols.

Run this on exported SFT JSONL, then on a small recorded model probe before
launching a full experiment. The probe must use the final prompts and settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mini_agent_rl.model.local import ActionParseError, parse_agent_action


def check_sft(path: Path) -> dict:
    counts = {"rows": 0, "search": 0, "answer": 0}
    for number, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row["messages"]
        target_index = row["target_assistant_index"]
        if not isinstance(target_index, int) or target_index < 0 or target_index >= len(messages):
            raise ValueError(f"row_{number}_invalid_target_index")
        target = messages[target_index]
        if target.get("role") != "assistant":
            raise ValueError(f"row_{number}_target_not_assistant")
        try:
            answer, tool = parse_agent_action(target["content"])
        except ActionParseError as exc:
            raise ValueError(f"row_{number}_target_protocol_invalid: {exc}") from exc
        counts["rows"] += 1
        counts["search" if tool else "answer"] += 1
    if not counts["rows"] or not counts["search"] or not counts["answer"]:
        raise ValueError("sft_requires_search_and_answer_actions")
    return counts


def check_probe(path: Path, settings_sha256: str, minimum_rate: float = .98) -> dict:
    """Check exact raw model outputs; parse success alone does not validate semantics."""
    total = valid = 0
    failures = []
    for number, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("settings_sha256") != settings_sha256:
            raise ValueError(f"probe_row_{number}_settings_hash_mismatch")
        if not isinstance(row.get("output_truncated"), bool):
            raise ValueError(f"probe_row_{number}_truncation_flag_missing")
        raw = row["raw_output"].split("<|im_end|>")[0].strip()
        total += 1
        try:
            parse_agent_action(raw)
            if row.get("parse_error"):
                raise ValueError("model_client_parse_error")
            if row.get("output_truncated") is True:
                raise ValueError("truncated")
            valid += 1
        except (ActionParseError, ValueError) as exc:
            failures.append({"row": number, "reason": str(exc)})
    if not total:
        raise ValueError("probe_empty")
    result = {"total": total, "valid": valid, "rate": valid / total, "failures": failures}
    if result["rate"] < minimum_rate:
        raise ValueError(f"probe_protocol_gate_failed: {valid}/{total} < {minimum_rate}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True,
                        help="JSONL raw outputs from the final model, prompt, and decoding settings")
    parser.add_argument("--settings", type=Path, required=True,
                        help="Frozen JSON settings used to generate every probe output")
    args = parser.parse_args()
    settings_sha256 = hashlib.sha256(args.settings.read_bytes()).hexdigest()
    result = {"settings_sha256": settings_sha256, "sft": check_sft(args.sft),
              "probe": check_probe(args.probe, settings_sha256)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
