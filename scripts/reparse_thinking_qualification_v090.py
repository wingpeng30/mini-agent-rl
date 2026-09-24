"""Recompute qualification metrics from preserved raw generations without GPU."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mini_agent_rl.model.local import ActionParseError, parse_agent_action, split_reasoning_action


def parse_raw(raw: str, thinking: bool):
    text = re.sub(r"<\|im_end\|>\s*<\|endoftext\|>\s*$", "", raw).strip()
    text = re.sub(r"<\|(?:im_end|endoftext)\|>\s*$", "", text).strip()
    reasoning, action = split_reasoning_action(text, thinking)
    content, tool = parse_agent_action(action)
    observed = {"kind": "search" if tool else "answer", "value": tool.arguments["query"] if tool else content}
    return reasoning, observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="reports/thinking-sft-v090-qualification")
    args = parser.parse_args()
    directory = Path(args.directory)
    aggregate = []
    for detail in sorted(directory.glob("checkpoint-*.jsonl")):
        records = []
        for line in detail.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            try:
                reasoning, observed = parse_raw(item["raw_text"], item["thinking"])
                item["reparsed_valid"] = True
                item["reparsed_error"] = None
                item["reparsed_observed"] = observed
                item["reparsed_direct_json"] = bool(item["thinking"] and reasoning is None)
            except ActionParseError as exc:
                item["reparsed_valid"] = False
                item["reparsed_error"] = str(exc)
                item["reparsed_observed"] = {"kind": "invalid", "value": None}
                item["reparsed_direct_json"] = False
            records.append(item)
        modes = {}
        for thinking in (False, True):
            rows = [item for item in records if item["thinking"] == thinking]
            answers = [item for item in rows if item["expected"]["kind"] == "answer"]
            modes["thinking" if thinking else "direct"] = {
                "count": len(rows),
                "legal_rate": sum(item["reparsed_valid"] for item in rows) / len(rows),
                "budget_exhaustion_rate": sum(bool(item["usage"].get("output_truncated")) for item in rows) / len(rows),
                "answer_em": sum(item["reparsed_valid"] and item["reparsed_observed"] == item["expected"] for item in answers) / len(answers),
                "valid_search": sum(item["reparsed_valid"] and item["reparsed_observed"]["kind"] == "search" for item in rows),
                "valid_answer": sum(item["reparsed_valid"] and item["reparsed_observed"]["kind"] == "answer" for item in rows),
                "direct_json_degradation": sum(item["reparsed_direct_json"] for item in rows),
            }
        report = {"source": str(detail), "modes": modes}
        output = detail.with_name(detail.stem + "-reparsed.json")
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        aggregate.append(report)
    (directory / "reparsed-summary.json").write_text(json.dumps({"candidates": aggregate}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
