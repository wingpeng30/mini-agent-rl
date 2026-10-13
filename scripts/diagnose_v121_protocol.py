"""Classify frozen v121 controller failures without running a model."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def classify(call: dict) -> str:
    raw = call["raw_output"].split("<|im_end|>")[0].strip()
    if not raw:
        return "empty"
    try:
        outer = json.loads(raw)
    except json.JSONDecodeError:
        if raw.startswith('{"action":"answer","answer":"{'):
            return "unescaped_nested_json"
        return "malformed_outer_json"
    if not isinstance(outer, dict):
        return "non_object"
    if outer.get("action") == "search":
        return "old_search_action"
    if outer.get("action") != "answer":
        return "wrong_outer_action"
    payload = outer.get("answer")
    if not isinstance(payload, str):
        return "non_string_answer"
    try:
        inner = json.loads(payload)
    except json.JSONDecodeError:
        return "invalid_inner_json"
    if not isinstance(inner, dict):
        return "non_object_inner_json"
    if call.get("check_valid") or call.get("query_review_valid"):
        return "accepted"
    return "valid_json_rejected_semantically"


def summarize(directory: Path, b_directory: Path | None = None) -> dict:
    report = {}
    for arm in "BCD":
        path = (b_directory if arm == "B" and b_directory else directory) / f"arm-{arm}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        calls = [call for row in rows for call in row["calls"]
                 if call["role"] in {"query_review", "sufficiency_check"}]
        reasons = collections.Counter(classify(call) for call in calls)
        lengths = [len(call["raw_output"].split("<|im_end|>")[0]) for call in calls]
        # The stored record does not include parse_error, output_truncated, or
        # generated token count. Do not infer truncation from character length.
        report[arm] = {
            "controller_calls": len(calls),
            "categories": dict(sorted(reasons.items())),
            "raw_character_length_min_max": [min(lengths), max(lengths)] if lengths else [],
            "truncation_determinable_from_record": False,
            "failure_examples": [
                {"category": category,
                 "raw_prefix": next(c["raw_output"][:240] for c in calls if classify(c) == category)}
                for category in sorted(reasons) if category != "accepted"
            ],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--b-directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory, args.b_directory)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
