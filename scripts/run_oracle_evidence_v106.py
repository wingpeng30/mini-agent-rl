"""Evaluate a frozen adapter after supplying the gold training-format evidence history.

This is a diagnostic upper bound on evidence availability. Gold answers and
support-title labels remain in audit fields and are never placed in messages.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from mini_agent_rl.prompts import DIRECT_SYSTEM
from mini_agent_rl.runtime.runner import _query_key
from mini_agent_rl.stop_probe import evaluate


def build_cases(data: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()][:limit]
    cases = []
    for row in rows:
        messages = row["messages"]
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"final message is not assistant: {row['metadata']['task_id']}")
        final = json.loads(messages[-1]["content"])
        if final.get("action") != "answer":
            raise ValueError(f"final message is not answer: {row['metadata']['task_id']}")
        visible = [dict(message) for message in messages[:-1]]
        visible[0] = {"role": "system", "content": DIRECT_SYSTEM}
        queries = []
        visible_titles = []
        for message in visible:
            if message.get("role") == "assistant":
                action = json.loads(message["content"])
                if action.get("action") == "search":
                    queries.append(_query_key(str(action.get("query", ""))))
            elif message.get("role") == "tool":
                payload = json.loads(message["content"])
                visible_titles.extend(str(doc.get("title", "")) for doc in payload.get("result", []))
        wanted = [str(title) for title in row["metadata"].get("support_titles", [])]
        wanted_fold = {title.casefold() for title in wanted}
        found_fold = {title.casefold() for title in visible_titles}
        matched = sorted(wanted_fold & found_fold)
        coverage = len(matched) / len(wanted_fold) if wanted_fold else None
        task_id = str(row["metadata"]["task_id"])
        cases.append({
            "case_id": f"{task_id}:oracle_evidence", "task_id": task_id,
            "condition": "two_hop", "messages": visible,
            "history_queries": queries, "gold_answer": str(row["metadata"]["gold_answer"]),
            "audit": {"history": "frozen SFT evidence history", "support_titles": wanted,
                      "support_count": len(wanted_fold), "matched_titles": matched,
                      "support_coverage": coverage,
                      "coverage_bucket": "complete" if wanted_fold and wanted_fold <= found_fold else ("partial" if matched else "none")},
            "environment": {"source_data_sha256": hashlib.sha256(data.read_bytes()).hexdigest()},
        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    cases = build_cases(args.data, args.limit)
    report = asyncio.run(evaluate(cases, args.output, args.adapter,
                                  max_input_tokens=args.max_input_tokens,
                                  observation_encoding="compact-v3", memory_limit_gb=21.0,
                                  model_path=args.model_path, enable_thinking=False,
                                  max_new_tokens=args.max_new_tokens, generation_timeout=600.0))
    summary = {
        "version": "v106-oracle-evidence", "case_count": len(cases),
        "source_data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "complete_support_cases": sum(case["audit"]["coverage_bucket"] == "complete" for case in cases),
        "report": report,
    }
    (args.output / "v106-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "two_hop": report["by_condition"]["two_hop"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
