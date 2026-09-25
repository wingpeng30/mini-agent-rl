"""Checkpointed, blind DeepSeek audit of v122 fixed-state action traces.

Only question, decision-time visible evidence, anonymous output, and retrieval
snippets are transmitted. Gold, scores, task IDs and arm labels stay local.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from mini_agent_rl.training.thinking_sft import _env
from scripts.audit_v121_arms_api import digest, make_trace, review_packet


def packets(report_dir: Path, tasks: dict) -> list[dict]:
    reports = {arm: json.loads((report_dir / f"fixed-{arm}.json").read_text(encoding="utf-8")) for arm in "ABCD"}
    identity = {report["settings_sha256"] for report in reports.values()}
    if len(identity) != 1:
        raise ValueError("fixed_reports_settings_mismatch")
    result = []
    for arm, report in reports.items():
        rows = {row["task_id"]: row for row in report["rows"]}
        if len(rows) != 50 or set(rows) != set(tasks) or report.get("complete_coverage") is not True:
            raise ValueError(f"fixed_{arm}_coverage_invalid")
        for task_id, row in sorted(rows.items()):
            question = str(tasks[task_id]["question"])
            case_id = digest("v122:" + arm + ":" + question)[:24]
            trace_id = "trace-" + digest(case_id + ":trace")[:12]
            result.append({"case_id":case_id,"question":question,
                           "traces":[make_trace(row, trace_id)],
                           "evidence_note":"Decision excerpts are the actual post-truncation evidence. Retrieval results are separate and cannot justify an earlier decision."})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir",type=Path,required=True)
    parser.add_argument("--tasks",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args()
    tasks = {row["id"]:row for row in (json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip())}
    prepared = packets(args.report_dir,tasks)
    packet_hash = hashlib.sha256(json.dumps(prepared,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps({"packets":len(prepared),"packet_sha256":packet_hash,
                          "gold_score_task_id_or_arm_sent":False},ensure_ascii=False))
        return
    config = _env(Path(__file__).resolve().parents[2] / ".env")
    endpoint = config.get("DEEPSEEK_API_BASE","").rstrip("/")
    if endpoint != "https://api.deepseek.com" or not config.get("DEEPSEEK_API_KEY"):
        raise ValueError("authorized_deepseek_endpoint_or_key_missing")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    existing = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()] if args.output.exists() else []
    if len({row["case_id"] for row in existing}) != len(existing):
        raise ValueError("duplicate_api_reviews")
    by_id = {packet["case_id"]:packet for packet in prepared}
    if any(row["case_id"] not in by_id or row.get("packet_sha256") !=
           digest(json.dumps(by_id[row["case_id"]],ensure_ascii=False,sort_keys=True)) for row in existing):
        raise ValueError("resume_packet_hash_mismatch")
    done = {row["case_id"] for row in existing}
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(review_packet,packet,endpoint,config):packet for packet in prepared if packet["case_id"] not in done}
        for future in as_completed(pending):
            record = future.result()
            with args.output.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+"\n")
                stream.flush()
            print(record["case_id"],"audited",flush=True)


if __name__ == "__main__":
    main()
