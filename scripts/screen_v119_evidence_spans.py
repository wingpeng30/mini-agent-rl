"""Exploratory candidate-recall screen on frozen v118 final-answer evidence.

Candidate generation reads only the question-time visible documents. Gold is
loaded afterward for scoring. No candidate is injected into a model or label.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.evaluation import answer_metrics
from scripts.merge_v118_semantic_reviews import sha

PROPER = re.compile(r"(?<!\w)(?:[A-Z][\w’'-]*|[A-Z]{2,})(?:\s+(?:[A-Z][\w’'-]*|[A-Z]{2,}|of|the|and|&|de|von)){0,5}")
DATE = re.compile(r"\b(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{4})\b")


def candidates(documents):
    """Return document-bound spans; do not use answer keys or references."""
    found = {}
    for document in documents:
        doc_id, body = document["id"], document["text"]
        for value in [document["title"], *(m.group().strip() for m in PROPER.finditer(body)),
                      *(m.group() for m in DATE.finditer(body))]:
            if value and (value in body or value == document["title"]):
                found.setdefault(value.casefold(), {"candidate": value, "document_id": doc_id})
    return list(found.values())


def run(args):
    report_path, tasks_path = Path(args.report), Path(args.private_tasks)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tasks = {item["id"]: item["answer"] for item in
             (json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    if report.get("version") != "v118" or report.get("mode") != "tasks" or not report.get("complete_coverage"):
        raise ValueError("frozen_v118_task_report_required")
    if len(report["rows"]) != 50 or set(tasks) != {r["task_id"] for r in report["rows"]}:
        raise ValueError("private_task_binding_mismatch")
    rows = []
    for task in report["rows"]:
        answer_step = next((step for step in reversed(task["transitions"])
                            if step["response"].get("content", "").strip() and not step["response"].get("tool_calls")), None)
        if answer_step is None:
            rows.append({"task_id": task["task_id"], "answered": False,
                         "original_em": bool(task["normalized_exact_match"])})
            continue
        usage = answer_step["response"]["usage"]
        if usage.get("lossy_body_trimmed") or usage.get("history_dropped") or usage.get("documents_deleted"):
            raise ValueError("lossy_final_answer_input_requires_exact_span_replay")
        documents = visible_documents(answer_step["messages"])
        spans = candidates(documents)
        gold = tasks[task["task_id"]]
        rows.append({"task_id": task["task_id"], "answered": True,
                     "original_em": bool(task["normalized_exact_match"]),
                     "gold_verbatim_in_document": any(gold.casefold() in doc["text"].casefold() for doc in documents),
                     "gold_normalized_candidate": any(
                         answer_metrics(span["candidate"], gold)["normalized_exact_match"] == 1 for span in spans),
                     "candidate_count": len(spans), "document_count": len(documents),
                     "prompt_truncated_losslessly": bool(usage.get("prompt_truncated"))})
    wrong = [row for row in rows if not row["original_em"]]
    wrong_answered = [row for row in wrong if row["answered"]]
    result = {"version": "v119-evidence-span-screen-v1", "exploratory_development_only": True,
              "report_sha256": sha(report_path), "private_tasks_sha256": sha(tasks_path),
              "wrong_tasks": len(wrong), "wrong_with_answer": len(wrong_answered),
              "gold_verbatim_visible_among_wrong_answers": sum(row["gold_verbatim_in_document"] for row in wrong_answered),
              "gold_in_span_candidates_among_wrong_answers": sum(row["gold_normalized_candidate"] for row in wrong_answered),
              "median_candidate_count_among_wrong_answers": statistics.median(row["candidate_count"] for row in wrong_answered),
              "max_candidate_count_among_wrong_answers": max(row["candidate_count"] for row in wrong_answered),
              "rows": rows, "independent100_opened": False, "final_test_accessed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("report", "private-tasks", "output"):
        parser.add_argument("--" + key, required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "report_sha256", "private_tasks_sha256"}},
                     ensure_ascii=False))
