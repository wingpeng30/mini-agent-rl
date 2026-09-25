"""Read-only replay analysis of v112/v118 divergence on the frozen internal 50.

Recorded v112 searches are historical outputs, not actions generated for v118.
The result locates promising intervention states; it is not a counterfactual
end-to-end score and may not be used for training or checkpoint selection.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from scripts.compare_v118_internal import ensure_same
from scripts.merge_v118_semantic_reviews import sha


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def search_steps(row):
    return [step for step in row["transitions"] if (step.get("response") or {}).get("tool_calls")]


def run(args):
    paths = {model: Path(getattr(args, model)) for model in ("v101", "v112", "v118")}
    reports = {model: read(path) for model, path in paths.items()}
    ensure_same(list(reports.values()), "tasks")
    rows = {model: {r["task_id"]: r for r in report["rows"]} for model, report in reports.items()}
    if any(len(v) != 50 for v in rows.values()) or len(set.intersection(*(set(v) for v in rows.values()))) != 50:
        raise ValueError("paired_50_required")
    counters = collections.Counter()
    differences = []
    for task_id, new in rows["v118"].items():
        ref = rows["v112"][task_id]
        if ref["normalized_exact_match"] != 1 or new["normalized_exact_match"] == 1:
            continue
        n_new, n_ref = len(search_steps(new)), len(search_steps(ref))
        same_second = (len(new["transitions"]) > 1 and len(ref["transitions"]) > 1 and
                       new["transitions"][1]["response"]["prompt_token_ids"] ==
                       ref["transitions"][1]["response"]["prompt_token_ids"])
        second = ref["transitions"][1] if len(ref["transitions"]) > 1 else None
        second_search = bool(second and (second.get("response") or {}).get("tool_calls"))
        documents = (second.get("observation") or {}).get("result", []) if second_search else []
        answer_in_second_result = bool(second_search and ref["final_answer"] and any(
            str(ref["final_answer"]).casefold() in str(doc.get("text", "")).casefold() for doc in documents))
        group = f"v118_{n_new}_vs_v112_{n_ref}_searches"
        counters[group] += 1
        if same_second:
            counters["same_second_prompt_token_ids"] += 1
        if answer_in_second_result:
            counters["v112_second_result_contains_its_answer"] += 1
        differences.append({"task_id": task_id, "v118_searches": n_new, "v112_searches": n_ref,
                            "same_second_prompt_token_ids": same_second,
                            "v112_second_action_search": second_search,
                            "v112_second_result_contains_its_answer": answer_in_second_result,
                            "v112_second_query": second["response"]["tool_calls"][0]["arguments"].get("query")
                            if second_search else None})
    result = {"version": "v119-retrieval-opportunity-v1", "diagnostic_only": True,
              "report_sha256": {model: sha(path) for model, path in paths.items()},
              "corpus_sha256": reports["v101"]["corpus_sha256"],
              "v112_correct_v118_wrong": len(differences), "counters": dict(sorted(counters.items())),
              "rows": differences, "independent100_opened": False, "final_test_accessed": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("v101", "v112", "v118", "output"):
        parser.add_argument("--" + key, required=True)
    result = run(parser.parse_args())
    print(json.dumps({"version": result["version"], "counters": result["counters"]}, ensure_ascii=False))
