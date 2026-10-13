"""Blind, evidence-linked review packet and counted semantic judgments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import run_v133_goal_u_v_evidence_preference as uv

CATEGORIES = ("sufficient", "missing_qualifier", "unsupported_or_wrong", "undetermined")


def files(phase):
    if phase == "baseline":
        return ["incumbent"], False
    if phase == "old":
        return ["incumbent", "42-a", "42-b", "43-a", "43-b"], False
    arm = uv.read(uv.OUT / "old-gate.json")["selected_arm"]
    if arm is None: raise RuntimeError("No fresh candidate")
    return ["incumbent", f"42-{arm}", f"43-{arm}"], True


def records(phase):
    labels, fresh = files(phase)
    cases = {x["case_id"]: x for x in (uv.fresh_cases() if fresh else uv.q.dev_cases())}
    output = []
    for label in labels:
        path = uv.OUT / (f"fresh-{label}.jsonl" if fresh else f"eval-{label}.jsonl")
        for row in uv.rows(path):
            case = cases[row["case_id"]]
            seen = {}
            for step in row["trace"]:
                for document in step.get("observation", {}).get("result", []):
                    seen[document["id"]] = document
            body = {"case_id": row["case_id"], "question": row["question"],
                    "answer": row["answer"] or "", "reason": row["reason"],
                    "visible_documents": [{"id": d["id"], "title": d["title"], "text": d["text"]}
                                          for d in seen.values()]}
            key = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            output.append((label, row["case_id"], case.get("stratum"), key, body))
    return labels, fresh, output


def packet(phase):
    _, _, data = records(phase)
    unique = {key: {"key": key, **body} for _, _, _, key, body in data}
    path = uv.OUT / f"{phase}-blind-packet.jsonl"
    if path.exists(): raise FileExistsError(path)
    with path.open("w", encoding="utf-8") as output:
        for key in sorted(unique):
            output.write(json.dumps(unique[key], ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"packet": str(path), "unique": len(unique), "outputs": len(data)}))


def summarize(phase):
    labels, fresh, data = records(phase)
    votes = uv.rows(uv.OUT / f"{phase}-blind-votes.jsonl")
    indexed = {item["key"]: item for item in votes}
    expected = {key: body for _, _, _, key, body in data}
    if len(votes) != len(indexed) or set(indexed) != set(expected):
        raise RuntimeError("Review votes incomplete, duplicated, or foreign")
    for key, item in indexed.items():
        if item.get("category") not in CATEGORIES or len(item.get("reason", "")) < 8:
            raise RuntimeError(f"Invalid or unexplained vote {key}")
        support_ids = {x["id"] for x in expected[key]["visible_documents"]}
        if not set(item.get("cited_doc_ids", [])).issubset(support_ids):
            raise RuntimeError(f"Vote cites invisible document {key}")
        if item["category"] in ("sufficient", "missing_qualifier", "unsupported_or_wrong") \
           and not item.get("cited_doc_ids"):
            raise RuntimeError(f"Evidence citation required {key}")
        if item["category"] == "unsupported_or_wrong" and item.get("subtype") not in ("one_sided", "other"):
            raise RuntimeError(f"Unsupported judgment needs one-sided/other subtype {key}")
    counts = {label: {**{category: 0 for category in CATEGORIES}, "one_sided_error": 0} for label in labels}
    if fresh:
        for label in labels:
            counts[label]["strata"] = {stratum: {**{category: 0 for category in CATEGORIES}, "one_sided_error": 0}
                                       for stratum in ("one_sided", "ordinary")}
    for label, _, stratum, key, _ in data:
        category = indexed[key]["category"]
        counts[label][category] += 1
        if fresh: counts[label]["strata"][stratum][category] += 1
        if indexed[key].get("subtype") == "one_sided":
            counts[label]["one_sided_error"] += 1
            if fresh: counts[label]["strata"][stratum]["one_sided_error"] += 1
    source_hashes = {label: uv.sha(uv.OUT / (f"fresh-{label}.jsonl" if fresh else f"eval-{label}.jsonl"))
                     for label in labels}
    audit = {"rubric_sha256": uv.sha(uv.RUBRIC), "packet_sha256": uv.sha(uv.OUT / f"{phase}-blind-packet.jsonl"),
             "votes_sha256": uv.sha(uv.OUT / f"{phase}-blind-votes.jsonl"),
             "source_hashes": source_hashes, "models": counts,
             "reviewer": "blind agent evidence review; not independent human gold"}
    if phase == "baseline":
        output = {"source_hash": source_hashes["incumbent"], "counts": counts["incumbent"],
                  "rubric_sha256": audit["rubric_sha256"], "votes_sha256": audit["votes_sha256"]}
        path = uv.OUT / "baseline-semantic-review.json"
    else:
        output = audit
        path = uv.OUT / ("fresh-semantic-review.json" if fresh else "old-semantic-review.json")
    if path.exists(): raise FileExistsError(path)
    uv.write(path, output)
    print(json.dumps({"review": str(path), "counts": counts}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("packet", "summarize"))
    parser.add_argument("phase", choices=("baseline", "old", "fresh"))
    args = parser.parse_args()
    if args.command == "packet": packet(args.phase)
    else: summarize(args.phase)


if __name__ == "__main__": main()
