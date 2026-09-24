"""Run structural and evidence-binding checks on the clean v114 blinded packets."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evidence-sft-v114-clean"
SOURCE = ROOT / "data/hotpot-full-train-v114"


def norm(value):
    return " ".join("".join(c.casefold() if c.isalnum() else " " for c in value).split())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tools(messages):
    documents = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            value = json.loads(message["content"])
        except Exception:
            continue
        for item in value.get("result", []):
            if all(isinstance(item.get(k), str) for k in ("id", "title", "text")):
                documents.append(item)
    return documents


def main():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    gates = json.loads((DATA / "candidate-gates.json").read_text(encoding="utf-8"))
    source_audit = json.loads((DATA / "source-audit.json").read_text(encoding="utf-8"))
    internal_freeze = json.loads((DATA / "internal-freeze.json").read_text(encoding="utf-8"))
    train_freeze = json.loads((DATA / "training-question-freeze.json").read_text(encoding="utf-8"))
    packets = load_jsonl(DATA / "blind-review-candidates.jsonl")
    private = load_jsonl(DATA / "private-key.jsonl")
    private_by_case = {row["case_id"]: row for row in private}

    selected_ids = set(internal_freeze["ids"]) | set(train_freeze["ids"])
    source_rows = {}
    for path in sorted(SOURCE.glob("train-*-of-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if row["id"] in selected_ids:
                    source_rows[row["id"]] = row
    assert set(source_rows) == selected_ids, "frozen IDs missing from official train shards"

    source_facts = {}
    for qid, row in source_rows.items():
        title_sentences = dict(zip(row["context"]["title"], row["context"]["sentences"]))
        source_facts[qid] = [
            (title, title_sentences[title][sent_id].strip())
            for title, sent_id in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"])
            if title in title_sentences and type(sent_id) is int and 0 <= sent_id < len(title_sentences[title])
        ]

    issues = []
    by_split = {"train": [], "internal": []}
    seen_cases = set()
    answer_evidence_failures = 0
    search_support_failures = 0
    ungrounded_queries = 0
    for packet in packets:
        case_id = packet["case_id"]
        key = private_by_case.get(case_id)
        if not key:
            issues.append({"case_id": case_id, "issue": "private_key_missing"})
            continue
        split, qid = key["split"], key["source_id"]
        by_split[split].append((packet, key))
        if case_id in seen_cases:
            issues.append({"case_id": case_id, "issue": "duplicate_case_id"})
        seen_cases.add(case_id)
        if qid not in selected_ids or qid not in source_rows:
            issues.append({"case_id": case_id, "issue": "question_missing_from_source_freeze"})
            continue
        if any(k in packet for k in ("source_id", "split", "gold_answer", "risk_tags", "source_support_titles_private")):
            issues.append({"case_id": case_id, "issue": "blind_packet_contains_private_fields"})
        if any(packet["review"].get(k) is not None for k in (
            "evidence_sufficient_for_action", "action_supported_or_justified", "search_query_grounded",
            "search_returns_relevant_new_evidence", "reviewer_id")):
            issues.append({"case_id": case_id, "issue": "review_was_not_blank_at_build"})
        visible = tools(packet["visible_messages"])
        visible_ids = {doc["id"] for doc in visible}
        if not set(packet["visible_evidence_ids"]).issubset(visible_ids):
            issues.append({"case_id": case_id, "issue": "visible_evidence_reference_invalid"})
        all_cited = visible + packet["retrieval_top3"]
        cited_ids = {doc["id"] for doc in all_cited}
        if not set(packet["evidence_refs"]).issubset(cited_ids):
            issues.append({"case_id": case_id, "issue": "citation_reference_invalid"})

        action = packet["proposed_action"]
        question = packet["question"]
        if action.get("action") == "answer":
            if action.get("answer") != key["gold_answer"]:
                issues.append({"case_id": case_id, "issue": "answer_target_differs_from_source"})
            missing = [(title, quote) for title, quote in source_facts[qid]
                       if not any(doc["title"] == title and quote in doc["text"] for doc in visible)]
            if missing:
                answer_evidence_failures += 1
                issues.append({"case_id": case_id, "issue": "answer_support_not_fully_visible", "missing_fact_count": len(missing)})
        elif action.get("action") == "search":
            query = action.get("query", "")
            input_text = question + " " + " ".join(doc["title"] + " " + doc["text"] for doc in visible)
            if not set(norm(query).split()).issubset(set(norm(input_text).split())):
                ungrounded_queries += 1
                issues.append({"case_id": case_id, "issue": "search_query_not_lexically_grounded"})
            before_ids = {doc["id"] for doc in visible}
            new_results = [doc for doc in packet["retrieval_top3"] if doc["id"] not in before_ids]
            missing_facts = [(title, quote) for title, quote in source_facts[qid]
                             if not any(doc["title"] == title and quote in doc["text"] for doc in visible)]
            useful_new = any(doc["title"] == title and quote in doc["text"]
                             for doc in new_results for title, quote in missing_facts)
            if not useful_new:
                search_support_failures += 1
                issues.append({"case_id": case_id, "issue": "search_top3_does_not_add_missing_support"})
        else:
            issues.append({"case_id": case_id, "issue": "unsupported_action_type"})

    train_ids = {key["source_id"] for _, key in by_split["train"]}
    internal_ids = {key["source_id"] for _, key in by_split["internal"]}
    if train_ids & internal_ids:
        issues.append({"issue": "question_level_split_leakage", "count": len(train_ids & internal_ids)})
    if not train_ids.issubset(set(train_freeze["ids"])) or not internal_ids.issubset(set(internal_freeze["ids"])):
        issues.append({"issue": "candidate_outside_question_freeze"})
    forbidden_paths = [p for p in source_audit.get("historical_artifacts_consulted", [])
                       if any(token in p.casefold() for token in ("final-test", "final_test", "independent100", "holdout"))]
    if forbidden_paths:
        issues.append({"issue": "forbidden_path_in_clean_history_scan", "paths": forbidden_paths})
    if source_audit.get("clean_build_final_test_accessed") is not False:
        issues.append({"issue": "clean_build_scope_not_false"})

    for filename, expected in manifest["files"].items():
        actual = hashlib.sha256((DATA / filename).read_bytes()).hexdigest()
        if actual != expected:
            issues.append({"file": filename, "issue": "manifest_sha256_mismatch"})

    blind = sorted(packets, key=lambda row: hashlib.sha256(("v114-review-a:" + row["case_id"]).encode()).hexdigest())
    blind_b = sorted(packets, key=lambda row: hashlib.sha256(("v114-review-b:" + row["case_id"]).encode()).hexdigest())
    (DATA / "reviewer-a.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in blind), encoding="utf-8")
    (DATA / "reviewer-b.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in blind_b), encoding="utf-8")

    result = {
        "version": "v114-structural-audit", "candidate_count": len(packets),
        "train_candidate_count": len(by_split["train"]), "internal_candidate_count": len(by_split["internal"]),
        "unique_case_ids": len(seen_cases), "train_questions_with_candidates": len(train_ids),
        "internal_questions_with_candidates": len(internal_ids), "question_level_split_disjoint": not bool(train_ids & internal_ids),
        "answer_support_failures": answer_evidence_failures, "search_support_failures": search_support_failures,
        "ungrounded_query_failures": ungrounded_queries, "forbidden_scan_paths": forbidden_paths,
        "independent100_content_read": False, "clean_build_final_test_accessed": False,
        "semantic_review_complete": False, "issues": issues,
        "structural_file_checks_passed": not issues,
        "candidate_gates": gates,
        "note": "Structural checks are not semantic double review or human ground truth.",
    }
    audit_path = DATA / "structural-audit.json"
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["files"]["reviewer-a.jsonl"] = hashlib.sha256((DATA / "reviewer-a.jsonl").read_bytes()).hexdigest()
    manifest["files"]["reviewer-b.jsonl"] = hashlib.sha256((DATA / "reviewer-b.jsonl").read_bytes()).hexdigest()
    manifest["files"]["structural-audit.json"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "candidate_count", "train_candidate_count", "internal_candidate_count", "train_questions_with_candidates",
        "internal_questions_with_candidates", "answer_support_failures", "search_support_failures",
        "ungrounded_query_failures", "forbidden_scan_paths", "structural_file_checks_passed",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
