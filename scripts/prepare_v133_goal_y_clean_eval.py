"""Freeze clean Goal Y evaluation material from Goal W's one-time candidate pools.

This program uses only source metadata and the independent eligibility ledger.
It does not read model outputs. Run `dev` now and `confirm` only after the
development selection gate has passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v133_goal_x_audit_transfer as x

W = ROOT / "runs/v133-goal-w-transfer/data"
LEDGERS = ROOT / "runs/v133-goal-w-transfer/eligibility-audit"
OUT = ROOT / "runs/v133-goal-y-three-class-transfer"
CATEGORIES = ("occupation", "entity_type", "source")
PER_CATEGORY = 16


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def freeze(split: str) -> dict:
    source = W / f"candidate-{split}.jsonl"
    corpus_source = W / f"corpus-{split}.json"
    ledger_path = LEDGERS / f"{split}-eligibility.jsonl"
    candidates = list(x.rows(source))
    ledger = list(x.rows(ledger_path))
    if len(candidates) != 128 or len(ledger) != 128:
        raise RuntimeError("Candidate or eligibility ledger is incomplete")
    if [c["case_id"] for c in candidates] != [v["case_id"] for v in ledger]:
        raise RuntimeError("Eligibility ledger is not in frozen candidate order")
    docs = json.loads(corpus_source.read_text(encoding="utf-8"))["documents"]
    by_id = {d["id"]: d for d in docs}
    if len(by_id) != len(docs):
        raise RuntimeError("Duplicate document ID")
    old_ids, old_questions, old_titles, _, index, _ = x.history_index()
    post = json.loads((x.OUT / "x0-history-exposure-index.json").read_text(encoding="utf-8"))
    old_titles |= set(post["post_g_title_union"]["normalized_titles"])
    selected, counts, rejects = [], Counter(), []
    for case, audit in zip(candidates, ledger):
        category = audit["category"]
        support = case["support_ids"]
        if audit["support_ids"] != support:
            raise RuntimeError(f"Support mapping changed: {case['case_id']}")
        reasons = []
        if audit["eligible"] is not True:
            reasons.append(audit["reason"] or "uncertain_or_ineligible")
        if category not in CATEGORIES:
            reasons.append("outside_three_categories")
        if case["source_id"] in old_ids or x.qhash(case["question"]) in old_questions:
            reasons.append("pre_g_historical_question")
        matches = index.get(("source_id", case["source_id"]), []) + index.get(("question_sha256", x.qhash(case["question"])), [])
        if any(m["tier"] != "metadata_only" for m in matches):
            reasons.append("historically_opened_question")
        if len(support) != 2 or any(i not in by_id for i in support):
            reasons.append("missing_support_document")
        elif any(x.norm(by_id[i]["title"]) in old_titles for i in support):
            reasons.append("historical_support_title")
        if reasons:
            rejects.append({"case_id": case["case_id"], "reasons": sorted(set(reasons)), "ledger_reason": audit["reason"]})
            continue
        if counts[category] < PER_CATEGORY:
            selected.append({**case, "attribute_category": category})
            counts[category] += 1
        else:
            rejects.append({"case_id": case["case_id"], "reasons": ["quota_filled"], "ledger_reason": audit["reason"]})
    if any(counts[k] != PER_CATEGORY for k in CATEGORIES):
        raise RuntimeError(f"Clean {split} category shortfall: {dict(counts)}")
    selected_ids = {d for case in selected for d in case["support_ids"]}
    clean_docs = [d for d in docs if x.norm(d["title"]) not in old_titles]
    if not selected_ids.issubset({d["id"] for d in clean_docs}):
        raise RuntimeError("Clean corpus lost a selected support document")
    if len({x.norm(d["title"]) for d in clean_docs}) != len(clean_docs):
        raise RuntimeError("Clean corpus contains duplicate normalized titles")
    OUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUT / f"{split}-cases.jsonl", selected)
    write_json(OUT / f"{split}-corpus.json", clean_docs)
    write_rows(OUT / f"{split}-exclusions.jsonl", rejects)
    manifest = {"split": split, "category_counts": dict(counts), "cases": len(selected),
                "corpus_documents": len(clean_docs), "source_hashes": {
                    "candidates": digest(source), "corpus": digest(corpus_source), "eligibility": digest(ledger_path),
                    "history": digest(x.OUT / "x0-history-exposure-index.json")},
                "output_hashes": {name: digest(OUT / f"{split}-{name}") for name in
                                  ("cases.jsonl", "corpus.json", "exclusions.jsonl")}}
    write_json(OUT / f"{split}-freeze.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=("dev", "confirm"))
    args = parser.parse_args()
    print(json.dumps(freeze(args.split), ensure_ascii=False, sort_keys=True))
