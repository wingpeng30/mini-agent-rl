#!/usr/bin/env python3
"""Validate the frozen Goal V fresh evidence set and provenance ledger."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED = {
    "occupation": {"total": 8, "one_sided": 4, "ordinary": 4},
    "nationality_geography": {"total": 8, "one_sided": 4, "ordinary": 4},
    "entity_type": {"total": 8, "one_sided": 4, "ordinary": 4},
    "other": {"total": 8, "one_sided": 4, "ordinary": 4},
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v133-goal-u-v-evidence-preference"))
    parser.add_argument("--old-corpus", type=Path, default=Path("runs/v133-goal-p-system/corpus.json"))
    args = parser.parse_args()

    corpus = json.loads((args.run_dir / "fresh-corpus.json").read_text(encoding="utf-8"))
    cases = read_jsonl(args.run_dir / "fresh-cases.jsonl")
    sources = read_jsonl(args.run_dir / "fresh-sources.jsonl")
    assert len(corpus) == 96, f"expected 96 docs, found {len(corpus)}"
    assert len(cases) == 32, f"expected 32 cases, found {len(cases)}"
    assert len(sources) == 96, f"expected 96 source rows, found {len(sources)}"
    docs = {doc["id"]: doc for doc in corpus}
    source_by_id = {source["doc_id"]: source for source in sources}
    assert len(docs) == len(corpus), "duplicate document ids"
    assert len(source_by_id) == len(sources), "duplicate source doc ids"
    assert set(docs) == set(source_by_id), "corpus/source ID mismatch"
    assert all(doc.get("title") and doc.get("text") for doc in corpus), "blank title or text"
    assert len({norm_title(doc["title"]) for doc in corpus}) == 96, "duplicate fresh titles"

    support_ids = [doc_id for case in cases for doc_id in case["support_ids"]]
    assert len(support_ids) == 64, "expected exactly two support docs per case"
    assert len(set(support_ids)) == 64, "support docs must be unique across all cases"
    assert all(doc_id in docs for doc_id in support_ids), "unknown support doc id"
    support_set = set(support_ids)
    assert len(set(docs) - support_set) == 32, "expected exactly 32 distractors"

    old_titles: set[str] = set()
    if args.old_corpus.exists():
        old_corpus = json.loads(args.old_corpus.read_text(encoding="utf-8"))
        old_titles = {norm_title(doc["title"]) for doc in old_corpus}
        overlap = sorted(doc["title"] for doc in corpus if norm_title(doc["title"]) in old_titles)
        assert not overlap, f"fresh corpus title overlaps old corpus: {overlap}"

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    case_ids: set[str] = set()
    for case in cases:
        assert case["case_id"] not in case_ids, f"duplicate case id {case['case_id']}"
        case_ids.add(case["case_id"])
        assert len(case["support_ids"]) == 2 and case["support_ids"][0] != case["support_ids"][1]
        assert case["official_answer"].strip() and case["question"].strip()
        category, stratum = case["attribute_category"], case["stratum"]
        assert category in EXPECTED, f"unexpected category {category}"
        assert stratum in {"one_sided", "ordinary"}, f"unexpected stratum {stratum}"
        counts[category]["total"] += 1
        counts[category][stratum] += 1
        if stratum == "one_sided":
            bait = case["one_sided_bait"]
            source_id = bait["source_doc_id"]
            assert source_id in case["support_ids"], f"bait source not a support doc: {case['case_id']}"
            other_id = next(doc_id for doc_id in case["support_ids"] if doc_id != source_id)
            phrase = bait["candidate_attribute"].casefold()
            assert phrase in docs[source_id]["text"].casefold(), f"bait not explicit in source: {case['case_id']}"
            assert phrase not in docs[other_id]["text"].casefold(), f"bait also explicit in other source: {case['case_id']}"
            assert bait.get("rationale"), f"missing bait rationale: {case['case_id']}"
        else:
            assert "one_sided_bait" not in case, f"ordinary case has bait: {case['case_id']}"
    assert {key: dict(value) for key, value in counts.items()} == EXPECTED, f"quota mismatch: {counts}"

    for doc_id, source in source_by_id.items():
        expected_role = "support" if doc_id in support_set else "distractor"
        assert source["role"] == expected_role, f"source role mismatch for {doc_id}"
        assert source["source_url"].startswith("https://en.wikipedia.org/wiki/"), f"noncanonical source URL for {doc_id}"
        assert source["source_excerpt"].strip(), f"missing verified excerpt for {doc_id}"
        assert len(source["source_excerpt"].split()) <= 25, f"source excerpt exceeds 25 words: {doc_id}"
        if source.get("revision_id") is None:
            assert source.get("page_id") is None and source.get("source_revision_timestamp") is None
        else:
            assert str(source["revision_id"]).isdigit(), f"invalid revision id for {doc_id}"

    print(json.dumps({
        "documents": len(corpus), "support_documents": len(support_set),
        "distractor_documents": len(docs) - len(support_set), "cases": len(cases),
        "categories": {key: dict(value) for key, value in counts.items()},
        "revision_ids": sum(source.get("revision_id") is not None for source in sources),
        "revision_unavailable": sum(source.get("revision_id") is None for source in sources),
        "old_title_overlaps": 0 if old_titles else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
