"""Post-hoc, label-blind query-policy diagnostics on already exposed v123 IDs."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from scripts.build_v114_sft_candidates import support_facts
from scripts.prepare_v116_data import full_rows
from scripts.v123_cpu_core import OUT as V123_OUT, ROOT, norm, read, sha, write
from mini_agent_rl.tools.search import _BM25Index

REPORT = ROOT / "data/evidence-sft-v124-forensic/goal-v128/g4-query-policy-linked-title-diagnostic-private.json"
TITLE_MENTION = re.compile(r"\b[A-Z][A-Za-z0-9'’.-]*(?:\s+(?:of|the|and|[A-Z][A-Za-z0-9'’.-]*)){0,6}\b")


def linked_title_candidates(question: str, documents: list[dict], title_lookup: dict[str, str],
                            used_queries: set[str]) -> list[str]:
    """Return corpus titles explicitly mentioned in currently visible documents."""
    question_norm = norm(question)
    current_titles = {norm(doc.get("title", "")) for doc in documents}
    found = []
    for rank, document in enumerate(documents):
        for match in TITLE_MENTION.finditer(document.get("text", "")):
            key = norm(match.group())
            title = title_lookup.get(key)
            # Exclude common short one-token accidents and known visible/query entities.
            if (title and len(key.replace(" ", "")) >= 6 and key not in current_titles
                    and key not in question_norm and key not in used_queries):
                found.append((rank, match.start(), -len(key), title))
    return [item[3] for item in sorted(set(found))]


def choose_next(question: str, visible: list[dict], used_queries: set[str], index,
                title_lookup: dict[str, str], excluded_doc_ids: set[str] | None = None
                ) -> tuple[str, list[dict]] | None:
    visible_ids = set(excluded_doc_ids or ()) | {doc["id"] for doc in visible}
    for title in linked_title_candidates(question, visible, title_lookup, used_queries):
        results = index.search(title, top_k=3)
        if any(doc["id"] not in visible_ids for doc in results):
            return title, results
    return None


def run() -> dict:
    freeze = V123_OUT / "internal/candidate-freeze.json"
    corpus_path = V123_OUT / "internal/candidate-corpus-private.json"
    ids = read(freeze)["new_candidate_ids"][:100]
    rows = full_rows(ids)  # Exposed development records; answer field is never accessed.
    corpus = read(corpus_path)
    title_lookup = {norm(doc["title"]): doc["title"] for doc in corpus if doc.get("title")}
    index = _BM25Index(corpus)
    counts = Counter()
    for qid in ids:
        row = rows[qid]
        facts = support_facts(row)  # Used only after the query path has been fixed.
        first = index.search(row["question"], top_k=3)
        evidence = list(first)
        visible = list(first)
        used = {norm(row["question"])}

        def hits() -> set[tuple[str, str]]:
            return {(title, quote) for title, quote in facts
                    if any(doc["title"] == title and quote in doc["text"] for doc in evidence)}

        stage_hits = [hits()]
        searches = 0
        for _ in range(2):  # at most three total searches, including the question query
            follow = choose_next(row["question"], visible, used, index, title_lookup,
                                 excluded_doc_ids={doc["id"] for doc in evidence})
            if follow is None:
                break
            query, results = follow
            searches += 1
            used.add(norm(query))
            visible = results
            seen = {doc["id"] for doc in evidence}
            evidence.extend(doc for doc in results if doc["id"] not in seen)
            stage_hits.append(hits())

        counts["questions"] += 1
        counts["question_grounded_followups_returned_new_document"] += searches >= 1
        counts["third_total_search_executed"] += searches >= 2
        counts["support_facts_after_first_search"] += bool(stage_hits[0])
        if len(stage_hits) > 1:
            counts["followup_added_missing_exact_support_fact"] += bool(stage_hits[1] - stage_hits[0])
            counts["all_exact_support_facts_after_second_total_search"] += bool(set(facts) <= stage_hits[1])
        if len(stage_hits) > 2:
            counts["third_search_added_missing_exact_support_fact"] += bool(stage_hits[2] - stage_hits[1])
        counts["all_exact_support_facts_after_up_to_three_total_searches"] += bool(set(facts) <= stage_hits[-1])
    inputs = [freeze, corpus_path, *sorted((ROOT / "data/hotpot-full-train-v114").glob("train-*.parquet")),
              Path(__file__)]
    return {"version": "v130-linked-title-query-policy-diagnostic-v1",
            "scope": "Exposed v123 new-candidate first 100 IDs only; development diagnostic, not acceptance set.",
            "policy": "After question search, choose the earliest corpus title explicitly mentioned in the latest visible top-3, absent from question/current titles/prior queries; continue only when top-3 adds an unseen document; at most three total searches.",
            "counts": dict(counts),
            "selection_or_query_used_answer_or_support_facts": False,
            "support_facts_used_only_posthoc": True,
            "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
            "new_pool_questions_inspected": 0, "training_allowed": False,
            "data_ready_for_experiment": False}


if __name__ == "__main__":
    result = run()
    write(REPORT, result)
    print(json.dumps({"counts": result["counts"], "report": REPORT.relative_to(ROOT).as_posix()}, ensure_ascii=True))
