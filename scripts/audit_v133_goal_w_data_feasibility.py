"""Aggregate-only pre-model support audit for frozen Goal W candidate pools."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/v133-goal-w-transfer/data"
OUT = ROOT / "runs/v133-goal-w-transfer/data-only-feasibility-audit.json"
SPLITS = ("train", "dev", "confirm")


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phrase_in(answer: str, text: str) -> bool:
    phrase = norm(answer)
    normalized_text = norm(text)
    return bool(phrase) and f" {phrase} " in f" {normalized_text} "


def run() -> dict:
    frozen_hashes = {}
    report = {"version": "goal-w-data-only-feasibility-audit-v1",
              "method": "normalized official answer whole-phrase presence independently in both referenced support-document texts; no inference or labels",
              "aggregate_only": True, "case_content_emitted": False, "splits": {}}
    for split in SPLITS:
        candidate_path = DATA / f"candidate-{split}.jsonl"
        corpus_path = DATA / f"corpus-{split}.json"
        frozen_hashes[candidate_path.name] = sha(candidate_path)
        frozen_hashes[corpus_path.name] = sha(corpus_path)
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        docs_by_id = {doc["id"]: doc for doc in corpus["documents"]}
        counters = defaultdict(lambda: {"cases": 0, "two_resolved_support_docs": 0,
                                        "answer_phrase_in_both_support_docs": 0,
                                        "answer_phrase_in_at_least_one_support_doc": 0})
        total = {"cases": 0, "two_resolved_support_docs": 0,
                 "answer_phrase_in_both_support_docs": 0,
                 "answer_phrase_in_at_least_one_support_doc": 0}
        for line in candidate_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            category = row["attribute_category"]
            stats = counters[category]
            total["cases"] += 1
            stats["cases"] += 1
            support_ids = row.get("support_ids", [])
            support_docs = [docs_by_id[key] for key in support_ids if key in docs_by_id]
            exactly_two = (len(support_ids) == 2 and len(support_docs) == 2
                           and len({norm(doc["title"]) for doc in support_docs}) == 2)
            if exactly_two:
                stats["two_resolved_support_docs"] += 1
                total["two_resolved_support_docs"] += 1
            supported = [phrase_in(row.get("official_answer", ""), doc["text"]) for doc in support_docs]
            if any(supported):
                stats["answer_phrase_in_at_least_one_support_doc"] += 1
                total["answer_phrase_in_at_least_one_support_doc"] += 1
            if exactly_two and all(supported):
                stats["answer_phrase_in_both_support_docs"] += 1
                total["answer_phrase_in_both_support_docs"] += 1
        report["splits"][split] = {"total": total, "by_category": dict(sorted(counters.items()))}
    report["frozen_input_sha256"] = frozen_hashes
    report["audit_script_sha256"] = sha(Path(__file__))
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {OUT}")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["audit_artifact_sha256"] = sha(OUT)
    return report


if __name__ == "__main__":
    result = run()
    # Only split/category aggregate counts and artifact hashes are printed.
    print(json.dumps({"splits": result["splits"], "audit_script_sha256": result["audit_script_sha256"],
                      "audit_artifact_sha256": result["audit_artifact_sha256"]}, ensure_ascii=False))
