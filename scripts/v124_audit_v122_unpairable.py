"""CPU-only, read-only replay of v122's 37 unpaired internal source questions.

This diagnoses constructor behavior, not semantic correctness or review approval.
Only the explicitly named v122 candidate files and approved official train shards are read.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_agent_rl.tools.search import _BM25Index  # noqa: E402
from scripts.build_v114_sft_candidates import build_examples, support_facts  # noqa: E402
from scripts.prepare_v116_data import full_rows  # noqa: E402

BASE = ROOT / "data/evidence-sft-v122-internal-candidates"
OUT = ROOT / "data/evidence-sft-v124-forensic/v122-unpairable-diagnostic-private.jsonl"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    freeze = json.loads((BASE / "candidate-freeze.json").read_text(encoding="utf-8"))
    construction = json.loads((BASE / "candidate-construction.json").read_text(encoding="utf-8"))
    corpus_path = BASE / "candidate-corpus-private.json"
    if file_sha(corpus_path) != freeze["corpus_sha256"] or construction["corpus_sha256"] != freeze["corpus_sha256"]:
        raise RuntimeError("frozen_corpus_hash_mismatch")
    ids = construction["unpairable_ids"]
    if len(ids) != 37 or len(set(ids)) != len(ids) or not set(ids) <= set(freeze["ids"]):
        raise RuntimeError("unpairable_id_inventory_changed")
    rows = full_rows(ids)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    index = _BM25Index(corpus)
    output = []
    for ordinal, source_id in enumerate(ids, 1):
        row = rows[source_id]
        examples = build_examples([row], index)
        kinds = Counter(e["state_kind"] for e in examples)
        answer = next((e for e in examples if e["state_kind"] == "supported_answer"), None)
        first_actions = [json.loads(m["content"]) for m in answer["messages"] if m["role"] == "assistant"] if answer else []
        if answer and len(first_actions) == 1 and kinds["nonempty_evidence_search"]:
            reason = "full_evidence_on_one_search_path_no_matching_nonempty_pair"
        elif answer:
            reason = "answer_exists_but_no_matching_nonempty_pair"
        elif not kinds["empty_evidence_search"]:
            reason = "question_queries_found_no_annotated_fact"
        else:
            reason = "no_supported_answer_path_under_frozen_retriever"
        facts = support_facts(row)
        present = [(title, quote) for title, quote in facts if any(d["title"] == title and quote in d["text"] for d in corpus)]
        output.append({
            "ordinal": ordinal, "source_id": source_id, "question": row["question"],
            "original_answer_private": row["answer"], "reason": reason,
            "annotated_fact_count": len(facts), "facts_present_in_frozen_corpus": len(present),
            "generated_empty_searches": kinds["empty_evidence_search"],
            "generated_nonempty_searches": kinds["nonempty_evidence_search"],
            "generated_supported_answers": kinds["supported_answer"],
            "chosen_answer_searches": len(first_actions),
            "chosen_answer_queries": [a["query"] for a in first_actions],
            "status": "structural_diagnosis_only_not_semantically_approved",
            "training_allowed": False,
        })
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"rows": len(output), "reasons": dict(Counter(r["reason"] for r in output)),
                      "missing_source_facts_in_corpus": sum(r["facts_present_in_frozen_corpus"] != r["annotated_fact_count"] for r in output),
                      "sha256": file_sha(OUT), "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
