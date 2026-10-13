"""Diagnose 13 v122 train and 22 v123 internal unpaired source questions.

Uses only explicitly named candidate artifacts and approved official train shards.
This is a structural CPU replay, not independent semantic approval.
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

OUT = ROOT / "data/evidence-sft-v124-forensic/remaining-unpaired-diagnostic-private.jsonl"
COHORTS = [
    ("v122_train_unpaired", ROOT / "data/evidence-sft-v122-train-candidates", "candidate-construction.json", 13),
    ("v123_internal_unpaired", ROOT / "data/evidence-sft-v123/internal", "structural-rejections-private.jsonl", 22),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ids(base: Path, name: str) -> tuple[list[str], dict[str, str]]:
    path = base / name
    if name.endswith("jsonl"):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert all(rec.get("reason") in ("no_pair_in_new_shared_corpus", "packet_invalid:token_limit") for rec in records)
        return [rec["id"] for rec in records], {rec["id"]: rec["reason"] for rec in records}
    ids = json.loads(path.read_text(encoding="utf-8"))["unpairable_ids"]
    return ids, {source_id: "unpairable_in_frozen_construction" for source_id in ids}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    output = []
    for cohort, base, id_file, expected in COHORTS:
        freeze = json.loads((base / "candidate-freeze.json").read_text(encoding="utf-8"))
        corpus_path = base / "candidate-corpus-private.json"
        assert sha(corpus_path) == freeze["corpus_sha256"]
        ids, recorded_reasons = load_ids(base, id_file)
        assert len(ids) == expected and len(set(ids)) == expected
        allowed_ids = set(freeze.get("ids", freeze.get("old_candidate_ids", []) + freeze.get("new_candidate_ids", [])))
        assert set(ids) <= allowed_ids
        source_rows = full_rows(ids)
        assert set(source_rows) == set(ids)
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        index = _BM25Index(corpus)
        for ordinal, source_id in enumerate(ids, 1):
            row = source_rows[source_id]
            examples = build_examples([row], index)
            kinds = Counter(ex["state_kind"] for ex in examples)
            answers = [ex for ex in examples if ex["state_kind"] == "supported_answer"]
            first_actions = []
            for answer in answers:
                first_actions.append([json.loads(m["content"]) for m in answer["messages"] if m["role"] == "assistant"])
            facts = support_facts(row)
            found = [(title, quote) for title, quote in facts
                     if any(d["title"] == title and quote in d["text"] for d in corpus)]
            if answers and kinds["nonempty_evidence_search"]:
                reason = "has_answer_and_search_states_but_no_frozen_pair"
            elif answers:
                reason = "answer_state_only"
            elif kinds["nonempty_evidence_search"] or kinds["empty_evidence_search"]:
                reason = "search_states_without_supported_answer"
            else:
                reason = "no_generated_state"
            output.append({
                "cohort": cohort,
                "ordinal": ordinal,
                "source_id": source_id,
                "question": row["question"],
                "original_answer_private": row["answer"],
                "reason": reason,
                "frozen_rejection_reason": recorded_reasons[source_id],
                "annotated_fact_count": len(facts),
                "facts_present_in_frozen_corpus": len(found),
                "generated_empty_searches": kinds["empty_evidence_search"],
                "generated_nonempty_searches": kinds["nonempty_evidence_search"],
                "generated_supported_answers": kinds["supported_answer"],
                "answer_path_search_counts": [len(a) for a in first_actions],
                "corpus_sha256": freeze["corpus_sha256"],
                "source_list_sha256": sha(base / id_file),
                "status": "structural_diagnosis_only_not_semantically_approved",
                "training_allowed": False,
            })
    assert len(output) == 35 and len({(r["cohort"], r["source_id"]) for r in output}) == 35
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in output), encoding="utf-8")
    print(json.dumps({
        "rows": len(output),
        "reasons_by_cohort": {cohort: dict(Counter(r["reason"] for r in output if r["cohort"] == cohort))
                              for cohort, *_ in COHORTS},
        "missing_annotated_facts": sum(r["facts_present_in_frozen_corpus"] < r["annotated_fact_count"] for r in output),
        "sha256": sha(OUT),
        "training_allowed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
