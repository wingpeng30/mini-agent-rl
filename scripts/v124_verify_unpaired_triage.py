"""Verify source binding and status of the 35 remaining unpaired triages."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.prepare_v116_data import full_rows
from scripts.build_v114_sft_candidates import support_facts

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v124-forensic"
FILES = {
    "v122_train_unpaired": BASE / "v122-train-unpaired-semantic-triage-private.jsonl",
    "v123_internal_unpaired": BASE / "v123-unpaired-semantic-triage-private.jsonl",
}


def readl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    diagnostic = readl(BASE / "remaining-unpaired-diagnostic-private.jsonl")
    assert len(diagnostic) == 35
    source = full_rows([r["source_id"] for r in diagnostic])
    triages = {cohort: readl(path) for cohort, path in FILES.items()}
    assert {cohort: len(rows) for cohort, rows in triages.items()} == {
        "v122_train_unpaired": 13, "v123_internal_unpaired": 22}
    assert len({(r["cohort"], r["source_id"]) for r in diagnostic}) == 35
    for cohort, records in triages.items():
        expected = [r for r in diagnostic if r["cohort"] == cohort]
        for ordinal, (d, t) in enumerate(zip(expected, records), 1):
            assert d["ordinal"] == t["ordinal"] == ordinal
            assert d["source_id"] == t["source_id"]
            assert d["question"] == t["question"] == source[d["source_id"]]["question"]
            assert d["original_answer_private"] == t["original_answer_private"] == source[d["source_id"]]["answer"]
            assert t["structural_rejection"] == d["frozen_rejection_reason"]
            facts = [{"title": title, "quote": quote} for title, quote in support_facts(source[d["source_id"]])]
            assert t["source_facts"] == facts
            assert t["source_facts_sha256"] == hashlib.sha256(
                json.dumps(facts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            assert t["disposition"] in {"replay_candidate", "rewrite_question", "exclude_original"}
            assert t["status"] == "primary_semantic_triage_only_not_independent_approval"
            assert t["training_allowed"] is False
    print(json.dumps({
        "verified": sum(map(len, triages.values())),
        "dispositions": {cohort: dict(Counter(r["disposition"] for r in records))
                         for cohort, records in triages.items()},
        "source_facts_bound": True, "training_allowed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
