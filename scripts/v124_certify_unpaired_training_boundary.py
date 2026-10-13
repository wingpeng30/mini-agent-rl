"""Bind the 35 unpaired-question triages to the frozen candidate diagnostics."""
from __future__ import annotations

import json
from collections import Counter

from scripts.v123_cpu_core import digest, readl, write
from scripts.v124_forensic import OUT

REPORT = OUT / "unpaired-training-boundary-private.json"


def main() -> None:
    diagnostic = readl(OUT / "remaining-unpaired-diagnostic-private.jsonl")
    triage_files = {
        "v122_train_unpaired": OUT / "v122-train-unpaired-semantic-triage-private.jsonl",
        "v123_internal_unpaired": OUT / "v123-unpaired-semantic-triage-private.jsonl",
    }
    if len(diagnostic) != 35 or len({(r["cohort"], r["source_id"]) for r in diagnostic}) != 35:
        raise RuntimeError("unpaired_diagnostic_coverage_mismatch")
    records = []
    by_cohort = {}
    for cohort, path in triage_files.items():
        source = [r for r in diagnostic if r["cohort"] == cohort]
        triage = readl(path)
        if len(source) != len(triage):
            raise RuntimeError("unpaired_triage_coverage_mismatch:" + cohort)
        counts = Counter()
        for d, t in zip(source, triage):
            if (d["source_id"] != t["source_id"] or d["ordinal"] != t["ordinal"] or
                    d["question"] != t["question"] or
                    d["frozen_rejection_reason"] != t["structural_rejection"] or
                    t["status"] != "primary_semantic_triage_only_not_independent_approval" or
                    t["training_allowed"] is not False):
                raise RuntimeError("unpaired_record_binding_mismatch")
            disposition = t["disposition"]
            counts[disposition] += 1
            records.append({"cohort": cohort, "ordinal": t["ordinal"],
                "source_id": t["source_id"], "source_id_sha256": digest(t["source_id"]),
                "question_sha256": digest(t["question"]),
                "frozen_structural_rejection": d["frozen_rejection_reason"],
                "triage_disposition": disposition,
                "status": "primary_triage_only_no_independent_approval",
                "training_allowed": False})
        by_cohort[cohort] = {"rows": len(triage), "dispositions": dict(counts)}
    if sum(len(readl(p)) for p in triage_files.values()) != len(diagnostic):
        raise RuntimeError("unpaired_total_mismatch")
    report = {
        "version": "v124-unpaired-training-boundary-v1",
        "diagnostic_sha256": digest(diagnostic),
        "triage_files": {p.name: digest(readl(p)) for p in triage_files.values()},
        "total_questions": len(diagnostic),
        "by_cohort": by_cohort,
        "overall_dispositions": dict(Counter(r["triage_disposition"] for r in records)),
        "complete_pairs_created": 0,
        "independent_reviews_added": 0,
        "policy": {
            "exclude_original": "exclude frozen wording from training; retain source record and rationale",
            "replay_candidate": "do not synthesize a teacher query in CPU-only audit; requires an authorized model-generated state and fresh review",
            "rewrite_question": "keep revised wording as a proposal only; all group states require replay and independent review",
            "v123_internal_unpaired": "keep separate from training and strategy tuning to preserve its internal-set role",
        },
        "records": records,
        "training_allowed": False,
    }
    write(REPORT, report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
