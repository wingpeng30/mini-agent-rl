"""Summarize what isolated repair/unpaired records can safely do next."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from scripts.v123_cpu_core import ROOT, readl, write
from scripts.v124_forensic import OUT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    sources = {
        "snapshot": OUT / "current-trainability-snapshot-private.json",
        "repair_links": OUT / "v125-v117-repair-trajectory-link-audit-private.json",
        "repair_luna": OUT / "v125-v117-repair-luna-validation-v2-private.json",
        "unpaired_boundary": OUT / "unpaired-training-boundary-private.json",
        "quarantine": OUT / "quarantine-processing-audit-private.json",
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    snapshot = json.loads(sources["snapshot"].read_text(encoding="utf-8"))
    links = json.loads(sources["repair_links"].read_text(encoding="utf-8"))
    luna = json.loads(sources["repair_luna"].read_text(encoding="utf-8"))
    unpaired = json.loads(sources["unpaired_boundary"].read_text(encoding="utf-8"))
    quarantine = json.loads(sources["quarantine"].read_text(encoding="utf-8"))

    triage_paths = [
        OUT / "v122-train-unpaired-semantic-triage-private.jsonl",
        OUT / "v123-unpaired-semantic-triage-private.jsonl",
    ]
    triage = {}
    for path in triage_paths:
        rows = readl(path)
        name = "v122_train" if "v122-" in path.name else "v123_internal"
        triage[name] = {
            "rows": len(rows),
            "dispositions": dict(Counter(r["disposition"] for r in rows)),
            "training_allowed": False,
        }

    report = {
        "version": "v125-isolated-data-processability-v1",
        "inputs": {str(path.relative_to(ROOT)): sha(path) for path in
                   [*sources.values(), *triage_paths]},
        "repair_replay": {
            "strict_action_result_citation_links": links["strict_action_result_citation_links"],
            "repair_pair_candidates": links["new_repair_pair_candidates_before_double_review"],
            "quarantined_strict_link_candidates": links["strict_link_candidates_still_quarantined"],
            "luna_reviewed_states": luna["reviewed_states"],
            "luna_mechanically_valid_states": luna["valid_states"],
            "luna_invalid_states": luna["invalid_states"],
            "luna_invalid_reasons": luna["invalid_reason_counts"],
            "luna_approved_complete_pairs": sum(
                row["luna_approved_pair"] for row in luna["pair_checks_private"]),
            "deepseek_states_reviewed": 0,
            "new_semantically_double_approved_pairs": 0,
            "training_allowed": False,
        },
        "unpaired_questions": {
            "total": unpaired["total_questions"],
            "by_cohort": triage,
            "existing_complete_pairs_created": unpaired["complete_pairs_created"],
            "independent_reviews_added": unpaired["independent_reviews_added"],
            "v123_internal_remains_excluded_from_training_and_strategy_tuning": True,
            "training_allowed": False,
        },
        "quarantine": {
            "states": quarantine["quarantine_states"],
            "by_cohort": quarantine["quarantine_by_cohort"],
            "source_rechecked": quarantine["source_rechecked_states"],
            "source_rechecked_still_held": quarantine["source_rechecked_still_held"],
            "training_allowed": False,
        },
        "safe_next_operations": [
            "CPU replay, tokenizer checks, source lineage checks, deduplication, and anonymous review packaging are allowed when inputs are hash-bound.",
            "A rewritten state must be a new version; rerun its full trajectory and bind each action to only the evidence visible at that time.",
            "An unpaired search state cannot be paired by writing a teacher query from gold or hindsight; it requires an authorized model-generated state and fresh review.",
            "A revised training candidate needs valid independent semantic approvals for both states; a one-sided vote or invalid-format vote adds zero training labels.",
            "Exposed v123 internal rows may be audited for the internal-set decision but stay out of training and strategy tuning.",
            "Quarantine is a reversible hold, not deletion; unresolved premise, source, time, ambiguity, or contamination issues remain excluded.",
        ],
        "training_allowed": False,
    }
    path = OUT / "v125-isolated-data-processability-private.json"
    if path.exists():
        raise FileExistsError(path)
    write(path, report)
    print(json.dumps({
        "version": report["version"],
        "repair_replay": report["repair_replay"],
        "unpaired_questions": report["unpaired_questions"],
        "quarantine": report["quarantine"],
        "training_allowed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
