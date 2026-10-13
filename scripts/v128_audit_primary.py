"""Read-only v128 recheck of the current primary ledger and citation bindings."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/evidence-sft-v124-forensic"
REPORT = OUT / "goal-v128" / "primary-ledger-audit-private.json"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def validate_citations(review: dict, source: dict, retrieved: list[dict]) -> int:
    """Validate nonempty, exact, single-document, action-time citation bindings."""
    action = source.get("proposed_action", {}).get("action")
    if action not in {"search", "answer"}:
        raise ValueError("source_action_invalid")
    citations = review.get("citations")
    if not isinstance(citations, list) or not citations:
        raise ValueError("primary_review_citations_required")
    visible = {
        "before_action": source.get("before_action", []),
        "retrieved_after_search": retrieved,
    }
    checked = 0
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError("citation_must_be_object")
        stage = citation.get("stage")
        quote = citation.get("quote")
        document_id = citation.get("document_id")
        if stage not in visible:
            raise ValueError("citation_stage_invalid")
        if action == "answer" and stage != "before_action":
            raise ValueError("answer_citation_after_action")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("citation_quote_empty")
        matches = [document for document in visible[stage]
                   if document.get("id") == document_id and quote in document.get("text", "")]
        if len(matches) != 1:
            raise ValueError("citation_not_exactly_bound_to_one_document")
        checked += 1
    return checked


def audit() -> dict:
    inventory_path = OUT / "inventory-private.jsonl"
    lineage_path = OUT / "lineage-private.jsonl"
    progress_path = OUT / "progress.json"
    inventory_rows = readl(inventory_path)
    inventory = {row["record_id"]: row for row in inventory_rows}
    if len(inventory) != len(inventory_rows):
        raise ValueError("duplicate_inventory_record_id")
    lineage_rows = readl(lineage_path)
    lineage = {row["record_id"]: row for row in lineage_rows}
    if len(lineage) != len(lineage_rows):
        raise ValueError("duplicate_lineage_record_id")

    batch_paths = sorted(OUT.glob("primary-batch-*.jsonl"))
    decisions = [row for path in batch_paths for row in readl(path)]
    primary = {row["record_id"]: row for row in decisions}
    if len(primary) != len(decisions):
        raise ValueError("duplicate_primary_review_record_id")
    if set(primary) - set(inventory):
        raise ValueError("primary_review_unknown_record_id")

    citations_checked = 0
    disposition_counts = Counter()
    question_groups: dict[str, set[str]] = {}
    for record_id, review in primary.items():
        source = inventory[record_id]
        if review.get("record_sha256") != source.get("record_sha256"):
            raise ValueError("primary_review_source_hash_mismatch")
        if review.get("independent_review") is not False or review.get("approved_training_label") is not False:
            raise ValueError("primary_review_mislabeled_as_independent_approval")
        if review.get("training_allowed") is not False:
            raise ValueError("primary_review_training_flag_invalid")
        matches = lineage.get(record_id, {}).get("matches", [])
        if len(matches) > 1:
            raise ValueError("primary_lineage_not_unique")
        retrieved = (matches[0]["retrieved_after_search"] if matches
                     else source.get("retrieved_after_search", []))
        citations_checked += validate_citations(review, source, retrieved)
        question_groups.setdefault(norm(source["question"]), set()).add(record_id)
        disposition_counts[review["disposition"]] += 1

    effective = dict(primary)
    amendment_paths = sorted(OUT.glob("primary-amendments*-private.jsonl"))
    amendments = [row for path in amendment_paths for row in readl(path)]
    if len({row["record_id"] for row in amendments}) != len(amendments):
        raise ValueError("duplicate_primary_amendment_record_id")
    for change in amendments:
        record_id = change["record_id"]
        if record_id not in primary or change.get("record_sha256") != primary[record_id]["record_sha256"]:
            raise ValueError("primary_amendment_source_mismatch")
        if change.get("superseded_primary_review_sha256") != digest(primary[record_id]):
            raise ValueError("primary_amendment_old_review_hash_mismatch")
        effective[record_id] = {**effective[record_id],
                                "disposition": change["effective_disposition"]}

    override_path = OUT / "primary-adjudication-overrides-private.jsonl"
    overrides = readl(override_path) if override_path.exists() else []
    if len({row["record_id"] for row in overrides}) != len(overrides):
        raise ValueError("duplicate_primary_override_record_id")
    for change in overrides:
        record_id = change["record_id"]
        if record_id not in effective or change.get("record_sha256") != inventory[record_id]["record_sha256"]:
            raise ValueError("primary_override_source_mismatch")
        if change.get("supersedes_effective_disposition") != effective[record_id]["disposition"]:
            raise ValueError("primary_override_chain_mismatch")
        if (change.get("independent_review") is not True or
                change.get("approved_training_label") is not False or
                change.get("training_allowed") is not False or not change.get("basis")):
            raise ValueError("primary_override_metadata_invalid")
        effective[record_id] = {**effective[record_id],
                                "disposition": change["effective_disposition"]}

    progress = json.loads(progress_path.read_text(encoding="utf-8-sig"))
    if progress.get("primary_reviewed") != len(decisions):
        raise ValueError("primary_progress_count_mismatch")
    if progress.get("primary_remaining") != len(inventory) - len(decisions):
        raise ValueError("primary_progress_remaining_mismatch")
    if progress.get("training_allowed") is not False or progress.get("repaired_double_approved") != 0:
        raise ValueError("primary_progress_authorization_mismatch")

    report = {
        "version": "v128-primary-ledger-audit-v1",
        "inputs": {
            "inventory": sha(inventory_path), "lineage": sha(lineage_path),
            "progress": sha(progress_path), "primary_batches": {
                path.relative_to(ROOT).as_posix(): sha(path) for path in batch_paths},
            "amendments": {path.relative_to(ROOT).as_posix(): sha(path) for path in amendment_paths},
            "overrides": sha(override_path) if override_path.exists() else None,
        },
        "inventory_rows": len(inventory_rows), "lineage_rows": len(lineage_rows),
        "primary_rows": len(decisions), "primary_unreviewed_rows": len(inventory) - len(decisions),
        "primary_citations_checked": citations_checked,
        "empty_or_whitespace_citations": 0,
        "citation_source_time_or_document_binding_errors": 0,
        "effective_dispositions": dict(Counter(row["disposition"] for row in effective.values())),
        "amendments": len(amendments), "adjudication_overrides": len(overrides),
        "questions_repeated_across_distinct_state_ids": sum(len(ids) > 1 for ids in question_groups.values()),
        "primary_is_independent_approval": False,
        "training_allowed": False,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = audit()
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"inputs", "effective_dispositions"}}, ensure_ascii=False))
