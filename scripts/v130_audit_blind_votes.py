"""Mechanically reconcile frozen v130 blind votes, the single citation correction, and API state."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

from scripts.v123_cpu_core import ROOT, sha, write
from scripts.v129_review_protocol import valid_vote

BASE = ROOT / "data/evidence-sft-v124-forensic/goal-v128"
AUDIT = BASE / "new-internal-audit"
REPORT = BASE / "g4-blind-review-disposition-v130-private.json"


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [(line, json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hash_vote_line(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _saved_vote_count(vote_file: dict) -> int:
    reviews = vote_file.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("saved_vote_reviews_not_array")
    return len(reviews)


def audit() -> dict:
    raw_votes = {}
    packets_by_id = {}
    batch_records = []
    for number in range(1, 11):
        batch = f"g4-batch-{number:02d}"
        bundle_path = AUDIT / "packets" / f"{batch}.json"
        vote_path = AUDIT / "luna" / f"{batch}-votes.jsonl"
        bundle = _read_json(bundle_path)
        packet_ids = {packet["case_id"] for packet in bundle["packets"]}
        rows = _read_jsonl(vote_path)
        votes = {vote["case_id"]: (line, vote) for line, vote in rows}
        if len(bundle["packets"]) != 20 or len(rows) != 20 or set(votes) != packet_ids:
            raise ValueError(f"luna_batch_coverage_mismatch:{batch}")
        for packet in bundle["packets"]:
            if packet["case_id"] in packets_by_id:
                raise ValueError("duplicate_case_id_across_batches")
            packets_by_id[packet["case_id"]] = packet
            raw_votes[packet["case_id"]] = votes[packet["case_id"]]
        batch_records.append({"batch": batch, "states": 20,
                              "packet_sha256": sha(bundle_path), "raw_vote_sha256": sha(vote_path)})

    correction_dir = AUDIT / "luna/corrections"
    ledger_rows = _read_jsonl(correction_dir / "citation-corrections-ledger.jsonl")
    ledger = {row["case_id"]: row for _, row in ledger_rows}
    if len(ledger) != len(ledger_rows):
        raise ValueError("duplicate_citation_correction_ledger_id")
    corrected = {}
    correction_files = {}
    for number in range(1, 11):
        batch = f"g4-batch-{number:02d}"
        path = correction_dir / f"{batch}-citation-corrections.jsonl"
        rows = _read_jsonl(path)
        correction_files[batch] = {vote["case_id"]: vote for _, vote in rows}

    invalid_before = Counter()
    invalid_after = Counter()
    approved_after = Counter()
    denied_after = Counter()
    all_labels = Counter()
    corrected_ids = []
    uncorrectable_ids = []
    votes_by_id = {}
    for case_id, packet in packets_by_id.items():
        line, raw_vote = raw_votes[case_id]
        valid_before, reason_before = valid_vote(packet, raw_vote)
        outcome = raw_vote
        entry = ledger.get(case_id)
        replacement = next((rows[case_id] for rows in correction_files.values() if case_id in rows), None)
        if not valid_before:
            invalid_before[(packet["proposed_action"]["action"], reason_before)] += 1
            if entry is None:
                raise ValueError("invalid_vote_missing_correction_disposition")
            if entry["original_vote_sha256"] != _hash_vote_line(raw_vote):
                raise ValueError("correction_ledger_original_vote_hash_mismatch")
            if entry["status"] == "corrected":
                if replacement is None:
                    raise ValueError("corrected_vote_file_missing")
                changed = {key for key in raw_vote if raw_vote[key] != replacement.get(key)}
                if changed != {"citations"}:
                    raise ValueError("correction_changed_non_citation_fields")
                old_citations, new_citations = raw_vote["citations"], replacement["citations"]
                if len(old_citations) != len(new_citations) or any(
                        any(old.get(k) != new.get(k) for k in ("stage", "step", "quote"))
                        for old, new in zip(old_citations, new_citations)):
                    raise ValueError("correction_changed_citation_content")
                if (entry.get("old_citation_document_ids") != [c["document_id"] for c in old_citations]
                        or entry.get("new_citation_document_ids") != [c["document_id"] for c in new_citations]):
                    raise ValueError("correction_ledger_binding_mismatch")
                corrected[case_id] = replacement
                outcome = replacement
                corrected_ids.append(case_id)
            elif entry["status"] == "uncorrectable":
                if replacement is not None:
                    raise ValueError("uncorrectable_vote_has_replacement")
                uncorrectable_ids.append(case_id)
            else:
                raise ValueError("unknown_correction_status")
        elif entry is not None:
            raise ValueError("correction_attempted_on_valid_vote")

        valid_after, reason_after = valid_vote(packet, outcome)
        if not valid_after:
            invalid_after[(packet["proposed_action"]["action"], reason_after)] += 1
        if valid_after and outcome["approved"]:
            approved_after[packet["proposed_action"]["action"]] += 1
            all_labels.update(set(outcome["risk_labels"]))
        elif valid_after:
            denied_after[packet["proposed_action"]["action"]] += 1
        votes_by_id[case_id] = (outcome, valid_after)

    questions = defaultdict(list)
    for case_id, packet in packets_by_id.items():
        vote, valid = votes_by_id[case_id]
        questions[packet["question"]].append((packet, vote, valid))
    strict_pairs = 0
    luna_approved_pairs = 0
    risk_intersection = Counter()
    for rows in questions.values():
        if len(rows) != 2 or {row[0]["proposed_action"]["action"] for row in rows} != {"search", "answer"}:
            raise ValueError("frozen_pair_shape_changed")
        strict_pairs += 1
        if all(valid and vote["approved"] for _, vote, valid in rows):
            luna_approved_pairs += 1
            risk_intersection.update(set.intersection(*(set(vote["risk_labels"]) for _, vote, _ in rows)))

    deepseek_dir = AUDIT / "deepseek"
    deepseek_status = "not_attempted"
    deepseek_batches = []
    if deepseek_dir.exists():
        for manifest_path in sorted(deepseek_dir.glob("*/manifest-private.json")):
            manifest = _read_json(manifest_path)
            batch_dir = deepseek_dir / manifest_path.parent.name
            journal = batch_dir / "request-journal-private.jsonl"
            events = [row for _, row in _read_jsonl(journal)] if journal.exists() else []
            received = any(row.get("event") == "response_received" for row in events)
            vote_path = batch_dir / "validated-review-private.json"
            valid_saved = vote_path.is_file()
            saved_vote_count = _saved_vote_count(_read_json(vote_path)) if valid_saved else 0
            state = "valid_votes_saved" if valid_saved and saved_vote_count else "response_or_outcome_unknown" if not received else "response_received_no_valid_votes"
            response_details = []
            for raw_path in sorted((batch_dir / "responses-private").glob("attempt-*-raw.json")):
                try:
                    raw = _read_json(raw_path)
                    choice = (raw.get("choices") or [{}])[0]
                    content = choice.get("message", {}).get("content")
                    response_details.append({"attempt": raw_path.stem.removeprefix("attempt-"),
                        "finish_reason": choice.get("finish_reason"),
                        "content_length": len(content) if isinstance(content, str) else None,
                        "prompt_tokens": (raw.get("usage") or {}).get("prompt_tokens"),
                        "completion_tokens": (raw.get("usage") or {}).get("completion_tokens"),
                        "reasoning_tokens": ((raw.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
                        "raw_response_sha256": sha(raw_path)})
                except (OSError, ValueError, KeyError, IndexError, TypeError):
                    response_details.append({"attempt": raw_path.stem.removeprefix("attempt-"),
                                             "raw_response_parse": "failed", "raw_response_sha256": sha(raw_path)})
            deepseek_batches.append({"batch": manifest.get("batch"), "payload_sha256": manifest.get("payload_sha256"),
                                    "state": state, "journal_sha256": sha(journal) if journal.exists() else None,
                                    "request_attempts": sum(row.get("event") == "sending" for row in events),
                                    "response_count": sum(row.get("event") == "response_received" for row in events),
                                    "validated_vote_count": saved_vote_count,
                                    "response_details": response_details,
                                    "valid_vote_file_present": valid_saved})
        deepseek_status = "no_valid_reviewer_votes_saved" if not any(
            row["state"] == "valid_votes_saved" for row in deepseek_batches) else "partial"

    result = {"version": "v130-blind-review-disposition-v1",
        "status": "blocked_insufficient_independent_dual_approval",
        "frozen_candidate": {"questions": len(questions), "states": len(packets_by_id),
                             "strict_pairs": strict_pairs, "screened_source_questions": 5000},
        "luna": {"reviewer": "one isolated GPT-6 Luna agent", "states_reviewed": len(raw_votes),
                 "raw_vote_coverage_complete": True, "raw_vote_sha256": {r["batch"]: r["raw_vote_sha256"] for r in batch_records},
                 "raw_approval_counts": {f"{action}:{str(approved).lower()}": count
                     for (action, approved), count in Counter(
                         (packets_by_id[cid]["proposed_action"]["action"], vote["approved"])
                         for cid, (_, vote) in raw_votes.items()).items()},
                 "invalid_before_correction": sum(invalid_before.values()),
                 "invalid_before_by_action_and_reason": {f"{a}:{r}": n for (a, r), n in invalid_before.items()},
                 "technical_correction_attempts": len(ledger), "technical_corrections_applied": len(corrected_ids),
                 "uncorrectable_invalid_votes": len(uncorrectable_ids),
                 "invalid_after_correction": sum(invalid_after.values()),
                 "valid_approved_states": dict(approved_after), "risk_labels_on_valid_approved_states": dict(all_labels),
                 "valid_denied_states": dict(denied_after),
                 "luna_complete_approved_pairs": luna_approved_pairs,
                 "risk_intersection_on_luna_approved_pairs": dict(risk_intersection),
                 "raw_files_sha256": {r["batch"]: r["raw_vote_sha256"] for r in batch_records},
                 "correction_ledger_sha256": sha(correction_dir / "citation-corrections-ledger.jsonl"),
                 "correction_file_sha256": {f"g4-batch-{i:02d}": sha(correction_dir / f"g4-batch-{i:02d}-citation-corrections.jsonl") for i in range(1, 11)}},
        "deepseek": {"status": deepseek_status, "batches": deepseek_batches,
                     "validated_states": sum(row["validated_vote_count"] for row in deepseek_batches),
                     "unknown_batches_must_not_be_retried": [row["batch"] for row in deepseek_batches if row["state"] == "response_or_outcome_unknown"]},
        "quality_diagnosis": {
            "search_actions_denied_by_luna": 100,
            "answer_actions_denied_by_luna": 98,
            "raw_and_mechanical_status": "All 100 raw search votes denied the proposed query, but 15 search citations remain mechanically invalid and excluded; 85 search denials are valid under v129. Of 100 answer votes, 98 were denied and 2 approved; after the one exact-document-ID correction, 93 denials and 2 approvals are mechanically valid, while 5 denials remain citation-invalid.",
            "primary_observed_pattern": "The 85 mechanically valid search denials point to a policy defect: linked-title extraction selected a title mention without verifying that it targeted the missing relation",
            "examples": [
                {"question": "Which Florida TV station broadcasts from Christmas all year round?", "candidate_query": "Mugby Junction"},
                {"question": "Who built the Union Church of Pocantico Hills and Kykuit?", "candidate_query": "County Tyrone"},
                {"question": "Collective Soul and Catfish and the Bottlemen are both what?", "candidate_query": "Panda Bear Meets the Grim Reaper"}
            ],
            "interpretation": "The pool is a deterministic CPU-linked-title scaffold with official answer proposals, not model rollouts. The blind Luna vote diagnoses a strong query-policy quality defect; it does not estimate model accuracy. One independent reviewer is insufficient for acceptance."
        },
        "training_allowed": False, "data_ready_for_experiment": False}
    write(REPORT, result)
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=True, indent=2))
