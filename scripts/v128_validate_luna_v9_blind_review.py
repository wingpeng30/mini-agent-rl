"""Validate a blind, independent Luna review against exact v9 packet evidence."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v124-forensic"
SOURCE = BASE / "pending-dual-review-v9"
REVIEW_DIR = BASE / "goal-v128/luna-independent-review-v1"
OUTPUT = REVIEW_DIR / "luna-review-output.json"
IMMUTABLE_INITIAL = REVIEW_DIR / "luna-review-output-attempt01-immutable.json"
CORRECTION = REVIEW_DIR / "luna-review-correction01.json"
EFFECTIVE_OUTPUT = REVIEW_DIR / "luna-review-output-effective.json"
MANIFEST = REVIEW_DIR / "input-manifest.json"
REPORT = BASE / "goal-v128/g2-luna-v9-independent-review-audit-private.json"
BATCHES = ("v9-batch-01", "v9-batch-02", "v9-batch-03")


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def document_map(state: dict) -> dict[tuple[str, str], str]:
    docs: dict[tuple[str, str], str] = {}
    for doc in state.get("pre_action_evidence", []):
        docs[("pre_action_evidence", doc["document_id"])] = doc.get("text", "")
    for doc in state.get("post_action_retrieval_results") or []:
        docs[("post_action_retrieval_results", doc["document_id"])] = doc.get("text", "")
    for message in state.get("visible_messages", []):
        if message.get("role") == "user":
            docs[("visible_messages", "user")] = message.get("content", "")
    return docs


def audit() -> dict:
    manifest = read_json(MANIFEST)
    inputs: dict[str, dict] = {}
    source_bindings: dict[str, str] = {}
    source_records: dict[str, str] = {}
    for i, batch in enumerate(BATCHES, 1):
        packet_path = SOURCE / f"{batch}-packet-private.json"
        copied_path = REVIEW_DIR / f"packet-{i:02d}.json"
        packet_raw, copied_raw = packet_path.read_bytes(), copied_path.read_bytes()
        if packet_raw != copied_raw:
            raise ValueError("blind_packet_copy_not_byte_identical")
        packet = json.loads(copied_raw.decode("utf-8-sig"))
        binding = read_json(SOURCE / f"{batch}-binding-private.json")
        payload = {k: v for k, v in packet.items() if k != "packet_sha256"}
        payload_sha = sha(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8"))
        if packet.get("packet_sha256") != payload_sha or binding.get("packet_sha256") != payload_sha:
            raise ValueError("source_binding_packet_hash_mismatch")
        rows = binding.get("source_mapping", [])
        state_to_source = {r["state_id"]: r["source_record_id"] for r in rows}
        packet_states = {s["state_id"]: s for g in packet["groups"] for s in g["states"]}
        if set(state_to_source) != set(packet_states):
            raise ValueError("binding_state_coverage_mismatch")
        inputs.update(packet_states)
        source_bindings.update(state_to_source)
        source_records[batch] = sha((SOURCE / f"{batch}-binding-private.json").read_bytes())
        mf = next((x for x in manifest["files"] if x["path"] == copied_path.name), None)
        if mf is None or mf.get("sha256") != sha(copied_raw):
            raise ValueError("review_input_manifest_hash_mismatch")
        if len(packet["groups"]) == 0:
            raise ValueError("empty_review_packet")
    instructions = REVIEW_DIR / "review-instructions.txt"
    instruction_row = next((x for x in manifest["files"] if x["path"] == instructions.name), None)
    if instruction_row is None or instruction_row.get("sha256") != sha(instructions.read_bytes()):
        raise ValueError("review_instructions_hash_mismatch")

    initial_raw = IMMUTABLE_INITIAL.read_bytes() if IMMUTABLE_INITIAL.exists() else OUTPUT.read_bytes()
    if sha(initial_raw) != "2241631a46a5c36715405b14137a5ffc12f12801d7b670848f6493139d49feb1":
        raise ValueError("initial_luna_review_output_hash_changed")
    output = json.loads(initial_raw.decode("utf-8-sig"))
    reviews = output.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("reviews_not_list")
    initial_by_id = {r.get("state_id"): r for r in reviews if isinstance(r, dict)}
    corrected_ids: list[str] = []
    effective_by_id = dict(initial_by_id)
    correction_sha = None
    if CORRECTION.exists():
        correction_raw = CORRECTION.read_bytes()
        correction = json.loads(correction_raw.decode("utf-8-sig"))
        correction_reviews = correction.get("reviews")
        if not isinstance(correction_reviews, list):
            raise ValueError("technical_correction_reviews_not_list")
        corrected_ids = [r.get("state_id") for r in correction_reviews if isinstance(r, dict)]
        if (len(corrected_ids) != len(correction_reviews) or len(corrected_ids) != 3
                or set(corrected_ids) != {"V9B03S010", "V9B03S012", "V9B03S014"}):
            raise ValueError("technical_correction_scope_mismatch")
        for replacement in correction_reviews:
            sid = replacement["state_id"]
            original = initial_by_id.get(sid)
            if original is None:
                raise ValueError("technical_correction_unknown_state")
            if any(replacement.get(k) != v for k, v in original.items() if k != "citations"):
                raise ValueError("technical_correction_changed_semantic_fields")
            old_cites = original.get("citations", [])
            new_cites = replacement.get("citations", [])
            if not isinstance(new_cites, list) or not all(c in new_cites for c in old_cites):
                raise ValueError("technical_correction_removed_or_changed_citations")
            effective_by_id[sid] = replacement
        correction_sha = sha(correction_raw)
    effective_reviews = [effective_by_id[r["state_id"]] for r in reviews]
    effective_raw = json.dumps({"reviews": effective_reviews}, ensure_ascii=False,
                               indent=2).encode("utf-8") + b"\n"
    EFFECTIVE_OUTPUT.write_bytes(effective_raw)
    state_ids = [r.get("state_id") for r in reviews if isinstance(r, dict)]
    if len(state_ids) != len(reviews) or len(state_ids) != len(set(state_ids)):
        raise ValueError("duplicate_or_malformed_review_id")
    if set(state_ids) != set(inputs):
        raise ValueError("review_coverage_incomplete_or_unknown")

    invalid: list[dict] = []
    decisions = Counter()
    per_record = {}
    repair_inventory = read_json(BASE / "goal-v128/g2-repair-inventory-private.json")
    repair_ids = set(repair_inventory["v117_repair_candidate_record_ids"])
    for review in effective_reviews:
        sid = review["state_id"]
        state = inputs[sid]
        decision = review.get("decision")
        decisions[decision] += 1
        reason = None
        if decision not in {"approve", "reject", "uncertain"}:
            reason = "decision_invalid"
        elif not isinstance(review.get("issues"), list) or not isinstance(review.get("rationale"), str):
            reason = "review_fields_invalid"
        elif any(review.get(k) not in (True, False, None) for k in
                 ("query_grounded", "relevant_new_evidence", "answer_supported")):
            reason = "review_diagnostic_field_invalid"
        citations = review.get("citations")
        if reason is None and (not isinstance(citations, list) or not citations):
            reason = "citations_missing"
        docs = document_map(state)
        if reason is None:
            for citation in citations:
                stage, doc_id, quote = (citation.get("stage"), citation.get("document_id"),
                                        citation.get("quote"))
                if (not isinstance(quote, str) or not quote.strip() or
                        (stage, doc_id) not in docs or quote not in docs[(stage, doc_id)]):
                    reason = "citation_not_exactly_bound"
                    break
        action = state["proposed_action"]["action"]
        stages = {c.get("stage") for c in citations} if isinstance(citations, list) else set()
        if reason is None and action == "search":
            expected_prior = "pre_action_evidence" if state.get("pre_action_evidence") else "visible_messages"
            if expected_prior not in stages:
                reason = "search_decision_missing_action_time_citation"
            elif state.get("post_action_retrieval_results") and "post_action_retrieval_results" not in stages:
                reason = "search_decision_missing_retrieved_result_citation"
        if reason is None and decision == "approve" and action == "answer":
            if not any(c.get("stage") == "pre_action_evidence" for c in citations):
                reason = "approved_answer_missing_pre_action_citation"
        if reason:
            invalid.append({"state_id": sid, "error": reason})
        rid = source_bindings[sid]
        per_record[rid] = {"state_id": sid, "decision": decision,
                           "mechanically_valid": reason is None,
                           "in_repair_inventory": rid in repair_ids,
                           "action": action}

    valid_approvals = [x for x in per_record.values()
                       if x["mechanically_valid"] and x["decision"] == "approve"]
    return {
        "version": "v128-luna-v9-blind-review-audit-v1",
        "reviewer": {"model": "gpt-6-luna", "fresh_agent_no_prior_turns": True,
                     "inputs_only": ["three copied anonymous packet files", "review-instructions.txt"]},
        "input_manifest_sha256": sha(MANIFEST.read_bytes()),
        "packet_file_sha256": [sha((REVIEW_DIR / f"packet-{i:02d}.json").read_bytes())
                                for i in range(1, 4)],
        "initial_output_sha256": sha(initial_raw),
        "technical_correction01_sha256": correction_sha,
        "technical_correction01_state_ids": corrected_ids,
        "effective_output_sha256": sha(EFFECTIVE_OUTPUT.read_bytes()),
        "binding_sha256": source_records,
        "output_sha256": sha(initial_raw),
        "expected_state_count": len(inputs),
        "reviewed_state_count": len(reviews),
        "unique_source_record_count": len(per_record),
        "repair_candidate_state_count": sum(x["in_repair_inventory"] for x in per_record.values()),
        "nonrepair_state_count": sum(not x["in_repair_inventory"] for x in per_record.values()),
        "decision_counts": dict(decisions),
        "mechanically_valid_review_count": len(reviews) - len(invalid),
        "mechanically_valid_approve_count": len(valid_approvals),
        "mechanically_invalid_count": len(invalid),
        "mechanically_invalid_review_ids": invalid,
        "record_dispositions": per_record,
        "semantic_double_review_granted": False,
        "training_allowed": False,
        "interpretation": [
            "These are fresh single-reviewer Luna opinions. Mechanical citation validation is not proof that a semantic judgment is correct.",
            "The one DeepSeek reviewer's endpoint/model identity remains unverified, so cross-vote agreement cannot resolve G3 or authorize training.",
            "Do not modify the original DeepSeek votes; any future semantic adjudication must be separately authorized and blind.",
        ],
    }


if __name__ == "__main__":
    report = audit()
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "expected_state_count", "reviewed_state_count", "repair_candidate_state_count",
        "nonrepair_state_count", "decision_counts", "mechanically_valid_review_count",
        "mechanically_valid_approve_count", "mechanically_invalid_count",
        "semantic_double_review_granted", "training_allowed",
    )}, ensure_ascii=False))
