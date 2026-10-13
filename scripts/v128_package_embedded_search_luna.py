"""Create a six-case, no-gold blind diagnostic packet for one Luna reviewer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.v128_audit_embedded_search_reuse import (
    EXTRACTED,
    GOAL,
    ROOT,
    read_jsonl,
    state_key,
)
from scripts.v128_review_protocol import contract_identity

AUDIT = GOAL / "g2-embedded-search-reuse-audit-private.json"
PACKET = GOAL / "g2-embedded-search-luna-packet-private.json"
PRIVATE_MAP = GOAL / "g2-embedded-search-luna-private-map.json"
MANIFEST = GOAL / "g2-embedded-search-luna-packet-manifest-private.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit() -> dict:
    audit_report = json.loads(AUDIT.read_text(encoding="utf-8-sig"))
    allowed = {
        row["state_sha256"] for row in audit_report["extracted_candidate_states"]
        if not row["exact_retained_candidate_state"]
        and not row["exact_frozen_search_target_record_ids"]
    }
    if len(allowed) != 6:
        raise ValueError("expected_six_deduplicated_embedded_search_candidates")

    sources = read_jsonl(EXTRACTED)
    selected = []
    for row in sources:
        key = state_key(row["visible_messages"], row["proposed_action"])
        if key in allowed:
            selected.append((key, row))
    if len(selected) != 6 or {key for key, _ in selected} != allowed:
        raise ValueError("deduplicated_candidate_mapping_not_one_to_one")

    # Hash-sort rather than preserve source row order; reviewer sees only case IDs.
    selected.sort(key=lambda pair: hashlib.sha256(
        ("v128-luna-blind-order-v1:" + pair[0]).encode()).hexdigest())
    cases, private_rows = [], []
    resolved_result_references = 0
    for index, (state_hash, source) in enumerate(selected, 1):
        case_id = f"case-{index:02d}"
        doc_names: dict[str, str] = {}

        source_before = source.get("before_action_evidence") or []
        before_by_id = {doc.get("id"): doc for doc in source_before
                        if isinstance(doc, dict) and isinstance(doc.get("id"), str)}

        def anonymize(docs: list[dict], *, resolve_prior_reference: bool = False) -> list[dict]:
            nonlocal resolved_result_references
            result = []
            for doc in docs:
                original_id = doc.get("id")
                if not isinstance(original_id, str):
                    raise ValueError("source_document_id_missing")
                title, text = doc.get("title"), doc.get("text")
                resolved_from_before = False
                if (not isinstance(title, str) or not title.strip()
                        or not isinstance(text, str) or not text.strip()):
                    prior = before_by_id.get(original_id) if resolve_prior_reference else None
                    if (not isinstance(prior, dict)
                            or not isinstance(prior.get("title"), str)
                            or not isinstance(prior.get("text"), str)
                            or not prior["title"].strip() or not prior["text"].strip()):
                        raise ValueError("result_reference_cannot_be_resolved_at_action_time")
                    title, text = prior["title"], prior["text"]
                    resolved_from_before = True
                    resolved_result_references += 1
                if original_id not in doc_names:
                    doc_names[original_id] = f"doc-{len(doc_names) + 1:02d}"
                projected = {
                    "id": doc_names[original_id],
                    "title": title,
                    "text": text,
                }
                if resolved_from_before:
                    projected["resolved_prior_reference"] = True
                result.append(projected)
            return result

        before = anonymize(source_before)
        retrieved = anonymize(source.get("retrieved_after_search") or [],
                              resolve_prior_reference=True)
        case = {
            "case_id": case_id,
            "question": source["question"],
            "before_action": before,
            "proposed_action": {"action": "search", "query": source["proposed_action"]["query"]},
            "retrieved_after_search": retrieved,
        }
        cases.append(case)
        private_rows.append({
            "case_id": case_id,
            "candidate_id_private": source["candidate_id_private"],
            "state_sha256": state_hash,
            "source_search_record_id_private": source["source_search_record_id_private"],
            "source_answer_record_id_private": source["source_answer_record_id_private"],
            "source_answer_row_sha256": source["source_answer_row_sha256"],
        })

    packet = {"protocol": contract_identity(), "cases": cases}
    serialized = json.dumps(packet, ensure_ascii=False, sort_keys=True).encode("utf-8")
    packet_hash = sha(serialized)
    private_map = {
        "version": "v128-embedded-search-luna-private-map-v1",
        "packet_sha256": packet_hash,
        "rows": private_rows,
        "training_allowed": False,
    }
    private_map_bytes = (json.dumps(private_map, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest = {
        "version": "v128-embedded-search-luna-packet-manifest-v1",
        "source_audit_sha256": sha(AUDIT.read_bytes()),
        "source_extraction_sha256": sha(EXTRACTED.read_bytes()),
        "protocol": contract_identity(),
        "packet_sha256": packet_hash,
        "private_map_sha256": sha(private_map_bytes),
        "case_count": len(cases),
        "retrieved_prior_document_references_resolved": resolved_result_references,
        "gold_fields_in_packet": False,
        "source_ids_in_packet": False,
        "answer_targets_in_packet": False,
        "other_reviewer_votes_in_packet": False,
        "review_kind": "single blind diagnostic; not dual approval",
        "training_allowed": False,
    }
    return {"packet": packet, "packet_bytes": serialized,
            "private_map": private_map, "manifest": manifest}


def write_packet() -> dict:
    result = audit()
    PACKET.write_bytes(result["packet_bytes"])
    PRIVATE_MAP.write_bytes((json.dumps(result["private_map"], ensure_ascii=False,
                                       indent=2) + "\n").encode("utf-8"))
    MANIFEST.write_text(json.dumps(result["manifest"], ensure_ascii=False,
                                   indent=2) + "\n", encoding="utf-8")
    return result["manifest"]


if __name__ == "__main__":
    print(json.dumps(write_packet(), ensure_ascii=False))
