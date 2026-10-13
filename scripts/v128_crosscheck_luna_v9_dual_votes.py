"""Cross-check v9 DeepSeek tickets with the new blind Luna votes, without promotion."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v124-forensic"
PACKETS = BASE / "pending-dual-review-v9"
REVIEW = BASE / "goal-v128/luna-independent-review-v1"
OUT = BASE / "goal-v128/g2-v9-independent-vote-crosscheck-private.json"
BATCHES = ("v9-batch-01", "v9-batch-02", "v9-batch-03")


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def citation_errors(state: dict, vote: dict) -> list[str]:
    errors = []
    action = state.get("proposed_action", {}).get("action")
    if vote.get("decision") not in {"approve", "reject", "uncertain"}:
        errors.append("decision_invalid")
    if not isinstance(vote.get("issues"), list) or not isinstance(vote.get("rationale"), str):
        errors.append("review_fields_invalid")
    for key in ("query_grounded", "relevant_new_evidence", "answer_supported"):
        if vote.get(key) not in (True, False, None):
            errors.append("diagnostic_field_invalid:" + key)
    citations = vote.get("citations")
    if not isinstance(citations, list) or not citations:
        return errors + ["citations_missing"]
    docs = {}
    for doc in state.get("pre_action_evidence", []):
        docs[("pre_action_evidence", doc.get("document_id"))] = doc.get("text", "")
    for doc in state.get("post_action_retrieval_results") or []:
        docs[("post_action_retrieval_results", doc.get("document_id"))] = doc.get("text", "")
    for msg in state.get("visible_messages", []):
        if msg.get("role") == "user":
            docs[("visible_messages", "user")] = msg.get("content", "")
    for c in citations:
        if not isinstance(c, dict):
            errors.append("citation_shape_invalid")
            continue
        key = (c.get("stage"), c.get("document_id"))
        quote = c.get("quote")
        if not isinstance(quote, str) or not quote.strip() or key not in docs or quote not in docs[key]:
            errors.append("citation_not_exactly_bound")
    stages = {c.get("stage") for c in citations if isinstance(c, dict)}
    if action == "search":
        prior = "pre_action_evidence" if state.get("pre_action_evidence") else "visible_messages"
        if prior not in stages:
            errors.append("search_missing_action_time_citation")
        if state.get("post_action_retrieval_results") and "post_action_retrieval_results" not in stages:
            errors.append("search_missing_retrieval_citation")
    elif action == "answer" and vote.get("decision") == "approve":
        if not state.get("pre_action_evidence") or "pre_action_evidence" not in stages:
            errors.append("approved_answer_missing_visible_evidence_citation")
    return errors


def audit() -> dict:
    luna_doc = read(REVIEW / "luna-review-output-effective.json")
    luna = {r["state_id"]: r for r in luna_doc["reviews"]}
    repair_ids = set(read(BASE / "goal-v128/g2-repair-inventory-private.json")[
                     "v117_repair_candidate_record_ids"])
    primary_dispute = read(BASE / "v9-batch02-primary-disagreement-private.json")
    held_ids = {r["state_id_in_blind_packet"] for r in primary_dispute["findings"]
                if r.get("primary_disagreement")}
    rows, packet_hashes, binding_hashes, ticket_hashes = {}, {}, {}, {}
    errors = []
    for i, batch in enumerate(BATCHES, 1):
        packet_path = PACKETS / f"{batch}-packet-private.json"
        packet = read(packet_path)
        binding = read(PACKETS / f"{batch}-binding-private.json")
        state_map = {s["state_id"]: s for g in packet["groups"] for s in g["states"]}
        source_map = {r["state_id"]: r["source_record_id"] for r in binding["source_mapping"]}
        if set(state_map) != set(source_map):
            raise ValueError("deepseek_source_binding_coverage_mismatch")
        if batch == "v9-batch-01":
            ticket_path = PACKETS / "deepseek-responses/v9-batch-01-citation-correction/citation-correction-validated-private.json"
        else:
            ticket_path = PACKETS / f"deepseek-responses/{batch}/validated-review-private.json"
        ticket = read(ticket_path)
        deepseek = {r["state_id"]: r for r in ticket["reviews"]}
        if set(deepseek) != set(state_map):
            raise ValueError("deepseek_ticket_state_coverage_mismatch")
        packet_hashes[batch] = sha(packet_path.read_bytes())
        binding_hashes[batch] = sha((PACKETS / f"{batch}-binding-private.json").read_bytes())
        ticket_hashes[batch] = sha(ticket_path.read_bytes())
        for sid, state in state_map.items():
            if sid not in luna:
                raise ValueError("missing_luna_review")
            d, l = deepseek[sid], luna[sid]
            de, le = citation_errors(state, d), citation_errors(state, l)
            rid = source_map[sid]
            if rid in rows:
                raise ValueError("duplicate_source_record_mapping")
            rows[rid] = {
                "state_id": sid, "batch": batch, "action": state["proposed_action"]["action"],
                "is_v117_repair_candidate": rid in repair_ids,
                "deepseek_decision": d["decision"], "deepseek_protocol_errors": de,
                "luna_decision": l["decision"], "luna_protocol_errors": le,
                "same_decision": d["decision"] == l["decision"],
                "both_approve_protocol_valid": (d["decision"] == "approve" and
                                                  l["decision"] == "approve" and not de and not le),
                "primary_disagreement_hold": sid in held_ids,
            }
    if set(luna) != {r["state_id"] for r in rows.values()}:
        raise ValueError("luna_and_crosscheck_state_coverage_mismatch")
    d_invalid = [sid for row in rows.values() if row["deepseek_protocol_errors"] for sid in [row["state_id"]]]
    l_invalid = [sid for row in rows.values() if row["luna_protocol_errors"] for sid in [row["state_id"]]]
    all_counts = Counter((r["is_v117_repair_candidate"], r["deepseek_decision"], r["luna_decision"],
                          r["both_approve_protocol_valid"], r["primary_disagreement_hold"])
                         for r in rows.values())
    repair_both = [r for r in rows.values() if r["is_v117_repair_candidate"] and r["both_approve_protocol_valid"]]
    repair_hold = [r for r in repair_both if r["primary_disagreement_hold"]]
    return {
        "version": "v128-v9-dual-vote-crosscheck-v1",
        "review_inputs_sha256": {"luna_effective_output": sha((REVIEW / "luna-review-output-effective.json").read_bytes()),
                                 "luna_initial_output": sha((REVIEW / "luna-review-output-attempt01-immutable.json").read_bytes()),
                                 "luna_correction01": sha((REVIEW / "luna-review-correction01.json").read_bytes()),
                                 "primary_disagreement_report": sha((BASE / "v9-batch02-primary-disagreement-private.json").read_bytes())},
        "source_packet_sha256": packet_hashes,
        "source_binding_sha256": binding_hashes,
        "deepseek_effective_ticket_sha256": ticket_hashes,
        "state_count": len(rows),
        "repair_candidate_states": sum(r["is_v117_repair_candidate"] for r in rows.values()),
        "nonrepair_states": sum(not r["is_v117_repair_candidate"] for r in rows.values()),
        "deepseek_protocol_invalid_count_under_current_citation_rule": len(d_invalid),
        "deepseek_protocol_invalid_state_ids": d_invalid,
        "luna_protocol_invalid_count": len(l_invalid),
        "luna_protocol_invalid_state_ids": l_invalid,
        "decision_cross_tab": [{"repair": k[0], "deepseek": k[1], "luna": k[2],
                                 "both_approve_protocol_valid": k[3], "primary_disagreement_hold": k[4],
                                 "count": v} for k, v in sorted(all_counts.items())],
        "repair_states_both_approve_and_citation_valid": len(repair_both),
        "repair_states_also_in_primary_disagreement_hold": len(repair_hold),
        "repair_states_provisionally_clean_of_old_hold": len(repair_both) - len(repair_hold),
        "all_source_record_dispositions": rows,
        "deepseek_endpoint_identity_verified": False,
        "all_votes_semantically_correct_proven": False,
        "training_eligible_states": 0,
        "training_allowed": False,
        "interpretation": [
            "Cross-approval is a mechanical intersection of the saved DeepSeek decisions and the blind Luna decisions under exact citation checks; it is not a guarantee of semantic correctness.",
            "The DeepSeek endpoint/model identity remains unverified; the seven pre-existing primary-disagreement holds remain held even where this Luna vote agrees with DeepSeek.",
            "No state is promoted to training or treated as a repaired action/result/answer trajectory by this cross-check.",
        ],
    }


if __name__ == "__main__":
    result = audit()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "state_count", "repair_candidate_states", "nonrepair_states",
        "deepseek_protocol_invalid_count_under_current_citation_rule", "luna_protocol_invalid_count",
        "repair_states_both_approve_and_citation_valid", "repair_states_also_in_primary_disagreement_hold",
        "repair_states_provisionally_clean_of_old_hold", "training_eligible_states", "training_allowed",
    )}, ensure_ascii=False))
