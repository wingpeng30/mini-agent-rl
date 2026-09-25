"""Validate two independent, blind v118 reviews and emit a fail-closed gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VALID = {"supported", "unsupported", "uncertain"}


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def readl(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_visible(packet, section, audit_id, alias):
    candidate = packet["anonymous_outputs"][alias]
    if section == "states":
        messages = candidate.get("pre_action_visible_messages", [])
    else:
        messages = candidate.get("answer_time_visible_messages", [])
    visible = []
    for message in messages:
        content = str(message.get("content", ""))
        if message.get("role") != "tool":
            visible.append(content)
            continue
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            visible.append(content)
            continue
        # Quotes are taken from the document text the model saw. Matching the
        # serialized JSON falsely rejects quotes containing escaped characters.
        for document in payload.get("result", []) if isinstance(payload, dict) else []:
            if isinstance(document, dict):
                visible.extend(str(document.get(key, "")) for key in ("title", "text"))
    return visible

def contains_quote(quote, texts):
    return bool(quote) and any(quote in text for text in texts)

def text_leaves(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from text_leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from text_leaves(child)

def candidate_search_results(packet, audit_id, alias):
    candidate = packet["anonymous_outputs"][alias]
    return list(text_leaves(candidate.get("post_action_retrieval_results", [])))


def index_reviews(review, field):
    rows = review.get(field)
    if not isinstance(rows, list): raise ValueError("review_section_missing:"+field)
    result = {row.get("audit_id"): (row.get("decision") if isinstance(row.get("decision"), dict) else row) for row in rows}
    if len(result) != len(rows) or None in result: raise ValueError("duplicate_or_missing_audit_id:"+field)
    return result


def validate_candidates(review_rows, packets, section, reviewer):
    if set(review_rows) != set(packets): raise ValueError("incomplete_review_coverage:"+reviewer+":"+section)
    normalized = {}
    invalid_quotes = []
    for audit_id, row in review_rows.items():
        candidates = row.get("candidates")
        if not isinstance(candidates, dict) or set(candidates) != {"A", "B", "C"}:
            raise ValueError("candidate_votes_missing:"+reviewer+":"+audit_id)
        normalized[audit_id] = {}
        for alias, decision in candidates.items():
            verdict = decision.get("evidence_verdict")
            if verdict not in VALID: raise ValueError("invalid_evidence_verdict:"+reviewer+":"+audit_id)
            quotes = decision.get("citations", [])
            if not isinstance(quotes, list) or any(not isinstance(q, str) for q in quotes):
                raise ValueError("invalid_citation_shape")
            visible = candidate_visible(packets[audit_id], section, audit_id, alias)
            invalid = [quote for quote in quotes if not contains_quote(quote, visible)]
            if verdict == "supported" and not quotes:
                invalid_quotes.append({"audit_id": audit_id, "alias": alias, "reviewer": reviewer, "reason": "support_without_quote"})
            if invalid:
                invalid_quotes.append({"audit_id": audit_id, "alias": alias, "reviewer": reviewer, "reason": "quote_not_in_action_time_input"})
            candidate = packets[audit_id]["anonymous_outputs"][alias]
            action = candidate.get("action") or {}
            if section == "states" and action.get("action") == "search":
                for field in ("search_necessity", "query_justified", "relevant_new_evidence"):
                    if decision.get(field) not in {"yes", "no", "uncertain", "not_evaluable"}:
                        raise ValueError("search_dimension_missing:"+reviewer+":"+audit_id+":"+field)
                result_quotes = decision.get("search_result_citations", [])
                if not isinstance(result_quotes, list) or any(not isinstance(q, str) for q in result_quotes):
                    raise ValueError("invalid_search_citation_shape")
                results = candidate_search_results(packets[audit_id], audit_id, alias)
                if any(not contains_quote(q, results) for q in result_quotes):
                    invalid_quotes.append({"audit_id": audit_id, "alias": alias, "reviewer": reviewer, "reason": "search_quote_not_in_retrieval_results"})
                if decision.get("relevant_new_evidence") == "yes" and not result_quotes:
                    invalid_quotes.append({"audit_id": audit_id, "alias": alias, "reviewer": reviewer, "reason": "search_benefit_without_quote"})
            normalized[audit_id][alias] = {"evidence_verdict": verdict, "citations": quotes,
                "search_necessity": decision.get("search_necessity"),
                "query_justified": decision.get("query_justified"),
                "relevant_new_evidence": decision.get("relevant_new_evidence")}
    return normalized, invalid_quotes


def merge(args):
    directory = Path(args.packet_dir)
    manifest = load(directory/"manifest.json")
    state_path, task_path = directory/"states-blind.jsonl", directory/"changed-tasks-blind.jsonl"
    if sha(state_path) != manifest.get("state_packet_sha256") or sha(task_path) != manifest.get("changed_task_packet_sha256"):
        raise ValueError("blind_packet_hash_mismatch")
    state_packets = {row["audit_id"]: row for row in readl(state_path)}
    task_packets = {row["audit_id"]: row for row in readl(task_path)}
    if len(state_packets) != 100: raise ValueError("must_review_all_100_internal_states")
    a, b = load(args.reviewer_a), load(args.reviewer_b)
    if (a.get("version") != "v118-independent-blind-review-v1" or b.get("version") != "v118-independent-blind-review-v1"
            or a.get("reviewer") != "luna_agent" or b.get("reviewer") != "deepseek_api"):
        raise ValueError("reviewer_identity_mismatch")
    if a.get("reviewer") == b.get("reviewer") or a.get("independent") is not True or b.get("independent") is not True:
        raise ValueError("reviewers_not_independent")
    packet_hash = sha(directory/"manifest.json")
    for review in (a, b):
        if review.get("packet_manifest_sha256") != packet_hash: raise ValueError("review_packet_binding_mismatch")
    a_state, a_bad_state = validate_candidates(index_reviews(a, "states"), state_packets, "states", "luna_agent")
    b_state, b_bad_state = validate_candidates(index_reviews(b, "states"), state_packets, "states", "deepseek_api")
    a_task, a_bad_task = validate_candidates(index_reviews(a, "changed_tasks"), task_packets, "changed_tasks", "luna_agent")
    b_task, b_bad_task = validate_candidates(index_reviews(b, "changed_tasks"), task_packets, "changed_tasks", "deepseek_api")
    private = load(args.private_mapping)
    if private.get("version") != "v118-private-review-map-v1": raise ValueError("private_mapping_version_mismatch")
    unresolved, regressions = [], []
    joined = []
    for section, packets, votes_a, votes_b, mapping in (
        ("states", state_packets, a_state, b_state, private["states"]),
        ("changed_tasks", task_packets, a_task, b_task, private["tasks"])):
        for audit_id, packet in packets.items():
            model_by_alias = mapping[audit_id]["anonymous_to_model"]
            by_model = {}
            for alias in "ABC":
                va, vb = votes_a[audit_id][alias], votes_b[audit_id][alias]
                status = va["evidence_verdict"] if va["evidence_verdict"] == vb["evidence_verdict"] else "disputed"
                if status == "uncertain" or status == "disputed":
                    unresolved.append({"section": section, "audit_id": audit_id, "alias": alias, "status": status})
                by_model[model_by_alias[alias]] = status
            joined.append({"section": section, "audit_id": audit_id, "verdicts_by_model": by_model})
            if by_model.get("v101") == "supported" and by_model.get("v118") == "unsupported":
                regressions.append({"section": section, "audit_id": audit_id})
    bad_quotes = a_bad_state+b_bad_state+a_bad_task+b_bad_task
    unresolved.extend({**row, "status": "invalid_citation"} for row in bad_quotes)
    result = {"version": "v118-semantic-gate-v1", "reviewers": 2, "independent_reviews": True,
        "reviewer_ids": [a["reviewer"], b["reviewer"]], "packet_manifest_sha256": packet_hash,
        "state_cases": len(state_packets), "changed_task_cases": len(task_packets),
        "each_reviewer_state_coverage": {"luna_agent": len(a_state), "deepseek_api": len(b_state)},
        "each_reviewer_changed_task_coverage": {"luna_agent": len(a_task), "deepseek_api": len(b_task)},
        "unresolved_disputes": len(unresolved), "confirmed_semantic_regressions": len(regressions),
        "unresolved_cases": unresolved, "regression_cases": regressions, "joined_reviews": joined,
        "independent100_opened": False, "final_test_accessed": False,
        "semantic_gate_passed": len(unresolved) == 0 and len(regressions) == 0}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("packet-dir", "reviewer-a", "reviewer-b", "private-mapping", "output"):
        parser.add_argument("--"+name, required=True)
    print(json.dumps(merge(parser.parse_args()), ensure_ascii=False, indent=2))
