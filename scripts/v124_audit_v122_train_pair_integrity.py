"""Recompute v122 train-supplement trajectory links from raw packets and reviews.

Read-only CPU audit. This does not accept a sample or authorize training.
"""
from __future__ import annotations

import json
from collections import Counter

from scripts.v123_cpu_core import ROOT, digest, readl, write
from scripts.v124_forensic import OUT
from scripts.finalize_v117 import valid_v117_vote

BASE = ROOT / "data/evidence-sft-v122-train-candidates"
REVIEW = BASE / "review"
REPORT = OUT / "v122-train-pair-integrity-audit-private.json"


def _action(packet: dict) -> dict:
    action = packet.get("proposed_action")
    if not isinstance(action, dict):
        raise ValueError("missing_proposed_action")
    return action


def _tool_result(message: dict) -> list[dict]:
    if message.get("role") != "tool":
        return []
    try:
        payload = json.loads(message.get("content", ""))
    except (TypeError, json.JSONDecodeError):
        return []
    result = payload.get("result", []) if isinstance(payload, dict) else []
    return result if isinstance(result, list) else []


def _linked(search: dict, answer: dict, answer_review: dict) -> dict:
    """Require exact search-state prefix, matching executed search, result and citation."""
    search_input = search["visible_messages"]
    answer_input = answer["visible_messages"]
    target = _action(search)
    prefix_pos = len(search_input)
    prefix_ok = answer_input[:prefix_pos] == search_input
    executed = (answer_input[prefix_pos] if len(answer_input) > prefix_pos else {})
    executed_tool = (answer_input[prefix_pos + 1] if len(answer_input) > prefix_pos + 1 else {})
    try:
        executed_action = json.loads(executed.get("content", ""))
    except (TypeError, json.JSONDecodeError):
        executed_action = None
    action_ok = executed.get("role") == "assistant" and executed_action == target
    observed = _tool_result(executed_tool)
    observed_ids = {d.get("id") for d in observed if isinstance(d, dict)}
    retrieved = search.get("retrieved_after_search") or []
    retrieved_ids = {d.get("id") for d in retrieved if isinstance(d, dict)}
    citations = answer_review.get("citations", [])
    cited_ids = {c.get("document_id") for c in citations
                 if isinstance(c, dict) and c.get("stage") == "before_action"}
    linked_ids = retrieved_ids & observed_ids & cited_ids
    return {
        "exact_search_prompt_prefix": prefix_ok,
        "search_action_replayed_in_answer": action_ok,
        "retrieved_document_present_in_answer_context": bool(retrieved_ids & observed_ids),
        "primary_answer_review_cites_search_result": bool(linked_ids),
        "linked_document_ids": sorted(linked_ids),
        "strict_link": prefix_ok and action_ok and bool(linked_ids),
    }


def main() -> None:
    packets = readl(REVIEW / "internal-packets.jsonl")
    packet_by_case = {p["case_id"]: p for p in packets}
    if len(packet_by_case) != len(packets) or len(packets) != 212:
        raise RuntimeError("candidate_packet_coverage_mismatch")

    inventory = {r["original_case_id"]: r for r in readl(OUT / "inventory-private.jsonl")
                 if r["cohort"] == "v122_train_candidates"}
    primary = {r["record_id"]: r for p in sorted(OUT.glob("primary-batch-*.jsonl"))
               for r in readl(p)}
    for path in sorted(OUT.glob("primary-amendments*-private.jsonl")):
        for change in readl(path):
            old = primary[change["record_id"]]
            primary[change["record_id"]] = {**old,
                "disposition": change["effective_disposition"]}
    for change in readl(OUT / "primary-adjudication-overrides-private.jsonl"):
        old = primary[change["record_id"]]
        if old["disposition"] != change["supersedes_effective_disposition"]:
            raise RuntimeError("primary_override_chain_mismatch")
        primary[change["record_id"]] = {**old,
            "disposition": change["effective_disposition"]}

    api = {}
    for path in sorted(REVIEW.glob("api-internal-batch-*.jsonl")):
        for row in readl(path):
            if row["case_id"] in api:
                raise RuntimeError("duplicate_api_vote")
            api[row["case_id"]] = row
    if set(api) != set(packet_by_case):
        raise RuntimeError("api_packet_case_coverage_mismatch")

    api_decisions = Counter()
    for case_id, row in api.items():
        vote = row.get("decision")
        valid, _ = valid_v117_vote(packet_by_case[case_id], vote)
        if row.get("packet_sha256") != digest(packet_by_case[case_id]):
            valid = False
        api_decisions["approve" if valid and vote.get("approved") is True else
                      "reject" if valid and vote.get("approved") is False else "invalid"] += 1

    source_to_cases = {r["source_id"]: r["case_ids"]
                       for r in readl(REVIEW / "eligible-private.jsonl")}
    if len(source_to_cases) != 106:
        raise RuntimeError("eligible_pair_inventory_mismatch")

    counts = Counter()
    pair_rows = []
    for source_id, case_ids in source_to_cases.items():
        if len(case_ids) != 2:
            raise RuntimeError("not_two_states_per_candidate")
        members = []
        for case_id in case_ids:
            packet = packet_by_case[case_id]
            inv = inventory[case_id]
            review = primary[inv["record_id"]]
            vote = api[case_id].get("decision")
            valid, validation = valid_v117_vote(packet, vote)
            if api[case_id].get("packet_sha256") != digest(packet):
                valid, validation = False, "api_packet_hash_mismatch"
            members.append({"case_id": case_id, "record_id": inv["record_id"],
                "packet": packet, "primary": review, "api": api[case_id],
                "api_valid": valid, "api_validation": validation})
        actions = {m["packet"]["proposed_action"].get("action") for m in members}
        if actions != {"search", "answer"}:
            raise RuntimeError("candidate_pair_actions_mismatch")
        search = next(m for m in members if m["packet"]["proposed_action"]["action"] == "search")
        answer = next(m for m in members if m["packet"]["proposed_action"]["action"] == "answer")
        link = _linked(search["packet"], answer["packet"], answer["primary"])
        primary_retained = all(m["primary"].get("disposition") == "retain_candidate"
                               for m in members)
        api_approved = all(m["api_valid"] and m["api"].get("decision", {}).get("approved") is True
                           for m in members)
        counts["candidate_pairs"] += 1
        if primary_retained:
            counts["both_primary_retained"] += 1
        if api_approved:
            counts["both_api_approved"] += 1
        if primary_retained and api_approved:
            counts["both_retained_and_api_approved"] += 1
        if link["strict_link"]:
            counts["strict_trajectory_links"] += 1
        if primary_retained and api_approved and link["strict_link"]:
            counts["both_retained_api_approved_strict_links"] += 1
        pair_rows.append({"source_id": source_id,
            "record_ids": [m["record_id"] for m in members],
            "primary_dispositions": [m["primary"].get("disposition") for m in members],
            "api_votes": [m["api"].get("decision", {}).get("approved")
                          if m["api"].get("decision") else None for m in members],
            "api_vote_validity": [m["api_valid"] for m in members],
            "trajectory_link": link,
            "eligible_for_independent_review": (primary_retained and api_approved
                                                 and link["strict_link"]),
            "training_allowed": False})

    report = {"version": "v124-v122-train-pair-integrity-v1",
        "source_packet_sha256": digest(packets),
        "primary_ledger_sha256": digest(primary),
        "api_case_coverage": len(api), "api_valid_votes": sum(
            1 for r in api.values() if valid_v117_vote(packet_by_case[r["case_id"]],
                r.get("decision"))[0]),
        "api_final_decisions": dict(api_decisions),
        "counts": dict(counts), "pairs": pair_rows,
        "independent_luna_review": "not_run",
        "training_allowed": False}
    write(REPORT, report)
    print(json.dumps({"counts": report["counts"],
        "api_case_coverage": report["api_case_coverage"],
        "api_valid_votes": report["api_valid_votes"],
        "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
