"""Fail-closed integrity check for the draft-only v7 Wardle repair."""
from __future__ import annotations

import hashlib
import json

from scripts.v123_cpu_core import ROOT, digest, readl, sha
from scripts.v124_forensic import OUT

IDS = [f"v117_train:{n:04d}" for n in (55, 92, 216, 217)]
QUESTION = (
    "According to the Wardle article, what country is named for the village "
    "described as being in Greater Manchester and among the foothills of the "
    "South Pennines?"
)
QUERY = "Wardle village Greater Manchester South Pennines country"


def main() -> None:
    drafts = readl(OUT / "repair-drafts-v7-private.jsonl")
    assert [r["record_id"] for r in drafts] == IDS
    assert all(r["status"] == "draft_only_requires_semantic_double_review" for r in drafts)
    assert all(r["approved_training_label"] is False and r["training_allowed"] is False for r in drafts)
    assert all(r["revised_row"]["messages"][1]["content"] == QUESTION for r in drafts)
    assert sum(json.loads(r["revised_row"]["messages"][r["revised_row"]["target_assistant_index"]]["content"])
               ["action"] == "search" for r in drafts) == 3
    for r in drafts:
        action = json.loads(r["revised_row"]["messages"][r["revised_row"]["target_assistant_index"]]["content"])
        if action["action"] == "search":
            assert action["query"] == QUERY
        else:
            assert action == {"action": "answer", "answer": "England"}

    replay = readl(OUT / "repair-draft-states-cpu-v2-private.jsonl")
    group = [g for g in replay if set(g["record_ids"]) == set(IDS)]
    assert len(group) == 1 and len(group[0]["states"]) == 4
    states = {s["record_id"]: s for s in group[0]["states"]}
    assert set(states) == set(IDS)
    draft_by_id = {r["record_id"]: r for r in drafts}
    for rid, state in states.items():
        assert state["within_token_limits"] is True
        assert state["original_row_sha256"] == draft_by_id[rid]["original_row_sha256"]
        assert state["revised_row_sha256"] == digest(draft_by_id[rid]["revised_row"])
    search_states = [s for s in states.values() if s["proposed_action"]["action"] == "search"]
    assert len(search_states) == 3
    assert all(s["proposed_action"]["query"] == QUERY for s in search_states)
    assert all(s["retrieved_after_search"][0]["title"] == "Wardle, Greater Manchester" for s in search_states)
    assert all("England" in s["retrieved_after_search"][0]["text"] for s in search_states)

    packet_path = OUT / "repair-v7-blind-review-packet-private.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    supplied_payload_sha = packet.pop("packet_sha256")
    canonical = json.dumps(packet, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    actual_payload_sha = hashlib.sha256(canonical).hexdigest()
    manifest = json.loads((OUT / "repair-v7-blind-review-manifest.json").read_text(encoding="utf-8"))
    assert actual_payload_sha == supplied_payload_sha == manifest["packet_payload_sha256"]
    assert sha(packet_path) == manifest["packet_file_sha256"]
    assert manifest["groups"] == 1 and manifest["states"] == 4
    packet_text = packet_path.read_text(encoding="utf-8")
    assert "v117_train:" not in packet_text
    assert all(key not in packet_text.lower() for key in ("gold_answer", "original_answer_private", "record_id"))
    assert manifest["training_allowed"] is False

    # Validate the fresh, blind Luna vote and the one permitted technical
    # correction. Citations must be verbatim substrings of packet evidence.
    first_attempt = json.loads((OUT / "repair-v7-luna-review-attempt1-private.json").read_text(
        encoding="utf-8"))
    luna = json.loads((OUT / "repair-v7-luna-review-final-private.json").read_text(encoding="utf-8"))
    packet_states = {s["state_id"]: s for s in packet["groups"][0]["states"]}
    assert first_attempt["context_mode"] == luna["context_mode"] == "fresh_no_history_fork_turns_none"
    assert luna["reviewer_model_selected"] == "gpt-6-luna"
    assert luna["packet_sha256"] == actual_payload_sha
    assert luna["technical_correction_only"] is True
    assert {r["state_id"] for r in first_attempt["states"]} == {r["state_id"] for r in luna["states"]} == set(packet_states)
    first_votes = {r["state_id"]: r["decision"] for r in first_attempt["states"]}
    final_votes = {r["state_id"]: r["decision"] for r in luna["states"]}
    assert first_votes == final_votes == {state_id: "approve" for state_id in packet_states}
    assert all(r["issues"] == next(x["issues"] for x in first_attempt["states"]
                                    if x["state_id"] == r["state_id"]) for r in luna["states"])
    for review in luna["states"]:
        state = packet_states[review["state_id"]]
        docs = {doc["document_id"]: doc["text"] for doc in
                state["pre_action_evidence"] + (state["post_action_retrieval_results"] or [])}
        for citation in review["evidence"]:
            assert citation["document_id"] in docs
            assert citation["quote"] in docs[citation["document_id"]]

    journal_rows = [json.loads(line) for line in
                    (OUT / "repair-v7-deepseek-review-journal-private.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in journal_rows] == ["sending", "outcome_unknown"]
    assert journal_rows[0]["packet_sha256"] == journal_rows[1]["packet_sha256"] == actual_payload_sha

    print(json.dumps({"draft_states": len(drafts), "replayed_groups": 1,
        "search_top1_direct_support": True, "token_limits": "passed",
        "blind_payload_and_file_hashes": "passed", "luna_vote": "4_approve_exact_quotes_verified",
        "deepseek_vote": "outcome_unknown_no_retry", "dual_approval": False,
        "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
