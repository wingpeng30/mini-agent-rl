"""Fail-closed audit record for the NXT TakeOver v8 repair candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.v123_cpu_core import sha
from scripts.v124_forensic import OUT


def main() -> None:
    packet_path = OUT / "repair-v8-blind-review-packet-private.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    supplied = packet.pop("packet_sha256")
    payload = json.dumps(packet, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    payload_sha = hashlib.sha256(payload).hexdigest()
    manifest = json.loads((OUT / "repair-v8-blind-review-manifest.json").read_text(encoding="utf-8"))
    assert supplied == payload_sha == manifest["packet_payload_sha256"]
    assert sha(packet_path) == manifest["packet_file_sha256"]
    assert manifest["states"] == 2 and manifest["groups"] == 1

    luna_path = OUT / "repair-v8-luna-review-correction-private.json"
    luna = json.loads(luna_path.read_text(encoding="utf-8"))
    assert luna["packet_sha256"] == payload_sha
    assert {x["state_id"] for x in luna["states"]} == {"V8S001", "V8S002"}
    states = {s["state_id"]: s for g in packet["groups"] for s in g["states"]}
    # The reviewer corrected document IDs, but still used noncanonical stages.
    # In V8S002 it also called the result a search result from a different
    # visible search even though the row explicitly records post-action top-3.
    bad_citations = []
    for review in luna["states"]:
        state = states[review["state_id"]]
        allowed = {}
        for d in state["pre_action_evidence"]:
            allowed[("pre_action_evidence", d["document_id"])] = d["text"]
        for d in state["post_action_retrieval_results"] or []:
            allowed[("post_action_retrieval_results", d["document_id"])] = d["text"]
        for message in state["visible_messages"]:
            if message["role"] == "user":
                allowed[("visible_messages", "user")] = message["content"]
        for citation in review["evidence"]:
            stage = citation["stage"]
            # Record the mismatch without normalizing an independent vote.
            key = (stage, citation["document_id"])
            if key not in allowed or citation["quote"] not in allowed.get(key, ""):
                bad_citations.append({"state_id": review["state_id"],
                                      "stage": stage,
                                      "document_id": citation["document_id"],
                                      "quote_exact_somewhere": any(
                                          citation["quote"] in text
                                          for (_, _), text in allowed.items())})
    assert bad_citations
    assert not (OUT / "repair-v8-deepseek-review-responses/validated-review.json").exists()
    journal = [json.loads(line) for line in
               (OUT / "repair-v8-deepseek-review-journal-private.jsonl").read_text(
                   encoding="utf-8").splitlines()]
    assert [x["status"] for x in journal] == [
        "batch_started", "sending", "response_received",
        "technical_validation_failed_one_fix_allowed", "sending",
        "response_received", "invalid_after_technical_correction"]
    assert journal[0]["packet_sha256"] == payload_sha
    assert journal[-1]["reason"] == "ValueError:citation_not_exact_or_not_visible"

    result = {
        "packet_payload_sha256": payload_sha,
        "packet_file_sha256": sha(packet_path),
        "states": 2,
        "luna_decisions": {x["state_id"]: x["decision"] for x in luna["states"]},
        "luna_vote_status": "invalid_citation_stage_binding_after_one_correction",
        "deepseek_vote_status": "invalid_citations_after_one_technical_correction",
        "independent_double_approval": False,
        "repair_candidate_status": "draft_only_unapproved",
        "training_allowed": False,
        "invalid_luna_citation_bindings": bad_citations,
        "invalid_deepseek_attempts": 2,
        "retry_allowed": False,
    }
    out_path = OUT / "repair-v8-audit-status-private.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
