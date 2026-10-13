"""Package only the v7 Wardle repair states into an anonymous blind packet."""
from __future__ import annotations

import hashlib
import json
import re

from scripts.v123_cpu_core import readl, sha, write
from scripts.v124_forensic import OUT
from scripts.v124_package_repair_blind_review import remap_docs, remap_messages

REPLAY = OUT / "repair-draft-states-cpu-v2-private.jsonl"
DRAFTS = OUT / "repair-drafts-v7-private.jsonl"
PACKET = OUT / "repair-v7-blind-review-packet-private.json"
BINDING = OUT / "repair-v7-blind-review-binding-private.json"
MANIFEST = OUT / "repair-v7-blind-review-manifest.json"


def main() -> None:
    drafts = readl(DRAFTS)
    expected = {r["record_id"] for r in drafts}
    replay_groups = readl(REPLAY)
    selected = [g for g in replay_groups if set(g["record_ids"]) & expected]
    if {rid for g in selected for rid in g["record_ids"]} != expected:
        raise RuntimeError("v7_replay_group_coverage_mismatch")
    if sum(len(g["states"]) for g in selected) != len(expected):
        raise RuntimeError("v7_replay_state_count_mismatch")
    if any(not all(s["within_token_limits"] for s in g["states"]) for g in selected):
        raise RuntimeError("v7_token_preflight_failed")
    if len(selected) != 1 or len(expected) != 4:
        raise RuntimeError("v7_packet_must_contain_one_complete_four_state_group")

    packet_groups, binding = [], []
    for group_index, group in enumerate(selected, 1):
        aliases: dict[str, str] = {}
        states = []
        for index, state in enumerate(group["states"], 1):
            anonymous_id = f"V7S{index:03d}"
            states.append({
                "state_id": anonymous_id,
                "visible_messages": remap_messages(state["visible_messages"], aliases),
                "pre_action_evidence": remap_docs(state["before_action_evidence"], aliases),
                "proposed_action": state["proposed_action"],
                "post_action_retrieval_results": remap_docs(state["retrieved_after_search"], aliases),
            })
            binding.append({
                "state_id": anonymous_id,
                "record_id": state["record_id"],
                "source_row_sha256": state["original_row_sha256"],
                "revised_row_sha256": state["revised_row_sha256"],
                "prompt_ids_sha256": state["prompt_ids_sha256"],
                "target_mask_sha256": state["target_mask_sha256"],
            })
        packet_groups.append({"pair_id": f"V7P{group_index:02d}",
                              "question": group["question"], "states": states})

    payload = {
        "version": "v124-v7-repair-blind-packet-v1",
        "review_instructions": (
            "Review each state independently using only visible messages and evidence. "
            "For searches check query grounding at action time, whether it targets a real "
            "missing relation, and whether top-3 adds relevant evidence. For answers check "
            "direct support, uniqueness and clarity of the question, exact answer fit, and "
            "no future-evidence or cross-document leap. Mark uncertainty explicitly."
        ),
        "groups": packet_groups,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    packet_hash = hashlib.sha256(encoded).hexdigest()
    write(PACKET, {**payload, "packet_sha256": packet_hash})
    packet_file_hash = sha(PACKET)
    packet_text = PACKET.read_text(encoding="utf-8")
    if "v117_train:" in packet_text or re.search(r"\b[0-9a-f]{20}\b", packet_text):
        raise RuntimeError("private_source_id_leak_in_v7_packet")
    write(BINDING, {
        "packet_payload_sha256": packet_hash,
        "packet_file_sha256": packet_file_hash,
        "replay_source_sha256": sha(REPLAY),
        "v7_drafts_sha256": sha(DRAFTS),
        "source_mapping": binding,
        "reviewer_sees_source_id": False,
        "training_allowed": False,
    })
    write(MANIFEST, {
        "version": "v124-v7-repair-blind-manifest-v1",
        "packet_payload_sha256": packet_hash,
        "packet_file_sha256": packet_file_hash,
        "packet_source_sha256": sha(REPLAY),
        "v7_drafts_sha256": sha(DRAFTS),
        "groups": len(packet_groups),
        "states": len(binding),
        "source_ids_removed": True,
        "source_document_ids_remapped": True,
        "gold_fields_absent": True,
        "other_votes_included": False,
        "training_allowed": False,
    })
    print(json.dumps({"groups": len(packet_groups), "states": len(binding),
                      "packet_payload_sha256": packet_hash,
                      "packet_file_sha256": packet_file_hash,
                      "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
