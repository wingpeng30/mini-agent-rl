"""Create a separate anonymous blind packet for v6 repair drafts only."""
from __future__ import annotations

import hashlib
import json
import re

from scripts.v123_cpu_core import readl, sha, write
from scripts.v124_forensic import OUT
from scripts.v124_package_repair_blind_review import remap_docs, remap_messages

REPLAY = OUT / "repair-draft-states-cpu-v2-private.jsonl"
DRAFTS = OUT / "repair-drafts-v6-private.jsonl"
PACKET = OUT / "repair-v6-blind-review-packet-private.json"
BINDING = OUT / "repair-v6-blind-review-binding-private.json"
MANIFEST = OUT / "repair-v6-blind-review-manifest.json"


def main() -> None:
    drafts = readl(DRAFTS)
    expected = {r["record_id"] for r in drafts}
    replay_groups = readl(REPLAY)
    selected = [g for g in replay_groups if set(g["record_ids"]) & expected]
    if {rid for g in selected for rid in g["record_ids"]} != expected:
        raise RuntimeError("v6_replay_group_coverage_mismatch")
    if sum(len(g["states"]) for g in selected) != len(expected):
        raise RuntimeError("v6_replay_state_count_mismatch")
    if any(not all(s["within_token_limits"] for s in g["states"]) for g in selected):
        raise RuntimeError("v6_token_preflight_failed")
    selected.sort(key=lambda g: hashlib.sha256(
        ("v124-v6-blind-group:" + g["question"]).encode("utf-8")).hexdigest())

    packet_groups, binding = [], []
    next_state = 0
    for group_index, group in enumerate(selected, 1):
        aliases: dict[str, str] = {}
        states = []
        for state in group["states"]:
            next_state += 1
            anonymous_id = f"V6S{next_state:03d}"
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
        packet_groups.append({"pair_id": f"V6P{group_index:02d}",
                              "question": group["question"], "states": states})

    payload = {
        "version": "v124-v6-repair-blind-packet-v1",
        "review_instructions": (
            "Review each state independently using only its visible messages and evidence. "
            "For search actions assess action-time query grounding, whether the query targets a "
            "missing relation, and whether the top-3 supplies relevant new evidence. For answers "
            "assess direct support, well-formedness and uniqueness of the question, and whether "
            "the answer follows without future-evidence or cross-document leakage. Mark uncertainty "
            "or unsupported claims explicitly. Do not infer hidden source answers or use outside facts."
        ),
        "groups": packet_groups,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    text = encoded.decode("utf-8")
    if "v117_train:" in text or re.search(r"\b[0-9a-f]{20}\b", text):
        raise RuntimeError("private_source_id_leak_in_v6_packet")
    packet_hash = hashlib.sha256(encoded).hexdigest()
    write(PACKET, {**payload, "packet_sha256": packet_hash})
    write(BINDING, {
        "packet_sha256": packet_hash,
        "replay_source_sha256": sha(REPLAY),
        "v6_drafts_sha256": sha(DRAFTS),
        "source_mapping": binding,
        "reviewer_sees_source_id": False,
        "training_allowed": False,
    })
    write(MANIFEST, {
        "version": "v124-v6-repair-blind-manifest-v1",
        "packet_sha256": packet_hash,
        "packet_source_sha256": sha(REPLAY),
        "v6_drafts_sha256": sha(DRAFTS),
        "groups": len(packet_groups),
        "states": len(binding),
        "source_ids_removed": True,
        "source_document_ids_remapped": True,
        "gold_fields_absent": True,
        "other_votes_included": False,
        "training_allowed": False,
    })
    print(json.dumps({"groups": len(packet_groups), "states": len(binding),
                      "packet_sha256": packet_hash, "training_allowed": False},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
