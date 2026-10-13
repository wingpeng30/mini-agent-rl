"""Package only the v8 NXT TakeOver repair group for blind review."""
from __future__ import annotations

import hashlib
import json
import re

from scripts.v123_cpu_core import readl, sha, write
from scripts.v124_forensic import OUT
from scripts.v124_package_repair_blind_review import remap_docs, remap_messages

REPLAY = OUT / "repair-draft-states-cpu-v2-private.jsonl"
DRAFTS = OUT / "repair-drafts-v8-private.jsonl"
PACKET = OUT / "repair-v8-blind-review-packet-private.json"
BINDING = OUT / "repair-v8-blind-review-binding-private.json"
MANIFEST = OUT / "repair-v8-blind-review-manifest.json"


def main() -> None:
    drafts = readl(DRAFTS)
    expected = {r["record_id"] for r in drafts}
    groups = [g for g in readl(REPLAY) if set(g["record_ids"]) & expected]
    if len(groups) != 1 or set(groups[0]["record_ids"]) != expected:
        raise RuntimeError("v8_complete_group_coverage_mismatch")
    group = groups[0]
    if len(group["states"]) != 2 or not group["token_limits_passed"]:
        raise RuntimeError("v8_pair_or_token_preflight_failed")
    states, binding = [], []
    aliases: dict[str, str] = {}
    for n, state in enumerate(group["states"], 1):
        anon = f"V8S{n:03d}"
        states.append({
            "state_id": anon,
            "visible_messages": remap_messages(state["visible_messages"], aliases),
            "pre_action_evidence": remap_docs(state["before_action_evidence"], aliases),
            "proposed_action": state["proposed_action"],
            "post_action_retrieval_results": remap_docs(state["retrieved_after_search"], aliases),
        })
        binding.append({"state_id": anon, "record_id": state["record_id"],
            "source_row_sha256": state["original_row_sha256"],
            "revised_row_sha256": state["revised_row_sha256"],
            "prompt_ids_sha256": state["prompt_ids_sha256"],
            "target_mask_sha256": state["target_mask_sha256"]})
    payload = {"version": "v124-v8-repair-blind-packet-v1",
        "review_instructions": (
            "Review each state independently using only visible messages and evidence. For the search, "
            "assess action-time query grounding, whether it targets a missing relation needed by the question, "
            "and whether top-3 supplies a relevant intermediate or final fact. For the answer, assess direct "
            "support, question clarity and uniqueness, answer fit, and no future-evidence or cross-document leap. "
            "Mark uncertainty explicitly; cite exact short quotes from one document at a time."
        ),
        "groups": [{"pair_id": "V8P01", "question": group["question"], "states": states}]}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
    payload_hash = hashlib.sha256(canonical).hexdigest()
    write(PACKET, {**payload, "packet_sha256": payload_hash})
    file_hash = sha(PACKET)
    packet_text = PACKET.read_text(encoding="utf-8")
    if "v117_train:" in packet_text or re.search(r"\b[0-9a-f]{20}\b", packet_text):
        raise RuntimeError("private_source_id_leak_in_v8_packet")
    write(BINDING, {"packet_payload_sha256": payload_hash, "packet_file_sha256": file_hash,
        "replay_source_sha256": sha(REPLAY), "v8_drafts_sha256": sha(DRAFTS),
        "source_mapping": binding, "reviewer_sees_source_id": False, "training_allowed": False})
    write(MANIFEST, {"version": "v124-v8-repair-blind-manifest-v1",
        "packet_payload_sha256": payload_hash, "packet_file_sha256": file_hash,
        "packet_source_sha256": sha(REPLAY), "v8_drafts_sha256": sha(DRAFTS),
        "groups": 1, "states": 2, "source_ids_removed": True,
        "source_document_ids_remapped": True, "gold_fields_absent": True,
        "other_votes_included": False, "training_allowed": False})
    print(json.dumps({"groups": 1, "states": 2, "packet_payload_sha256": payload_hash,
        "packet_file_sha256": file_hash, "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
