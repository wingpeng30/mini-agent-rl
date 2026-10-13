"""Prepare blind, stage-explicit packets for still-unreviewed v124 drafts.

This is offline packaging only; it sends no data to an API and creates no vote.
Groups already attempted as v7/v8 are excluded to prevent accidental reruns.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict

from scripts.v123_cpu_core import digest, readl, sha, write
from scripts.v124_forensic import OUT
from scripts.v124_package_repair_blind_review import remap_docs, remap_messages


REPLAY = OUT / "repair-draft-states-cpu-v2-private.jsonl"
DEST = OUT / "pending-dual-review-v9"
EXCLUDED = {"v117_train:0055", "v117_train:0092", "v117_train:0216", "v117_train:0217",
            "v117_train:0027", "v117_train:0251"}
MAX_STATES = 20


def main() -> None:
    replay_groups = readl(REPLAY)
    ordered = sorted(replay_groups, key=lambda g: hashlib.sha256(
        ("v124-pending-review:" + g["question"]).encode("utf-8")).hexdigest())
    selected = [g for g in ordered if not set(g["record_ids"]).issubset(EXCLUDED)]
    if any(set(g["record_ids"]) & EXCLUDED for g in selected):
        raise RuntimeError("previously_attempted_group_must_not_be_repacked")
    if {rid for g in ordered if set(g["record_ids"]).issubset(EXCLUDED) for rid in g["record_ids"]} != EXCLUDED:
        raise RuntimeError("prior_attempt_exclusion_mismatch")
    if any(not all(s.get("within_token_limits") for s in g["states"]) for g in selected):
        raise RuntimeError("token_preflight_failed")

    batches: list[list[dict]] = []
    current: list[dict] = []
    count = 0
    for group in selected:
        size = len(group["states"])
        if size > MAX_STATES:
            raise RuntimeError("single_question_group_exceeds_batch_limit")
        if count and count + size > MAX_STATES:
            batches.append(current)
            current, count = [], 0
        current.append(group)
        count += size
    if current:
        batches.append(current)
    if not batches:
        raise RuntimeError("no_pending_review_groups")
    if DEST.exists():
        raise RuntimeError("destination_exists_refuse_overwrite")
    DEST.mkdir(parents=True)

    manifests = []
    for batch_no, batch in enumerate(batches, 1):
        batch_id = f"v9-batch-{batch_no:02d}"
        packet_groups, mapping, seen_ids = [], [], set()
        for group_no, group in enumerate(batch, 1):
            aliases: dict[str, str] = {}
            states = []
            for state in group["states"]:
                anonymous_id = f"V9B{batch_no:02d}S{len(seen_ids) + 1:03d}"
                if anonymous_id in seen_ids:
                    raise RuntimeError("anonymous_id_duplicate")
                seen_ids.add(anonymous_id)
                states.append({
                    "state_id": anonymous_id,
                    "visible_messages": remap_messages(state["visible_messages"], aliases),
                    "pre_action_evidence": remap_docs(state["before_action_evidence"], aliases),
                    "proposed_action": state["proposed_action"],
                    "post_action_retrieval_results": remap_docs(state.get("retrieved_after_search"), aliases),
                })
                mapping.append({
                    "state_id": anonymous_id,
                    "source_record_id": state["record_id"],
                    "original_row_sha256": state["original_row_sha256"],
                    "revised_row_sha256": state["revised_row_sha256"],
                    "prompt_ids_sha256": state["prompt_ids_sha256"],
                    "target_mask_sha256": state["target_mask_sha256"],
                })
            packet_groups.append({"pair_id": f"V9B{batch_no:02d}P{group_no:02d}",
                                  "question": group["question"], "states": states})
        payload = {
            "version": "v124-repair-blind-packet-v2",
            "review_instructions": (
                "Review each state independently using only its visible messages and evidence. "
                "For search, judge query grounding at that action time, whether the query targets a "
                "real missing relation, whether top-3 supplies relevant new evidence, and whether the "
                "action is needed. For answers, judge direct support, unique question meaning, exact "
                "answer fit, and absence of future-evidence or cross-document leakage. Mark uncertainty "
                "explicitly. Return one review per state. Citation schema is strict: each citation must "
                "have stage exactly one of 'visible_messages', 'pre_action_evidence', or "
                "'post_action_retrieval_results'. Use document_id 'user' only for the visible user "
                "message; otherwise copy the exact D-number in that stage. Quote a verbatim substring "
                "from that exact stage/document. Do not invent stage names, use a state ID as document_id, "
                "join documents in one quote, or infer hidden labels. Output JSON with reviews:[{state_id, "
                "decision:'approve'|'reject'|'uncertain', issues:[...], rationale:'...', citations:[{stage, "
                "document_id, quote}], query_grounded:true|false|null, relevant_new_evidence:true|false|null, "
                "answer_supported:true|false|null}]."
            ),
            "groups": packet_groups,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        payload_hash = hashlib.sha256(encoded).hexdigest()
        packet_path = DEST / f"{batch_id}-packet-private.json"
        binding_path = DEST / f"{batch_id}-binding-private.json"
        manifest_path = DEST / f"{batch_id}-manifest.json"
        packet_obj = {**payload, "packet_sha256": payload_hash}
        write(packet_path, packet_obj)
        content = packet_path.read_text(encoding="utf-8")
        if "v117_train:" in content or re.search(r"\b[0-9a-f]{20}\b", content):
            raise RuntimeError("source_id_leak_in_blind_packet")
        file_hash = sha(packet_path)
        write(binding_path, {"packet_sha256": payload_hash,
                             "source_replay_sha256": sha(REPLAY),
                             "source_mapping": mapping,
                             "reviewer_sees_source_mapping": False,
                             "training_allowed": False})
        manifest = {
            "batch_id": batch_id,
            "packet_payload_sha256": payload_hash,
            "packet_file_sha256": file_hash,
            "groups": len(packet_groups),
            "states": len(mapping),
            "maximum_states_per_batch": MAX_STATES,
            "all_groups_complete": True,
            "source_ids_removed": True,
            "document_ids_remapped": True,
            "gold_metadata_absent": True,
            "prior_v7_v8_attempts_excluded": True,
            "review_attempted": False,
            "training_allowed": False,
        }
        write(manifest_path, manifest)
        manifests.append(manifest)
    summary = {
        "version": "v124-pending-repair-review-packages-v1",
        "source_replay_sha256": sha(REPLAY),
        "excluded_previous_attempt_states": sorted(EXCLUDED),
        "batches": manifests,
        "total_groups": sum(m["groups"] for m in manifests),
        "total_states": sum(m["states"] for m in manifests),
        "review_attempted": False,
        "training_allowed": False,
    }
    write(DEST / "index-private.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "batches"} |
                     {"batches": [{k: m[k] for k in ("batch_id", "groups", "states",
                                                       "packet_payload_sha256", "packet_file_sha256")}
                                   for m in manifests]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
