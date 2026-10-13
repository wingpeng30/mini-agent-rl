"""Create one source-anonymous, evidence-complete packet for repair-state review."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from scripts.v123_cpu_core import digest, readl, sha, write
from scripts.v124_forensic import OUT

SOURCE = OUT / "repair-draft-states-cpu-v2-private.jsonl"
PACKET = OUT / "repair-draft-review-packet-private.json"
BINDING = OUT / "repair-draft-review-binding-private.json"
MANIFEST = OUT / "repair-draft-review-manifest.json"


def remap_messages(messages: list[dict], aliases: dict[str, str]) -> list[dict]:
    result = []
    for message in messages:
        item = dict(message)
        if item.get("role") == "tool":
            try:
                payload = json.loads(item["content"])
                for doc in payload.get("result", []):
                    if not isinstance(doc, dict):
                        continue
                    old = doc.get("id")
                    if old:
                        alias = aliases.setdefault(old, f"D{len(aliases) + 1}")
                        doc["id"] = alias
                        if "source" in doc:
                            doc["source"] = alias
                        if "ref" in doc:
                            doc["ref"] = alias
                item["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            except (KeyError, TypeError, ValueError):
                pass
        result.append(item)
    return result


def remap_docs(docs: list[dict] | None, aliases: dict[str, str]):
    if docs is None:
        return None
    result = []
    for doc in docs:
        old = doc.get("document_id") or doc.get("id")
        alias = aliases.setdefault(old, f"D{len(aliases) + 1}") if old else None
        result.append({"document_id": alias, "title": doc.get("title"), "text": doc.get("text")})
    return result


def main() -> None:
    groups = readl(SOURCE)
    if len(groups) != 18 or any(not g["token_limits_passed"] for g in groups):
        raise RuntimeError("repair_candidate_preflight_failed")
    # Stable but source-independent ordering; keep complete same-question groups.
    ordered = sorted(groups, key=lambda g: hashlib.sha256(
        ("v124-blind-group:" + g["question"]).encode("utf-8")).hexdigest())
    packet_groups, binding, used_state_ids = [], [], set()
    for group_index, group in enumerate(ordered, 1):
        alias_by_doc: dict[str, str] = {}
        states = []
        for state_index, state in enumerate(group["states"], 1):
            anon_id = f"A{len(used_state_ids) + 1:03d}"
            if anon_id in used_state_ids:
                raise RuntimeError("duplicate_anonymous_state_id")
            used_state_ids.add(anon_id)
            states.append({"state_id": anon_id,
                "visible_messages": remap_messages(state["visible_messages"], alias_by_doc),
                "pre_action_evidence": remap_docs(state["before_action_evidence"], alias_by_doc),
                "proposed_action": state["proposed_action"],
                "post_action_retrieval_results": remap_docs(state["retrieved_after_search"], alias_by_doc)})
            binding.append({"state_id": anon_id, "record_id": state["record_id"],
                "original_row_sha256": state["original_row_sha256"],
                "revised_row_sha256": state["revised_row_sha256"],
                "prompt_ids_sha256": state["prompt_ids_sha256"],
                "target_mask_sha256": state["target_mask_sha256"],
                "reviewer_sees_source_id": False})
        packet_groups.append({"pair_id": f"P{group_index:02d}",
            "question": group["question"], "states": states})
    payload = {"version": "v124-repair-independent-review-packet-v1",
        "review_instructions": "Review each proposed action using only that state's visible messages and evidence. For search actions assess action-time query grounding, whether retrieved results supply a relevant missing fact, and whether the action is needed. For answers assess direct support and whether the question is well-formed and uniquely answerable. Mark uncertainty or unsupported claims explicitly. Do not infer hidden source answers.",
        "groups": packet_groups}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    content = encoded.decode("utf-8")
    if "v117_train:" in content or re.search(r"\b[0-9a-f]{20}\b", content):
        raise RuntimeError("private_source_id_leak_in_blind_packet")
    packet_hash = hashlib.sha256(encoded).hexdigest()
    packet_obj = {**payload, "packet_sha256": packet_hash}
    # Hash is over payload before adding the hash field.
    write(PACKET, packet_obj)
    write(BINDING, {"packet_sha256": packet_hash, "mapping": binding,
                    "source_sha256": sha(SOURCE), "training_allowed": False})
    write(MANIFEST, {"version": "v124-repair-independent-review-manifest-v1",
        "packet_sha256": packet_hash, "packet_source_sha256": sha(SOURCE),
        "groups": len(packet_groups), "states": len(binding),
        "blind_source_ids_removed": True, "gold_fields_absent": True,
        "reviewer_votes_included": False, "training_allowed": False})
    print(json.dumps({"groups": len(packet_groups), "states": len(binding),
        "packet_sha256": packet_hash, "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
