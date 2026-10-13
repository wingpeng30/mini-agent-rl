"""Build one anonymous, action-time-complete review packet for v117 repairs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.v123_cpu_core import digest, readl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/evidence-sft-v124-forensic"


def main():
    candidates = json.loads((OUT / "v117-repair-candidates-cpu-private.json").read_text(encoding="utf-8"))
    inventory = {row["record_id"]: row for row in readl(OUT / "inventory-private.jsonl")}
    lineage = {row["record_id"]: row for row in readl(OUT / "lineage-private.jsonl")}
    states, mapping = [], []
    for group_index, group in enumerate(candidates["candidates"], 1):
        for state_index, state in enumerate(group["states"], 1):
            record_id = state["record_id"]
            source = inventory[record_id]
            docs, alias_by_id = [], {}

            def alias(doc_id):
                if doc_id not in alias_by_id:
                    alias_by_id[doc_id] = f"D{len(alias_by_id) + 1}"
                return alias_by_id[doc_id]

            before = []
            for document in source["before_action"]:
                before.append({"document_id": alias(document["id"]),
                               "title": document["title"], "text": document["text"]})
            action = state["target_action"]
            after = []
            if action.get("action") == "search":
                matches = lineage[record_id]["matches"]
                if len(matches) != 1 or not isinstance(matches[0].get("retrieved_after_search"), list):
                    raise RuntimeError(f"missing_frozen_search_result:{record_id}")
                for document in matches[0]["retrieved_after_search"]:
                    after.append({"document_id": alias(document["id"]),
                                  "title": document["title"], "text": document["text"]})
            messages = []
            for message in state["revised_visible_messages"]:
                item = {"role": message["role"], "content": message["content"]}
                if item["role"] == "tool":
                    try:
                        payload = json.loads(item["content"])
                        for document in payload.get("result", []):
                            if isinstance(document, dict) and document.get("id"):
                                old_id = document["id"]
                                document["id"] = alias(old_id)
                                if isinstance(document.get("source"), str):
                                    document["source"] = alias(old_id)
                        item["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    except (ValueError, TypeError):
                        pass
                messages.append(item)
            audit_id = f"G{group_index}-S{state_index}"
            states.append({"audit_id": audit_id, "question_group": f"G{group_index}",
                           "question": group["revised_question"],
                           "visible_messages": messages,
                           "pre_action_evidence": before,
                           "proposed_action": action,
                           "post_action_retrieval_results": after if action.get("action") == "search" else None})
            mapping.append({"audit_id": audit_id, "record_id": record_id,
                            "source_record_sha256": state["source_record_sha256"],
                            "revised_prompt_ids_sha256": state["prompt_ids_sha256"],
                            "reviewer_sees_source_id": False})

    packet = {"version": "v124-anonymous-repair-review-packet-v1",
              "states": states, "training_allowed": False}
    encoded = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    packet_hash = hashlib.sha256(encoded).hexdigest()
    if len(states) != 10 or len({row["audit_id"] for row in states}) != 10:
        raise RuntimeError("blind_packet_count_or_id_error")
    serialized = encoded.decode("utf-8")
    if any(row["record_id"] in serialized for row in mapping):
        raise RuntimeError("source_id_leak_in_blind_packet")
    manifest = {"version": "v124-anonymous-repair-review-manifest-v1",
                "packet_sha256": packet_hash, "states": len(states), "groups": 5,
                "search_states_with_actual_results": sum(
                    row["proposed_action"].get("action") == "search"
                    and bool(row["post_action_retrieval_results"]) for row in states),
                "answer_states": sum(row["proposed_action"].get("action") == "answer" for row in states),
                "source_file_sha256": candidates["source_sha256"],
                "training_allowed": False}
    (OUT / "v117-repair-blind-packet-private.json").write_bytes(encoded + b"\n")
    (OUT / "v117-repair-blind-packet-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "v117-repair-blind-packet-binding-private.json").write_text(
        json.dumps({"packet_sha256": packet_hash, "mapping": mapping,
                    "training_allowed": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
