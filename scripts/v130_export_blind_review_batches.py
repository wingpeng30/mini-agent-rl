"""Export the one frozen internal pool as hash-bound anonymous blind-review batches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.v123_cpu_core import ROOT, read, readl, sha, write
from scripts.v129_review_protocol import contract_identity
from scripts.v130_deepseek_review import _payload_hash

BASE = ROOT / "data/evidence-sft-v124-forensic/goal-v128"
FROZEN = BASE / "g4-new-pool/frozen-v130"
OUT = BASE / "new-internal-audit/packets"
MAX_STATES_PER_BATCH = 20


def export() -> dict:
    audit_path = BASE / "g4-frozen-replay-audit-v130-private.json"
    audit = read(audit_path)
    freeze_path = FROZEN / "candidate-freeze-private.json"
    freeze = read(freeze_path)
    packets = readl(FROZEN / "review/internal-packets-private.jsonl")
    source_map = readl(FROZEN / "review/source-map-private.jsonl")
    if audit.get("passed") is not True or audit.get("training_allowed") is not False:
        raise ValueError("independent_freeze_replay_not_passed")
    if (audit.get("input_sha256", {}).get("data/evidence-sft-v124-forensic/goal-v128/g4-new-pool/frozen-v130/review/internal-packets-private.jsonl")
            != sha(FROZEN / "review/internal-packets-private.jsonl")):
        raise ValueError("audit_does_not_bind_frozen_packets")
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError("blind_review_export_already_exists_refuse_overwrite")
    if len(packets) != 200 or len(source_map) != 200 or len({p["case_id"] for p in packets}) != 200:
        raise ValueError("frozen_pool_packet_count_invalid")

    grouped: dict[str, list[dict]] = {}
    for source, packet in zip(source_map, packets):
        grouped.setdefault(source["source_id"], []).append(packet)
    pairs = []
    for source_id, rows in grouped.items():
        if (len(rows) != 2 or {p["proposed_action"]["action"] for p in rows} != {"search", "answer"}
                or len({p["question"] for p in rows}) != 1):
            raise ValueError("strict_pair_required_for_review_export")
        rows.sort(key=lambda p: 0 if p["proposed_action"]["action"] == "search" else 1)
        pairs.append((hashlib.sha256(("v130-review-batch:" + source_id).encode()).hexdigest(), rows))
    pairs.sort(key=lambda item: item[0])

    protocol = contract_identity()
    freeze_hash = sha(freeze_path)
    batch_records = []
    OUT.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(pairs), MAX_STATES_PER_BATCH // 2):
        number = offset // (MAX_STATES_PER_BATCH // 2) + 1
        blind_packets = []
        for _, pair in pairs[offset:offset + MAX_STATES_PER_BATCH // 2]:
            for packet in pair:
                action = packet["proposed_action"]
                clean_action = ({"action": "search", "query": action["query"]}
                                if action["action"] == "search"
                                else {"action": "answer", "answer": action["answer"]})
                def clean_docs(rows):
                    return [{"id": row["id"], "title": row.get("title", ""), "text": row["text"]}
                            for row in rows if isinstance(row.get("text"), str)]
                blind_packets.append({
                    "case_id": packet["case_id"],
                    "question": packet["question"],
                    "before_action": clean_docs(packet["before_action"]),
                    "proposed_action": clean_action,
                    "retrieved_after_search": clean_docs(packet["retrieved_after_search"] or []),
                    "protocol": {"version": protocol["version"]},
                })
        if len(blind_packets) > MAX_STATES_PER_BATCH:
            raise AssertionError("batch_state_limit_exceeded")
        bundle = {"version": "v130-anonymous-independent-review-bundle-v1",
                  "batch": f"g4-batch-{number:02d}", "review_authorized": True,
                  "training_allowed": False, "freeze_sha256": freeze_hash,
                  "protocol": protocol, "packets": blind_packets,
                  "payload_sha256": _payload_hash(blind_packets)}
        path = OUT / f"g4-batch-{number:02d}.json"
        write(path, bundle)
        batch_records.append({"batch": bundle["batch"], "path": path.relative_to(ROOT).as_posix(),
                              "states": len(blind_packets), "packet_file_sha256": sha(path),
                              "payload_sha256": bundle["payload_sha256"]})
    result = {"version": "v130-blind-review-export-v1", "status": "ready_for_two_independent_reviewers",
              "freeze_sha256": freeze_hash, "protocol": protocol,
              "frozen_internal_questions": 100, "frozen_states": 200,
              "batch_state_limit": MAX_STATES_PER_BATCH, "batches": batch_records,
              "training_allowed": False, "data_ready_for_experiment": False}
    write(BASE / "g4-blind-review-export-v130-private.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(export(), ensure_ascii=True, indent=2))
