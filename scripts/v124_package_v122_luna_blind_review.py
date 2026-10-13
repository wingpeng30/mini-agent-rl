"""Build a narrow, anonymous Luna packet from viable v122 train pairs.

Only states with effective primary retain + a valid approving DeepSeek vote in
both members and a verified message-level search/result/citation link are
included. The reviewer receives none of the selection reason or outcome data.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.v123_cpu_core import ROOT, digest, readl, write, writel
from scripts.v124_forensic import OUT

SOURCE = ROOT / "data/evidence-sft-v122-train-candidates/review"
AUDIT = OUT / "v122-train-pair-integrity-audit-private.json"
DEST = OUT / "pending-luna-v122-train"


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    packets = readl(SOURCE / "internal-packets.jsonl")
    packet_by_case = {p["case_id"]: p for p in packets}
    eligible = readl(SOURCE / "eligible-private.jsonl")
    inventory_by_case = {r["original_case_id"]: r["record_id"]
                         for r in readl(OUT / "inventory-private.jsonl")
                         if r["cohort"] == "v122_train_candidates"}
    pair_audit = {r["source_id"]: r for r in audit["pairs"]}
    chosen = [r for r in eligible
              if pair_audit[r["source_id"]]["eligible_for_independent_review"]]
    if len(chosen) != 28 or len(chosen) * 2 != 56:
        raise RuntimeError("v122_blind_review_candidate_count_changed")
    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.mkdir(parents=True)

    anonymous, bindings = [], []
    ordinal = 0
    for group_no, row in enumerate(chosen, 1):
        group_id = f"q{group_no:03d}"
        members = [packet_by_case[c] for c in row["case_ids"]]
        if {p["proposed_action"]["action"] for p in members} != {"search", "answer"}:
            raise RuntimeError("candidate_group_action_mismatch")
        # Preserve pair order as search then answer, without exposing source IDs.
        members.sort(key=lambda p: 0 if p["proposed_action"]["action"] == "search" else 1)
        for packet in members:
            ordinal += 1
            anon_id = f"s{ordinal:03d}"
            public = {
                "case_id": anon_id,
                "group_id": group_id,
                "question": packet["question"],
                "proposed_action": packet["proposed_action"],
                "before_action": packet["before_action"],
                "retrieved_after_search": packet["retrieved_after_search"],
            }
            public["packet_sha256"] = digest(public)
            anonymous.append(public)
            bindings.append({"case_id": anon_id, "group_id": group_id,
                "source_id": row["source_id"], "original_case_id": packet["case_id"],
                "record_id": inventory_by_case[packet["case_id"]],
                "original_packet_sha256": packet["packet_sha256"],
                "anonymous_packet_sha256": public["packet_sha256"]})

    # Three complete-pair batches preserve group boundaries and cap each at 20 states.
    batch_specs = [("work-luna-01.jsonl", 20), ("work-luna-02.jsonl", 20),
                   ("work-luna-03.jsonl", 16)]
    cursor = 0
    batch_manifest = {}
    for filename, count in batch_specs:
        batch = anonymous[cursor:cursor + count]
        if len(batch) != count or count % 2:
            raise RuntimeError("blind_batch_split_or_size_error")
        writel(DEST / filename, batch)
        batch_manifest[filename] = {"states": len(batch), "sha256": digest(batch)}
        cursor += count
    if cursor != len(anonymous):
        raise RuntimeError("blind_batch_coverage_mismatch")
    writel(DEST / "blind-binding-private.jsonl", bindings)
    manifest = {"version": "v124-v122-luna-blind-packet-v1",
        "anonymous_states": len(anonymous), "anonymous_groups": len(chosen),
        "packet_payload_sha256": digest(anonymous), "batches": batch_manifest,
        "binding_sha256": digest(bindings), "selection_basis_private":
            "effective primary retain, valid approving DeepSeek vote for both states, and exact raw-message search/result/answer-citation link",
        "review_status": "not_run", "training_allowed": False}
    write(DEST / "manifest-private.json", manifest)
    print(json.dumps({"anonymous_states": len(anonymous),
        "anonymous_groups": len(chosen), "batches": batch_manifest,
        "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
