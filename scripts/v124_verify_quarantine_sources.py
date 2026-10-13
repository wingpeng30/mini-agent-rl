"""Recheck two quarantined v117 search states against exact v114 lineage."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.v123_cpu_core import ROOT, digest, readl, write
from scripts.v124_forensic import OUT


def main() -> None:
    inventory = {r["record_id"]: r for r in readl(OUT / "inventory-private.jsonl")}
    lineage = {r["record_id"]: r for r in readl(OUT / "lineage-private.jsonl")}
    source_rows = readl(ROOT / "data/evidence-sft-v114-review/accepted.jsonl")
    by_case = {}
    for row in source_rows:
        packet = row.get("packet", row)
        if packet.get("case_id"):
            by_case[packet["case_id"]] = packet

    findings = []
    expected = {
        "v117_train:0149": {
            "case_id": "efac02413814a394d938",
            "top_titles": {"Rome", "Paganico Sabino", "Campagnano di Roma"},
            "before_titles": set(),
            "classification": "ambiguous_target_no_unique_common_oracle",
        },
        "v117_train:0321": {
            "case_id": "0befaf447f6e6411089f",
            "top_titles": {"Mary Williamson Averell", "American black bear", "Harriman State Park (Idaho)"},
            "before_titles": {"Mary Williamson Averell", "Harriman State Park (Idaho)", "Harriman State Park (New York)"},
            "classification": "same_name_park_ambiguity_no_answer_peer",
        },
    }
    for record_id, spec in expected.items():
        inv, link = inventory[record_id], lineage[record_id]
        assert link["exact_prompt_action_match"] is True
        match = next(m for m in link["matches"] if m["source_case_id"] == spec["case_id"])
        packet = by_case[spec["case_id"]]
        assert digest(packet) == match["source_packet_sha256"]
        assert digest([packet["visible_messages"], packet["proposed_action"]]) == digest(
            [inv["visible_messages"], inv["proposed_action"]])
        top = packet.get("retrieved_after_search", packet.get("retrieval_top3"))
        before = packet.get("before_action", packet.get("visible_evidence"))
        top_titles = {d["title"] for d in top}
        before_titles = {d["title"] for d in before}
        assert top_titles == spec["top_titles"]
        assert before_titles == spec["before_titles"]
        if record_id.endswith("0149"):
            assert inv["before_action"] == []
        else:
            assert {"Harriman State Park (Idaho)", "Harriman State Park (New York)"}.issubset(
                {d["title"] for d in inv["before_action"]})
            same_question = [r for r in inventory.values()
                             if r["cohort"] == "v117_train" and
                             r["question"].strip().lower() == inv["question"].strip().lower()]
            assert len(same_question) == 1
        findings.append({
            "record_id": record_id,
            "record_sha256": inv["record_sha256"],
            "v114_source_case_id": spec["case_id"],
            "v114_packet_sha256": match["source_packet_sha256"],
            "exact_prompt_action_match": True,
            "historical_top3_titles": sorted(top_titles),
            "historical_before_titles": sorted(before_titles),
            "disposition": "retain_quarantine_for_training",
            "diagnosis": spec["classification"],
        })

    report = {"version": "v124-quarantine-lineage-recheck-v1",
              "records": findings,
              "training_allowed": False,
              "sealed_assets_accessed": False}
    write(OUT / "quarantine-source-review-private.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
