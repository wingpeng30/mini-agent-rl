"""Create an anonymous, one-reviewer packet for strict v117 repair-link candidates."""
from __future__ import annotations

import hashlib
import json

from scripts.v123_cpu_core import ROOT, digest, readl, write, writel
from scripts.v124_forensic import OUT
from scripts.v125_review_protocol import SYSTEM

DEST = OUT / "pending-luna-v125-v117-repair"


def main() -> None:
    if DEST.exists():
        raise FileExistsError("refuse_to_overwrite_v125_repair_review_packet")
    link_path = OUT / "v125-v117-repair-trajectory-link-audit-private.json"
    trajectories_path = OUT / "repair-draft-trajectories-cpu-private.jsonl"
    report = json.loads(link_path.read_text(encoding="utf-8"))
    if report["original_record_bindings_pass"] is not True:
        raise ValueError("repair_source_binding_failed")
    inventory = {r["record_id"]: r for r in readl(OUT / "inventory-private.jsonl")}
    reviews = {r["record_id"]: r for p in sorted(OUT.glob("primary-batch-*.jsonl"))
               for r in readl(p)}
    for p in sorted(OUT.glob("primary-amendments*-private.jsonl")):
        for change in readl(p):
            reviews[change["record_id"]] = {
                **reviews[change["record_id"]],
                "disposition": change["effective_disposition"],
            }
    for change in readl(OUT / "primary-adjudication-overrides-private.jsonl"):
        reviews[change["record_id"]] = {
            **reviews[change["record_id"]],
            "disposition": change["effective_disposition"],
        }
    trajectory_groups = readl(trajectories_path)
    pair_candidates = []
    state_by_id = {}
    for group in trajectory_groups:
        for state in group["states"]:
            state_by_id[state["record_id"]] = state
    for finding in report["findings"]:
        for link in finding["strict_action_result_citation_links"]:
            if (link["new_question_vs_retained_core"] and
                    link["search_disposition"] == "repair_candidate" and
                    link["answer_disposition"] == "repair_candidate"):
                pair_candidates.append((link["search_record_id"], link["answer_record_id"]))
    if len(pair_candidates) != 3 or len(set(pair_candidates)) != 3:
        raise ValueError("strict_repair_candidate_count_changed")

    source_packets = []
    bindings = []
    for rid_pair in pair_candidates:
        for record_id in rid_pair:
            state = state_by_id[record_id]
            if (state["original_row_sha256"] != inventory[record_id]["source_row_sha256"] or
                    reviews[record_id]["disposition"] != "repair_candidate"):
                raise ValueError("repair_state_binding_or_disposition_changed")
            payload = {
                "question": state["question"],
                "visible_messages": state["visible_messages"],
                "before_action": state["before_action_evidence"],
                "proposed_action": state["proposed_action"],
                "retrieved_after_search": state["retrieved_after_search"]
                    if state["proposed_action"]["action"] == "search" else [],
            }
            source_packets.append((record_id, payload))
    source_packets.sort(key=lambda item: hashlib.sha256(
        (item[1]["question"] + digest(item[1]["proposed_action"])).encode()).hexdigest())
    anonymous, bindings = [], []
    for index, (record_id, payload) in enumerate(source_packets, 1):
        case_id = f"R{index:03d}"
        public = {"case_id": case_id, **payload}
        anonymous.append(public)
        bindings.append({"case_id": case_id, "record_id": record_id,
                         "source_row_sha256": inventory[record_id]["source_row_sha256"],
                         "review_packet_sha256": digest(public)})

    DEST.mkdir(parents=True)
    packet_path = DEST / "work-luna-01.jsonl"
    binding_path = DEST / "binding-private.jsonl"
    rubric_path = DEST / "rubric.txt"
    writel(packet_path, anonymous)
    writel(binding_path, bindings)
    rubric_path.write_text(SYSTEM + "\n", encoding="utf-8")
    manifest = {
        "version": "v125-luna-v117-repair-packet-v1",
        "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "binding_sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        "rubric_sha256": hashlib.sha256(rubric_path.read_bytes()).hexdigest(),
        "source_trajectory_sha256": hashlib.sha256(trajectories_path.read_bytes()).hexdigest(),
        "strict_link_report_sha256": hashlib.sha256(link_path.read_bytes()).hexdigest(),
        "states": len(anonymous), "questions": len({p["question"] for p in anonymous}),
        "gold_or_source_ids_in_packet": False, "prior_votes_in_packet": False,
        "reviewer_has_pair_ids_or_quotas": False, "training_allowed": False,
    }
    write(DEST / "manifest-private.json", manifest)
    print(json.dumps({"states": len(anonymous), "questions": manifest["questions"],
        "packet_sha256": manifest["packet_sha256"], "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
