"""Package all five strict-linked v117 repair candidates for a fresh blind vote."""
from __future__ import annotations

import hashlib
import json

from scripts.v123_cpu_core import ROOT, digest, readl, write, writel
from scripts.v124_forensic import OUT
from scripts.v126_review_protocol import SYSTEM


DEST = OUT / "pending-review-v126-v117-repair"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError("refuse_to_overwrite_v126_independent_review_packet")
    trajectories_path = OUT / "repair-draft-trajectories-cpu-private.jsonl"
    links_path = OUT / "v125-v117-repair-trajectory-link-audit-private.json"
    replay_path = OUT / "v126-quarantined-repair-replay-candidates-private.jsonl"
    inventory_path = OUT / "inventory-private.jsonl"
    inventory = {r["record_id"]: r for r in readl(inventory_path)}
    trajectories = readl(trajectories_path)
    by_id = {s["record_id"]: s for g in trajectories for s in g["states"]}
    links = json.loads(links_path.read_text(encoding="utf-8"))

    pairs = []
    for finding in links["findings"]:
        for link in finding["strict_action_result_citation_links"]:
            if link["new_question_vs_retained_core"]:
                pairs.append((link["search_record_id"], link["answer_record_id"]))
    if len(pairs) != 5 or len(set(pairs)) != 5:
        raise ValueError("expected_five_strict_disjoint_repair_pairs")

    quarantined = {s["record_id_private"]: s
                   for group in readl(replay_path) for s in group["states"]}
    payloads = []
    source_by_case = []
    for search_id, answer_id in pairs:
        for rid in (search_id, answer_id):
            if rid in quarantined:
                s = quarantined[rid]
                payload = {
                    "question": s["question"],
                    "visible_messages": s["visible_messages"],
                    "before_action": s["before_action_evidence"],
                    "proposed_action": s["proposed_action"],
                    "retrieved_after_search": s["retrieved_after_search"]
                        if s["proposed_action"]["action"] == "search" else [],
                }
                row_sha = s["original_row_sha256"]
            else:
                s = by_id[rid]
                payload = {
                    "question": s["question"],
                    "visible_messages": s["visible_messages"],
                    "before_action": s["before_action_evidence"],
                    "proposed_action": s["proposed_action"],
                    "retrieved_after_search": s["retrieved_after_search"]
                        if s["proposed_action"]["action"] == "search" else [],
                }
                row_sha = s["original_row_sha256"]
            if inventory[rid]["source_row_sha256"] != row_sha:
                raise ValueError("source_row_binding_mismatch")
            payloads.append((rid, payload, row_sha))

    payloads.sort(key=lambda item: hashlib.sha256(
        (item[1]["question"] + digest(item[1]["proposed_action"])).encode()).hexdigest())
    public, binding = [], []
    for index, (rid, payload, row_sha) in enumerate(payloads, 1):
        case_id = f"I{index:03d}"
        packet = {"case_id": case_id, **payload}
        public.append(packet)
        binding.append({"case_id": case_id, "record_id": rid,
                        "source_row_sha256": row_sha,
                        "review_packet_sha256": digest(packet)})
    if len(public) != 10 or len({p["question"] for p in public}) != 5:
        raise ValueError("review_packet_pair_count_mismatch")

    DEST.mkdir(parents=True)
    packet_path = DEST / "work-review-01.jsonl"
    binding_path = DEST / "binding-private.jsonl"
    rubric_path = DEST / "rubric.txt"
    writel(packet_path, public)
    writel(binding_path, binding)
    rubric_path.write_text(SYSTEM + "\n", encoding="utf-8")
    manifest = {
        "version": "v126-independent-v117-repair-packet-v1",
        "packet_sha256": sha(packet_path),
        "binding_sha256": sha(binding_path),
        "rubric_sha256": sha(rubric_path),
        "trajectory_source_sha256": sha(trajectories_path),
        "strict_link_report_sha256": sha(links_path),
        "quarantine_replay_sha256": sha(replay_path),
        "states": len(public), "questions": len({p["question"] for p in public}),
        "includes_prior_review_decisions": False,
        "includes_source_ids_or_cohort_labels": False,
        "reviewer_has_pair_ids_or_quota": False,
        "review_status": "prepared_not_submitted",
        "training_allowed": False,
    }
    write(DEST / "manifest-private.json", manifest)
    print(json.dumps({"states": manifest["states"], "questions": manifest["questions"],
        "packet_sha256": manifest["packet_sha256"],
        "review_status": manifest["review_status"], "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
