"""Build two bounded, anonymous v126 review batches for recovered v117 pairs."""
from __future__ import annotations

import hashlib
import json

from scripts.v123_cpu_core import ROOT, digest, readl, write, writel
from scripts.v124_forensic import OUT
from scripts.v126_review_protocol import SYSTEM
from mini_agent_rl.answer_verifier import visible_documents
from scripts.build_v114_sft_candidates import norm


DEST = OUT / "pending-review-v126-pair-replay"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packet_payload(state):
    return {
        "question": state["question"],
        "visible_messages": state["visible_messages"],
        "before_action": state["before_action_evidence"],
        "proposed_action": state["proposed_action"],
        "retrieved_after_search": state.get("retrieved_after_search")
            if state["proposed_action"].get("action") == "search" else [],
    }


def main():
    if DEST.exists():
        raise FileExistsError("refuse_to_overwrite_v126_pair_review_packet")
    source_path = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
    trajectory_path = OUT / "repair-draft-trajectories-cpu-private.jsonl"
    repair_link_path = OUT / "v125-v117-repair-trajectory-link-audit-private.json"
    quarantine_path = OUT / "v126-quarantined-repair-replay-candidates-private.jsonl"
    dedupe_path = OUT / "v126-deduped-pair-replay-candidates-private.jsonl"
    extraction_path = OUT / "v126-retained-linked-search-state-candidates-private.jsonl"
    extraction_manifest_path = OUT / "v126-retained-linked-search-extraction-manifest-private.json"
    inventory_path = OUT / "inventory-private.jsonl"
    sources = readl(source_path)
    by_id = {f"v117_train:{i:04d}": row for i, row in enumerate(sources, 1)}
    inventory = {r["record_id"]: r for r in readl(inventory_path)}
    trajectories = readl(trajectory_path)
    traj_states = {s["record_id"]: s for g in trajectories for s in g["states"]}
    candidate_pairs = []

    # Ten exact action instances extracted from actual answer transcripts.
    for search in readl(extraction_path):
        answer_id = search["answer_state_record_id_private"]
        answer_row = by_id[answer_id]
        if digest(answer_row) != inventory[answer_id]["source_row_sha256"]:
            raise ValueError("extracted_answer_source_hash_mismatch")
        target_index = answer_row["target_assistant_index"]
        answer_messages = answer_row["messages"][:target_index]
        answer_state = {
            "question": answer_row["messages"][1]["content"],
            "visible_messages": answer_messages,
            "before_action_evidence": visible_documents(answer_messages),
            "proposed_action": json.loads(answer_row["messages"][target_index]["content"]),
            "retrieved_after_search": [],
            "record_id_private": answer_id,
            "source_row_sha256": digest(answer_row),
        }
        search_state = {
            **search,
            "record_id_private": search["candidate_id_private"],
            "source_row_sha256": search["source_answer_row_sha256"],
        }
        candidate_pairs.append(("trajectory_extraction", [search_state, answer_state]))

    # Three strict repair pairs that are outside the retained core.
    links = json.loads(repair_link_path.read_text(encoding="utf-8"))
    repair_ids = []
    for finding in links["findings"]:
        for link in finding["strict_action_result_citation_links"]:
            if (link["search_disposition"] == "repair_candidate" and
                    link["answer_disposition"] == "repair_candidate"):
                repair_ids.append((link["search_record_id"], link["answer_record_id"]))
    if len(repair_ids) != 3:
        raise ValueError("strict_repair_pair_count_changed")
    for search_id, answer_id in repair_ids:
        states = [traj_states[search_id], traj_states[answer_id]]
        candidate_pairs.append(("strict_repair", [
            {**state, "record_id_private": state["record_id"],
             "source_row_sha256": state["original_row_sha256"]} for state in states]))

    # Two source-attributed question replays of formerly quarantined pairs.
    for group in readl(quarantine_path):
        candidate_pairs.append(("quarantine_repair", [
            {**state, "record_id_private": state["record_id_private"],
             "source_row_sha256": state["original_row_sha256"]} for state in group["states"]]))

    # The Taylor duplicate group is already represented by the exact trajectory
    # extraction from its answer transcript; include only the distinct Ridl pair.
    extraction_by_question = {norm(r["question"]): r for r in readl(extraction_path)}
    overlap_resolved = []
    for group in readl(dedupe_path):
        if norm(group["question"]) in extraction_by_question:
            extracted_search = extraction_by_question[norm(group["question"])]
            extracted_answer_row = by_id[extracted_search["answer_state_record_id_private"]]
            extracted_answer_target = extracted_answer_row["target_assistant_index"]
            dedupe_search = next(s for s in group["states"]
                                 if s["proposed_action"].get("action") == "search")
            dedupe_answer = next(s for s in group["states"]
                                 if s["proposed_action"].get("action") == "answer")
            equivalent = (
                extracted_search["question"] == group["question"] and
                extracted_search["visible_messages"] == dedupe_search["visible_messages"] and
                extracted_search["proposed_action"] == dedupe_search["proposed_action"] and
                extracted_search["retrieved_after_search"] == dedupe_search["retrieved_after_search"] and
                extracted_answer_row["messages"][:extracted_answer_target] ==
                    dedupe_answer["visible_messages"] and
                json.loads(extracted_answer_row["messages"][extracted_answer_target]["content"]) ==
                    dedupe_answer["proposed_action"])
            if not equivalent:
                raise ValueError("dedupe_extraction_overlap_content_mismatch")
            overlap_resolved.append(group["record_ids_private"])
            continue
        candidate_pairs.append(("dedupe_repair", [
            {**state, "record_id_private": state["record_id_private"],
             "source_row_sha256": state["source_row_sha256"]} for state in group["states"]]))

    if len(candidate_pairs) != 16:
        raise ValueError("expected_16_unique_candidate_pairs")
    all_questions = {pair[1][0]["question"] for pair in candidate_pairs}
    if len(all_questions) != 16:
        raise ValueError("candidate_question_overlap")

    case_rows = []
    for group_type, states in candidate_pairs:
        if len(states) != 2 or {s["proposed_action"].get("action") for s in states} != {"search", "answer"}:
            raise ValueError("candidate_pair_action_shape_invalid")
        question = states[0]["question"]
        if states[1]["question"] != question:
            raise ValueError("pair_question_mismatch")
        for state in states:
            record_id = state["record_id_private"]
            if not record_id.startswith("derived_search_from:"):
                row = by_id[record_id]
                if digest(row) != state["source_row_sha256"]:
                    raise ValueError("packet_source_row_hash_mismatch")
            payload = packet_payload(state)
            payload["source_row_sha256_private"] = state["source_row_sha256"]
            case_rows.append((group_type, record_id, payload))

    case_rows.sort(key=lambda row: hashlib.sha256(
        (row[2]["question"] + digest(row[2]["proposed_action"])).encode()).hexdigest())
    anonymous = [{"case_id": f"P{i:03d}",
                  **{k: v for k, v in payload.items() if not k.endswith("_private")}}
                 for i, (_, _, payload) in enumerate(case_rows, 1)]
    bindings = [{"case_id": f"P{i:03d}", "candidate_group_type_private": group_type,
                 "record_id_private": record_id,
                 "source_row_sha256": payload["source_row_sha256_private"],
                 "review_packet_sha256": digest(anonymous[i - 1])}
                for i, (group_type, record_id, payload) in enumerate(case_rows, 1)]
    if len(anonymous) != 32 or len({p["case_id"] for p in anonymous}) != 32:
        raise ValueError("review_case_count_mismatch")

    DEST.mkdir(parents=True)
    rubric_path = DEST / "rubric.txt"
    rubric_path.write_text(SYSTEM + "\n", encoding="utf-8")
    batch1_path, batch2_path = DEST / "work-01.jsonl", DEST / "work-02.jsonl"
    writel(batch1_path, anonymous[:20])
    writel(batch2_path, anonymous[20:])
    binding_path = DEST / "binding-private.jsonl"
    writel(binding_path, bindings)
    manifest = {
        "version": "v126-pair-replay-review-packet-v1",
        "source_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                           for p in (source_path, trajectory_path, repair_link_path,
                                     quarantine_path, dedupe_path, extraction_path,
                                     extraction_manifest_path, inventory_path)},
        "batches": {
            "work-01.jsonl": {"states": 20, "questions": 10, "sha256": sha(batch1_path)},
            "work-02.jsonl": {"states": 12, "questions": 6, "sha256": sha(batch2_path)},
        },
        "duplicate_pair_candidates_excluded_after_content_equivalence": overlap_resolved,
        "rubric_sha256": sha(rubric_path), "binding_sha256": sha(binding_path),
        "states": len(anonymous), "questions": len(all_questions),
        "prior_review_decisions_in_packet": False,
        "source_ids_in_packet": False,
        "gold_answers_separate_from_proposed_actions": False,
        "review_status": "prepared_not_submitted_no_api_credentials",
        "training_allowed": False,
    }
    write(DEST / "manifest-private.json", manifest)
    print(json.dumps({"states": manifest["states"], "questions": manifest["questions"],
        "batches": manifest["batches"], "review_status": manifest["review_status"],
        "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
