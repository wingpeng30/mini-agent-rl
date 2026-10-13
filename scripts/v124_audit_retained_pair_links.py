"""Measure query/result/citation links between retained v117 search and answer states."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.v123_cpu_core import digest, norm, readl, write
from scripts.v124_forensic import OUT

TRAIN = Path(__file__).resolve().parents[1] / "data/evidence-sft-v117/frozen/train.jsonl"


def _effective_reviews():
    rows = [r for p in sorted(OUT.glob("primary-batch-*.jsonl")) for r in readl(p)]
    effective = {r["record_id"]: r for r in rows}
    for p in sorted(OUT.glob("primary-amendments*-private.jsonl")):
        for change in readl(p):
            effective[change["record_id"]] = {
                **effective[change["record_id"]],
                "disposition": change["effective_disposition"],
            }
    for change in readl(OUT / "primary-adjudication-overrides-private.jsonl"):
        effective[change["record_id"]] = {
            **effective[change["record_id"]],
            "disposition": change["effective_disposition"],
        }
    return effective


def _search_doc_ids(lineage_row):
    if not lineage_row["exact_prompt_action_match"] or len(lineage_row["matches"]) != 1:
        return set()
    return {d["id"] for d in lineage_row["matches"][0]["retrieved_after_search"]}


def main() -> None:
    source_rows = readl(TRAIN)
    inventory_rows = readl(OUT / "inventory-private.jsonl")
    inventory = {r["record_id"]: r for r in inventory_rows}
    lineage_rows = readl(OUT / "lineage-private.jsonl")
    lineage = {r["record_id"]: r for r in lineage_rows}
    reviews = _effective_reviews()
    assert len(source_rows) == 345 and set(reviews) == set(inventory)

    retained = []
    for ordinal, row in enumerate(source_rows, 1):
        rid = f"v117_train:{ordinal:04d}"
        if reviews[rid]["disposition"] != "retain_candidate":
            continue
        target = row["target_assistant_index"]
        action = json.loads(row["messages"][target]["content"])
        retained.append({
            "record_id": rid,
            "question_norm": norm(inventory[rid]["question"]),
            "row": row,
            "target_index": target,
            "action": action,
            "review": reviews[rid],
        })

    searches = [r for r in retained if r["action"].get("action") == "search"]
    answers = [r for r in retained if r["action"].get("action") == "answer"]
    linked = []
    for answer in answers:
        messages = answer["row"]["messages"]
        cited_ids = {c["document_id"] for c in answer["review"].get("citations", [])
                     if c.get("stage") == "before_action"}
        for search in searches:
            if search["question_norm"] != answer["question_norm"]:
                continue
            if search["target_index"] >= answer["target_index"]:
                continue
            docs_from_search = _search_doc_ids(lineage[search["record_id"]])
            if not docs_from_search:
                continue
            for pos in range(answer["target_index"]):
                if messages[pos]["role"] != "assistant" or pos + 1 >= answer["target_index"]:
                    continue
                try:
                    action = json.loads(messages[pos]["content"])
                except (TypeError, json.JSONDecodeError):
                    continue
                if action != search["action"] or action.get("action") != "search":
                    continue
                if messages[pos + 1]["role"] != "tool":
                    continue
                try:
                    observation = json.loads(messages[pos + 1]["content"])
                except (TypeError, json.JSONDecodeError):
                    continue
                observed_ids = {d.get("id") for d in observation.get("result", [])}
                shared_ids = docs_from_search & observed_ids
                supported_ids = shared_ids & cited_ids
                if not shared_ids:
                    continue
                search_prompt = search["row"]["messages"][:search["target_index"]]
                answer_prefix = messages[:pos]
                linked.append({
                    "search_record_id": search["record_id"],
                    "answer_record_id": answer["record_id"],
                    "normalized_question_sha256": digest(search["question_norm"]),
                    "same_source_case_id": bool(
                        set(m["source_case_id"] for m in lineage[search["record_id"]]["matches"])
                        & set(m["source_case_id"] for m in lineage[answer["record_id"]]["matches"])),
                    "search_prompt_exactly_matches_answer_prefix": search_prompt == answer_prefix,
                    "search_action_position_in_answer": pos,
                    "shared_retrieved_document_ids": sorted(shared_ids),
                    "answer_cited_shared_document_ids": sorted(supported_ids),
                    "answer_evidence_linked": bool(supported_ids),
                })

    by_q = defaultdict(list)
    for row in retained:
        by_q[row["question_norm"]].append(row)
    old_preflight_pairs = []
    for q, rows in by_q.items():
        if len(rows) == 2 and {r["action"].get("action") for r in rows} == {"search", "answer"}:
            old_preflight_pairs.append(q)
    linked_q = {row["normalized_question_sha256"] for row in linked if row["answer_evidence_linked"]}
    strict_old_pairs = [q for q in old_preflight_pairs if digest(q) in linked_q]
    exact_prefix_q = {row["normalized_question_sha256"] for row in linked
                      if row["answer_evidence_linked"]
                      and row["search_prompt_exactly_matches_answer_prefix"]}
    result = {
        "version": "v124-retained-pair-link-audit-v1",
        "source_sha256": digest(source_rows),
        "primary_inventory_sha256": digest(inventory_rows),
        "lineage_payload_sha256": digest(lineage_rows),
        "retained_states": len(retained),
        "normalized_question_groups": len(by_q),
        "retained_search_states": len(searches),
        "retained_answer_states": len(answers),
        "legacy_question_only_one_search_one_answer_groups": len(old_preflight_pairs),
        "action_and_observation_links_with_answer_citation": sum(
            row["answer_evidence_linked"] for row in linked),
        "unique_questions_with_linked_search_answer_evidence": len(linked_q),
        "legacy_pair_groups_with_linked_search_answer_evidence": len(strict_old_pairs),
        "exact_full_prompt_prefix_links_with_answer_citation": len(exact_prefix_q),
        "links": linked,
        "unlinked_legacy_pair_question_hashes": sorted(
            digest(q) for q in old_preflight_pairs if digest(q) not in linked_q),
        "interpretation": [
            "A shared source-case ID is not required or assumed to be a trajectory ID.",
            "An action/result/citation link is weaker than identical full prompt-prefix continuity.",
            "The report counts only retrieved documents cited by the retained answer review.",
            "No pairing, label, or training authorization is created by this diagnostic.",
        ],
        "training_allowed": False,
    }
    write(OUT / "retained-pair-link-audit-private.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"links", "unlinked_legacy_pair_question_hashes"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
