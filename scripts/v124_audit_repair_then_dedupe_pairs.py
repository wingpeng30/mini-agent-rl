"""Audit existing message-level links in the seven repair-then-deduplicate groups."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.v123_cpu_core import digest, norm, readl, write
from scripts.v124_forensic import OUT

TRAIN = Path(__file__).resolve().parents[1] / "data/evidence-sft-v117/frozen/train.jsonl"


def main() -> None:
    source_rows = readl(TRAIN)
    inventory_rows = readl(OUT / "inventory-private.jsonl")
    inventory = {r["record_id"]: r for r in inventory_rows}
    lineage = {r["record_id"]: r for r in readl(OUT / "lineage-private.jsonl")}
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
    rows_by_id = {f"v117_train:{i:04d}": row for i, row in enumerate(source_rows, 1)}
    backlog_path = OUT / "v117-remaining-repair-backlog-private.json"
    backlog = json.loads(backlog_path.read_text(encoding="utf-8"))
    findings = []

    for group in backlog["groups"]:
        if group["work_class"] != "repair_question_or_target_then_deduplicate":
            continue
        ids = [s["record_id"] for s in group["all_same_normalized_question_states"]]
        states = []
        for rid in ids:
            row = rows_by_id[rid]
            index = row["target_assistant_index"]
            states.append({
                "record_id": rid,
                "question_key": norm(inventory[rid]["question"]),
                "row": row,
                "target_index": index,
                "action": json.loads(row["messages"][index]["content"]),
                "effective_disposition": reviews[rid]["disposition"],
                "issues": reviews[rid]["issues"],
                "retrieval_lineage_exact": lineage[rid]["exact_prompt_action_match"],
                "retrieval_matches": lineage[rid]["matches"],
                "answer_citations": reviews[rid].get("citations", []),
            })
        links = []
        for search in (s for s in states if s["action"].get("action") == "search"):
            if not search["retrieval_lineage_exact"] or len(search["retrieval_matches"]) != 1:
                continue
            retrieved_ids = {d["id"] for d in search["retrieval_matches"][0]["retrieved_after_search"]}
            if not retrieved_ids:
                continue
            for answer in (s for s in states if s["action"].get("action") == "answer"
                           and search["target_index"] < s["target_index"]):
                messages = answer["row"]["messages"]
                cited_ids = {c["document_id"] for c in answer["answer_citations"]
                             if c.get("stage") == "before_action"}
                for pos in range(answer["target_index"]):
                    if messages[pos]["role"] != "assistant" or pos + 1 >= answer["target_index"]:
                        continue
                    try:
                        prior_action = json.loads(messages[pos]["content"])
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if prior_action != search["action"] or messages[pos + 1]["role"] != "tool":
                        continue
                    try:
                        observation = json.loads(messages[pos + 1]["content"])
                    except (TypeError, json.JSONDecodeError):
                        continue
                    observed_ids = {d.get("id") for d in observation.get("result", [])}
                    cited_shared = retrieved_ids & observed_ids & cited_ids
                    if not cited_shared:
                        continue
                    links.append({
                        "search_record_id": search["record_id"],
                        "search_disposition": search["effective_disposition"],
                        "answer_record_id": answer["record_id"],
                        "answer_disposition": answer["effective_disposition"],
                        "exact_search_prompt_prefix_match":
                            search["row"]["messages"][:search["target_index"]] == messages[:pos],
                        "shared_cited_document_ids": sorted(cited_shared),
                        "both_states_retained": (search["effective_disposition"] == "retain_candidate"
                                                  and answer["effective_disposition"] == "retain_candidate"),
                    })
                    break
        findings.append({
            "normalized_question_sha256": digest(norm(inventory[ids[0]]["question"])),
            "question": group["normalized_question_for_audit"],
            "flagged_state_ids": [s["record_id"] for s in group["flagged_states"]],
            "states": [{k: s[k] for k in ("record_id", "action", "effective_disposition", "issues")}
                       for s in states],
            "message_level_supported_links": links,
            "has_retained_linked_pair": any(x["both_states_retained"] for x in links),
            "has_exact_prefix_linked_pair": any(x["exact_search_prompt_prefix_match"] for x in links),
            "training_allowed": False,
        })
    result = {
        "version": "v124-repair-dedupe-message-pair-audit-v1",
        "backlog_payload_sha256": digest(backlog),
        "groups": len(findings),
        "groups_with_message_level_supported_links": sum(bool(f["message_level_supported_links"])
                                                          for f in findings),
        "groups_with_retained_linked_pair": sum(f["has_retained_linked_pair"] for f in findings),
        "groups_with_exact_prefix_linked_pair": sum(f["has_exact_prefix_linked_pair"]
                                                     for f in findings),
        "findings": findings,
        "interpretation": [
            "A message-level evidence link does not repair a false or ambiguous question.",
            "Only exact prompt-prefix links are trajectory-continuous under this audit.",
            "No repair, deduplication, semantic approval, or training label is created.",
        ],
        "training_allowed": False,
    }
    write(OUT / "repair-dedupe-message-pair-audit-private.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
