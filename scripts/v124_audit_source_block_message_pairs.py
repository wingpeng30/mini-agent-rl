"""Check source-block groups against actual answer transcripts and citations."""
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
    lineage_rows = readl(OUT / "lineage-private.jsonl")
    lineage = {r["record_id"]: r for r in lineage_rows}
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
        if group["work_class"] != "evidence_or_target_block_requires_source_resolution":
            continue
        ids = [s["record_id"] for s in group["all_same_normalized_question_states"]]
        states = []
        for rid in ids:
            row = rows_by_id[rid]
            idx = row["target_assistant_index"]
            action = json.loads(row["messages"][idx]["content"])
            states.append({"record_id": rid, "action": action, "row": row,
                           "target_index": idx,
                           "disposition": reviews[rid]["disposition"]})
        searches = [s for s in states if s["action"].get("action") == "search"]
        retained_answers = [s for s in states if s["action"].get("action") == "answer"
                            and s["disposition"] == "retain_candidate"]
        links = []
        for search in searches:
            matches = lineage[search["record_id"]]["matches"]
            if not lineage[search["record_id"]]["exact_prompt_action_match"] or len(matches) != 1:
                continue
            retrieved_ids = {d["id"] for d in matches[0]["retrieved_after_search"]}
            for answer in retained_answers:
                messages = answer["row"]["messages"]
                cited_ids = {c["document_id"] for c in reviews[answer["record_id"]].get("citations", [])
                             if c.get("stage") == "before_action"}
                for pos in range(answer["target_index"]):
                    if messages[pos]["role"] != "assistant" or pos + 1 >= answer["target_index"]:
                        continue
                    try:
                        observed_action = json.loads(messages[pos]["content"])
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if observed_action != search["action"] or messages[pos + 1]["role"] != "tool":
                        continue
                    try:
                        observation = json.loads(messages[pos + 1]["content"])
                    except (TypeError, json.JSONDecodeError):
                        continue
                    observed_ids = {d.get("id") for d in observation.get("result", [])}
                    supported_ids = retrieved_ids & observed_ids & cited_ids
                    if not supported_ids:
                        continue
                    links.append({
                        "search_record_id": search["record_id"],
                        "answer_record_id": answer["record_id"],
                        "answer_transcript_action_index": pos,
                        "exact_search_prompt_prefix_match":
                            search["row"]["messages"][:search["target_index"]] == messages[:pos],
                        "shared_cited_document_ids": sorted(supported_ids),
                    })
                    break
        findings.append({
            "normalized_question_sha256": digest(norm(inventory[ids[0]]["question"])),
            "question": group["normalized_question_for_audit"],
            "state_ids": ids,
            "retained_answer_ids": [a["record_id"] for a in retained_answers],
            "message_level_search_answer_citation_links": links,
            "has_message_level_supported_pair": bool(links),
            "training_allowed": False,
        })
    result = {
        "version": "v124-source-block-message-pair-audit-v1",
        "backlog_payload_sha256": digest(backlog),
        "groups": len(findings),
        "groups_with_message_level_supported_pair": sum(
            f["has_message_level_supported_pair"] for f in findings),
        "groups_with_exact_prompt_prefix_link": sum(
            any(p["exact_search_prompt_prefix_match"] for p in f["message_level_search_answer_citation_links"])
            for f in findings),
        "findings": findings,
        "interpretation": [
            "This check uses the actual v117 message transcript, not source-case IDs as trajectory IDs.",
            "A link requires the search action to appear before a retained answer, a retrieved document ID to appear in that observation, and the answer review to cite that document.",
            "A message-level link is diagnostic only; it does not approve a repair or create a training pair.",
        ],
        "training_allowed": False,
    }
    write(OUT / "source-block-message-pair-audit-private.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
