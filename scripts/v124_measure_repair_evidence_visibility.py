"""Mechanical visibility inventory for replayed v117 repair drafts.

This report does not decide semantic correctness; it only binds exact answer
strings and newly retrieved document text to the frozen replay states.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.v123_cpu_core import ROOT, digest, readl, sha, write
from scripts.v124_forensic import OUT


def main() -> None:
    replay_path = OUT / "repair-draft-states-cpu-v2-private.jsonl"
    reconciliation_path = OUT / "repair-draft-reconciliation-cpu-v2-private.json"
    groups = readl(replay_path)
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    assert len(groups) == reconciliation["eligible_revision_groups"] == 35
    findings = []
    answer_count = search_count = answer_literal_count = search_new_doc_count = 0
    for group in groups:
        state_findings = []
        for state in group["states"]:
            action = state["proposed_action"]
            before = state.get("before_action_evidence", [])
            after = state.get("retrieved_after_search") or []
            if action["action"] == "answer":
                answer_count += 1
                answer = action["answer"]
                cited_in = [doc["id"] for doc in before
                            if answer.casefold() in doc.get("text", "").casefold()]
                answer_literal_count += bool(cited_in)
                state_findings.append({
                    "record_id": state["record_id"],
                    "action": "answer",
                    "answer": answer,
                    "literal_substring_documents": cited_in,
                    "semantic_support_established": False,
                    "interpretation": "literal_text_match_is_a_locator_only",
                })
            elif action["action"] == "search":
                search_count += 1
                before_ids = {doc["id"] for doc in before}
                novel = [doc for doc in after if doc["id"] not in before_ids]
                search_new_doc_count += bool(novel)
                state_findings.append({
                    "record_id": state["record_id"],
                    "action": "search",
                    "query": action.get("query"),
                    "novel_top3_documents": [
                        {"id": doc["id"], "title": doc["title"],
                         "text_sha256": digest(doc["text"])} for doc in novel],
                    "query_grounding_established": False,
                    "relation_relevance_established": False,
                })
            else:
                raise ValueError(f"unsupported_action:{action.get('action')}")
        findings.append({
            "record_ids": group["record_ids"],
            "question": group["question"],
            "states": state_findings,
            "semantic_decision": "not_made_by_this_mechanical_report",
        })
    report = {
        "version": "v124-repair-evidence-visibility-v1",
        "replay_path": str(replay_path.relative_to(ROOT)),
        "replay_sha256": sha(replay_path),
        "reconciliation_path": str(reconciliation_path.relative_to(ROOT)),
        "reconciliation_sha256": sha(reconciliation_path),
        "groups": len(groups),
        "states": sum(len(g["states"]) for g in groups),
        "answer_states": answer_count,
        "answer_states_with_exact_answer_substring_in_one_visible_document": answer_literal_count,
        "search_states": search_count,
        "search_states_with_at_least_one_new_top3_document": search_new_doc_count,
        "limitations": [
            "A literal answer match does not prove the answer is semantically correct or uniquely entailed.",
            "A newly retrieved document does not prove the query was grounded or the document relevant.",
            "No state receives an approval or training label from this report.",
        ],
        "findings": findings,
        "training_allowed": False,
    }
    write(OUT / "repair-evidence-visibility-private.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
