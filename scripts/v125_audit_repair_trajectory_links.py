"""Strictly link repaired v117 search/answer replays without promoting either state."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.build_v114_sft_candidates import norm
from scripts.v123_cpu_core import ROOT, readl, write
from scripts.v124_forensic import OUT


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    trajectory_path = OUT / "repair-draft-trajectories-cpu-private.jsonl"
    core_path = OUT / "train-core-candidate-private.jsonl"
    inventory_path = OUT / "inventory-private.jsonl"
    trajectories = readl(trajectory_path)
    core = readl(core_path)
    inventory = {r["record_id"]: r for r in readl(inventory_path)}
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
    core_questions = {norm(r["messages"][1]["content"]) for r in core}
    findings = []
    identity_bindings_pass = True
    for group in trajectories:
        states = group["states"]
        for state in states:
            identity_bindings_pass &= (
                state["record_id"] in inventory and
                state["original_row_sha256"] == inventory[state["record_id"]]["source_row_sha256"])
        searches = [s for s in states if s["proposed_action"].get("action") == "search"]
        answers = [s for s in states if s["proposed_action"].get("action") == "answer"]
        pair_links = []
        for search in searches:
            result_ids = {d["id"] for d in search.get("retrieved_after_search", [])}
            for answer in answers:
                if norm(search["question"]) != norm(answer["question"]):
                    continue
                answer_review = reviews[answer["record_id"]]
                cited = [c for c in answer_review.get("citations", [])
                         if c.get("stage") == "before_action"]
                visible_docs = {d["id"]: d["text"] for d in answer.get("before_action_evidence", [])}
                cited_docs = {c["document_id"] for c in cited
                              if c.get("document_id") in visible_docs and
                              c.get("quote") and c["quote"] in visible_docs[c["document_id"]]}
                if not (result_ids & cited_docs):
                    continue
                messages = answer["visible_messages"]
                for position, message in enumerate(messages[:-1]):
                    if message.get("role") != "assistant" or messages[position + 1].get("role") != "tool":
                        continue
                    try:
                        action = json.loads(message.get("content", ""))
                        observation = json.loads(messages[position + 1].get("content", ""))
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if action != search["proposed_action"] or messages[:position] != search["visible_messages"]:
                        continue
                    observed_ids = {d.get("id") for d in observation.get("result", [])}
                    shared = observed_ids & result_ids & cited_docs
                    if shared:
                        pair_links.append({
                            "search_record_id": search["record_id"],
                            "answer_record_id": answer["record_id"],
                            "search_disposition": reviews[search["record_id"]]["disposition"],
                            "answer_disposition": answer_review["disposition"],
                            "exact_search_prompt_prefix": True,
                            "retrieved_docs_visible_and_cited": sorted(shared),
                            "both_states_retained": (
                                reviews[search["record_id"]]["disposition"] == "retain_candidate" and
                                answer_review["disposition"] == "retain_candidate"),
                            "new_question_vs_retained_core": norm(answer["question"]) not in core_questions,
                            "semantic_double_review": "not_run",
                            "training_allowed": False,
                        })
                        break
                    break
        findings.append({
            "record_ids": group["record_ids"],
            "question_sha256": hashlib.sha256(norm(group["question"]).encode()).hexdigest(),
            "contains_search_and_answer": group["contains_search_and_answer"],
            "token_limits_passed": group["token_limits_passed"],
            "strict_action_result_citation_links": pair_links,
            "strict_link_count": len(pair_links),
            "training_allowed": False,
        })
    links = [link for finding in findings for link in finding["strict_action_result_citation_links"]]
    unique_new = {link["answer_record_id"] for link in links if link["new_question_vs_retained_core"]}
    repair_new = {link["answer_record_id"] for link in links
                  if link["new_question_vs_retained_core"] and
                  link["search_disposition"] == "repair_candidate" and
                  link["answer_disposition"] == "repair_candidate"}
    quarantined = {link["answer_record_id"] for link in links
                   if "quarantine_pending_repair" in
                   {link["search_disposition"], link["answer_disposition"]}}
    report = {
        "version": "v125-v117-repair-trajectory-link-audit-v1",
        "inputs": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
                   for path in (trajectory_path, core_path, inventory_path)},
        "trajectory_groups": len(trajectories),
        "groups_with_search_and_answer": sum(g["contains_search_and_answer"] for g in trajectories),
        "strict_action_result_citation_links": len(links),
        "strict_links_both_retained": sum(link["both_states_retained"] for link in links),
        "strict_links_with_nonretained_disposition": sum(not link["both_states_retained"] for link in links),
        "new_question_strict_link_candidates": len(unique_new),
        "new_repair_pair_candidates_before_double_review": len(repair_new),
        "strict_link_candidates_still_quarantined": len(quarantined),
        "original_record_bindings_pass": identity_bindings_pass,
        "all_tokens_pass": all(g["token_limits_passed"] for g in trajectories),
        "semantic_double_review": "not_run",
        "findings": findings,
        "training_allowed": False,
    }
    write(OUT / "v125-v117-repair-trajectory-link-audit-private.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
