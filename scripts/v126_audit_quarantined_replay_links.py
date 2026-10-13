"""Recheck action/result/citation continuity for v126 quarantine replays."""
from __future__ import annotations

import hashlib
import json

from scripts.v123_cpu_core import ROOT, readl, write
from scripts.v124_forensic import OUT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effective_reviews():
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
    return reviews


def main():
    replay_path = OUT / "v126-quarantined-repair-replay-candidates-private.jsonl"
    manifest_path = OUT / "v126-quarantined-repair-replay-manifest-private.json"
    replay = readl(replay_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reviews = effective_reviews()
    review_paths = [*sorted(OUT.glob("primary-batch-*.jsonl")),
                    *sorted(OUT.glob("primary-amendments*-private.jsonl")),
                    OUT / "primary-adjudication-overrides-private.jsonl"]
    findings = []
    for pair in replay:
        states = {s["record_id_private"]: s for s in pair["states"]}
        searches = [s for s in pair["states"] if s["proposed_action"].get("action") == "search"]
        answers = [s for s in pair["states"] if s["proposed_action"].get("action") == "answer"]
        if len(searches) != 1 or len(answers) != 1:
            raise ValueError("not_exactly_one_search_and_answer")
        search, answer = searches[0], answers[0]
        search_ids = {d["id"] for d in search.get("retrieved_after_search") or []}
        answer_docs = {d["id"]: d["text"] for d in answer["before_action_evidence"]}
        citations = [c for c in reviews[answer["record_id_private"]].get("citations", [])
                     if c.get("stage") == "before_action"]
        cited_ids = {c["document_id"] for c in citations
                     if c.get("document_id") in answer_docs and c.get("quote") and
                     c["quote"] in answer_docs[c["document_id"]]}
        matched_action_result = False
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
            if search_ids & observed_ids & cited_ids:
                matched_action_result = True
                break
        findings.append({
            "record_ids_private": pair["record_ids_private"],
            "question_unchanged_between_states": search["question"] == answer["question"],
            "exact_search_prompt_prefix": any(
                m.get("role") == "assistant" and
                (i + 1 < len(messages)) and messages[i + 1].get("role") == "tool" and
                messages[:i] == search["visible_messages"] and
                json.loads(m["content"]) == search["proposed_action"]
                for i, m in enumerate(messages[:-1]) if m.get("role") == "assistant"),
            "retrieval_and_cited_answer_document_linked": matched_action_result,
            "token_limits_passed": all(s["within_token_limits"] for s in pair["states"]),
            "independent_semantic_review": "not_run",
            "training_allowed": False,
        })
    report = {
        "version": "v126-quarantined-repair-link-audit-v1",
        "inputs": {
            str(replay_path.relative_to(ROOT)): sha(replay_path),
            str(manifest_path.relative_to(ROOT)): sha(manifest_path),
            **{p.relative_to(ROOT).as_posix(): sha(p) for p in review_paths},
        },
        "candidate_pairs": len(findings),
        "strict_trajectory_linked_pairs": sum(
            f["question_unchanged_between_states"] and f["exact_search_prompt_prefix"] and
            f["retrieval_and_cited_answer_document_linked"] and f["token_limits_passed"]
            for f in findings),
        "findings_private": findings,
        "independent_semantic_review": "not_run",
        "training_allowed": False,
    }
    out = OUT / "v126-quarantined-repair-link-audit-private.json"
    if out.exists():
        raise FileExistsError(out)
    write(out, report)
    print(json.dumps({"candidate_pairs": report["candidate_pairs"],
        "strict_trajectory_linked_pairs": report["strict_trajectory_linked_pairs"],
        "independent_semantic_review": "not_run", "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
