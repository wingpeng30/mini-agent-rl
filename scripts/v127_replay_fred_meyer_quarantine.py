"""Rebuild one source-attributed v117 quarantine candidate from its raw trace.

This script does CPU/token replay only. It neither changes the original row nor
promotes the candidate into an approved dataset.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoProcessor

from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.training.sft import encode_assistant_only
from scripts.prepare_v116_review import PROCESSOR, render
from scripts.v123_cpu_core import ROOT, digest, readl, write, writel
from scripts.v124_forensic import OUT


QUESTION = (
    "According to the Hollywood Theatre article, which company does it identify "
    "as occupying the first suburban store of the company founded in Portland in 1922?"
)
SOURCE_ID = "v117_train:0179"
OUT_CANDIDATES = OUT / "v127-fred-meyer-quarantine-replay-private.jsonl"
OUT_REPORT = OUT / "v127-fred-meyer-quarantine-replay-report-private.json"
OUT_PACKET = OUT / "pending-review-v127-fred-meyer"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if any(p.exists() for p in (OUT_CANDIDATES, OUT_REPORT, OUT_PACKET)):
        raise FileExistsError("refuse_to_overwrite_v127_fred_meyer_replay")

    source_path = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
    inventory_path = OUT / "inventory-private.jsonl"
    source_rows = readl(source_path)
    inventory = {r["record_id"]: r for r in readl(inventory_path)}
    row = source_rows[178]
    if digest(row) != inventory[SOURCE_ID]["source_row_sha256"]:
        raise ValueError("source_row_hash_mismatch")
    messages = row["messages"]
    if row["target_assistant_index"] != 4:
        raise ValueError("unexpected_answer_action_position")
    search_action = json.loads(messages[2]["content"])
    observation = json.loads(messages[3]["content"])
    answer_action = json.loads(messages[4]["content"])
    if search_action != {"action": "search", "query": "Portland"}:
        raise ValueError("observed_search_action_changed")
    if observation.get("ok") is not True or answer_action.get("action") != "answer":
        raise ValueError("source_trace_action_observation_invalid")
    result_docs = observation.get("result") or []
    hollywood = next((d for d in result_docs if d.get("title") == "Hollywood Theatre (Portland, Oregon)"), None)
    fred_meyer = next((d for d in result_docs if d.get("title") == "Fred Meyer"), None)
    if not hollywood or not fred_meyer:
        raise ValueError("required_multihop_documents_missing_from_observed_result")
    if "first suburban Fred Meyer store, which is currently occupied by Rite Aid" not in hollywood.get("text", ""):
        raise ValueError("source_answer_statement_changed")
    if "founded in 1922 in Portland, Oregon" not in fred_meyer.get("text", ""):
        raise ValueError("entity_resolution_statement_missing")
    if answer_action.get("answer") != "Rite Aid":
        raise ValueError("observed_answer_changed")

    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    search_messages = [dict(m) for m in messages[:2]]
    answer_messages = [dict(m) for m in messages[:4]]
    search_messages[1]["content"] = QUESTION
    answer_messages[1]["content"] = QUESTION

    states = []
    for kind, prefix, action, retrieved in (
        ("search", search_messages, search_action, result_docs),
        ("answer", answer_messages, answer_action, []),
    ):
        prompt_ids = render(processor, [SimpleNamespace(**m) for m in prefix])
        target = {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}
        encoded = encode_assistant_only(processor, prefix + [target], 2176, len(prefix), False)
        target_tokens = sum(token != -100 for token in encoded["labels"])
        within = len(prompt_ids) <= 2048 and target_tokens <= 128 and len(encoded["input_ids"]) <= 2176
        states.append({
            "case_id": f"v127f-{kind}",
            "question": QUESTION,
            "visible_messages": prefix,
            "before_action": visible_documents(prefix),
            "proposed_action": action,
            "retrieved_after_search": retrieved,
            "prompt_tokens": len(prompt_ids),
            "target_tokens": target_tokens,
            "total_tokens": len(encoded["input_ids"]),
            "within_token_limits": within,
        })
    if not all(s["within_token_limits"] for s in states):
        raise ValueError("token_limit_failure")
    if states[0]["before_action"]:
        raise ValueError("unexpected_pre_search_evidence")
    answer_before_ids = {d["id"] for d in states[1]["before_action"]}
    if not {hollywood["id"], fred_meyer["id"]} <= answer_before_ids:
        raise ValueError("answer_input_missing_multihop_evidence")

    # Reviewer packet is anonymous and contains neither source IDs nor prior votes.
    packet_rows = [{
        "case_id": state["case_id"],
        "question": state["question"],
        "visible_messages": state["visible_messages"],
        "before_action": state["before_action"],
        "proposed_action": state["proposed_action"],
        "retrieved_after_search": state["retrieved_after_search"],
    } for state in states]
    OUT_PACKET.mkdir(parents=True)
    work_path = OUT_PACKET / "work.jsonl"
    writel(work_path, packet_rows)
    rubric_path = OUT / "pending-review-v126-pair-replay" / "rubric.txt"
    if not rubric_path.is_file():
        raise FileNotFoundError("frozen_v126_review_rubric_missing")
    (OUT_PACKET / "rubric.txt").write_bytes(rubric_path.read_bytes())

    writel(OUT_CANDIDATES, [{
        "source_record_id_private": SOURCE_ID,
        "source_row_sha256": digest(row),
        "original_question": messages[1]["content"],
        "revised_question": QUESTION,
        "revised_question_sha256": hashlib.sha256(QUESTION.encode()).hexdigest(),
        "search_action_observed_unchanged": search_action,
        "answer_action_observed_unchanged": answer_action,
        "search_result_sha256": hashlib.sha256(json.dumps(observation, sort_keys=True).encode()).hexdigest(),
        "states": states,
        "status": "source_attributed_replay_candidate_pending_independent_review",
        "semantic_review": "not_run",
        "independent_double_review": "not_run",
        "training_allowed": False,
    }])
    write(OUT_REPORT, {
        "version": "v127-fred-meyer-quarantine-replay-v1",
        "inputs": {
            "data/evidence-sft-v117/frozen/train.jsonl": sha(source_path),
            "data/evidence-sft-v124-forensic/inventory-private.jsonl": sha(inventory_path),
            "scripts/v126_review_protocol.py": sha(ROOT / "scripts/v126_review_protocol.py"),
        },
        "source_record_private": SOURCE_ID,
        "original_state_disposition": "quarantine_pending_repair",
        "observed_search_action_reused_without_change": True,
        "observed_answer_reused_without_change": True,
        "retrieval_has_both_entity_link_and_occupant_statement": True,
        "new_query_or_answer_authored": False,
        "question_rewritten_for_source_attribution": True,
        "token_limits_passed": True,
        "states": 2,
        "questions": 1,
        "conditional_ceiling_if_both_states_pass_independent_reviews": {
            "states": 292,
            "questions": 182,
            "strict_complete_pairs": 54,
            "remaining_to_original_training_gate": {
                "states": 103,
                "questions": 29,
                "strict_complete_pairs": 46,
            },
        },
        "independent_reviews": 0,
        "training_allowed": False,
        "review_packet_sha256": sha(work_path),
        "rubric_sha256": sha(OUT_PACKET / "rubric.txt"),
    })
    print(json.dumps({
        "candidate_states": 2,
        "token_limits_passed": True,
        "source_attribution_repair": True,
        "independent_double_review": "not_run",
        "training_allowed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
