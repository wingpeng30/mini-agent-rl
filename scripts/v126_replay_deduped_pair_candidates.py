"""Replay two already-observed v117 pair candidates after safe dedup/question repair."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from transformers import AutoProcessor

from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.training.sft import encode_assistant_only
from scripts.prepare_v116_review import PROCESSOR, render
from scripts.v123_cpu_core import ROOT, digest, readl, write, writel
from scripts.v124_forensic import OUT


TARGETS = [
    {
        "search_id": "v117_train:0284", "answer_id": "v117_train:0036",
        "question": "Taylor McLam composed music for what brand that was founded in 1940?",
        "candidate_type": "select_exact_prefix_member_of_duplicate_search_cluster",
    },
    {
        "search_id": "v117_train:0306", "answer_id": "v117_train:0308",
        "question": "What was the original name of the college where Jack Ridl was a professor of English?",
        "candidate_type": "past_tense_question_repair_then_deduplicate",
    },
]


def main():
    out_path = OUT / "v126-deduped-pair-replay-candidates-private.jsonl"
    report_path = OUT / "v126-deduped-pair-replay-report-private.json"
    manifest_path = OUT / "v126-deduped-pair-replay-manifest-private.json"
    if any(p.exists() for p in (out_path, report_path, manifest_path)):
        raise FileExistsError("refuse_to_overwrite_v126_deduped_pair_replay")

    source_path = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
    pair_audit_path = OUT / "repair-dedupe-message-pair-audit-private.json"
    inventory_path = OUT / "inventory-private.jsonl"
    rows = readl(source_path)
    by_id = {f"v117_train:{i:04d}": row for i, row in enumerate(rows, 1)}
    inventory = {r["record_id"]: r for r in readl(inventory_path)}
    pair_audit = json.loads(pair_audit_path.read_text(encoding="utf-8"))
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
    audit_groups = {tuple(g["flagged_state_ids"]): g for g in pair_audit["findings"]}
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    output, findings = [], []

    for spec in TARGETS:
        search_id, answer_id = spec["search_id"], spec["answer_id"]
        search_row, answer_row = by_id[search_id], by_id[answer_id]
        for rid, row in ((search_id, search_row), (answer_id, answer_row)):
            if digest(row) != inventory[rid]["source_row_sha256"]:
                raise ValueError("original_source_hash_mismatch")
        if search_row["messages"][1]["content"].casefold() != answer_row["messages"][1]["content"].casefold():
            raise ValueError("source_questions_do_not_match")

        # The exact prefix match is recovered from the actual answer transcript;
        # do not copy a search row with a different history.
        answer_messages = json.loads(json.dumps(answer_row["messages"], ensure_ascii=False))
        answer_messages[1]["content"] = spec["question"]
        answer_target_index = answer_row["target_assistant_index"]
        linked_search_positions = []
        for position, message in enumerate(answer_messages[:answer_target_index]):
            if message.get("role") != "assistant" or position + 1 >= len(answer_messages):
                continue
            if answer_messages[position + 1].get("role") != "tool":
                continue
            try:
                action = json.loads(message["content"])
            except json.JSONDecodeError:
                continue
            expected_action = json.loads(search_row["messages"][search_row["target_assistant_index"]]["content"])
            if action == expected_action and action.get("action") == "search":
                linked_search_positions.append(position)
        if len(linked_search_positions) != 1:
            raise ValueError("unique_matching_search_action_not_found_in_answer_trajectory")
        search_position = linked_search_positions[0]
        search_row_messages = json.loads(json.dumps(search_row["messages"], ensure_ascii=False))
        search_row_messages[1]["content"] = spec["question"]
        search_row_prompt = search_row_messages[:search_row["target_assistant_index"]]
        exact_search_row_prefix = search_row_prompt == answer_messages[:search_position]
        if not exact_search_row_prefix:
            raise ValueError("chosen_search_row_prefix_not_exact_after_replay")
        tool_observation = json.loads(answer_messages[search_position + 1]["content"])
        if tool_observation.get("ok") is not True:
            raise ValueError("search_tool_observation_not_ok")

        states = []
        # The search target is extracted from the answer's exact recorded prefix.
        search_prompt = answer_messages[:search_position]
        search_action_message = answer_messages[search_position]
        search_prompt_ids = render(processor, [SimpleNamespace(**m) for m in search_prompt])
        search_target = {"role": "assistant", "content": search_action_message["content"]}
        search_encoded = encode_assistant_only(processor, search_prompt + [search_target], 2176,
                                               len(search_prompt), False)
        search_target_tokens = sum(token != -100 for token in search_encoded["labels"])
        search_within = (len(search_prompt_ids) <= 2048 and search_target_tokens <= 128 and
                         len(search_encoded["input_ids"]) <= 2176)
        states.append({
            "record_id_private": search_id,
            "source_row_sha256": digest(search_row),
            "question": spec["question"],
            "visible_messages": search_prompt,
            "before_action_evidence": visible_documents(search_prompt),
            "proposed_action": json.loads(search_action_message["content"]),
            "retrieved_after_search": tool_observation.get("result", []),
            "prompt_tokens": len(search_prompt_ids), "target_tokens": search_target_tokens,
            "total_tokens": len(search_encoded["input_ids"]), "within_token_limits": search_within,
            "prompt_ids_sha256": hashlib.sha256(json.dumps(search_prompt_ids).encode()).hexdigest(),
            "target_mask_sha256": hashlib.sha256(json.dumps(search_encoded["labels"]).encode()).hexdigest(),
            "training_allowed": False,
        })

        # Rebuild the answer prompt from its recorded transcript with only the
        # corrected question text changed.
        answer_prompt = answer_messages[:answer_target_index]
        answer_prompt_ids = render(processor, [SimpleNamespace(**m) for m in answer_prompt])
        answer_target = {"role": "assistant", "content": answer_messages[answer_target_index]["content"]}
        answer_encoded = encode_assistant_only(processor, answer_prompt + [answer_target], 2176,
                                               len(answer_prompt), False)
        answer_target_tokens = sum(token != -100 for token in answer_encoded["labels"])
        answer_within = (len(answer_prompt_ids) <= 2048 and answer_target_tokens <= 128 and
                         len(answer_encoded["input_ids"]) <= 2176)
        answer_action = json.loads(answer_target["content"])
        if answer_action.get("action") != "answer":
            raise ValueError("linked_target_is_not_answer")
        states.append({
            "record_id_private": answer_id,
            "source_row_sha256": digest(answer_row),
            "question": spec["question"],
            "visible_messages": answer_prompt,
            "before_action_evidence": visible_documents(answer_prompt),
            "proposed_action": answer_action,
            "retrieved_after_search": [],
            "prompt_tokens": len(answer_prompt_ids), "target_tokens": answer_target_tokens,
            "total_tokens": len(answer_encoded["input_ids"]), "within_token_limits": answer_within,
            "prompt_ids_sha256": hashlib.sha256(json.dumps(answer_prompt_ids).encode()).hexdigest(),
            "target_mask_sha256": hashlib.sha256(json.dumps(answer_encoded["labels"]).encode()).hexdigest(),
            "training_allowed": False,
        })

        # Check that the answer's citation to this result remains exact.
        answer_docs = {d["id"]: d["text"] for d in states[1]["before_action_evidence"]}
        result_ids = {d.get("id") for d in tool_observation.get("result", [])}
        citations = [c for c in reviews[answer_id].get("citations", [])
                     if c.get("stage") in ("before_action", "retrieved_after_search")]
        exact_shared = {c["document_id"] for c in citations
                        if c.get("document_id") in answer_docs and c.get("quote") and
                        c["quote"] in answer_docs[c["document_id"]] and
                        c["document_id"] in result_ids}
        if not exact_shared:
            raise ValueError("answer_citation_not_linked_to_extracted_result")
        candidate_group = {
            "candidate_type": spec["candidate_type"],
            "question": spec["question"],
            "record_ids_private": [search_id, answer_id],
            "states": states,
            "strict_search_answer_prefix": exact_search_row_prefix,
            "tool_result_cited_by_answer": True,
            "exact_shared_citation_ids_private": sorted(exact_shared),
            "token_limits_passed": search_within and answer_within,
            "dedupe_action": "select this one linked search instance; never export duplicate copies",
            "status": "candidate_requires_fresh_semantic_review",
            "training_allowed": False,
        }
        output.append(candidate_group)
        findings.append({
            "record_ids_private": [search_id, answer_id],
            "candidate_type": spec["candidate_type"],
            "exact_search_prefix_in_answer": exact_search_row_prefix,
            "actual_tool_result_cited": True,
            "token_limits_passed": search_within and answer_within,
            "historical_semantic_review_reusable_for_revised_input": False,
            "new_independent_reviews": 0,
            "training_allowed": False,
        })

    report = {
        "version": "v126-deduped-pair-replay-v1",
        "inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (source_path, pair_audit_path, inventory_path)},
        "candidate_pairs": len(output),
        "strict_replay_links": sum(f["exact_search_prefix_in_answer"] and f["actual_tool_result_cited"]
                                    and f["token_limits_passed"] for f in findings),
        "new_question_pairs": 1,
        "core_replacement_candidates": 1,
        "new_independent_reviews": 0,
        "findings_private": findings,
        "training_allowed": False,
    }
    manifest = {
        "version": "v126-deduped-pair-replay-manifest-v1",
        "inputs": report["inputs"],
        "candidate_pairs": len(output), "candidate_states": sum(len(g["states"]) for g in output),
        "queries_authored_or_changed": 0,
        "one_existing_core_search_would_be_replaced_after_review": True,
        "training_allowed": False,
    }
    writel(out_path, output)
    write(report_path, report)
    write(manifest_path, manifest)
    print(json.dumps({"candidate_pairs": report["candidate_pairs"],
        "strict_replay_links": report["strict_replay_links"],
        "all_token_limits_passed": all(f["token_limits_passed"] for f in findings),
        "new_independent_reviews": 0, "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
