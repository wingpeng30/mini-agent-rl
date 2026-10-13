"""Extract search labels from their actual answer trajectory prefixes."""
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


def main() -> None:
    output_path = OUT / "v126-retained-linked-search-state-candidates-private.jsonl"
    report_path = OUT / "v126-retained-linked-search-extraction-report-private.json"
    manifest_path = OUT / "v126-retained-linked-search-extraction-manifest-private.json"
    if any(p.exists() for p in (output_path, report_path, manifest_path)):
        raise FileExistsError("refuse_to_overwrite_v126_trajectory_extraction")

    source_path = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
    pair_audit_path = OUT / "retained-pair-link-audit-private.json"
    inventory_path = OUT / "inventory-private.jsonl"
    sources = readl(source_path)
    by_id = {f"v117_train:{index:04d}": row for index, row in enumerate(sources, 1)}
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

    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    output, findings = [], []
    for link in pair_audit["links"]:
        if link["search_prompt_exactly_matches_answer_prefix"]:
            continue
        search_id, answer_id = link["search_record_id"], link["answer_record_id"]
        if not link["answer_evidence_linked"]:
            continue
        if search_id not in by_id or answer_id not in by_id:
            raise ValueError("source_record_missing")
        search_row, answer_row = by_id[search_id], by_id[answer_id]
        if any(digest(row) != inventory[rid]["source_row_sha256"]
               for rid, row in ((search_id, search_row), (answer_id, answer_row))):
            raise ValueError("source_row_hash_mismatch")
        if any(reviews[rid]["disposition"] != "retain_candidate"
               for rid in (search_id, answer_id)):
            raise ValueError("source_rows_not_both_retained")
        if search_row["messages"][1]["content"] != answer_row["messages"][1]["content"]:
            raise ValueError("normalized_question_only_match_not_exact")

        search_index = link["search_action_position_in_answer"]
        if type(search_index) is not int or search_index < 2:
            raise ValueError("invalid_search_action_position")
        messages = answer_row["messages"]
        if search_index + 1 >= len(messages):
            raise ValueError("missing_tool_observation_after_search")
        action_message, tool_message = messages[search_index:search_index + 2]
        if action_message["role"] != "assistant" or tool_message["role"] != "tool":
            raise ValueError("search_action_observation_role_mismatch")
        action = json.loads(action_message["content"])
        observation = json.loads(tool_message["content"])
        if action.get("action") != "search" or observation.get("ok") is not True:
            raise ValueError("embedded_search_action_or_result_invalid")
        original_target_index = search_row["target_assistant_index"]
        original_search_action = json.loads(search_row["messages"][original_target_index]["content"])
        if action != original_search_action:
            raise ValueError("embedded_action_differs_from_separately_reviewed_search")

        prompt = messages[:search_index]
        if not prompt or prompt[1]["content"] != answer_row["messages"][1]["content"]:
            raise ValueError("extracted_prefix_question_mismatch")
        prompt_ids = render(processor, [SimpleNamespace(**m) for m in prompt])
        target = {"role": "assistant", "content": action_message["content"]}
        encoded = encode_assistant_only(processor, prompt + [target], 2176,
                                        len(prompt), False)
        target_tokens = sum(token != -100 for token in encoded["labels"])
        within = len(prompt_ids) <= 2048 and target_tokens <= 128 and len(encoded["input_ids"]) <= 2176

        answer_target_index = answer_row["target_assistant_index"]
        answer_action = json.loads(answer_row["messages"][answer_target_index]["content"])
        if answer_action.get("action") != "answer":
            raise ValueError("linked_answer_row_target_is_not_answer")
        retrieved = observation.get("result")
        cited_ids = set(link["answer_cited_shared_document_ids"])
        result_ids = {doc.get("id") for doc in retrieved or []}
        if not (cited_ids & result_ids):
            raise ValueError("embedded_result_does_not_contain_cited_answer_document")
        prompt_docs = {d["id"]: d["text"] for d in visible_documents(answer_row["messages"][:answer_target_index])}
        answer_review_citations = [c for c in reviews[answer_id].get("citations", [])
                                   if c.get("stage") == "before_action"]
        exact_cited_shared = {c["document_id"] for c in answer_review_citations
                              if c.get("document_id") in prompt_docs and c.get("quote") and
                              c["quote"] in prompt_docs[c["document_id"]] and
                              c["document_id"] in result_ids}
        if not exact_cited_shared:
            raise ValueError("no_exact_answer_citation_to_embedded_retrieval")

        synthetic_id = f"derived_search_from:{answer_id}:step{search_index:02d}"
        output.append({
            "candidate_id_private": synthetic_id,
            "source_search_record_id_private": search_id,
            "source_answer_record_id_private": answer_id,
            "source_answer_row_sha256": digest(answer_row),
            "question": answer_row["messages"][1]["content"],
            "visible_messages": prompt,
            "before_action_evidence": visible_documents(prompt),
            "proposed_action": action,
            "retrieved_after_search": retrieved,
            "original_action_message_index": search_index,
            "answer_state_record_id_private": answer_id,
            "answer_action": answer_action,
            "answer_cited_shared_document_ids_private": sorted(exact_cited_shared),
            "prompt_tokens": len(prompt_ids), "target_tokens": target_tokens,
            "total_tokens": len(encoded["input_ids"]), "within_token_limits": within,
            "prompt_ids_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
            "target_mask_sha256": hashlib.sha256(json.dumps(encoded["labels"]).encode()).hexdigest(),
            "status": "observed_trajectory_extract_requires_new_semantic_review",
            "query_authored_or_changed": False,
            "training_allowed": False,
        })
        findings.append({
            "candidate_id_private": synthetic_id,
            "source_search_record_id_private": search_id,
            "source_answer_record_id_private": answer_id,
            "same_question_exact": True,
            "actual_action_and_immediate_tool_result": True,
            "action_matches_separately_reviewed_search_row": True,
            "answer_cited_document_in_embedded_result": True,
            "exact_answer_citation_visible": True,
            "token_limits_passed": within,
            "semantic_review": "not_run",
            "training_allowed": False,
        })

    review_paths = [*sorted(OUT.glob("primary-batch-*.jsonl")),
                    *sorted(OUT.glob("primary-amendments*-private.jsonl")),
                    OUT / "primary-adjudication-overrides-private.jsonl"]
    manifest = {
        "version": "v126-retained-linked-search-extraction-v1",
        "inputs": {
            str(source_path.relative_to(ROOT)): hashlib.sha256(source_path.read_bytes()).hexdigest(),
            str(pair_audit_path.relative_to(ROOT)): hashlib.sha256(pair_audit_path.read_bytes()).hexdigest(),
            str(inventory_path.relative_to(ROOT)): hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            **{p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in review_paths},
        },
        "processor_path": str(PROCESSOR), "local_files_only": True,
        "candidate_search_states": len(output), "new_queries_authored": 0,
        "training_allowed": False,
    }
    writel(output_path, output)
    write(manifest_path, manifest)
    report = {
        "version": "v126-retained-linked-search-extraction-report-v1",
        "candidate_search_states": len(output),
        "strict_action_result_citation_links": sum(
            f["actual_action_and_immediate_tool_result"] and
            f["answer_cited_document_in_embedded_result"] and
            f["exact_answer_citation_visible"] and f["token_limits_passed"]
            for f in findings),
        "search_queries_authored": 0,
        "semantically_reviewed_candidates": 0,
        "findings_private": findings,
        "training_allowed": False,
    }
    write(report_path, report)
    print(json.dumps({"candidate_search_states": report["candidate_search_states"],
        "strict_action_result_citation_links": report["strict_action_result_citation_links"],
        "all_token_limits_passed": all(f["token_limits_passed"] for f in findings),
        "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
