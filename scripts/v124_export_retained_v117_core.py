"""Export the exact, still-retained v117 states into a non-authorized safe core."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoProcessor

from scripts.build_v114_sft_candidates import norm
from scripts.prepare_v116_review import PROCESSOR, render
from scripts.v123_cpu_core import ROOT, digest, readl, sha, write
from scripts.v124_forensic import OUT
from mini_agent_rl.training.sft import encode_assistant_only

TRAIN = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
OUTPUT = OUT / "train-core-candidate-private.jsonl"
REPORT = OUT / "train-core-candidate-preflight-private.json"


def main() -> None:
    source_rows = readl(TRAIN)
    inventory_rows = readl(OUT / "inventory-private.jsonl")
    inventory = {r["record_id"]: r for r in inventory_rows}
    primary = {r["record_id"]: r for p in sorted(OUT.glob("primary-batch-*.jsonl"))
               for r in readl(p)}
    for pattern in ("primary-amendments*-private.jsonl",):
        for change in sorted(OUT.glob(pattern)):
            for row in readl(change):
                primary[row["record_id"]] = {**primary[row["record_id"]],
                    "disposition": row["effective_disposition"]}
    for change in readl(OUT / "primary-adjudication-overrides-private.jsonl"):
        primary[change["record_id"]] = {**primary[change["record_id"]],
            "disposition": change["effective_disposition"]}
    legacy = json.loads((OUT / "v117-legacy-review-reconciliation-private.json").read_text(encoding="utf-8"))
    legacy_by_id = {r["record_id"]: r for r in legacy["records"]}
    if len(source_rows) != 345 or len(legacy_by_id) != 345:
        raise RuntimeError("v117_legacy_coverage_mismatch")
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    lineage = {r["record_id"]: r for r in readl(OUT / "lineage-private.jsonl")}
    v114 = {r["case_id"]: r for r in readl(ROOT / "data/evidence-sft-v114-review/accepted.jsonl")}
    v116 = {r["case_id"]: r for r in readl(ROOT / "data/evidence-sft-v116/review/train-packets.jsonl")}
    exports, risk_counts, actions, search_counts = [], Counter(), Counter(), Counter()
    source_risk_counts = {"v114_independent_intersection": Counter(),
                          "v116_inherited_provisional": Counter()}
    groups = defaultdict(list)
    retained_states = []
    token_rows = []
    for ordinal, row in enumerate(source_rows, 1):
        rid = f"v117_train:{ordinal:04d}"
        inv = inventory[rid]
        if digest(row) != inv["source_row_sha256"]:
            raise RuntimeError("frozen_source_row_changed:" + rid)
        if primary[rid]["disposition"] != "retain_candidate":
            continue
        ticket = legacy_by_id[rid]
        if not ticket["original_content_exact"] or not ticket["valid_dual_approval"]:
            raise RuntimeError("legacy_double_review_not_reusable:" + rid)
        target_index = row["target_assistant_index"]
        action = json.loads(row["messages"][target_index]["content"])
        prompt = row["messages"][:target_index]
        prompt_ids = render(processor, [SimpleNamespace(**m) for m in prompt])
        if len(prompt_ids) > 2048:
            raise RuntimeError("retained_prompt_token_limit:" + rid)
        target_msg = {"role": "assistant", "content": row["messages"][target_index]["content"]}
        encoded = encode_assistant_only(processor, prompt + [target_msg], 2176,
                                        len(prompt), False)
        target_tokens = sum(value != -100 for value in encoded["labels"])
        if target_tokens > 128 or len(encoded["input_ids"]) > 2176:
            raise RuntimeError("retained_sequence_token_limit:" + rid)
        token_rows.append({"record_id": rid, "prompt_tokens": len(prompt_ids),
                           "target_tokens": target_tokens, "total_tokens": len(encoded["input_ids"]),
                           "prompt_ids_sha256": digest(prompt_ids),
                           "target_mask_sha256": digest(encoded["labels"])})
        actions[action["action"]] += 1
        question_key = norm(inv["question"])
        groups[question_key].append((rid, action["action"]))
        matches = lineage[rid]["matches"]
        if len(matches) != 1:
            raise RuntimeError("legacy_retrieval_lineage_not_unique:" + rid)
        match = matches[0]
        if action["action"] == "search":
            after = match["retrieved_after_search"]
            search_counts["nonempty" if after else "empty"] += 1
        if match["source_path"].endswith("accepted.jsonl"):
            packet = v114[match["source_case_id"]]
            if not packet["review_a"]["approved"] or not packet["review_b"]["approved"]:
                raise RuntimeError("v114_legacy_vote_not_approved:" + rid)
            tags = set(packet["review_a"]["risk_labels"]) & set(packet["review_b"]["risk_labels"])
            source = "v114_independent_intersection"
        else:
            packet = v116[match["source_case_id"]]
            tags = set(packet.get("risk_tags_provisional", []))
            source = "v116_inherited_provisional"
        for tag in tags:
            risk_counts[tag] += 1
            source_risk_counts[source][tag] += 1
        exports.append(row)
        retained_states.append({
            "record_id": rid,
            "question_key": question_key,
            "row": row,
            "target_index": target_index,
            "action": action,
            "primary_review": primary[rid],
            "retrieved_after_search": match["retrieved_after_search"],
        })
    if len(exports) != 228:
        raise RuntimeError("unexpected_retained_core_count")
    answer_fraction = actions["answer"] / len(exports)
    nonempty_fraction = search_counts["nonempty"] / len(exports)
    empty_fraction = search_counts["empty"] / len(exports)
    question_only_pairs = sum(
        len(items) == 2 and {a for _, a in items} == {"search", "answer"}
        for items in groups.values())
    pair_links = []
    for search_state in retained_states:
        if search_state["action"].get("action") != "search":
            continue
        search_prefix = search_state["row"]["messages"][:search_state["target_index"]]
        retrieved_ids = {d["id"] for d in search_state["retrieved_after_search"]}
        if not retrieved_ids:
            continue
        for answer_state in retained_states:
            if (answer_state["question_key"] != search_state["question_key"]
                    or answer_state["action"].get("action") != "answer"
                    or search_state["target_index"] >= answer_state["target_index"]):
                continue
            messages = answer_state["row"]["messages"]
            answer_citations = {
                c["document_id"]
                for c in answer_state["primary_review"].get("citations", [])
                if c.get("stage") == "before_action"
            }
            for pos in range(answer_state["target_index"]):
                if messages[pos]["role"] != "assistant" or pos + 1 >= answer_state["target_index"]:
                    continue
                try:
                    prior_action = json.loads(messages[pos]["content"])
                except (TypeError, json.JSONDecodeError):
                    continue
                if prior_action != search_state["action"] or messages[pos + 1]["role"] != "tool":
                    continue
                try:
                    observation = json.loads(messages[pos + 1]["content"])
                except (TypeError, json.JSONDecodeError):
                    continue
                observed_ids = {d.get("id") for d in observation.get("result", [])}
                answer_supported_ids = retrieved_ids & observed_ids & answer_citations
                if not answer_supported_ids:
                    continue
                pair_links.append({
                    "search_record_id": search_state["record_id"],
                    "answer_record_id": answer_state["record_id"],
                    "normalized_question_sha256": digest(search_state["question_key"]),
                    "exact_search_prompt_prefix_match":
                        search_prefix == messages[:pos],
                    "cited_search_result_document_ids": sorted(answer_supported_ids),
                })
                break
    linked_question_hashes = {
        p["normalized_question_sha256"] for p in pair_links
        if p["exact_search_prompt_prefix_match"]
    }
    complete_pairs = len(linked_question_hashes)
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                                   for row in exports), encoding="utf-8")
    report = {"version": "v124-retained-v117-core-preflight-v1",
        "source_sha256": sha(TRAIN), "source_states": len(source_rows),
        "primary_verification_sha256": sha(OUT / "primary-verification.json"),
        "legacy_reconciliation_sha256": sha(OUT / "v117-legacy-review-reconciliation-private.json"),
        "output_sha256": sha(OUTPUT), "selected_states": len(exports),
        "distinct_questions": len(groups),
        "question_only_one_search_one_answer_groups": question_only_pairs,
        "complete_search_answer_groups": complete_pairs,
        "complete_pair_definition": (
            "same normalized question; search target action is present before the answer; "
            "the search state prompt exactly matches the answer transcript prefix; "
            "the answer transcript contains a result document from that search; "
            "the primary answer review cites that same visible document"),
        "verified_pair_links": [p for p in pair_links if p["exact_search_prompt_prefix_match"]],
        "action_result_citation_links_without_exact_prefix": sum(
            not p["exact_search_prompt_prefix_match"] for p in pair_links),
        "actions": dict(actions), "search_results": dict(search_counts),
        "answer_fraction": round(answer_fraction, 6),
        "nonempty_search_fraction_of_states": round(nonempty_fraction, 6),
        "empty_search_fraction_of_states": round(empty_fraction, 6),
        "risk_counts": dict(risk_counts), "risk_count_sources": {
            k: dict(v) for k, v in source_risk_counts.items()},
        "token_replay": {"prompt_states_rechecked": len(token_rows),
                         "all_prompts_under_2048": all(r["prompt_tokens"] <= 2048 for r in token_rows),
                         "all_targets_under_128": all(r["target_tokens"] <= 128 for r in token_rows),
                         "all_sequences_under_2176": all(r["total_tokens"] <= 2176 for r in token_rows),
                         "target_sequences": "replayed locally with the frozen processor/tokenizer; no model loaded"},
        "core_checks": {"exact_legacy_dual_review_for_every_selected_state": True,
            "all_selected_states_primary_retained": True,
            "action_distribution_45_to_55_answer": 0.45 <= answer_fraction <= 0.55,
            "nonempty_search_at_least_40_percent_states": nonempty_fraction >= 0.40,
            "empty_search_at_most_10_percent_states": empty_fraction <= 0.10,
            "four_risk_categories_at_least_20": all(risk_counts[r] >= 20 for r in
                ("relation", "time", "scope", "measurement")),
            "complete_pairs_at_least_50": complete_pairs >= 50},
        "full_user_plan_gate": {"required_states": 395, "required_questions": 211,
            "required_complete_pairs": 100, "current_states": len(exports),
            "current_questions": len(groups), "current_complete_pairs": complete_pairs,
            "passed": False},
        "new_internal_set_gate": "not_run; requires 50 independently double-reviewed candidate pairs",
        "training_allowed": False}
    write(REPORT, report)
    print(json.dumps({"selected_states": len(exports), "questions": len(groups),
        "complete_pairs": complete_pairs, "actions": dict(actions),
        "risk_counts": dict(risk_counts), "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
