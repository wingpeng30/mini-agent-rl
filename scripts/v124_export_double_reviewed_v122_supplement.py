"""Export only the passing v122 supplement pairs as a non-final candidate."""
from __future__ import annotations

import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from types import SimpleNamespace

from transformers import AutoProcessor

from mini_agent_rl.training.sft import encode_assistant_only
from scripts.finalize_v116 import export
from scripts.prepare_v116_review import PROCESSOR, render
from scripts.v123_cpu_core import ROOT, digest, readl, sha, write, writel
from scripts.v124_forensic import OUT
from scripts.finalize_v117 import valid_v117_vote

TRAIN = ROOT / "data/evidence-sft-v122-train-candidates"
REVIEW = TRAIN / "review"
OUTPUT = OUT / "double-reviewed-v122-supplement-candidate-private.jsonl"
MANIFEST = OUT / "double-reviewed-v122-supplement-manifest-private.json"
PREFLIGHT = OUT / "double-reviewed-v122-supplement-preflight-private.json"


def main() -> None:
    if OUTPUT.exists() or MANIFEST.exists() or PREFLIGHT.exists():
        raise FileExistsError("refuse_to_overwrite_v122_supplement_export")
    validation = json.loads((OUT / "v122-train-luna-review-validation-private.json").read_text(
        encoding="utf-8"))
    pair_audit = json.loads((OUT / "v122-train-pair-integrity-audit-private.json").read_text(
        encoding="utf-8"))
    bindings = readl(OUT / "pending-luna-v122-train/blind-binding-private.jsonl")
    blind_votes = {r["case_id"]: r for r in readl(
        OUT / "pending-luna-v122-train/luna-decisions-private.jsonl")}
    original_packets = {r["case_id"]: r for r in readl(REVIEW / "internal-packets.jsonl")}
    eligible = {r["source_id"]: r["case_ids"] for r in readl(REVIEW / "eligible-private.jsonl")}
    api = {}
    for path in sorted(REVIEW.glob("api-internal-batch-*.jsonl")):
        for row in readl(path):
            api[row["case_id"]] = row
    pair_findings = {r["source_id"]: r for r in pair_audit["pairs"]}
    validation_pairs = {r["source_id"]: r for r in validation["pair_rows"]}
    blind_by_source = {}
    for binding in bindings:
        blind_by_source.setdefault(binding["source_id"], []).append(binding)

    selected_ids = validation["selected_source_ids_private"]
    if (len(selected_ids) != 25 or len(set(selected_ids)) != 25 or
            validation["selected_25_pair_gate"] is not True or
            validation["selected_relation_risk_states"] < 20):
        raise RuntimeError("supplement_review_gate_not_met")

    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    primary = {r["record_id"]: r for p in sorted(OUT.glob("primary-batch-*.jsonl"))
               for r in readl(p)}
    for path in sorted(OUT.glob("primary-amendments*-private.jsonl")):
        for change in readl(path):
            primary[change["record_id"]] = {**primary[change["record_id"]],
                "disposition": change["effective_disposition"]}
    for change in readl(OUT / "primary-adjudication-overrides-private.jsonl"):
        primary[change["record_id"]] = {**primary[change["record_id"]],
            "disposition": change["effective_disposition"]}
    inventory_by_case = {r["original_case_id"]: r for r in readl(OUT / "inventory-private.jsonl")
                         if r["cohort"] == "v122_train_candidates"}

    output_rows, token_checks = [], []
    for source_id in selected_ids:
        cases = eligible[source_id]
        if (len(cases) != 2 or
                not pair_findings[source_id]["eligible_for_independent_review"] or
                not validation_pairs[source_id]["independent_double_approved_pair"]):
            raise RuntimeError("selected_pair_not_currently_approved")
        pair_packets = [original_packets[c] for c in cases]
        if {p["proposed_action"]["action"] for p in pair_packets} != {"search", "answer"}:
            raise RuntimeError("selected_pair_actions_mismatch")
        for packet in sorted(pair_packets,
                             key=lambda p: 0 if p["proposed_action"]["action"] == "search" else 1):
            api_row = api[packet["case_id"]]
            api_vote = api_row.get("decision")
            api_valid, _ = valid_v117_vote(packet, api_vote)
            if api_row.get("packet_sha256") != digest(packet):
                api_valid = False
            if not api_valid or api_vote.get("approved") is not True:
                raise RuntimeError("selected_state_deepseek_vote_invalid")
            bind = next(b for b in blind_by_source[source_id]
                        if b["original_case_id"] == packet["case_id"])
            validated_state = next(s for s in validation_pairs[source_id]["states"]
                                   if s["case_id"] == bind["case_id"])
            inventory_state = inventory_by_case[packet["case_id"]]
            if (inventory_state["record_id"] != bind["record_id"] or
                    inventory_state["record_id"] != validated_state["record_id"]):
                raise RuntimeError("selected_state_primary_binding_mismatch")
            luna_vote = blind_votes[bind["case_id"]]
            lv, _ = valid_v117_vote({**packet, "case_id": bind["case_id"]}, luna_vote)
            if not lv or luna_vote.get("approved") is not True:
                raise RuntimeError("selected_state_luna_vote_invalid")
            primary_review = primary[bind["record_id"]]
            if primary_review.get("disposition") != "retain_candidate":
                raise RuntimeError("selected_state_not_primary_retained")

            row = export(packet)
            target_index = row["target_assistant_index"]
            prompt = row["messages"][:target_index]
            prompt_ids = render(processor, [SimpleNamespace(**m) for m in prompt])
            target_message = {"role": "assistant", "content": row["messages"][target_index]["content"]}
            encoded = encode_assistant_only(processor, prompt + [target_message], 2176,
                                            len(prompt), False)
            target_tokens = sum(token != -100 for token in encoded["labels"])
            if len(prompt_ids) > 2048 or target_tokens > 128 or len(encoded["input_ids"]) > 2176:
                raise RuntimeError("selected_state_token_limit_failed")
            output_rows.append(row)
            token_checks.append({"record_id": bind["record_id"],
                "source_id": source_id, "action": packet["proposed_action"]["action"],
                "prompt_tokens": len(prompt_ids), "target_tokens": target_tokens,
                "sequence_tokens": len(encoded["input_ids"]),
                "prompt_ids_sha256": digest(prompt_ids),
                "target_mask_sha256": digest(encoded["labels"])})

    if len(output_rows) != 50 or len({digest(r) for r in output_rows}) != 50:
        raise RuntimeError("supplement_export_state_count_or_duplicate_failure")
    writel(OUTPUT, output_rows)
    manifest = {
        "version": "v124-double-reviewed-v122-supplement-candidate-v1",
        "states": 50, "questions": 25, "strict_complete_pairs": 25,
        "source_ids_sha256": digest(selected_ids),
        "source_packet_sha256": digest([original_packets[c]
            for sid in selected_ids for c in eligible[sid]]),
        "pair_audit_sha256": sha(OUT / "v122-train-pair-integrity-audit-private.json"),
        "luna_validation_sha256": sha(OUT / "v122-train-luna-review-validation-private.json"),
        "output_sha256": sha(OUTPUT),
        "tokenizer": "frozen processor loaded local_files_only; no model weights loaded",
        "tokens": {"all_prompts_under_2048": all(r["prompt_tokens"] <= 2048 for r in token_checks),
            "all_targets_under_128": all(r["target_tokens"] <= 128 for r in token_checks),
            "all_sequences_under_2176": all(r["sequence_tokens"] <= 2176 for r in token_checks),
            "rows": token_checks},
        "candidate_status": "double_reviewed_supplement_only_not_complete_training_set",
        "training_allowed": False,
    }
    write(MANIFEST, manifest)
    v117 = json.loads((OUT / "train-core-candidate-preflight-private.json").read_text(
        encoding="utf-8"))
    preflight = {
        "version": "v124-v117-core-plus-v122-supplement-preflight-v1",
        "v117_core_states": v117["selected_states"],
        "v117_core_questions": v117["distinct_questions"],
        "v117_core_strict_pairs": v117["complete_search_answer_groups"],
        "supplement_states": 50, "supplement_questions": 25,
        "supplement_strict_pairs": 25,
        "combined_states": v117["selected_states"] + 50,
        "combined_questions": v117["distinct_questions"] + 25,
        "combined_strict_pairs": v117["complete_search_answer_groups"] + 25,
        "required": {"states": 395, "questions": 211, "strict_pairs": 100},
        "deficits": {"states": 395 - (v117["selected_states"] + 50),
            "questions": 211 - (v117["distinct_questions"] + 25),
            "strict_pairs": 100 - (v117["complete_search_answer_groups"] + 25)},
        "passes_full_data_gate": False,
        "internal_set_gate": "failed_by_frozen_v123_insufficiency_certificate",
        "training_allowed": False,
    }
    write(PREFLIGHT, preflight)
    print(json.dumps({"states": len(output_rows), "questions": 25,
        "strict_complete_pairs": 25, "tokens_passed": all(
            r["prompt_tokens"] <= 2048 and r["target_tokens"] <= 128 and
            r["sequence_tokens"] <= 2176 for r in token_checks),
        "full_gate_deficits": preflight["deficits"], "training_allowed": False},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
