"""Freeze v133 Goal C data from existing valid pairs plus a disjoint dev set.

CPU only. Reuses the 25 sample-level double-approved v122 pairs, extracting
their answer actions, and pairs them with 16 exact-support official-train dev
items created by the bounded v133 screen. No reviewers or model are called.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoProcessor

from scripts.build_v133_goal_c_data import OUT as NEW_POOL, norm, read_json, read_jsonl, sha_bytes, sha_file
from scripts.finalize_v117 import valid_v117_vote
from scripts.prepare_v116_review import PROCESSOR, render
from scripts.v123_cpu_core import digest, readl
from mini_agent_rl.training.sft import encode_assistant_only

FORENSIC = ROOT / "data/evidence-sft-v124-forensic"
V122 = ROOT / "data/evidence-sft-v122-train-candidates"
V122_REVIEW = V122 / "review"
SUPPLEMENT = FORENSIC / "double-reviewed-v122-supplement-candidate-private.jsonl"
SUPPLEMENT_MANIFEST = FORENSIC / "double-reviewed-v122-supplement-manifest-private.json"
VALIDATION = FORENSIC / "v122-train-luna-review-validation-private.json"
PAIR_AUDIT = FORENSIC / "v122-train-pair-integrity-audit-private.json"
OUT = NEW_POOL / "frozen"


def jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refuse_to_overwrite:{OUT}")
    supplement_manifest = read_json(SUPPLEMENT_MANIFEST)
    if sha_file(SUPPLEMENT) != supplement_manifest["output_sha256"]:
        raise ValueError("v122_supplement_export_hash_mismatch")
    if supplement_manifest.get("strict_complete_pairs") != 25 or supplement_manifest.get("training_allowed") is not False:
        raise ValueError("unexpected_v122_supplement_manifest")

    validation = read_json(VALIDATION)
    pair_audit = read_json(PAIR_AUDIT)
    selected_ids = validation["selected_source_ids_private"]
    if len(selected_ids) != 25 or len(set(selected_ids)) != 25 or validation["selected_25_pair_gate"] is not True:
        raise ValueError("v122_selected_pair_gate_mismatch")
    frozen_v122_ids = set(read_json(V122 / "candidate-freeze.json")["ids"])
    if not set(selected_ids).issubset(frozen_v122_ids):
        raise ValueError("v122_approved_ids_missing_from_source_freeze")
    validation_pairs = {x["source_id"]: x for x in validation["pair_rows"]}
    pair_findings = {x["source_id"]: x for x in pair_audit["pairs"]}

    bindings = readl(FORENSIC / "pending-luna-v122-train/blind-binding-private.jsonl")
    binding_by_source_original = {(x["source_id"], x["original_case_id"]): x for x in bindings}
    blind_votes = {x["case_id"]: x for x in readl(FORENSIC / "pending-luna-v122-train/luna-decisions-private.jsonl")}
    original_packets = {x["case_id"]: x for x in readl(V122_REVIEW / "internal-packets.jsonl")}
    eligible = {x["source_id"]: x["case_ids"] for x in readl(V122_REVIEW / "eligible-private.jsonl")}
    api_votes = {}
    for path in sorted(V122_REVIEW.glob("api-internal-batch-*.jsonl")):
        for row in readl(path):
            api_votes[row["case_id"]] = row

    primary = {x["record_id"]: x for path in sorted(FORENSIC.glob("primary-batch-*.jsonl")) for x in readl(path)}
    for path in sorted(FORENSIC.glob("primary-amendments*-private.jsonl")):
        for change in readl(path):
            primary[change["record_id"]] = {**primary[change["record_id"]], "disposition": change["effective_disposition"]}
    for change in readl(FORENSIC / "primary-adjudication-overrides-private.jsonl"):
        primary[change["record_id"]] = {**primary[change["record_id"]], "disposition": change["effective_disposition"]}

    supplement_rows = jsonl_rows(SUPPLEMENT)
    token_rows = supplement_manifest["tokens"]["rows"]
    if len(supplement_rows) != 50 or len(token_rows) != 50:
        raise ValueError("v122_supplement_state_or_token_count_mismatch")

    train_rows, train_receipts = [], []
    seen_answer_ids = set()
    for row, token in zip(supplement_rows, token_rows):
        target_i = int(row["target_assistant_index"])
        action = json.loads(row["messages"][target_i]["content"])
        if action.get("action") != "answer":
            continue
        source_id, record_id = token["source_id"], token["record_id"]
        if source_id not in set(selected_ids) or source_id in seen_answer_ids:
            raise ValueError("v122_answer_row_source_or_duplicate_mismatch")
        seen_answer_ids.add(source_id)
        finding = pair_findings[source_id]
        pair = validation_pairs[source_id]
        if not finding.get("eligible_for_independent_review") or not pair.get("independent_double_approved_pair"):
            raise ValueError("v122_pair_not_sample_level_approved")
        pair_state = next((s for s in pair["states"] if s["record_id"] == record_id), None)
        if not pair_state or pair_state["primary_disposition"] != "retain_candidate":
            raise ValueError("answer_state_primary_vote_not_retained")
        if not (pair_state["api_valid"] and pair_state["api_approved"] and pair_state["luna_valid"] and pair_state["luna_approved"]):
            raise ValueError("answer_state_missing_valid_dual_approval")
        expected_records = set(finding["record_ids"])
        if record_id not in expected_records or row["target_assistant_index"] < 2:
            raise ValueError("answer_state_not_bound_to_strict_pair")
        source_cases = eligible[source_id]
        packet = next((original_packets[c] for c in source_cases if original_packets[c]["proposed_action"]["action"] == "answer"), None)
        if not packet:
            raise ValueError("original_answer_packet_missing")
        api = api_votes[packet["case_id"]]
        valid, _ = valid_v117_vote(packet, api.get("decision"))
        if api.get("packet_sha256") != digest(packet):
            valid = False
        if not valid or api["decision"].get("approved") is not True:
            raise ValueError("deepseek_answer_vote_failed_revalidation")
        binding = next(b for b in bindings if b["source_id"] == source_id and b["original_case_id"] == packet["case_id"])
        luna_vote = blind_votes[binding["case_id"]]
        luna_valid, _ = valid_v117_vote({**packet, "case_id": binding["case_id"]}, luna_vote)
        if not luna_valid or luna_vote.get("approved") is not True:
            raise ValueError("luna_answer_vote_failed_revalidation")
        primary_review = primary.get(record_id)
        if not primary_review or primary_review.get("disposition") != "retain_candidate":
            raise ValueError("answer_state_primary_ledger_not_retained")

        docs = []
        for message in row["messages"][:target_i]:
            if message["role"] == "tool":
                docs.extend(json.loads(message["content"]).get("result", []))
        evidence = " ".join(doc.get("text", "") for doc in docs)
        answer = str(action.get("answer", ""))
        if not norm(answer) or norm(answer) not in norm(evidence):
            raise ValueError("approved_answer_not_literal_in_action_time_evidence")
        if token["source_id"] != source_id or token["action"] != "answer":
            raise ValueError("v122_token_receipt_binding_mismatch")
        train_rows.append(row)
        train_receipts.append({
            "source_id": source_id,
            "record_id": record_id,
            "original_deepseek_case_id": packet["case_id"],
            "luna_blind_case_id": binding["case_id"],
            "deepseek_packet_sha256": api["packet_sha256"],
            "luna_original_packet_sha256": binding["original_packet_sha256"],
            "luna_anonymous_packet_sha256": binding["anonymous_packet_sha256"],
            "deepseek_vote_valid_and_approved": True,
            "luna_vote_valid_and_approved": True,
            "deepseek_risk_labels": pair_state["api_risk_labels"],
            "luna_risk_labels": pair_state["luna_risk_labels"],
            "risk_intersection": pair_state["risk_intersection"],
            "primary_disposition": "retain_candidate",
            "strict_pair_link": True,
            "answer_literal_in_visible_evidence": True,
            "prompt_tokens": token["prompt_tokens"],
            "target_tokens": token["target_tokens"],
            "sequence_tokens": token["sequence_tokens"],
            "prompt_ids_sha256": token["prompt_ids_sha256"],
            "target_mask_sha256": token["target_mask_sha256"],
        })
    if len(train_rows) != 25 or seen_answer_ids != set(selected_ids):
        raise ValueError("expected_one_answer_state_per_approved_pair")

    candidates = jsonl_rows(NEW_POOL / "candidate-supervision-private.pending-review.jsonl")
    source_map = jsonl_rows(NEW_POOL / "source-map-private.jsonl")
    role_by_id = {x["item_id"]: x for x in source_map}
    dev = []
    dev_receipts = []
    for row in candidates:
        link = role_by_id[row["item_id"]]
        if link["precommitted_role"] != "dev_candidate":
            continue
        dev.append({k: row[k] for k in ("messages", "target_assistant_index", "thinking_enabled")})
        dev_receipts.append({
            "item_id": row["item_id"], "source_id": row["source_id"],
            "source_shard": link["source_shard"], "source_row_index": link["source_row_index"],
            "question_sha256": link["question_sha256"],
            "label_source": "official HotpotQA distractor train answer with exact supporting_fact sentence links",
            "double_review_votes": "not collected; dev only, not train-eligible",
            "prompt_tokens": row["tokens"]["prompt"], "target_tokens": row["tokens"]["target"],
            "sequence_tokens": row["tokens"]["full"],
        })
    if len(dev) != 16:
        raise ValueError("expected_16_precommitted_dev_items")
    train_questions = {norm(row["messages"][1]["content"]) for row in train_rows}
    dev_questions = {norm(row["messages"][1]["content"]) for row in dev}
    train_source_ids = set(selected_ids)
    dev_source_ids = {x["source_id"] for x in dev_receipts}
    if (len(train_questions) != 25 or len(dev_questions) != 16 or train_questions & dev_questions
            or train_source_ids & dev_source_ids):
        raise ValueError("train_dev_question_overlap_or_duplicate")

    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True, trust_remote_code=False)
    token_rechecks = []
    for row in train_rows + dev:
        target_i = int(row["target_assistant_index"])
        prompt = row["messages"][:target_i]
        prompt_ids = render(processor, [SimpleNamespace(**m) for m in prompt])
        target = {"role": "assistant", "content": row["messages"][target_i]["content"]}
        encoded = encode_assistant_only(processor, prompt + [target], 2176, len(prompt), False)
        target_tokens = sum(token != -100 for token in encoded["labels"])
        if len(prompt_ids) > 2048 or target_tokens > 128 or len(encoded["input_ids"]) > 2176:
            raise ValueError("final_goal_c_token_limit_failed")
        token_rechecks.append({"prompt_tokens": len(prompt_ids), "target_tokens": target_tokens, "sequence_tokens": len(encoded["input_ids"])})
    for index in range(len(train_rows)):
        expected, actual = train_receipts[index], token_rechecks[index]
        if (actual["prompt_tokens"], actual["target_tokens"], actual["sequence_tokens"]) != (expected["prompt_tokens"], expected["target_tokens"], expected["sequence_tokens"]):
            raise ValueError("reused_train_token_signature_length_mismatch")
        expected["rechecked_with_frozen_processor"] = True
    for index, receipt in enumerate(dev_receipts):
        receipt["frozen_processor_prompt_tokens"] = token_rechecks[len(train_rows) + index]["prompt_tokens"]
        receipt["frozen_processor_target_tokens"] = token_rechecks[len(train_rows) + index]["target_tokens"]
        receipt["frozen_processor_sequence_tokens"] = token_rechecks[len(train_rows) + index]["sequence_tokens"]

    OUT.mkdir()
    train_path, dev_path = OUT / "train-candidate.jsonl", OUT / "dev-candidate.jsonl"
    train_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in train_rows), encoding="utf-8")
    dev_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in dev), encoding="utf-8")
    (OUT / "train-review-index-private.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in train_receipts), encoding="utf-8")
    (OUT / "dev-source-index-private.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in dev_receipts), encoding="utf-8")

    config_path = ROOT / "configs/v133-goal-c-small-sft.json"
    config = read_json(config_path)
    config["status"] = "data_frozen_pending_separate_user_training_authorization"
    config.pop("data_pending_review", None)
    config.pop("review_packet_pending_authorization", None)
    config.pop("approved_train_data_after_reviews", None)
    config.pop("approved_dev_data_after_reviews", None)
    config["train_data"] = "data/evidence-sft-v133-goal-c-private/frozen/train-candidate.jsonl"
    config["dev_data"] = "data/evidence-sft-v133-goal-c-private/frozen/dev-candidate.jsonl"
    config["precommitted_candidate_counts"] = {
        "training_from_existing_valid_dual_reviewed_pairs": 25,
        "dev_from_new_official_train_support_items": 16,
        "new_pool_rows_reviewed_or_used_for_training": 0,
        "unreviewed_reserve_rows_used": 0,
    }
    config["training"]["max_steps"] = 7
    config["training"]["save_steps"] = 7
    config["training"]["eval_steps"] = 7
    config["training"]["warmup_ratio"] = 0.14285714285714285
    config["training"]["epochs"] = 1
    config["review_policy"] = "All 25 training answers reuse previously validated DeepSeek+Luna approvals and primary retention; no new review calls. Dev labels come from official source answers and exact source supporting facts; dev labels are never used as training labels."
    config["new_pool_disposition"] = "The 32 unreviewed training candidates and 16 unreviewed reserves are not used; the prepared 48-item review packet is superseded and was never sent. Only the 16 precommitted dev items are used from the new pool."
    config["training_allowed"] = False
    config["model_inference_allowed"] = False
    config["external_review_calls_allowed"] = False
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def hash_file(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    split_manifest = {
        "version": "v133-goal-c-reused-reviewed-train-freeze-v1",
        "status": "sample_level_data_verified; training_and_inference_not_authorized",
        "hypothesis": "When the question's requested attribute/value is explicitly present in visible evidence, one answer action should return that requested value concisely rather than a related entity or partial value.",
        "training_source": {
            "candidate_file": str(SUPPLEMENT.relative_to(ROOT)).replace("\\", "/"),
            "candidate_sha256": hash_file(SUPPLEMENT),
            "candidate_manifest_sha256": hash_file(SUPPLEMENT_MANIFEST),
            "pair_audit_sha256": hash_file(PAIR_AUDIT),
            "luna_validation_sha256": hash_file(VALIDATION),
            "preflight_training_allowed": False,
            "preflight_gate_reason": "legacy 395-state/211-question/100-pair scale gate and frozen v123 internal-set gate; v133 sample-level reuse does not claim those old gates passed",
            "selected_training_answers": len(train_rows),
            "strict_pairs_with_valid_deepseek_luna_and_primary_retention": len(train_receipts),
            "literal_visible_answer_support_count": len(train_rows),
        },
        "development_source": {
            "candidate_pool_manifest_sha256": hash_file(NEW_POOL / "manifest-private.json"),
            "source_split": "official HotpotQA distractor train",
            "count": len(dev),
            "double_reviewed": False,
            "selection_rule": "the precommitted 16 dev_candidate rows from the fixed 300-row v133 screen",
            "support_reference_and_answer_checked": True,
            "never_used_for_training": True,
        },
        "splits": {
            "train_sha256": hash_file(train_path),
            "train_rows": len(train_rows),
            "dev_sha256": hash_file(dev_path),
            "dev_rows": len(dev),
            "train_review_index_sha256": hash_file(OUT / "train-review-index-private.jsonl"),
            "dev_source_index_sha256": hash_file(OUT / "dev-source-index-private.jsonl"),
            "train_dev_question_overlap": 0,
            "train_dev_source_id_overlap": 0,
        },
        "baseline": {
            "model_path": config["base_model"],
            "v118_adapter_path": config["base_adapter"],
            "v118_adapter_sha256": config["base_adapter_sha256"],
            "tokenizer_sha256": read_json(NEW_POOL / "manifest-private.json")["baseline_assets"]["tokenizer_sha256"],
            "config_sha256": hash_file(config_path),
        },
        "limits": {"screened_train_source_rows": 300, "constructed_candidates": 64, "no_pool_expansion": True},
        "model_calls": 0,
        "sft_started": False,
        "inference_started": False,
        "training_allowed": False,
        "sealed_data_accessed": False,
    }
    (OUT / "freeze-manifest-private.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pool_manifest_path = NEW_POOL / "manifest-private.json"
    pool_manifest = read_json(pool_manifest_path)
    pool_manifest["current_disposition"] = {
        "train_candidates": "not_used; replaced by 25 existing valid dual-reviewed v122 answer states",
        "dev_candidates": "16 precommitted rows reused as official-source evidence-only development cases",
        "reserve_candidates": "not_used",
        "prepared_48_item_review_packet": "superseded_without_submission; zero API or agent review calls",
    }
    pool_manifest["status"] = "source_pool_verified; current train/dev freeze uses reviewed reuse plus 16 dev rows"
    pool_manifest["training_allowed"] = False
    pool_manifest["sft_started"] = False
    pool_manifest["new_model_inference_started"] = False
    pool_manifest_path.write_text(json.dumps(pool_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split_manifest["development_source"]["candidate_pool_manifest_sha256"] = hash_file(pool_manifest_path)
    (OUT / "freeze-manifest-private.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"train_rows": len(train_rows), "dev_rows": len(dev), "votes_valid": len(train_receipts), "literal_supported_train": len(train_rows), "overlap": 0, "train_sha256": split_manifest["splits"]["train_sha256"], "dev_sha256": split_manifest["splits"]["dev_sha256"], "training_allowed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
