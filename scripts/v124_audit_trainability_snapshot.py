"""Recompute the current v124 data-readiness boundary without changing labels."""
from __future__ import annotations

import json
import hashlib
from collections import Counter

from scripts.build_v114_sft_candidates import norm
from scripts.v123_cpu_core import ROOT, digest, readl, write
from scripts.v124_forensic import OUT
from scripts.v124_verify_primary import verify as verify_primary


def _count_jsonl(path):
    rows = readl(path)
    return rows


def _sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    primary = verify_primary()
    inventory = {r["record_id"]: r for r in readl(OUT / "inventory-private.jsonl")}
    reviews = [r for p in sorted(OUT.glob("primary-batch-*.jsonl")) for r in readl(p)]
    effective = {r["record_id"]: r for r in reviews}
    for p in sorted(OUT.glob("primary-amendments*-private.jsonl")):
        for change in readl(p):
            effective[change["record_id"]] = {
                **effective[change["record_id"]],
                "disposition": change["effective_disposition"],
            }
    for change in readl(OUT / "primary-adjudication-overrides-private.jsonl"):
        effective[change["record_id"]] = {
            **effective[change["record_id"]],
            "disposition": change["effective_disposition"],
        }
    assert set(effective) == set(inventory)

    quarantine_by_cohort = Counter(
        inventory[rid]["cohort"] for rid, review in effective.items()
        if review["disposition"] == "quarantine_pending_repair"
    )
    disposition_by_cohort = {}
    for cohort in sorted({r["cohort"] for r in inventory.values()}):
        disposition_by_cohort[cohort] = dict(Counter(
            review["disposition"] for rid, review in effective.items()
            if inventory[rid]["cohort"] == cohort
        ))

    triage_paths = [
        OUT / "v122-train-unpaired-semantic-triage-private.jsonl",
        OUT / "v123-unpaired-semantic-triage-private.jsonl",
    ]
    triage_by_cohort = {}
    for path in triage_paths:
        rows = _count_jsonl(path)
        assert all(r["status"] == "primary_semantic_triage_only_not_independent_approval"
                   and r["training_allowed"] is False for r in rows)
        name = "v122_train_unpaired" if "v122-" in path.name else "v123_internal_unpaired"
        counts = dict(Counter(r["disposition"] for r in rows))
        triage_by_cohort[name] = {"rows": len(rows), "dispositions": counts}
    backlog_path = OUT / "v117-remaining-repair-backlog-private.json"
    backlog = json.loads(backlog_path.read_text(encoding="utf-8"))
    backlog_statuses = Counter(
        s["effective_primary_disposition"]
        for group in backlog["groups"] for s in group["flagged_states"]
    )
    assert sum(backlog_statuses.values()) == backlog["unresolved_flagged_states"]
    preflight_path = OUT / "train-core-candidate-preflight-private.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    candidate_path = OUT / "train-core-candidate-private.jsonl"
    assert _sha_file(candidate_path) == preflight["output_sha256"]
    assert preflight["training_allowed"] is False
    assert preflight["full_user_plan_gate"]["passed"] is False

    source_block_path = OUT / "source-block-case-audit-private.json"
    source_block = json.loads(source_block_path.read_text(encoding="utf-8"))
    assert source_block["backlog_payload_sha256"] == digest(backlog)
    assert source_block["groups"] == backlog["work_classes"][
        "evidence_or_target_block_requires_source_resolution"]
    duplicate_path = OUT / "duplicate-pair-lineage-audit-private.json"
    duplicate_audit = json.loads(duplicate_path.read_text(encoding="utf-8"))
    assert duplicate_audit["backlog_payload_sha256"] == digest(backlog)
    assert duplicate_audit["groups"] == backlog["work_classes"][
        "deduplicate_at_export_after_pair_check"]
    pair_link_path = OUT / "retained-pair-link-audit-private.json"
    pair_link_audit = json.loads(pair_link_path.read_text(encoding="utf-8"))
    assert pair_link_audit["legacy_question_only_one_search_one_answer_groups"] == preflight[
        "question_only_one_search_one_answer_groups"]
    assert pair_link_audit["exact_full_prompt_prefix_links_with_answer_citation"] == preflight[
        "complete_search_answer_groups"]
    source_message_pair_path = OUT / "source-block-message-pair-audit-private.json"
    source_message_pair_audit = json.loads(source_message_pair_path.read_text(encoding="utf-8"))
    assert source_message_pair_audit["backlog_payload_sha256"] == digest(backlog)
    assert source_message_pair_audit["groups"] == source_block["groups"]

    repair_dedupe_path = OUT / "repair-dedupe-message-pair-audit-private.json"
    repair_dedupe = json.loads(repair_dedupe_path.read_text(encoding="utf-8"))
    assert repair_dedupe["backlog_payload_sha256"] == digest(backlog)
    quarantine_path = OUT / "quarantine-processing-audit-private.json"
    quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
    assert quarantine["quarantine_states"] == 120
    assert quarantine["training_allowed"] is False
    unpaired_boundary_path = OUT / "unpaired-training-boundary-private.json"
    unpaired_boundary = json.loads(unpaired_boundary_path.read_text(encoding="utf-8"))
    assert unpaired_boundary["total_questions"] == 35
    assert unpaired_boundary["training_allowed"] is False

    v122_pair_path = OUT / "v122-train-pair-integrity-audit-private.json"
    v122_pairs = json.loads(v122_pair_path.read_text(encoding="utf-8"))
    assert v122_pairs["api_valid_votes"] == 199
    assert v122_pairs["api_final_decisions"] == {"approve": 145, "reject": 54, "invalid": 13}
    assert v122_pairs["counts"]["candidate_pairs"] == 106
    assert v122_pairs["counts"]["strict_trajectory_links"] == 106

    luna_validation_path = OUT / "v122-train-luna-review-validation-private.json"
    luna_validation = json.loads(luna_validation_path.read_text(encoding="utf-8"))
    assert luna_validation["luna_reviews_received"] == 56
    assert luna_validation["luna_valid_votes"] == 56
    assert luna_validation["independent_double_approved_pairs"] == 27
    assert luna_validation["selected_25_pair_gate"] is True
    assert luna_validation["selected_relation_risk_states"] >= 20
    assert luna_validation["training_allowed"] is False

    supplement_manifest_path = OUT / "double-reviewed-v122-supplement-manifest-private.json"
    supplement_preflight_path = OUT / "double-reviewed-v122-supplement-preflight-private.json"
    supplement_path = OUT / "double-reviewed-v122-supplement-candidate-private.jsonl"
    supplement_manifest = json.loads(supplement_manifest_path.read_text(encoding="utf-8"))
    supplement_preflight = json.loads(supplement_preflight_path.read_text(encoding="utf-8"))
    supplement_rows = readl(supplement_path)
    assert _sha_file(supplement_path) == supplement_manifest["output_sha256"]
    assert len(supplement_rows) == supplement_manifest["states"] == 50
    assert supplement_manifest["questions"] == 25
    assert supplement_manifest["strict_complete_pairs"] == 25
    assert supplement_manifest["tokens"]["all_prompts_under_2048"] is True
    assert supplement_manifest["tokens"]["all_targets_under_128"] is True
    assert supplement_manifest["tokens"]["all_sequences_under_2176"] is True
    assert supplement_manifest["training_allowed"] is False
    assert supplement_preflight["training_allowed"] is False
    assert supplement_preflight["passes_full_data_gate"] is False

    exposed_sentinel_path = OUT / "v125-exposed-v123-review-sentinel-audit-private.json"
    exposed_sentinels = json.loads(exposed_sentinel_path.read_text(encoding="utf-8"))
    assert exposed_sentinels["votes_changed"] is False
    assert exposed_sentinels["training_allowed"] is False
    assert exposed_sentinels["multi_document_false_denial_candidates"] == 2
    assert exposed_sentinels["direct_evidence_false_denial_candidate"] == 1
    assert exposed_sentinels["redundant_search_false_approval_candidates"] == 2

    repair_yield_path = OUT / "v125-v117-repair-yield-bound-private.json"
    repair_yield = json.loads(repair_yield_path.read_text(encoding="utf-8"))
    repair_links_path = OUT / "v125-v117-repair-trajectory-link-audit-private.json"
    repair_links = json.loads(repair_links_path.read_text(encoding="utf-8"))
    assert repair_yield["existing_repair_backlog"]["current_cpu_replays"]["states"] == 61
    assert repair_yield["existing_repair_backlog"]["rebuilt_trajectory_replays"]["groups"] == 19
    assert repair_yield["existing_repair_backlog"]["rebuilt_trajectory_replays"][
        "groups_with_search_and_answer"] == 11
    assert repair_yield["existing_repair_backlog"]["rebuilt_trajectory_replays"][
        "new_question_candidates_upper_before_strict_pair_recheck"] == 9
    assert repair_links["strict_action_result_citation_links"] == 5
    assert repair_links["new_repair_pair_candidates_before_double_review"] == 3
    assert repair_links["strict_link_candidates_still_quarantined"] == 2
    assert repair_links["original_record_bindings_pass"] is True
    assert repair_yield["training_allowed"] is False

    core_rows = readl(candidate_path)
    core_questions = {norm(r["messages"][1]["content"]) for r in core_rows}
    eligible = readl(ROOT / "data/evidence-sft-v122-train-candidates/review/eligible-private.jsonl")
    candidate_packets = {r["case_id"]: r for r in readl(
        ROOT / "data/evidence-sft-v122-train-candidates/review/internal-packets.jsonl")}
    questions_by_source = {}
    for pair in eligible:
        qs = {norm(candidate_packets[c]["question"]) for c in pair["case_ids"]}
        if len(qs) != 1:
            raise RuntimeError("candidate_pair_question_mismatch")
        questions_by_source[pair["source_id"]] = next(iter(qs))
    selected_ids = luna_validation["selected_source_ids_private"]
    selected_questions = {questions_by_source[sid] for sid in selected_ids}
    if len(selected_ids) != 25 or len(selected_questions) != 25:
        raise RuntimeError("selected_supplement_question_count_mismatch")
    history = json.loads((ROOT / "data/evidence-sft-v122-train-candidates/history-private.json").read_text(
        encoding="utf-8"))
    internal_freeze = json.loads((ROOT / "data/evidence-sft-v122-internal-candidates/candidate-freeze.json").read_text(
        encoding="utf-8"))
    selected_source_ids = set(selected_ids)
    assert not (selected_source_ids & (set(history["ids"]) | set(internal_freeze["ids"])))
    assert not (selected_questions & core_questions)
    combined = {
        "states": len(core_rows) + 50,
        "questions": len(core_questions | selected_questions),
        "strict_complete_pairs": preflight["complete_search_answer_groups"] + 25,
        "questions_overlap_with_retained_core": len(selected_questions & core_questions),
        "questions_overlap_with_exposure_history": len(selected_questions & set(history["normalized_questions"])),
        "fully_double_reviewed_pairs": luna_validation["independent_double_approved_pairs"],
        "selected_pair_states": 50,
        "deficits_to_original_full_gate": {
            "states": max(0, 395 - (len(core_rows) + 50)),
            "questions": max(0, 211 - len(core_questions | selected_questions)),
            "strict_complete_pairs": max(0, 100 -
                (preflight["complete_search_answer_groups"] + 25)),
        },
        "passes_original_full_gate": (len(core_rows) + 50 >= 395 and
            len(core_questions | selected_questions) >= 211 and
            preflight["complete_search_answer_groups"] + 25 >= 100),
        "training_allowed": False,
    }

    internal_gate_path = ROOT / "data/evidence-sft-v123/internal/insufficiency-certificate.json"
    internal_gate = json.loads(internal_gate_path.read_text(encoding="utf-8"))
    assert internal_gate["possible_complete_pairs_upper"] < internal_gate["required_complete_pairs"]

    v9_files = sorted((OUT / "pending-dual-review-v9" / "deepseek-responses").glob(
        "v9-batch-*/validated-review-private.json"))
    v9_decisions = Counter()
    v9_states = 0
    for path in v9_files:
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["model"] == "deepseek-flash" and report["training_allowed"] is False
        v9_states += len(report["reviews"])
        v9_decisions.update(r["decision"] for r in report["reviews"])
    v9_disputes_path = OUT / "v9-batch02-primary-disagreement-private.json"
    v9_disputes = json.loads(v9_disputes_path.read_text(encoding="utf-8"))

    inputs = [
        OUT / "inventory-manifest.json",
        OUT / "progress.json",
        OUT / "v117-remaining-repair-backlog-private.json",
        source_block_path,
        duplicate_path,
        pair_link_path,
        source_message_pair_path,
        repair_dedupe_path,
        quarantine_path,
        unpaired_boundary_path,
        v122_pair_path,
        luna_validation_path,
        supplement_manifest_path,
        supplement_preflight_path,
        supplement_path,
        exposed_sentinel_path,
        repair_yield_path,
        repair_links_path,
        OUT / "pending-luna-v122-train/manifest-private.json",
        OUT / "pending-luna-v122-train/luna-decisions-private.jsonl",
        *[OUT / "pending-luna-v122-train" / n for n in
          ("work-luna-01.jsonl", "work-luna-02.jsonl", "work-luna-03.jsonl")],
        internal_gate_path,
        *triage_paths,
        preflight_path,
        candidate_path,
        v9_disputes_path,
        *v9_files,
    ]
    report = {
        "version": "v124-current-trainability-snapshot-v5",
        "primary": primary,
        "dispositions_by_cohort": disposition_by_cohort,
        "quarantine_states_by_cohort": dict(sorted(quarantine_by_cohort.items())),
        "unpaired_primary_triage": triage_by_cohort,
        "v117_repair_backlog": {
            "groups": len(backlog["groups"]),
            "flagged_states": backlog["unresolved_flagged_states"],
            "effective_dispositions": dict(backlog_statuses),
            "work_classes": backlog["work_classes"],
            "training_allowed": False,
        },
        "v117_repair_then_dedupe_message_audit": {
            "groups": repair_dedupe["groups"],
            "groups_with_message_level_supported_links": repair_dedupe[
                "groups_with_message_level_supported_links"],
            "groups_with_retained_linked_pair": repair_dedupe[
                "groups_with_retained_linked_pair"],
            "groups_with_exact_prefix_linked_pair": repair_dedupe[
                "groups_with_exact_prefix_linked_pair"],
            "training_allowed": False,
        },
        "quarantine_processing": {
            "states": quarantine["quarantine_states"],
            "by_cohort": quarantine["quarantine_by_cohort"],
            "source_rechecked_states": quarantine["source_rechecked_states"],
            "source_rechecked_still_held": quarantine["source_rechecked_still_held"],
            "issue_families_multi_label": quarantine["multi_label_issue_families"],
            "training_allowed": False,
        },
        "unpaired_training_boundary": {
            "questions": unpaired_boundary["total_questions"],
            "dispositions": unpaired_boundary["overall_dispositions"],
            "complete_pairs_created": unpaired_boundary["complete_pairs_created"],
            "independent_reviews_added": unpaired_boundary["independent_reviews_added"],
            "training_allowed": False,
        },
        "v117_source_block_case_audit": {
            "groups": source_block["groups"],
            "groups_with_shared_exact_case_id": source_block["groups_with_shared_exact_case_id"],
            "interpretation": "lineage identifiers only; not used as trajectory pairing evidence",
            "training_allowed": False,
        },
        "v117_source_block_message_pair_audit": {
            "groups": source_message_pair_audit["groups"],
            "groups_with_message_level_supported_pair": source_message_pair_audit[
                "groups_with_message_level_supported_pair"],
            "groups_with_exact_prompt_prefix_link": source_message_pair_audit[
                "groups_with_exact_prompt_prefix_link"],
            "training_allowed": False,
        },
        "v117_duplicate_pair_lineage_audit": {
            "groups": duplicate_audit["groups"],
            "flagged_search_states": duplicate_audit["flagged_search_states"],
            "searches_with_action_and_top3_match": duplicate_audit[
                "flagged_searches_with_action_and_top3_match"],
            "searches_with_same_case_retained_peer": duplicate_audit[
                "flagged_searches_with_same_case_retained_peer"],
            "searches_with_same_case_answer_peer": duplicate_audit[
                "flagged_searches_with_same_case_answer_peer"],
            "exact_prompt_action_duplicate_clusters": duplicate_audit[
                "exact_prompt_action_duplicate_clusters"],
            "export_dedup_ready": False,
            "training_allowed": False,
        },
        "v117_core_candidate": {
            "states": preflight["selected_states"],
            "questions": preflight["distinct_questions"],
            "complete_pairs": preflight["complete_search_answer_groups"],
            "question_only_one_search_one_answer_groups": preflight[
                "question_only_one_search_one_answer_groups"],
            "pair_link_definition": preflight["complete_pair_definition"],
            "action_result_citation_link_count": pair_link_audit[
                "action_and_observation_links_with_answer_citation"],
            "unique_questions_with_linked_search_answer_evidence": pair_link_audit[
                "unique_questions_with_linked_search_answer_evidence"],
            "unique_questions_with_exact_prompt_prefix_links": pair_link_audit[
                "exact_full_prompt_prefix_links_with_answer_citation"],
            "required": preflight["full_user_plan_gate"],
            "core_checks": preflight["core_checks"],
            "training_allowed": False,
        },
        "v122_train_supplement": {
            "candidate_pairs": v122_pairs["counts"]["candidate_pairs"],
            "strict_trajectory_links": v122_pairs["counts"]["strict_trajectory_links"],
            "primary_retained_and_deepseek_approved_pairs":
                v122_pairs["counts"]["both_retained_and_api_approved"],
            "deepseek": {"states": v122_pairs["api_case_coverage"],
                "valid": v122_pairs["api_valid_votes"],
                "decisions": v122_pairs["api_final_decisions"],
                "historical_rows_with_technical_error_metadata":
                    luna_validation["deepseek"]["records_with_historical_technical_errors"]},
            "luna": {"valid_states": luna_validation["luna_valid_votes"],
                "decisions": luna_validation["luna_decisions"],
                "double_approved_pairs": luna_validation[
                    "independent_double_approved_pairs"],
                "selected_pairs": len(selected_ids),
                "selected_relation_risk_states": luna_validation[
                    "selected_relation_risk_states"],
                "selected_pool_gate_passed": luna_validation[
                    "reviewed_pool_gate_passed"]},
            "clean_combination_with_v117_retained_core": combined,
            "exported_double_reviewed_subset": {
                "states": len(supplement_rows),
                "questions": supplement_manifest["questions"],
                "strict_complete_pairs": supplement_manifest["strict_complete_pairs"],
                "token_checks_passed": all(supplement_manifest["tokens"][k] for k in (
                    "all_prompts_under_2048", "all_targets_under_128",
                    "all_sequences_under_2176")),
                "combined_preflight_deficits": supplement_preflight["deficits"],
                "training_allowed": False,
            },
            "training_allowed": False,
        },
        "v123_exposed_review_sentinels": {
            "multi_document_false_denial_candidates": exposed_sentinels[
                "multi_document_false_denial_candidates"],
            "direct_evidence_false_denial_candidates": exposed_sentinels[
                "direct_evidence_false_denial_candidate"],
            "redundant_search_false_approval_candidates": exposed_sentinels[
                "redundant_search_false_approval_candidates"],
            "votes_changed": False,
            "training_or_strategy_tuning_use": False,
            "fresh_independent_reaudit_required": True,
            "training_allowed": False,
        },
        "v117_repair_replay_yield": {
            "backlog_states": repair_yield["existing_repair_backlog"]["flagged_states"],
            "cpu_replay_groups": repair_yield["existing_repair_backlog"][
                "current_cpu_replays"]["groups"],
            "cpu_replay_states": repair_yield["existing_repair_backlog"][
                "current_cpu_replays"]["states"],
            "rebuilt_trajectory_groups": repair_yield["existing_repair_backlog"][
                "rebuilt_trajectory_replays"]["groups"],
            "candidate_paired_trajectory_groups": repair_yield["existing_repair_backlog"][
                "rebuilt_trajectory_replays"]["groups_with_search_and_answer"],
            "new_question_candidate_upper_before_strict_pair_recheck": repair_yield[
                "existing_repair_backlog"]["rebuilt_trajectory_replays"][
                    "new_question_candidates_upper_before_strict_pair_recheck"],
            "strict_action_result_citation_links": repair_links[
                "strict_action_result_citation_links"],
            "new_repair_pair_candidates_before_double_review": repair_links[
                "new_repair_pair_candidates_before_double_review"],
            "strict_link_candidates_still_quarantined": repair_links[
                "strict_link_candidates_still_quarantined"],
            "reviews_pending_or_invalid": True,
            "training_allowed": False,
        },
        "v123_internal_set_gate": {
            "conclusion": internal_gate["conclusion"],
            "complete_pairs": internal_gate["current_complete_double_approved_pairs"],
            "possible_pairs_upper": internal_gate["possible_complete_pairs_upper"],
            "required_pairs": internal_gate["required_complete_pairs"],
            "risk_upper": internal_gate["possible_risk_state_upper"],
            "training_allowed": False,
        },
        "v9_deepseek_single_review": {
            "states": v9_states,
            "decisions": dict(v9_decisions),
            "batch02_dispute_records": v9_disputes["disputed_states"],
            "independent_double_approval": False,
        },
        "inputs": {p.relative_to(OUT.parents[1]).as_posix(): _sha_file(p)
                   for p in inputs},
        "interpretation": [
            "Quarantine is an exclusion from training pending source or semantic repair, not deletion.",
            "Primary semantic triage and CPU replay are not independent approvals.",
            "Only exact, unchanged historical dual-reviewed states may reuse those votes.",
            "A candidate file passing tokenizer and legacy-vote checks does not override unmet dataset gates.",
            "Historical DeepSeek error metadata records attempted technical corrections; it is not itself the final vote validity result.",
            "The v122 supplement locally passes its 25-pair/20-relation double-review gate, but the strictly retained v117 core plus the supplement remains below the original total data gate.",
            "The frozen v123 internal pool also has a certified upper bound below its required gate; no training authorization is inferred from the repaired supplement.",
        ],
        "training_allowed": False,
    }
    write(OUT / "current-trainability-snapshot-private.json", report)
    print(json.dumps({
        "primary": primary,
        "quarantine_states_by_cohort": report["quarantine_states_by_cohort"],
        "unpaired_primary_triage": triage_by_cohort,
        "v117_repair_backlog": report["v117_repair_backlog"],
        "v122_train_supplement": report["v122_train_supplement"],
        "v123_exposed_review_sentinels": report["v123_exposed_review_sentinels"],
        "v117_repair_replay_yield": report["v117_repair_replay_yield"],
        "v123_internal_set_gate": report["v123_internal_set_gate"],
        "v117_source_block_case_audit": report["v117_source_block_case_audit"],
        "v117_source_block_message_pair_audit": report[
            "v117_source_block_message_pair_audit"],
        "v117_duplicate_pair_lineage_audit": report[
            "v117_duplicate_pair_lineage_audit"],
        "v117_core_candidate": report["v117_core_candidate"],
        "v9_deepseek_single_review": report["v9_deepseek_single_review"],
        "training_allowed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
