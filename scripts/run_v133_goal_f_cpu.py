"""Bounded CPU-only inventory for Goal F; never loads model weights."""
from __future__ import annotations

import hashlib
import json
import os
import re
import string
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoProcessor
from mini_agent_rl.training.sft import encode_assistant_only

OUT = ROOT / "runs/v133-goal-f-cpu"
MODEL = Path("/root/autodl-tmp/qwen_08b")
FILES = {
    "core": ROOT / "data/evidence-sft-v124-forensic/train-core-candidate-private.jsonl",
    "core_preflight": ROOT / "data/evidence-sft-v124-forensic/train-core-candidate-preflight-private.json",
    "legacy_reconciliation": ROOT / "data/evidence-sft-v124-forensic/v117-legacy-review-reconciliation-private.json",
    "primary_verification": ROOT / "data/evidence-sft-v124-forensic/primary-verification.json",
    "supplement": ROOT / "data/evidence-sft-v124-forensic/double-reviewed-v122-supplement-candidate-private.jsonl",
    "supplement_manifest": ROOT / "data/evidence-sft-v124-forensic/double-reviewed-v122-supplement-manifest-private.json",
    "supplement_preflight": ROOT / "data/evidence-sft-v124-forensic/double-reviewed-v122-supplement-preflight-private.json",
    "reuse_audit": ROOT / "data/evidence-sft-v124-forensic/goal-v128/g4-v123-reuse-v129-audit-v2-private.json",
    "reuse_packets": ROOT / "data/evidence-sft-v123/internal/review/internal-packets.jsonl",
    "reuse_source_map": ROOT / "data/evidence-sft-v123/internal/review/source-map-private.jsonl",
    "reuse_vote_map": ROOT / "data/evidence-sft-v123/internal/review/old-vote-reuse-private.jsonl",
    "goal_c_train": ROOT / "data/evidence-sft-v133-goal-c-private/frozen/train-candidate.jsonl",
    "goal_c_dev": ROOT / "data/evidence-sft-v133-goal-c-private/frozen/dev-candidate.jsonl",
    "goal_c_train_index": ROOT / "data/evidence-sft-v133-goal-c-private/frozen/train-review-index-private.jsonl",
    "goal_c_dev_index": ROOT / "data/evidence-sft-v133-goal-c-private/frozen/dev-source-index-private.jsonl",
    "goal_c_freeze": ROOT / "data/evidence-sft-v133-goal-c-private/frozen/freeze-manifest-private.json",
    "goal_e_summary": ROOT / "runs/v133-goal-e-diagnostic/summary.json",
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    value = "".join(c if c not in string.punctuation else " " for c in value)
    return " ".join(value.split())


def category(question: str) -> tuple[str, str]:
    q = norm(question)
    date_comparison = (
        re.search(r"\b(who|which) (was )?(born|formed|founded|established|released|published|created) first\b", q)
        or re.search(r"\bwhich .{0,90}\b(formed|founded|established|born|debuted|released|published|created) earlier\b", q)
        or re.search(r"\bwhich .{0,90}\b(formed|founded|established|born|debuted|released|published|created) first\b", q)
        or re.search(r"\bwhich came first\b", q)
        or re.search(r"\b(first|earlier) of (the )?(two|these)\b", q)
    )
    if date_comparison:
        return "date_comparison", "explicitly asks which of two events/entities happened first or earlier"
    common_value = (
        re.search(r"\b(share|shared) (which|what)\b.{0,60}\b(genre|category|type|class)\b", q)
        or re.search(r"\bwhat .{0,30}\b(do|does) .{0,120}\b(and|both) .{0,120}\bhave in common\b", q)
        or re.search(r"\bwhat .{0,30}\b(both|same)\b.{0,120}\b(genre|category|type|class|profession|occupation|nationality|country|region|team)\b", q)
        or re.search(r"\b(share|same) (the )?(same )?(genre|category|type|class|profession|occupation|nationality|country|region|team)\b", q)
        or re.search(r"\bwhat were both .{0,120}\b", q)
        or re.search(r"\bwhich .{0,100}\b(and|both) .{0,100}\bshare\b.{0,40}\bgenre\b", q)
        or re.search(r"\bwhat .{0,60}\bare both .{0,120}\b(located|found)\b", q)
    )
    if common_value:
        return "common_category", "asks for a category/property shared by two entities"
    if re.search(r"\b(who|whom|whose|what|which|where|when|how long|how many|how much|what year|what nationality)\b", q):
        return "relation_attribute", "asks for an entity, relation, or attribute value"
    return "other", "no target-pattern match; manual category review required"


def answer_action(row: dict):
    i = int(row["target_assistant_index"])
    try:
        action = json.loads(row["messages"][i]["content"])
    except (KeyError, IndexError, json.JSONDecodeError):
        return None
    return action if action.get("action") == "answer" and isinstance(action.get("answer"), str) else None


def literal_in_evidence(answer: str, evidence: str) -> bool:
    needle = norm(answer).split()
    haystack = norm(evidence).split()
    if not needle:
        return False
    return any(haystack[i:i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def visible_docs(row: dict) -> list[dict]:
    docs = []
    for msg in row["messages"][:int(row["target_assistant_index"] )]:
        if msg["role"] == "tool":
            try:
                docs.extend(json.loads(msg["content"]).get("result", []))
            except json.JSONDecodeError:
                pass
    return docs


def extract_entities(question: str, docs: list[dict]) -> list[str]:
    # Conservative literal names from question and evidence titles. This is a
    # grouping aid, not a named-entity model; unparsed cases remain unverified.
    text = question + " " + " ".join(str(d.get("title", "")) for d in docs)
    stop = {"who", "what", "which", "where", "when", "how", "the", "a", "an", "in", "on", "of", "to", "for", "from", "and", "or", "is", "was", "are", "were", "as", "by", "with", "that", "this", "it", "he", "she", "they", "their", "his", "her", "does", "did", "has", "have", "had", "other", "former", "american"}
    found = set()
    for match in re.finditer(r"\b[A-Z][\w.’'-]*(?:\s+[A-Z][\w.’'-]*){0,5}", text):
        phrase = norm(match.group())
        if phrase and phrase not in stop and any(token not in stop for token in phrase.split()):
            found.add(phrase)
    return sorted(found)


def item_record(row: dict, origin: str, qid: str, review_status: str, exposure: str, source_id: str | None = None):
    action = answer_action(row)
    question = row["messages"][1]["content"]
    docs = visible_docs(row)
    evidence = " ".join(str(d.get("text", "")) for d in docs)
    literal = bool(action and literal_in_evidence(action["answer"], evidence))
    skill, reason = category(question)
    target = int(row["target_assistant_index"])
    return {
        "origin": origin, "record_id": qid, "source_id": source_id,
        "question": question, "question_sha256": hashlib.sha256(norm(question).encode()).hexdigest(),
        "category": skill, "category_basis": reason, "target_action": action["action"] if action else "non_answer_or_invalid",
        "answer": action["answer"] if action else None, "answer_literal_in_visible_evidence": literal,
        "visible_doc_ids": sorted({str(d.get("id")) for d in docs if d.get("id")}),
        "visible_titles": sorted({norm(str(d.get("title", ""))) for d in docs if d.get("title")}),
        "entity_keys_heuristic": extract_entities(question, docs),
        "review_status": review_status, "exposure": exposure,
        "_row_for_tokenizer_only": row,
        "target_index": target,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {name: sha(path) for name, path in FILES.items()}
    hashes.update({f"tokenizer:{name}": sha(MODEL / name) for name in TOKENIZER_FILES})
    core_pf = read_json(FILES["core_preflight"])
    core = read_jsonl(FILES["core"])
    if (hashes["core"] != core_pf["output_sha256"] or len(core) != 228
            or core_pf["selected_states"] != 228 or core_pf["training_allowed"] is not False):
        raise ValueError("core_frozen_asset_mismatch")
    legacy = read_json(FILES["legacy_reconciliation"])
    if legacy["training_allowed"] is not False or len(legacy["records"]) != 345:
        raise ValueError("legacy_reconciliation_unexpected")
    primary = read_json(FILES["primary_verification"])
    if primary["reviewed"] != primary["total"] or primary["training_allowed"] is not False:
        raise ValueError("primary_verification_unexpected")
    sup_pf = read_json(FILES["supplement_preflight"])
    sup_manifest = read_json(FILES["supplement_manifest"])
    supplement = read_jsonl(FILES["supplement"])
    if (hashes["supplement"] != sup_manifest["output_sha256"] or len(supplement) != 50
            or sup_manifest["strict_complete_pairs"] != 25 or sup_manifest["training_allowed"] is not False):
        raise ValueError("supplement_frozen_asset_mismatch")

    goal_c_train = read_jsonl(FILES["goal_c_train"])
    goal_c_dev = read_jsonl(FILES["goal_c_dev"])
    train_index = read_jsonl(FILES["goal_c_train_index"])
    dev_index = read_jsonl(FILES["goal_c_dev_index"])
    freeze = read_json(FILES["goal_c_freeze"])
    if (len(goal_c_train) != 25 or len(goal_c_dev) != 16 or len(train_index) != 25 or len(dev_index) != 16
            or hashes["goal_c_train"] != freeze["splits"]["train_sha256"]
            or hashes["goal_c_dev"] != freeze["splits"]["dev_sha256"]):
        raise ValueError("goal_c_provenance_mismatch")

    reuse = read_json(FILES["reuse_audit"])
    for path_key in ("data/evidence-sft-v123/internal/review/internal-packets.jsonl",
                     "data/evidence-sft-v123/internal/review/source-map-private.jsonl",
                     "data/evidence-sft-v123/internal/review/old-vote-reuse-private.jsonl"):
        if reuse["input_sha256"].get(path_key) != hashes["reuse_packets" if "internal-packets" in path_key else "reuse_source_map" if "source-map" in path_key else "reuse_vote_map"]:
            raise ValueError("reuse_audit_input_hash_mismatch:" + path_key)
    reuse_packets = {p["case_id"]: p for p in read_jsonl(FILES["reuse_packets"])}
    source_map = {p["case_id"]: p for p in read_jsonl(FILES["reuse_source_map"])}
    vote_map = {p["new_case_id"]: p for p in read_jsonl(FILES["reuse_vote_map"])}
    reuse_rows = []
    for case_id, audit in reuse["case_audit"].items():
        if case_id not in reuse_packets or case_id not in source_map or case_id not in vote_map:
            continue
        packet = reuse_packets[case_id]
        action = audit["action"]
        packet_row = {"messages": packet["visible_messages"], "target_assistant_index": len(packet["visible_messages"]),
                      "thinking_enabled": False}
        # Review packets carry proposed action separately from visible context.
        packet_row["messages"] = packet["visible_messages"] + [{"role": "assistant", "content": json.dumps(packet["proposed_action"], ensure_ascii=False)}]
        packet_row["target_assistant_index"] = len(packet["visible_messages"])
        reuse_rows.append(item_record(packet_row, "v123_reuse", case_id,
            "old_vote_reuse_mechanically_valid" if audit["api_valid_v129_approved"] and audit["luna_valid_v129_approved"] and vote_map[case_id]["same_reviewed_input"] else "not_valid_dual_approved",
            "exposure_unknown; prior v123 internal review; no evidence of independent split", audit["source_id"]))

    train_qhashes = {hashlib.sha256(norm(r["messages"][1]["content"]).encode()).hexdigest() for r in goal_c_train}
    dev_qhashes = {hashlib.sha256(norm(r["messages"][1]["content"]).encode()).hexdigest() for r in goal_c_dev}
    supplement_tokens = sup_manifest["tokens"]["rows"]
    if len(supplement_tokens) != len(supplement):
        raise ValueError("supplement_token_receipt_count_mismatch")
    inventory = []
    core_by_question = defaultdict(list)
    for i, row in enumerate(core):
        question = row["messages"][1]["content"]
        qhash = hashlib.sha256(norm(question).encode()).hexdigest()
        rec = item_record(row, "v117_core", f"core:{i:03d}", "frozen_preflight_all_selected_primary_retained_and_exact_legacy_dual_approval", "known_v117_train_lineage; exposed_to_v118", f"v117_question:{qhash}")
        inventory.append(rec)
        core_by_question[qhash].append(rec)
    supplement_by_question = defaultdict(list)
    for i, (row, receipt) in enumerate(zip(supplement, supplement_tokens)):
        question = row["messages"][1]["content"]
        qhash = hashlib.sha256(norm(question).encode()).hexdigest()
        rec = item_record(row, "v122_supplement", f"supplement:{i:03d}", "frozen_25_complete_double_reviewed_pairs_primary_retained", "known_goal_c_train_exposure", receipt.get("source_id"))
        rec["source_id"] = receipt.get("source_id")
        if qhash not in train_qhashes:
            raise ValueError("supplement_missing_from_goal_c_training:")
        inventory.append(rec)
        supplement_by_question[qhash].append(rec)
    for rec in reuse_rows:
        qhash = rec["question_sha256"]
        if qhash in train_qhashes:
            rec["exposure"] = "known_goal_c_train_question_overlap"
        elif qhash in dev_qhashes:
            rec["exposure"] = "known_goal_c_dev_question_overlap; regression_only"
        inventory.append(rec)

    goal_c_train_source_ids = {str(x["source_id"]) for x in train_index if x.get("source_id")}
    goal_c_dev_source_ids = {str(x["source_id"]) for x in dev_index if x.get("source_id")}
    goal_c_train_entity_keys = set().union(*(set(extract_entities(r["messages"][1]["content"], visible_docs(r))) for r in goal_c_train))
    goal_c_dev_entity_keys = set().union(*(set(extract_entities(r["messages"][1]["content"], visible_docs(r))) for r in goal_c_dev))
    candidate_train_source_ids = {str(x["source_id"]) for x in inventory if x["origin"] == "v122_supplement" and x.get("source_id")}
    candidate_reuse_source_ids = {str(x["source_id"]) for x in inventory if x["origin"] == "v123_reuse" and x.get("source_id")}
    candidate_train_entities = set().union(*(set(x["entity_keys_heuristic"]) for x in inventory if x["origin"] in ("v117_core", "v122_supplement")))
    candidate_reuse_entities = set().union(*(set(x["entity_keys_heuristic"]) for x in inventory if x["origin"] == "v123_reuse"))

    # Question-level groups are the independent units for the inventory. Keep
    # only exact question duplicates joined here; entity overlap is reported
    # separately and blocks a clean evaluation split.
    groups = {}
    by_origin = defaultdict(dict)
    for rec in inventory:
        key = rec["question_sha256"]
        group = by_origin[rec["origin"]].setdefault(key, {"origin": rec["origin"], "question_sha256": key,
            "question": rec["question"], "category": rec["category"], "category_basis": rec["category_basis"],
            "source_ids": set(), "record_ids": [], "actions": set(), "answers": set(), "doc_ids": set(),
            "titles": set(), "entities": set(), "review_statuses": set(), "exposures": set(), "literal_supported": True})
        if rec["source_id"]:
            group["source_ids"].add(str(rec["source_id"]))
        group["record_ids"].append(rec["record_id"])
        group["actions"].add(rec["target_action"])
        if rec["answer"]:
            group["answers"].add(rec["answer"])
        group["doc_ids"].update(rec["visible_doc_ids"])
        group["titles"].update(rec["visible_titles"])
        group["entities"].update(rec["entity_keys_heuristic"])
        group["review_statuses"].add(rec["review_status"])
        group["exposures"].add(rec["exposure"])
        if rec["target_action"] == "answer":
            group["literal_supported"] &= rec["answer_literal_in_visible_evidence"]
    for origin, mapping in by_origin.items():
        for key, value in mapping.items():
            groups[f"{origin}:{key}"] = value
    for value in groups.values():
        for key in ("source_ids", "record_ids", "actions", "answers", "doc_ids", "titles", "entities", "review_statuses", "exposures"):
            value[key] = sorted(value[key])

    goal_e = read_json(FILES["goal_e_summary"])
    # Exposed sources: v117 candidates have the v118 source lineage; supplement
    # question hashes match the frozen Goal C train set; Goal E confirms these
    # 25 rows were trained and separately evaluated the 16 dev rows.
    candidates = []
    for group in groups.values():
        training_eligible = (group["origin"] in ("v117_core", "v122_supplement")
                             and group["category"] != "other" and "answer" in group["actions"]
                             and group["literal_supported"] and group["review_statuses"] != ["not_valid_dual_approved"])
        # Current records with known model/train exposure can only serve as
        # training or regression data, never as an independent evaluation row.
        candidates.append({**group, "training_candidate_for_diagnostic_only": training_eligible,
            "independent_evaluation_candidate": False,
            "evaluation_block_reason": "known training exposure or unknown v123 exposure; no verified unseen group in whitelist"})

    component_counts = {}
    for skill in sorted({x["category"] for x in candidates}):
        subset = [x for x in candidates if x["category"] == skill]
        parent = list(range(len(subset)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union(i, j):
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                shared = (set(subset[i]["source_ids"]) & set(subset[j]["source_ids"])
                          or set(subset[i]["doc_ids"]) & set(subset[j]["doc_ids"])
                          or set(subset[i]["entities"]) & set(subset[j]["entities"]))
                if shared:
                    union(i, j)
        component_counts[skill] = len({find(i) for i in range(len(subset))})

    counts = defaultdict(Counter)
    for c in candidates:
        counts[c["category"]][c["origin"]] += 1
    summary = {
        "status": "inventory_complete_cpu_only",
        "gpu_used": False, "model_weights_loaded": False, "inference_or_training": False,
        "candidate_state_counts": {"v117_core": len(core), "v122_supplement": len(supplement), "v123_audited_mappings": len(reuse_rows)},
        "candidate_question_groups": {o: len(m) for o, m in by_origin.items()},
        "v117_core_preflight": {"questions": core_pf["distinct_questions"], "states": core_pf["selected_states"], "complete_search_answer_pairs": core_pf["complete_search_answer_groups"], "all_primary_retained_and_legacy_dual_approved": True},
        "v122_supplement_preflight": {"questions": sup_pf["supplement_questions"], "states": sup_pf["supplement_states"], "complete_double_reviewed_pairs": sup_manifest["strict_complete_pairs"], "goal_c_train_question_overlap": len(supplement_by_question), "source_pairs_available_for_new_training": 0},
        "v123_reuse_audit": {"mapped_states": reuse["mapped_states"], "complete_dual_approved_pairs_under_v129": reuse["complete_dual_approved_pairs_under_v129"], "training_eligible_states": reuse["training_eligible_states"], "training_allowed": reuse["training_allowed"], "classification_rows": len(reuse_rows), "exposure": "unknown except exact Goal C overlaps"},
        "goal_c_e_exposure": {"known_train_questions": len(goal_c_train), "known_dev_questions": len(goal_c_dev), "goal_e_status": goal_e["status"], "goal_e_models": list(goal_e["models"])},
        "overlap_checks": {
            "supplement_source_ids_in_goal_c_train": len(candidate_train_source_ids & goal_c_train_source_ids),
            "v123_source_ids_in_goal_c_train": len(candidate_reuse_source_ids & goal_c_train_source_ids),
            "v123_source_ids_in_goal_c_dev": len(candidate_reuse_source_ids & goal_c_dev_source_ids),
            "core_supplement_entity_keys_overlapping_goal_c_train": len(candidate_train_entities & goal_c_train_entity_keys),
            "v123_entity_keys_overlapping_goal_c_train": len(candidate_reuse_entities & goal_c_train_entity_keys),
            "v123_entity_keys_overlapping_goal_c_dev": len(candidate_reuse_entities & goal_c_dev_entity_keys),
            "entity_extraction_method": "literal capitalized names in questions and retrieved document titles; approximate collision screen, not semantic entity resolution",
        },
        "category_question_groups": {k: dict(v) for k, v in counts.items()},
        "source_entity_connected_groups_heuristic": component_counts,
        "candidate_counts_by_category_and_origin": {},
        "independent_evaluation_groups_available": 0,
        "evaluation_isolation_verified": False,
        "selection_rule_result": "no_target_can_form_train_and_source_entity_disjoint_independent_evaluation_from_this_whitelist",
        "status_flags": {"cpu_ready": False, "evaluation_isolation_verified": False,
                         "training_experiment_executable": False,
                         "missing_conditions": ["new independently sourced and exposure-verified evaluation questions", "prove no overlap with v117/v118/v122/v123 or prior development exposure"]},
        "input_sha256": hashes,
    }

    # Tokenizer-check the in-scope answer states once to identify what a future
    # matching experiment can actually encode. This is CPU-only and never
    # instantiates the model weights.
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    token_checks = []
    for rec in inventory:
        if rec["target_action"] != "answer":
            continue
        row = rec.get("_row_for_tokenizer_only")
        if row is None:
            continue
        target = int(row["target_assistant_index"])
        prompt = row["messages"][:target]
        prompt_ids = processor.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if isinstance(prompt_ids, dict):
            prompt_ids = prompt_ids["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        encoded = encode_assistant_only(processor, row["messages"], 2176, target, False)
        positions = [i for i, x in enumerate(encoded["labels"]) if x != -100]
        prefix_match = encoded["input_ids"][:positions[0]] == list(prompt_ids)
        rendered = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # Tool content is JSON-encoded in the chat template, so the raw evidence
        # sentence may be escaped and not occur as a literal substring. Confirm
        # it survives JSON parsing and that the exact tokenized prompt is the
        # supervised sequence prefix.
        docs = visible_docs(row)
        evidence_docs = [d for d in docs if isinstance(d.get("text"), str) and d["text"].strip()]
        evidence_visible = bool(evidence_docs) and prefix_match
        action = answer_action(row)
        answer = action["answer"]
        answer_content = row["messages"][target]["content"]
        offsets = processor.tokenizer(answer_content, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        answer_json = json.dumps(answer, ensure_ascii=False)[1:-1]
        answer_start = answer_content.find(answer_json, answer_content.find('"answer"'))
        answer_end = answer_start + len(answer)
        answer_tokens = sum(start >= answer_start and end <= answer_end for start, end in offsets)
        token_checks.append({"origin": rec["origin"], "record_id": rec["record_id"],
            "question_sha256": rec["question_sha256"], "prompt_tokens": len(prompt_ids),
            "sequence_tokens": len(encoded["input_ids"]), "target_tokens": len(positions),
            "prefix_match": prefix_match, "evidence_visible": evidence_visible,
            "answer_tokens": answer_tokens, "answer_boundary_parseable": answer_tokens > 0,
            "within_frozen_limits": len(prompt_ids) <= 2048 and len(positions) <= 128 and len(encoded["input_ids"]) <= 2176})

    with (OUT / "candidate-inventory.jsonl").open("w", encoding="utf-8") as f:
        for row in candidates:
            safe_row = {k: v for k, v in row.items() if k != "_row_for_tokenizer_only"}
            f.write(json.dumps(safe_row, ensure_ascii=False, sort_keys=True) + "\n")
    with (OUT / "tokenizer-preflight.jsonl").open("w", encoding="utf-8") as f:
        for row in token_checks:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary["tokenizer_preflight"] = {"answer_states_checked": len(token_checks),
        "prefix_match_all": all(x["prefix_match"] for x in token_checks),
        "evidence_visible_all": all(x["evidence_visible"] for x in token_checks),
        "answer_boundaries_parseable_all": all(x["answer_boundary_parseable"] for x in token_checks),
        "within_limits_all": all(x["within_frozen_limits"] for x in token_checks)}
    state_counts = defaultdict(Counter)
    for rec in inventory:
        key = (rec["origin"], category(rec["question"])[0])
        state_counts[key]["states"] += 1
        if rec["target_action"] == "answer":
            state_counts[key]["answer_states"] += 1
            state_counts[key]["literal_answer_supported"] += int(rec["answer_literal_in_visible_evidence"])
        if rec["review_status"] in ("old_vote_reuse_mechanically_valid", "frozen_preflight_all_selected_primary_retained_and_exact_legacy_dual_approval", "frozen_25_complete_double_reviewed_pairs_primary_retained"):
            state_counts[key]["vote_valid_and_retained_or_reusable"] += 1
    summary["candidate_counts_by_category_and_origin"] = {
        f"{origin}:{skill}": dict(value) for (origin, skill), value in state_counts.items()
    }
    summary["status_flags"] = {
        "cpu_ready": summary["tokenizer_preflight"]["prefix_match_all"]
                     and summary["tokenizer_preflight"]["evidence_visible_all"]
                     and summary["tokenizer_preflight"]["answer_boundaries_parseable_all"]
                     and summary["tokenizer_preflight"]["within_limits_all"],
        "evaluation_isolation_verified": False,
        "training_experiment_executable": False,
        "missing_conditions": [
            "new independently sourced and exposure-verified evaluation questions",
            "prove no overlap with v117/v118/v122/v123 or prior development exposure",
        ],
    }
    summary["next_step"] = (
        "No current category meets the train-plus-independent-evaluation requirement. Reuse only the already qualified existing answer rows as regression/training candidates; acquire a small source-disjoint evaluation set for one selected target."
    )
    # Earlier preflight checks only searched for literal evidence substrings in
    # rendered chat text; escaped tool JSON made that overly strict. The actual
    # check now uses parsed tool evidence plus exact generation/label prefix IDs.
    summary["tokenizer_preflight"]["rendered_json_escape_note"] = "Evidence is JSON-escaped by the chat template. Visibility is established by parsing each action-time tool result and verifying the exact serialized prompt token IDs equal the training prefix."
    (OUT / "inventory-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "counts": summary["candidate_question_groups"],
        "categories": summary["category_question_groups"], "tokenizer": summary["tokenizer_preflight"],
        "evaluation_isolation_verified": summary["evaluation_isolation_verified"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
