"""Build bounded train-only measurement-risk candidate packets from unused Hotpot train questions."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoProcessor

from build_v114_sft_candidates import (
    ROOT, OUT as CLEAN, TRAIN_FILES, build_examples, build_index, collect_history,
    load_full, norm, public_example, risk_tags, row_meta_batches, state_hash,
)
from prepare_v114_reviews import MODEL, canonical, sha
from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.domain import Message
from mini_agent_rl.model.context import ContextBudgetError, budget_messages
from mini_agent_rl.training.sft import encode_assistant_only

WORK = ROOT / "data/evidence-sft-v114-review"
SEED = "v114-training-measurement-supplement-v1:"
MEASURE_REQUEST = re.compile(
    r"\bhow many\b|\bhow much\b|\bhow long\b|\bwhat number\b|\bpopulation\b|\binhabitants\b|"
    r"\blength\b|\bheight\b|\bweight\b|\bdistance\b|\bpercentage\b|\bpercent\b|\bduration\b",
    re.I,
)
SUBJECTIVE = re.compile(r"\b(more well[- ]known|better known|most famous|more famous|best known)\b", re.I)
TARGET_STATES = 60
MAX_CANDIDATE_QUESTIONS = 100


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError(f"immutable_artifact_changed:{path.name}")
    if not path.exists():
        path.write_bytes(raw)


def main() -> None:
    source_manifest = load_json(CLEAN / "manifest.json")
    for name, expected in source_manifest["files"].items():
        if sha((CLEAN / name).read_bytes()) != expected:
            raise RuntimeError(f"clean_source_manifest_mismatch:{name}")
    base_train_freeze = load_json(CLEAN / "training-question-freeze.json")
    base_internal_freeze = load_json(CLEAN / "internal-freeze.json")
    v2_internal = load_json(WORK / "internal-freeze-v2.json")
    proposal_private_path = WORK / "internal-replacement-proposals-private.jsonl"
    excluded_internal_proposals = {row["source_id"] for row in
                                   (json.loads(line) for line in proposal_private_path.read_text(encoding="utf-8").splitlines())}
    reserve_rows = [json.loads(line) for line in (WORK / "reserve-private.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    excluded_ids = (set(base_train_freeze["ids"]) | set(base_internal_freeze["ids"]) |
                    set(v2_internal["ids"]) | excluded_internal_proposals |
                    {row["source_id"] for row in reserve_rows})
    history, _ = collect_history()
    history_norms = {item for item in history if not item.startswith("__id__")}
    history_ids = {item.removeprefix("__id__") for item in history if item.startswith("__id__")}
    metadata: dict[str, dict] = {}
    eligible = []
    for row in row_meta_batches():
        qid, question = row["id"], row["question"]
        qnorm = norm(question)
        if (qid in excluded_ids or qid.casefold() in history_ids or qnorm in history_norms
                or len(question) > 240 or not MEASURE_REQUEST.search(question) or SUBJECTIVE.search(question)
                or "measurement" not in risk_tags(question)):
            continue
        metadata[qid] = {"id": qid, "question": question, "type": row.get("type")}
        eligible.append(metadata[qid])
    eligible.sort(key=lambda row: (hashlib.sha256((SEED + row["id"]).encode()).hexdigest(), row["id"]))
    selected_examples = []
    selected_questions: list[dict] = []
    # Expand by fixed hash order until enough measured, action-diverse states exist.
    for start in range(0, min(len(eligible), MAX_CANDIDATE_QUESTIONS), 25):
        end = min(start + 25, min(len(eligible), MAX_CANDIDATE_QUESTIONS))
        batch_meta = eligible[:end]
        rows = list(load_full({row["id"] for row in batch_meta}).values())
        documents, index = build_index(rows)
        del documents
        examples = build_examples(rows, index)
        by_question: dict[str, list[dict]] = defaultdict(list)
        for example in examples:
            by_question[example["qid"]].append(example)
        chosen = []
        chosen_q = []
        for meta in batch_meta:
            qid = meta["id"]
            available = by_question.get(qid, [])
            answers = sorted((ex for ex in available if ex["state_kind"] == "supported_answer"),
                             key=lambda ex: state_hash(ex))
            searches = sorted((ex for ex in available if ex["state_kind"] == "nonempty_evidence_search"),
                              key=lambda ex: state_hash(ex))
            group = answers[:1] + searches[:2]
            if group:
                chosen_q.append(meta)
                chosen.extend(group)
            if len(chosen) >= TARGET_STATES:
                break
        selected_examples, selected_questions = chosen, chosen_q
        if len(selected_examples) >= TARGET_STATES:
            break
    if len(selected_examples) < 30:
        raise RuntimeError(f"measurement_candidate_pool_too_small:{len(selected_examples)}")
    selected_examples = selected_examples[:100]
    selected_qids = {ex["qid"] for ex in selected_examples}
    selected_questions = [row for row in selected_questions if row["id"] in selected_qids]
    if not selected_qids or not selected_qids.issubset(excluded_ids | {row["id"] for row in eligible}):
        raise RuntimeError("measurement_question_selection_invalid")
    if selected_qids & set(base_internal_freeze["ids"]) or selected_qids & set(v2_internal["ids"]):
        raise RuntimeError("measurement_internal_isolation_collision")

    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)

    def render(messages: list[Message]) -> list[int]:
        value = processor.apply_chat_template(
            [{"role": msg.role, "content": msg.content} for msg in messages],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if isinstance(value, dict):
            value = value["input_ids"]
        if value and isinstance(value[0], list):
            value = value[0]
        return value

    public_rows, private_rows = [], []
    for ordinal, example in enumerate(selected_examples):
        public = public_example(example, ordinal)
        messages = [Message(**msg) for msg in public["visible_messages"]]
        budgeted = budget_messages(messages, 2048, lambda ms: len(render(ms)),
                                   lambda text: len(processor.tokenizer.encode(text, add_special_tokens=False)),
                                   observation_encoding="compact-v3")
        actual = [{"role": msg.role, "content": msg.content} for msg in budgeted.messages]
        token_ids = render(budgeted.messages)
        target = {"role": "assistant", "content": json.dumps(public["proposed_action"], ensure_ascii=False, separators=(",", ":"))}
        encoded = encode_assistant_only(processor, actual + [target], 2176, len(actual), False)
        target_tokens = sum(label != -100 for label in encoded["labels"])
        if len(token_ids) > 2048 or target_tokens > 128:
            continue
        evidence = visible_documents(actual)
        if not {doc["id"] for doc in evidence}.issubset(set(public["visible_evidence_ids"])):
            raise RuntimeError("measurement_supplement_unexpected_visible_evidence")
        packet = {"case_id": public["case_id"], "question": public["question"],
                  "visible_messages": actual, "proposed_action": public["proposed_action"],
                  "visible_evidence": evidence, "retrieval_top3": public["retrieval_top3"],
                  "prompt_sha256": sha(str(token_ids).encode()), "prompt_tokens": len(token_ids),
                  "target_tokens": target_tokens, "preflight_error": None}
        packet["packet_sha256"] = sha(canonical(packet))
        public_rows.append(packet)
        private_rows.append({"case_id": packet["case_id"], "source_id": example["qid"],
                             "split": "train", "state_kind": example["state_kind"],
                             "risk_tags": example["risk_tags"]})
    if len(public_rows) < 30 or len({row["case_id"] for row in public_rows}) != len(public_rows):
        raise RuntimeError(f"measurement_supplement_preflight_or_duplicate_failure:{len(public_rows)}")
    existing_ids = set()
    for path in WORK.glob("*packets.jsonl"):
        existing_ids.update(row.get("case_id") for row in
                            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
    if existing_ids & {row["case_id"] for row in public_rows}:
        raise RuntimeError("measurement_supplement_case_id_collision")

    old_ids = set(base_train_freeze["ids"])
    old_hashes = set(base_train_freeze["question_hashes"])
    added_hashes = {hashlib.sha256(norm(row["question"]).encode()).hexdigest() for row in selected_questions}
    if old_hashes & added_hashes:
        raise RuntimeError("measurement_question_norm_collision")
    train_freeze_v2 = {
        "version": "v114-training-question-freeze-v2",
        "source_split": base_train_freeze["source_split"],
        "parent_freeze_sha256": sha((CLEAN / "training-question-freeze.json").read_bytes()),
        "question_count": len(old_ids | selected_qids),
        "ids": sorted(old_ids | selected_qids),
        "question_hashes": sorted(old_hashes | added_hashes),
        "added_source_ids": sorted(selected_qids),
        "selection_rule": "retain v114 training freeze v1 and add fixed SHA256-ordered, historically unseen official-training questions with explicit numeric measurement requests; build only evidence-supported answers and nonempty evidence search states; do not inspect evaluation data",
        "selection_seed": SEED,
        "candidate_count": len(public_rows),
        "source_manifest_sha256": sha((CLEAN / "manifest.json").read_bytes()),
        "independent100_content_read": False,
        "final_test_accessed": False,
    }
    write_once(WORK / "training-question-freeze-v2.json",
               (json.dumps(train_freeze_v2, ensure_ascii=False, indent=2) + "\n").encode())
    payload = b"".join(canonical(row) + b"\n" for row in public_rows)
    private_payload = b"".join(canonical(row) + b"\n" for row in private_rows)
    for reviewer in ("a", "b"):
        write_once(WORK / f"measurement-supplement-{reviewer}-packets.jsonl", payload)
    write_once(WORK / "measurement-supplement-private.jsonl", private_payload)
    summary = {"version": "v114-measurement-supplement-v1", "candidate_states": len(public_rows),
               "source_questions": len(selected_qids),
               "action_kinds": dict(Counter(row["state_kind"] for row in private_rows)),
               "source_manifest_sha256": sha((CLEAN / "manifest.json").read_bytes()),
               "training_freeze_sha256": sha((WORK / "training-question-freeze-v2.json").read_bytes()),
               "packet_sha256": sha(payload), "private_mapping_sha256": sha(private_payload),
               "independent100_content_read": False, "final_test_accessed": False}
    write_once(WORK / "measurement-supplement-manifest.json",
               (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
