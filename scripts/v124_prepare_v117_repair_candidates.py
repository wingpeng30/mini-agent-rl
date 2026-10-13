"""Prepare CPU-token-replayed, non-training revisions for recoverable v117 cases.

Only user wording is changed. Search queries, frozen observations and action
targets remain byte-for-byte equivalent, so no retrieval result is fabricated.
Every output remains a draft until fresh independent review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoProcessor
from mini_agent_rl.training.sft import encode_assistant_only
from scripts.prepare_v116_review import PROCESSOR, canonical, render

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
OUT = ROOT / "data/evidence-sft-v124-forensic"
REVISIONS = {
    "v117_train:0001": (
        "According to the cited First Lady of Ghana source, who was identified as the current holder of the title previously held by Nana Konadu Agyeman-Rawlings?",
        "source-scoped current-office claim; preserves the historical predecessor link and limits currentness to the cited source snapshot",
    ),
    "v117_train:0050": (
        "How many inhabitants did the retrieved population passage report for the city near Alberta Ferretti's studio?",
        "asks for the source-reported value rather than asserting a timeless population count",
    ),
    "v117_train:0067": (
        "According to the retrieved passage about the former first suburban Fred Meyer store, which company did the passage identify as occupying it?",
        "scopes the time-sensitive occupant claim to the cited passage",
    ),
    "v117_train:0165": (
        "According to the supplied WLAX syndicated-programming list, which long-running American arbitration-based reality court show was included?",
        "scopes the time-sensitive lineup claim to the supplied list",
    ),
    "v117_train:0174": (
        "What is the name of Michael Hayden's best-known commission at the airport on Chicago's Far Northwest Side?",
        "removes the unsupported assumption that the commission was a painting",
    ),
}
QUESTION_GROUPS = {
    "v117_train:0001": ["v117_train:0001", "v117_train:0167"],
    "v117_train:0050": ["v117_train:0050", "v117_train:0161"],
    "v117_train:0067": ["v117_train:0067", "v117_train:0179", "v117_train:0227"],
    "v117_train:0165": ["v117_train:0165", "v117_train:0209"],
    "v117_train:0174": ["v117_train:0174"],
}


def sha(value) -> str:
    if isinstance(value, bytes):
        raw = value
    else:
        raw = canonical(value)
    return hashlib.sha256(raw).hexdigest()


def readl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main():
    rows = readl(SOURCE)
    inventory = {row["record_id"]: row for row in readl(OUT / "inventory-private.jsonl")}
    if len(rows) != 345:
        raise RuntimeError("unexpected_v117_train_state_count")
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    prepared = []
    checked = set()
    for anchor, (new_question, rationale) in REVISIONS.items():
        ids = QUESTION_GROUPS[anchor]
        if any(record_id in checked for record_id in ids):
            raise RuntimeError("overlapping_question_group")
        group_rows = []
        old_questions = set()
        for record_id in ids:
            ordinal = int(record_id.split(":", 1)[1])
            row = rows[ordinal - 1]
            old_question = row["messages"][1]["content"]
            old_questions.add(old_question)
            if inventory[record_id]["question"] != old_question:
                raise RuntimeError(f"inventory_question_mismatch:{record_id}")
            messages = json.loads(json.dumps(row["messages"], ensure_ascii=False))
            target_index = row["target_assistant_index"]
            target = json.loads(messages[target_index]["content"])
            messages[1]["content"] = new_question
            prompt = messages[:target_index]
            prompt_ids = render(processor, [type("M", (), message)() for message in prompt])
            target_message = {"role": "assistant", "content": json.dumps(
                target, ensure_ascii=False, separators=(",", ":"))}
            encoded = encode_assistant_only(processor, prompt + [target_message],
                                            2176, len(prompt), False)
            target_tokens = sum(token != -100 for token in encoded["labels"])
            if len(prompt_ids) > 2048 or target_tokens > 128 or len(encoded["input_ids"]) > 2176:
                raise RuntimeError(f"token_limit:{record_id}")
            old_searches = [json.loads(m["content"]) for m in row["messages"][:target_index]
                            if m.get("role") == "assistant" and m["content"].lstrip().startswith("{")]
            retrieved = inventory[record_id].get("retrieved_after_search")
            if target.get("action") == "search" and not isinstance(retrieved, list):
                # Search results are recorded on the subsequent answer state.
                peers = [inventory[peer] for peer in ids if peer != record_id]
                if not any(peer.get("before_action") for peer in peers):
                    raise RuntimeError(f"search_transition_missing_observation:{record_id}")
            if target.get("action") == "search" and not target.get("query", "").strip():
                raise RuntimeError(f"empty_search_query:{record_id}")
            group_rows.append({
                "record_id": record_id,
                "source_record_sha256": inventory[record_id]["record_sha256"],
                "original_row_sha256": sha(row),
                "original_question": old_question,
                "revised_question": new_question,
                "revised_visible_messages": prompt,
                "target_action": target,
                "prompt_tokens": len(prompt_ids),
                "target_tokens": target_tokens,
                "total_tokens": len(encoded["input_ids"]),
                "prompt_ids_sha256": sha(canonical(prompt_ids)),
                "sequence_ids_sha256": sha(canonical(encoded["input_ids"])),
                "target_mask_sha256": sha(canonical(encoded["labels"])),
                "query_unchanged": True,
                "retrieval_replay": "reuse_exact_frozen_observation_query_unchanged",
                "review_status": "draft_only_fresh_independent_double_review_required",
                "training_allowed": False,
            })
        if len(old_questions) != 1:
            raise RuntimeError(f"source_question_group_not_identical:{anchor}")
        checked.update(ids)
        prepared.append({
            "revision_id": "candidate-" + anchor.replace(":", "-"),
            "anchor_record_id": anchor,
            "record_ids": ids,
            "revision_rationale": rationale,
            "source_question": next(iter(old_questions)),
            "revised_question": new_question,
            "states": group_rows,
            "same_question_group_complete": True,
            "training_allowed": False,
        })

    report = {
        "version": "v124-v117-repair-candidates-cpu-v1",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "processor_path": str(PROCESSOR),
        "processor_mode": "local_files_only; tokenizer and chat template only; no model loaded or generated",
        "question_groups": len(prepared),
        "states": sum(len(row["states"]) for row in prepared),
        "search_queries_changed": 0,
        "retrieval_replayed": 0,
        "retrieval_reuse_basis": "For every search action, the search query and frozen source corpus/observation are unchanged; this CPU pass validates token rendering only and does not claim a new retrieval result.",
        "candidates": prepared,
        "training_allowed": False,
    }
    target = OUT / "v117-repair-candidates-cpu-private.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"question_groups": report["question_groups"],
                      "states": report["states"], "report": str(target),
                      "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
