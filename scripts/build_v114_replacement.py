"""Freeze a deterministic clean replacement for the ambiguous internal question."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from transformers import AutoProcessor

from build_v114_sft_candidates import (
    OUT as SOURCE, ROOT, TRAIN_FILES, build_examples, build_index, collect_history,
    documents_for, load_full, norm, public_example, risk_tags, support_facts,
)
from prepare_v114_reviews import MODEL, canonical, sha
from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.domain import Message
from mini_agent_rl.model.context import budget_messages
from mini_agent_rl.training.sft import encode_assistant_only

WORK = ROOT / "data/evidence-sft-v114-review"
OLD_BAD_ID = "5a81a76655429903bc27b994"
SEED = "v114-internal-replacement-v1:"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def freeze(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError(f"frozen_artifact_changed:{path.name}")
    if not path.exists():
        path.write_bytes(data)


def main():
    manifest = read_json(SOURCE / "manifest.json")
    for filename, expected in manifest["files"].items():
        if sha((SOURCE / filename).read_bytes()) != expected:
            raise RuntimeError(f"source_manifest_hash_mismatch:{filename}")

    old_internal = read_json(SOURCE / "internal-freeze.json")
    old_train = read_json(SOURCE / "training-question-freeze.json")
    old_ids = set(old_internal["ids"])
    train_ids = set(old_train["ids"])
    if OLD_BAD_ID not in old_ids:
        raise RuntimeError("expected_ambiguous_question_not_in_old_freeze")
    old_norms = set()

    history, _ = collect_history()
    history_q = {v for v in history if not v.startswith("__id__")}
    history_ids = {v.removeprefix("__id__") for v in history if v.startswith("__id__")}
    unique = {}
    for path in TRAIN_FILES:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["id", "question", "type"], batch_size=4096):
            for row in batch.to_pylist():
                qid, question = str(row["id"]), row["question"]
                question_norm = norm(question)
                if qid.casefold() in history_ids or question_norm in history_q or question_norm in unique:
                    continue
                unique[question_norm] = {"id": qid, "question": question, "type": row.get("type"),
                                         "risk_tags": risk_tags(question)}
    old_meta = [row for row in unique.values() if row["id"] in old_ids]
    if len(old_meta) != 50:
        raise RuntimeError("old_internal_freeze_not_source_verified")
    old_norms = {norm(row["question"]) for row in old_meta}
    replacement_pool = [row for row in unique.values()
                        if row["id"] not in old_ids | train_ids
                        and norm(row["question"]) not in old_norms
                        and "measurement" in row["risk_tags"]]
    if not replacement_pool:
        raise RuntimeError("no_clean_measurement_replacement_candidate")
    replacement_pool.sort(key=lambda row: (hashlib.sha256((SEED + row["id"]).encode()).hexdigest(), row["id"]))
    replacement = replacement_pool[0]
    new_ids = (old_ids - {OLD_BAD_ID}) | {replacement["id"]}
    if len(new_ids) != 50:
        raise RuntimeError("replacement_freeze_wrong_size")
    risk_counts = Counter(tag for row in unique.values() if row["id"] in new_ids for tag in row["risk_tags"])
    freeze_obj = {
        "version": "v114-internal-freeze-v2",
        "source_split": "official HotpotQA distractor train",
        "question_count": 50,
        "ids": sorted(new_ids),
        "question_hashes": sorted(hashlib.sha256(norm(row["question"]).encode()).hexdigest()
                                   for row in unique.values() if row["id"] in new_ids),
        "risk_question_counts": dict(risk_counts),
        "selection_rule": "retain the prior 49 unambiguous IDs; replace only the dual-rejected ambiguous Roosevelt Park ID with the first measurement-risk eligible unused training-source ID under SHA256(v114-internal-replacement-v1:<source id>) order; exclude all prior local exposure, old internal IDs, and frozen training IDs before selection",
        "replaced_source_id": OLD_BAD_ID,
        "replacement_source_id": replacement["id"],
        "replacement_question_sha256": hashlib.sha256(norm(replacement["question"]).encode()).hexdigest(),
        "candidate_quality_used_for_selection": False,
        "independent100_content_read": False,
        "final_test_accessed": False,
    }
    freeze(WORK / "internal-freeze-v2.json", json.dumps(freeze_obj, ensure_ascii=False, indent=2).encode() + b"\n")

    source_rows = load_full(new_ids)
    corpus, index = build_index(list(source_rows.values()))
    examples = build_examples([source_rows[replacement["id"]]], index)
    if not examples:
        raise RuntimeError("frozen_replacement_has_no_reproducible_candidate_state")
    by_kind = {}
    for example in examples:
        by_kind.setdefault(example["state_kind"], []).append(example)
    selected = []
    for kind in ("supported_answer", "nonempty_evidence_search", "empty_evidence_search"):
        candidates = sorted(by_kind.get(kind, []), key=lambda e: hashlib.sha256(
            ("v114-replacement-state:" + e["qid"] + ":" + e["state_kind"] + ":" + str(e["action"])).encode()
        ).hexdigest())
        selected.extend(candidates[:1] if kind != "nonempty_evidence_search" else candidates[:4])
    public = [public_example(example, i) for i, example in enumerate(selected)]
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    packets = []
    private = []
    for row, example in zip(public, selected):
        messages = [Message(**m) for m in row["visible_messages"]]
        budgeted = budget_messages(messages, 2048,
                                   lambda ms: len(processor.apply_chat_template(
                                       [{"role": m.role, "content": m.content} for m in ms],
                                       tokenize=True, add_generation_prompt=True, enable_thinking=False)[0]),
                                   lambda s: len(processor.tokenizer.encode(s, add_special_tokens=False)),
                                   observation_encoding="compact-v3")
        actual = [{"role": m.role, "content": m.content} for m in budgeted.messages]
        token_ids = processor.apply_chat_template(actual, tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False)[0]
        target = {"role": "assistant", "content": json.dumps(row["proposed_action"], ensure_ascii=False, separators=(",", ":"))}
        encoded = encode_assistant_only(processor, actual + [target], 2176, len(actual), False)
        target_tokens = sum(value != -100 for value in encoded["labels"])
        if len(token_ids) > 2048 or target_tokens > 128:
            raise RuntimeError("replacement_candidate_exceeds_frozen_token_budget")
        docs = visible_documents(actual)
        packet = {"case_id": row["case_id"], "question": row["question"], "visible_messages": actual,
                  "proposed_action": row["proposed_action"], "visible_evidence": docs,
                  "retrieval_top3": row["retrieval_top3"],
                  "prompt_sha256": sha(bytes(str(token_ids), "utf-8")),
                  "prompt_tokens": len(token_ids), "target_tokens": target_tokens, "preflight_error": None}
        packet["packet_sha256"] = sha(canonical(packet))
        packets.append(packet)
        private.append({"case_id": row["case_id"], "source_id": replacement["id"], "split": "internal",
                        "state_kind": example["state_kind"], "risk_tags": example["risk_tags"]})
    payload = b"".join(canonical(row) + b"\n" for row in packets)
    for reviewer in ("a", "b"):
        freeze(WORK / f"replacement-{reviewer}-packets.jsonl", payload)
    freeze(WORK / "replacement-private.jsonl", b"".join(canonical(row) + b"\n" for row in private))
    replacement_record = next(row for row in unique.values() if row["id"] == replacement["id"])
    freeze(WORK / "replacement-question.json", json.dumps({
        "source_id": replacement["id"], "question": replacement["question"],
        "normalized_question_sha256": freeze_obj["replacement_question_sha256"],
        "risk_tags": replacement_record["risk_tags"], "order_sha256": hashlib.sha256((SEED + replacement["id"]).encode()).hexdigest(),
        "packet_count": len(packets), "packet_sha256": sha(payload),
    }, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"replacement_id": replacement["id"], "risk_tags": replacement["risk_tags"],
                      "packet_count": len(packets), "state_kinds": dict(Counter(e["state_kind"] for e in selected))}))


if __name__ == "__main__":
    main()
