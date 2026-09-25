"""Recompute v116 review packet token IDs, target masks and visible evidence."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from transformers import AutoProcessor

from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.training.sft import encode_assistant_only
from scripts.prepare_v116_review import WORK, PROCESSOR, canonical, loadl, render

OLD = WORK.parent / "evidence-sft-v114-review/accepted.jsonl"


def check(packet, processor):
    prompt = packet["visible_messages"]
    prompt_ids = render(processor, [type("M", (), m)() for m in prompt])
    action = packet["proposed_action"]
    target = {"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))}
    encoded = encode_assistant_only(processor, prompt + [target], 2176, len(prompt), False)
    labels = encoded["labels"]
    target_count = sum(token != -100 for token in labels)
    expected = {
        "prompt_tokens": len(prompt_ids), "target_tokens": target_count,
        "total_tokens": len(encoded["input_ids"]),
        "prompt_ids_sha256": hashlib.sha256(canonical(prompt_ids)).hexdigest(),
        "sequence_ids_sha256": hashlib.sha256(canonical(encoded["input_ids"])).hexdigest(),
        "target_mask_sha256": hashlib.sha256(canonical(labels)).hexdigest(),
    }
    if any(packet.get(k) != v for k, v in expected.items()):
        raise ValueError("token_or_mask_mismatch:" + packet["case_id"])
    if len(prompt_ids) > 2048 or target_count > 128 or len(encoded["input_ids"]) > 2176:
        raise ValueError("token_limit:" + packet["case_id"])
    if visible_documents(prompt) != packet["before_action"]:
        raise ValueError("visible_evidence_mismatch:" + packet["case_id"])
    if len({(d["id"], d["text"]) for d in packet["before_action"]}) != len(packet["before_action"]):
        raise ValueError("duplicate_visible_document:" + packet["case_id"])
    return expected


def main():
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    review = WORK / "review"
    counts = Counter()
    for split in ("train", "internal"):
        packets = loadl(review / f"{split}-packets.jsonl")
        for packet in packets:
            check(packet, processor)
            counts[split] += 1
    for row in loadl(OLD):
        if row["split"] != "train":
            continue
        packet = row["packet"]
        prompt = packet["visible_messages"]
        prompt_ids = render(processor, [type("M", (), m)() for m in prompt])
        target = {"role": "assistant", "content": json.dumps(packet["proposed_action"], ensure_ascii=False,
                                                      separators=(",", ":"))}
        encoded = encode_assistant_only(processor, prompt + [target], 2176, len(prompt), False)
        target_tokens = sum(x != -100 for x in encoded["labels"])
        if (hashlib.sha256(str(prompt_ids).encode()).hexdigest() != packet["prompt_sha256"]
                or len(prompt_ids) != packet["prompt_tokens"] or target_tokens != packet["target_tokens"]
                or len(prompt_ids) > 2048 or target_tokens > 128 or len(encoded["input_ids"]) > 2176
                or visible_documents(prompt) != packet["visible_evidence"]):
            raise ValueError("old_approved_packet_token_mismatch:" + packet["case_id"])
        counts["reused_v114_train"] += 1
    inputs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (review / "train-packets.jsonl", review / "internal-packets.jsonl", OLD)}
    report = {"version": "v116-local-token-replay-v1", "counts": dict(counts),
              "input_sha256": inputs, "all_exact": True, "remote_exact_replay_pending": True}
    (WORK / "token-replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
