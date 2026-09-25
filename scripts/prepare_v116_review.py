"""Freeze bounded, runtime-visible v116 blind review packets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoProcessor

from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.domain import Message
from mini_agent_rl.model.context import budget_messages
from mini_agent_rl.training.sft import encode_assistant_only

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/evidence-sft-v116"
PROCESSOR = Path("D:/qwen_08b")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loadl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render(processor, messages):
    result = processor.apply_chat_template([{"role": m.role, "content": m.content} for m in messages],
                                           tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if isinstance(result, dict):
        result = result["input_ids"]
    return result[0] if result and isinstance(result[0], list) else result


def packet(row, processor):
    source = row["packet"]
    original = [Message(**m) for m in source.get("visible_messages", source.get("messages", []))]
    budgeted = budget_messages(original, 2048, lambda ms: len(render(processor, ms)),
                               lambda s: len(processor.tokenizer.encode(s, add_special_tokens=False)),
                               observation_encoding="compact-v3")
    actual = [{"role": m.role, "content": m.content} for m in budgeted.messages]
    token_ids = render(processor, budgeted.messages)
    action = source.get("proposed_action", source.get("action"))
    if not isinstance(action, dict) or action.get("action") not in {"answer", "search"}:
        raise ValueError("invalid_action")
    target = {"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))}
    encoded = encode_assistant_only(processor, actual + [target], 2176, len(actual), False)
    target_tokens = sum(x != -100 for x in encoded["labels"])
    if len(token_ids) > 2048 or target_tokens > 128 or len(encoded["input_ids"]) > 2176:
        raise ValueError("token_limit")
    before = visible_documents(actual)
    after = source.get("retrieval_top3") if action["action"] == "search" else None
    if action["action"] == "search" and not isinstance(after, list):
        raise ValueError("missing_search_results")
    if row.get("split", "internal") == "internal" and action["action"] == "search" and not before:
        raise ValueError("internal_search_must_have_nonempty_evidence")
    result = {"case_id": source["case_id"], "question": source["question"],
              "visible_messages": actual, "before_action": before,
              "proposed_action": action, "retrieved_after_search": after,
              "risk_tags_provisional": row.get("risk_tags_provisional", []),
              "prompt_tokens": len(token_ids), "target_tokens": target_tokens,
              "total_tokens": len(encoded["input_ids"]),
              "prompt_ids_sha256": hashlib.sha256(canonical(token_ids)).hexdigest(),
              "sequence_ids_sha256": hashlib.sha256(canonical(encoded["input_ids"])).hexdigest(),
              "target_mask_sha256": hashlib.sha256(canonical(encoded["labels"])).hexdigest(),
              "origin": row["source"]}
    result["packet_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main():
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    if not json.loads((WORK / "history-inventory.json").read_text(encoding="utf-8"))["ready"]:
        raise RuntimeError("history_gate_failed")
    work = WORK / "review";work.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "v116-review-packets-v1", "source_files": {}, "packets": {}}
    for split, name in (("train", "train-supplement-private.jsonl"),
                        ("internal", "internal-candidate-private.jsonl")):
        source = WORK / name
        manifest["source_files"][name] = sha(source)
        records = []
        for row in loadl(source):
            if row.get("split", split) != split:
                raise ValueError("split_mismatch")
            records.append(packet(row, processor))
        if len({r["case_id"] for r in records}) != len(records):
            raise ValueError("duplicate_case_id")
        out = work / f"{split}-packets.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records),
                       encoding="utf-8")
        manifest["packets"][split] = {"count": len(records), "sha256": sha(out)}
        for start in range(0, len(records), 20):
            batch = work / f"{split}-batch-{start // 20 + 1:02}.jsonl"
            batch.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                                     for r in records[start:start + 20]), encoding="utf-8")
    write(work / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
