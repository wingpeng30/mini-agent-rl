"""Add a question-grounded search state for the one uncovered frozen question."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoProcessor

from mini_agent_rl.domain import Message
from mini_agent_rl.model.context import budget_messages
from mini_agent_rl.prompts import DIRECT_SYSTEM
from mini_agent_rl.tools.search import _BM25Index
from mini_agent_rl.training.sft import encode_assistant_only

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data/evidence-sft-v114-clean"
WORK = ROOT / "data/evidence-sft-v114-review"
SOURCE = ROOT / "data/hotpot-full-train-v114"
MODEL = Path("D:/qwen_08b")


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def freeze(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"refuse_to_change_frozen_artifact:{path.name}")
    if not path.exists():
        path.write_bytes(content)


def main() -> None:
    frozen = set(json.loads((CLEAN / "internal-freeze.json").read_text(encoding="utf-8"))["ids"])
    key = [json.loads(line) for line in (CLEAN / "private-key.jsonl").read_text(encoding="utf-8").splitlines() if line]
    covered = {row["source_id"] for row in key if row["split"] == "internal"}
    missing = frozen - covered
    if len(missing) != 1:
        raise RuntimeError(f"expected_one_missing_frozen_question:{len(missing)}")
    qid = next(iter(missing))
    question = None
    for path in sorted(SOURCE.glob("train-*-of-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=["id", "question"], batch_size=4096):
            for row in batch.to_pylist():
                if row["id"] == qid:
                    question = row["question"]
    if not question:
        raise RuntimeError("missing_question_not_in_train_source")
    docs = [json.loads(line) for line in (CLEAN / "internal-corpus.jsonl").read_text(encoding="utf-8").splitlines() if line]
    results = _BM25Index(docs).search(question, top_k=3)
    if len(results) != 3:
        raise RuntimeError("expected_real_top3")
    messages = [Message(role="system", content=DIRECT_SYSTEM), Message(role="user", content=question)]
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)

    def render(ms):
        value = processor.apply_chat_template([{"role": m.role, "content": m.content} for m in ms],
                                              tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if isinstance(value, dict):
            value = value["input_ids"]
        return value[0] if value and isinstance(value[0], list) else value

    budgeted = budget_messages(messages, 2048, lambda ms: len(render(ms)),
                               lambda text: len(processor.tokenizer.encode(text, add_special_tokens=False)),
                               observation_encoding="compact-v3")
    actual = [{"role": m.role, "content": m.content} for m in budgeted.messages]
    action = {"action": "search", "query": question}
    target = {"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))}
    encoded = encode_assistant_only(processor, actual + [target], 2176, len(actual), False)
    target_tokens = sum(v != -100 for v in encoded["labels"])
    if target_tokens > 128:
        raise RuntimeError("target_exceeds_128")
    case_id = sha(("v114-gap-search:" + qid + ":" + question).encode())[:20]
    packet = {"case_id": case_id, "question": question, "visible_messages": actual,
              "proposed_action": action, "visible_evidence": [], "retrieval_top3": results,
              "prompt_sha256": sha(bytes(str(render(budgeted.messages)), "utf-8")),
              "prompt_tokens": len(render(budgeted.messages)), "target_tokens": target_tokens,
              "preflight_error": None}
    packet["packet_sha256"] = sha(canonical(packet))
    WORK.mkdir(parents=True, exist_ok=True)
    for reviewer in ("a", "b"):
        freeze(WORK / f"supplement-gap-{reviewer}-packets.jsonl", canonical(packet) + b"\n")
    freeze(WORK / "supplement-gap-private.json", canonical({"case_id": case_id, "source_id": qid, "split": "internal"}) + b"\n")
    print(json.dumps({"case_id": case_id, "prompt_tokens": packet["prompt_tokens"],
                      "retrieved_titles": [d["title"] for d in results]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
