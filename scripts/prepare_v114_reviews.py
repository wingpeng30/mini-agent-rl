"""Freeze runtime-visible, blinded v114 packets without changing source candidates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoProcessor

from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.domain import Message
from mini_agent_rl.model.context import ContextBudgetError, budget_messages
from mini_agent_rl.training.sft import encode_assistant_only

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/evidence-sft-v114-clean"
OUT = ROOT / "data/evidence-sft-v114-review"
MODEL = Path("D:/qwen_08b")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def freeze(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"frozen_review_artifact_changed:{path}")
        return
    path.write_bytes(data)


def main() -> None:
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if sha((SOURCE / name).read_bytes()) != expected:
            raise RuntimeError(f"source_hash_mismatch:{name}")
    if not MODEL.is_dir():
        raise RuntimeError("local_tokenizer_missing")
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    source = load(SOURCE / "blind-review-candidates.jsonl")
    if len(source) != 419 or len({row["case_id"] for row in source}) != 419:
        raise RuntimeError("expected_419_unique_candidates")

    def render(messages: list[Message]) -> list[int]:
        value = processor.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if isinstance(value, dict):
            value = value["input_ids"]
        if len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        if not isinstance(value, list) or not value or any(type(t) is not int or t < 0 for t in value):
            raise ValueError("invalid_prompt_tokens")
        return value

    packets: list[dict] = []
    failures = []
    for row in source:
        reason = None
        try:
            original = [Message(**m) for m in row["visible_messages"]]
            budgeted = budget_messages(original, 2048, lambda ms: len(render(ms)),
                                      lambda s: len(processor.tokenizer.encode(s, add_special_tokens=False)),
                                      observation_encoding="compact-v3")
            actual = [{"role": m.role, "content": m.content} for m in budgeted.messages]
            token_ids = render(budgeted.messages)
            if len(token_ids) > 2048:
                raise ValueError("input_exceeds_2048")
            target = {"role": "assistant", "content": json.dumps(row["proposed_action"], ensure_ascii=False, separators=(",", ":"))}
            encoded = encode_assistant_only(processor, actual + [target], 2176, len(actual), False)
            target_tokens = sum(v != -100 for v in encoded["labels"])
            if target_tokens > 128:
                raise ValueError("target_exceeds_128")
            docs = visible_documents(actual)
            actual_ids = {d["id"] for d in docs}
            if not actual_ids.issubset(set(row["visible_evidence_ids"])):
                raise ValueError("unexpected_visible_document")
            query = row["proposed_action"].get("query")
            old_queries = []
            for m in actual:
                if m["role"] == "assistant":
                    try:
                        action = json.loads(m["content"])
                    except (ValueError, TypeError):
                        continue
                    if action.get("action") == "search":
                        old_queries.append(action.get("query", ""))
            if query and query.strip().casefold() in {q.strip().casefold() for q in old_queries}:
                raise ValueError("repeated_query")
            packed = {
                "case_id": row["case_id"], "question": row["question"],
                "visible_messages": actual, "proposed_action": row["proposed_action"],
                "visible_evidence": docs, "retrieval_top3": row["retrieval_top3"],
                "prompt_sha256": sha(bytes(str(token_ids), "utf-8")),
                "prompt_tokens": len(token_ids), "target_tokens": target_tokens,
                "preflight_error": None,
            }
        except (ContextBudgetError, ValueError, KeyError, TypeError) as exc:
            reason = f"{type(exc).__name__}:{exc}"
            packed = {
                "case_id": row["case_id"], "question": row["question"],
                "visible_messages": [], "proposed_action": row["proposed_action"],
                "visible_evidence": [], "retrieval_top3": [],
                "prompt_sha256": None, "prompt_tokens": None, "target_tokens": None,
                "preflight_error": reason,
            }
        packed["packet_sha256"] = sha(canonical(packed))
        packets.append(packed)
        if reason:
            failures.append({"case_id": row["case_id"], "reason": reason})
    OUT.mkdir(parents=True, exist_ok=True)
    for reviewer in ("a", "b"):
        ordered = sorted(packets, key=lambda p: sha((f"v114-review-{reviewer}:" + p["case_id"]).encode()))
        freeze(OUT / f"reviewer-{reviewer}-packets.jsonl",
               b"".join(canonical(p) + b"\n" for p in ordered))
    report = {
        "version": "v114-runtime-visible-v1", "source_manifest_sha256": sha((SOURCE / "manifest.json").read_bytes()),
        "candidate_count": len(packets), "preflight_failures": failures,
        "packet_files_sha256": {f"reviewer-{name}-packets.jsonl": sha((OUT / f"reviewer-{name}-packets.jsonl").read_bytes()) for name in ("a", "b")},
        "model_tokenizer": str(MODEL),
        "model_tokenizer_sha256": sha((MODEL / "tokenizer.json").read_bytes()),
        "context": {"prompt": "direct-v090", "observation_encoding": "compact-v3", "max_input_tokens": 2048, "max_target_tokens": 128},
    }
    freeze(OUT / "preflight.json", json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"candidates": len(packets), "preflight_failed": len(failures), "max_prompt_tokens": max((p["prompt_tokens"] or 0) for p in packets)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
