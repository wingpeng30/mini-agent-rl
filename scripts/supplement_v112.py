"""Uniform, unapproved question-only initial searches for frozen internal questions."""
from __future__ import annotations

import argparse
import hashlib
import json

try:
    from .build_v112 import ROOT, OUT, digest, norm, provisional_risks, question, read, write_rows
except ImportError:
    from build_v112 import ROOT, OUT, digest, norm, provisional_risks, question, read, write_rows


def build(model_path: str = "D:/qwen_08b") -> dict:
    from transformers import AutoProcessor
    from mini_agent_rl.prompts import DIRECT_SYSTEM
    from mini_agent_rl.tools.search import _BM25Index
    from mini_agent_rl.training.sft import encode_assistant_only

    outputs = [OUT / "supplement-initial.jsonl", OUT / "blind-packet-12.jsonl"]
    if any(path.exists() for path in outputs):
        raise RuntimeError("refuse_to_overwrite_frozen_supplement")
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    source = ROOT / "data/hotpot-agent-v041/train.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise RuntimeError("frozen_training_source_changed")
    internal = manifest["internal_ids"]
    if len(internal) != 50 or len(set(internal)) != 50:
        raise RuntimeError("require_exactly_50_frozen_internal_questions")
    source_rows = {row["metadata"]["task_id"]: row for row in read(source)}
    corpus = json.loads((OUT / "train-corpus.json").read_text(encoding="utf-8"))
    index = _BM25Index(corpus)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    candidates, packets = [], []
    for task_id in internal:
        q = question(source_rows[task_id])
        actual = [{"role": "system", "content": DIRECT_SYSTEM}, {"role": "user", "content": q}]
        prompt_ids = processor.apply_chat_template(actual, tokenize=True, add_generation_prompt=True,
                                                   enable_thinking=False)
        if isinstance(prompt_ids, dict):
            prompt_ids = prompt_ids["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        action = {"action": "search", "query": q}
        full = actual + [{"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))}]
        encoded = encode_assistant_only(processor, full, 2176, 2, False)
        target_tokens = sum(token != -100 for token in encoded["labels"])
        if len(prompt_ids) > 2048 or target_tokens > 128:
            raise ValueError(f"supplement_token_budget:{task_id}:input={len(prompt_ids)}:target={target_tokens}")
        hits = [{key: doc[key] for key in ("id", "title", "text")} for doc in index.search(q, top_k=5)]
        row = {"task_id": task_id, "question": q, "messages": full, "target_assistant_index": 2,
               "thinking_enabled": False, "target": {"action": "search", "value": q}, "step": 0,
               "risk_labels": [], "provisional_risks": provisional_risks(q), "evidence": [],
               "evidence_hash": digest([]), "state_hash": digest({"question": norm(q), "messages": actual}),
               "reviews": [], "split": "validation", "variant": "uniform-initial-question-search",
               "prompt_token_ids": prompt_ids, "prompt_tokens": len(prompt_ids), "target_tokens": target_tokens,
               "retrieval_results": hits,
               "metadata": {"source": "data/hotpot-agent-v041/train.jsonl",
                            "source_kind": "uniform_initial_search_unapproved"}}
        candidates.append(row)
        packets.append({"state_hash": row["state_hash"], "question": q, "current_evidence": [],
                        "proposed_action": row["target"], "previous_queries": [], "retrieval_results": hits})
    if len({row["state_hash"] for row in candidates}) != 50:
        raise RuntimeError("duplicate_initial_state")
    # All 50 must preflight before either frozen artifact is written.
    write_rows(outputs[0], candidates)
    write_rows(outputs[1], packets)
    return {"states": len(candidates), "questions": len(internal), "approved": 0,
            "max_input_tokens": max(row["prompt_tokens"] for row in candidates),
            "max_target_tokens": max(row["target_tokens"] for row in candidates),
            "training_permitted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="D:/qwen_08b")
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
