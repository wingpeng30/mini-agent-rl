"""Re-encode frozen review states with the local processor; no model loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.v123_cpu_core import OUT, readl, sha, write


def replay() -> dict:
    from transformers import AutoProcessor
    from mini_agent_rl.domain import Message
    from scripts.prepare_v116_review import render, encode_assistant_only

    path = OUT / "internal/review/internal-packets.jsonl"
    packets = readl(path)
    tokenizer_dir = Path("D:/qwen_08b")
    tokenizer = tokenizer_dir / "tokenizer.json"
    if not tokenizer.is_file():
        raise ValueError("local_tokenizer_missing")
    processor = AutoProcessor.from_pretrained(str(tokenizer_dir), local_files_only=True)
    mismatches = []
    for p in packets:
        visible = p["visible_messages"]
        target = {"role": "assistant", "content": json.dumps(p["proposed_action"],
                  ensure_ascii=False, separators=(",", ":"))}
        prompt_ids = render(processor, [Message(**m) for m in visible])
        encoded = encode_assistant_only(processor, visible + [target], 2176, len(visible), False)
        canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True,
                                               separators=(",", ":")).encode()
        observed = {"prompt_tokens": len(prompt_ids),
                    "target_tokens": sum(x != -100 for x in encoded["labels"]),
                    "total_tokens": len(encoded["input_ids"]),
                    "prompt_ids_sha256": hashlib.sha256(canonical(prompt_ids)).hexdigest(),
                    "sequence_ids_sha256": hashlib.sha256(canonical(encoded["input_ids"])).hexdigest(),
                    "target_mask_sha256": hashlib.sha256(canonical(encoded["labels"])).hexdigest()}
        if any(p.get(key) != value for key, value in observed.items()):
            mismatches.append({"case_id": p["case_id"], "fields": [key for key, value in observed.items()
                                                                   if p.get(key) != value]})
    report = {"version": "v123-token-replay-v1", "states": len(packets),
              "packet_sha256": sha(path), "tokenizer_json_sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
              "mismatch_count": len(mismatches), "mismatches": mismatches,
              "all_exact": len(packets) == 246 and not mismatches,
              "model_inference": "not_run", "training_allowed": False}
    write(OUT / "internal/token-replay.json", report)
    return report
