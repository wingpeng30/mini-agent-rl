"""Replay exact v117 internal token IDs, masks, and truncated evidence."""
from __future__ import annotations

import json

from transformers import AutoProcessor

from scripts.prepare_v116_review import PROCESSOR, sha
from scripts.prepare_v117 import OUT, ROOT
from scripts.prepare_v117_review import loadl, write
from scripts.verify_v116_tokens import check


def main():
    old = json.loads((ROOT / "data/evidence-sft-v116/token-replay.json").read_text(encoding="utf-8"))
    if not old.get("all_exact") or old["counts"] != {"train": 60, "internal": 100, "reused_v114_train": 332}:
        raise RuntimeError("reused_training_token_baseline_invalid")
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    path = OUT / "review/internal-packets.jsonl"
    packets = loadl(path)
    for packet in packets:
        check(packet, processor)
    report = {"version": "v117-local-token-replay-v1", "internal_states": len(packets),
              "packet_sha256": sha(path), "reused_training_report_sha256":
              sha(ROOT / "data/evidence-sft-v116/token-replay.json"),
              "all_exact": True, "remote_exact_replay_pending": True}
    write(OUT / "token-replay.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
