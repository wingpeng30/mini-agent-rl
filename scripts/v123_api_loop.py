"""Bounded sequential CPU audit batches with gates recomputed after every batch."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v123_api_review import audit_batch
from scripts.v123_review import next_batch, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batches", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_batches <= 12:
        parser.error("max-batches must be between 1 and 12")
    for _ in range(args.max_batches):
        before = status()
        if before["stop_for_risk_bound"] or before["stop_for_pair_bound"] or before["review_complete"]:
            print(json.dumps({"stage": "stop", "status": before}, ensure_ascii=False), flush=True)
            return
        batch = next_batch("api")
        if batch.get("status") in {"stopped", "blocked"}:
            print(json.dumps({"stage": "stop", "batch": batch, "status": before}, ensure_ascii=False), flush=True)
            return
        result = audit_batch(batch["batch"])
        after = status()
        print(json.dumps({"stage": "completed", "batch": batch["batch"], "result": result,
                          "status": after}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
