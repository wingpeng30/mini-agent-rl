"""Explicit stage runner for v123 CPU work.  No generator or trainer is imported."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v123_cpu_core import OUT, baseline, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", default="check", choices=(
        "check", "recheck", "diagnose", "third-search", "coverage", "prepare-internal", "token-replay", "review-status", "next-api", "audit-api",
        "next-luna", "finalize-internal", "assess-train", "train-status", "train-next-luna", "train-finalize"))
    parser.add_argument("--batch", help="Exact work-api batch filename for audit-api")
    args = parser.parse_args()
    if args.stage == "check":
        result = baseline()
        write(OUT / "baseline.json", result)
    elif args.stage == "recheck":
        from scripts.v123_recheck import recheck
        result = recheck()
    elif args.stage == "diagnose":
        from scripts.v123_diagnose import diagnose
        result = diagnose()
    elif args.stage == "third-search":
        from scripts.v123_third_search import analyze
        result = analyze()
    elif args.stage == "coverage":
        from scripts.v123_coverage import analyze
        result = analyze()
    elif args.stage == "prepare-internal":
        from scripts.v123_prepare_internal import prepare
        result = prepare()
    elif args.stage == "token-replay":
        from scripts.v123_token_replay import replay
        result = replay()
    elif args.stage in {"review-status", "next-api", "next-luna", "finalize-internal"}:
        from scripts.v123_review import status, next_batch, finalize_internal
        result = (status() if args.stage == "review-status" else
                  next_batch("api" if args.stage == "next-api" else "luna") if args.stage in {"next-api", "next-luna"} else
                  finalize_internal())
    elif args.stage == "audit-api":
        if not args.batch:
            parser.error("--batch is required for audit-api")
        from scripts.v123_api_review import audit_batch
        result = audit_batch(args.batch)
    elif args.stage == "assess-train":
        from scripts.v123_train_assess import assess
        result = assess()
    else:
        from scripts.v123_train_review import review_status, next_luna_batch, finalize
        result = (review_status() if args.stage == "train-status" else
                  next_luna_batch() if args.stage == "train-next-luna" else finalize())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
