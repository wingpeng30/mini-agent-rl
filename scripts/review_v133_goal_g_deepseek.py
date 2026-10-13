"""One-shot, bounded DeepSeek review of frozen anonymous Goal G packets.

Runs where the existing restricted credential is configured. No key is printed,
copied to AutoDL, or stored in the result directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.v130_deepseek_review import (
    MODEL, _credentials, _no_redirect_opener, run_batch,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-g-cpu"
PACKETS = OUT / "review-packets.jsonl"
EXPECTED_SHA = "c5c44da23a275e76089f67d93b39e25e0e1e9a706628a317007f501aa7e4da26"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-index", type=int, choices=range(4), required=True)
    parser.add_argument("--direct-connection", action="store_true")
    args = parser.parse_args()
    actual = hashlib.sha256(PACKETS.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA:
        raise ValueError("frozen_anonymous_packets_changed")
    packets = [json.loads(x) for x in PACKETS.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(packets) != 19:
        raise ValueError("packet_count_changed")
    ranges = ((0, 5), (5, 10), (10, 15), (15, 19))
    lo, hi = ranges[args.batch_index]
    group = packets[lo:hi]
    key, base = _credentials()
    if not key or base != "https://api.deepseek.com":
        raise RuntimeError("existing_official_deepseek_credential_unavailable")
    batch_dir = OUT / "deepseek" / f"g-{lo:02d}-{hi-1:02d}"
    result = run_batch(batch=f"g-{lo:02d}-{hi-1:02d}", packets=group,
        packet_file_sha256=actual, freeze_sha256=actual, batch_dir=batch_dir,
        key=key, base=base,
        opener=_no_redirect_opener(direct_connection=args.direct_connection),
        direct_connection=args.direct_connection)
    raw = sorted((batch_dir / "responses-private").glob("attempt-*-raw.json"))[-1]
    response_model = json.loads(raw.read_text(encoding="utf-8")).get("model")
    model_identity_verified = response_model == MODEL
    identity = {"requested_model": MODEL, "response_model": response_model,
                "model_identity_verified": model_identity_verified,
                "packet_sha256": actual, "review_count": len(result["reviews"]),
                "approved": sum(v["approved"] for v in result["reviews"])}
    (batch_dir / "identity-check.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(identity))


if __name__ == "__main__":
    main()
