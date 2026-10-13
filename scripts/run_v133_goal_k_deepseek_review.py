"""Send the one frozen Goal K anonymous output-review batch to DeepSeek."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
package = types.ModuleType("scripts")
package.__path__ = [str(ROOT / "scripts")]
sys.modules["scripts"] = package

from scripts import v130_deepseek_review as ds

OUT = ROOT / "runs/v133-goal-k-attribution"
PACKET = OUT / "review-packets-anonymous.json"
EXPECTED_PACKET_SHA = "b1899b6469bf558de7b717c94f4fac3e6af036d90d6012cd0928f0cec26d85b7"
EXPECTED_EVAL_SHA = "42531fc1dbd2c3cb1fbec12e85523483c1c0b40546c55a0a6d622474d9648d29"
BATCH = "goal-k-output-losses-01"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", help="send this one approved anonymous review batch")
    args = parser.parse_args()
    if not args.send:
        print(json.dumps({"status": "not_sent", "reason": "explicit --send required"}))
        return
    if not PACKET.is_file():
        raise FileNotFoundError("frozen_review_packet_missing")
    packet_sha = sha(PACKET)
    if EXPECTED_PACKET_SHA and packet_sha != EXPECTED_PACKET_SHA:
        raise ValueError("review_packet_hash_mismatch")
    bundle = json.loads(PACKET.read_text(encoding="utf-8"))
    if bundle.get("training_allowed") is not False or bundle.get("review_authorized") is not True:
        raise ValueError("review_packet_not_authorized_or_training_flag_invalid")
    if bundle.get("freeze_sha256") != EXPECTED_EVAL_SHA:
        raise ValueError("evaluation_freeze_binding_mismatch")
    packets = bundle.get("packets")
    if [p.get("case_id") for p in packets or []] != ["J-loss-A", "J-loss-B"]:
        raise ValueError("review_case_allowlist_mismatch")
    if any("official_answer" in p or "source_id" in p or "arm" in p for p in packets):
        raise ValueError("anonymous_packet_contains_private_mapping")
    if bundle.get("payload_sha256") != ds._payload_hash(packets):
        raise ValueError("payload_hash_mismatch")
    if bundle.get("protocol") != ds.contract_identity():
        raise ValueError("review_protocol_identity_mismatch")
    key, configured = ds._credentials()
    if not key or configured != "https://api.deepseek.com":
        raise RuntimeError("approved_deepseek_credentials_or_host_missing")
    result = ds.run_batch(
        batch=BATCH,
        packets=packets,
        packet_file_sha256=packet_sha,
        freeze_sha256=EXPECTED_EVAL_SHA,
        batch_dir=OUT / "deepseek" / BATCH,
        key=key,
        opener=ds._no_redirect_opener(),
        direct_connection=False,
    )
    print(json.dumps({"status": "valid_vote_saved", "batch": BATCH,
                      "cases": len(result["reviews"]), "training_allowed": False,
                      "packet_sha256": packet_sha,
                      "payload_sha256": result["payload_sha256"]}))


if __name__ == "__main__":
    main()
