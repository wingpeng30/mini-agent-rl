"""One-shot DeepSeek review for Goal L's frozen anonymous answer candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import v130_deepseek_review as ds

OUT = ROOT / "runs/v133-goal-l-answer-policy"
SOURCE_PACKET = OUT / "review-packets-anonymous.json"
PACKET = OUT / "deepseek-review-bundle.json"
MANIFEST = OUT / "freeze-manifest.json"
BATCH = "goal-l-answer-policy-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="send the single preauthorized Goal L review batch")
    args = ap.parse_args()
    if not args.send:
        print(json.dumps({"status": "not_sent", "training_allowed": False}))
        return
    source = json.loads(SOURCE_PACKET.read_text(encoding="utf-8"))
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    packets = source.get("packets", [])
    if (source.get("max_cases") != 8 or not packets or len(packets) > 8):
        raise ValueError("review_budget_or_authorization_failed")
    if any(not re.fullmatch(r"L-review-[0-9]{2}", p.get("case_id", "")) for p in packets):
        raise ValueError("review_alias_allowlist_failed")
    private_keys = {"arm", "policy", "gold", "official_answer", "source_id", "prediction", "selection_reason"}
    if any(private_keys & set(p) for p in packets):
        raise ValueError("private_condition_or_label_in_anonymous_packet")
    key, base = ds._credentials()
    if not key or base != "https://api.deepseek.com":
        raise RuntimeError("approved_deepseek_credentials_or_host_missing")
    bundle = {"training_allowed": False, "review_authorized": True,
        "freeze_sha256": frozen["prompts_sha256"], "protocol": ds.contract_identity(),
        "packets": packets, "payload_sha256": ds._payload_hash(packets)}
    PACKET.write_text(json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if bundle["payload_sha256"] != ds._payload_hash(packets) or bundle["protocol"] != ds.contract_identity():
        raise ValueError("review_bundle_freeze_failed")
    packet_sha = hashlib.sha256(PACKET.read_bytes()).hexdigest()
    batch_dir = OUT / "deepseek" / BATCH
    if batch_dir.exists():
        raise RuntimeError("batch_exists_stop_without_retry")
    result = ds.run_batch(batch=BATCH, packets=packets, packet_file_sha256=packet_sha,
        freeze_sha256=frozen["prompts_sha256"], batch_dir=batch_dir, key=key,
        opener=ds._no_redirect_opener(), direct_connection=False)
    print(json.dumps({"status": "valid_votes_saved", "batch": BATCH,
        "valid_vote_count": len(result["reviews"]), "packet_sha256": packet_sha,
        "payload_sha256": result["payload_sha256"], "training_allowed": False}))


if __name__ == "__main__":
    main()
