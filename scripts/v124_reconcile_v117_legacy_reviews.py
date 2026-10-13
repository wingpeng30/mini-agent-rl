"""Reconcile historical reviews with v117 frozen training states.

Read-only with respect to source assets. Writes one audit report under the
isolated v124 forensic directory and never authorizes training.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/evidence-sft-v124-forensic"


def readl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def digest(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def projection(documents):
    return [{key: document.get(key) for key in ("id", "title", "text")}
            for document in documents]


def main():
    inventory = {row["record_id"]: row for row in readl(OUT / "inventory-private.jsonl")
                 if row["cohort"] == "v117_train"}
    lineage = {row["record_id"]: row for row in readl(OUT / "lineage-private.jsonl")
               if row["record_id"].startswith("v117_train:")}
    if len(inventory) != 345 or set(inventory) != set(lineage):
        raise RuntimeError("v117_train_inventory_lineage_mismatch")

    v114 = {row["case_id"]: row for row in readl(ROOT / "data/evidence-sft-v114-review/accepted.jsonl")}
    v116_review = ROOT / "data/evidence-sft-v116/review"
    v116_packets = {row["case_id"]: row for row in readl(v116_review / "train-packets.jsonl")}
    v116_api = {row["case_id"]: row for row in readl(v116_review / "api-train.jsonl")}
    for row in readl(v116_review / "api-retry-train.jsonl"):
        if row["case_id"] not in v116_api or v116_api[row["case_id"]].get("decision") is not None:
            raise RuntimeError("invalid_v116_retry_lineage")
        v116_api[row["case_id"]] = row
    v116_luna = {}
    for path in sorted(v116_review.glob("luna-train-batch-*.jsonl")):
        for row in readl(path):
            if row["case_id"] in v116_luna:
                raise RuntimeError("duplicate_v116_luna_vote")
            v116_luna[row["case_id"]] = row
    for row in readl(v116_review / "luna-corrections-train.jsonl"):
        if row["case_id"] not in v116_luna:
            raise RuntimeError("orphan_v116_luna_correction")
        v116_luna[row["case_id"]] = row
    packet_file_sha = sha(v116_review / "train-packets.jsonl")

    counts = Counter()
    records = []
    for record_id in sorted(inventory):
        current = inventory[record_id]
        matches = lineage[record_id]["matches"]
        if len(matches) != 1:
            raise RuntimeError(f"non_unique_lineage:{record_id}")
        match = matches[0]
        case_id = match["source_case_id"]
        if "v114-review/accepted.jsonl" in match["source_path"]:
            accepted = v114.get(case_id)
            if not accepted:
                raise RuntimeError(f"missing_v114_accepted:{record_id}")
            packet = accepted["packet"]
            reviewers = [accepted.get("review_a"), accepted.get("review_b")]
            exact = (current["visible_messages"] == packet["visible_messages"]
                     and current["proposed_action"] == packet["proposed_action"]
                     and projection(current["before_action"]) == projection(packet["visible_evidence"])
                     and projection(match["retrieved_after_search"]) == projection(packet["retrieval_top3"]))
            valid_reviews = all(
                isinstance(vote, dict) and vote.get("approved") is True
                and vote.get("packet_sha256") == packet.get("packet_sha256")
                and vote.get("reviewer_id")
                and len(vote.get("citations", [])) > 0
                and not vote.get("_invalid_errors")
                for vote in reviewers
            ) and reviewers[0].get("reviewer_id") != reviewers[1].get("reviewer_id")
            source = "v114_two_reviewer_acceptance"
            detail = {"reviewer_ids": [v.get("reviewer_id") for v in reviewers],
                      "reviewer_packet_hashes_match": valid_reviews}
        elif "v116/review/train-packets.jsonl" in match["source_path"]:
            packet = v116_packets.get(case_id)
            api_record, luna = v116_api.get(case_id), v116_luna.get(case_id)
            if not packet or not api_record or not luna:
                raise RuntimeError(f"missing_v116_vote:{record_id}")
            decision = api_record.get("decision")
            exact = (current["visible_messages"] == packet["visible_messages"]
                     and current["proposed_action"] == packet["proposed_action"])
            api_ok = (bool(decision and decision.get("approved") is True)
                      and api_record.get("packet_sha256") == digest(packet)
                      and api_record.get("input_file_sha256") == packet_file_sha
                      and api_record.get("reviewer") == "deepseek-flash-independent")
            luna_ok = (luna.get("approved") is True
                       and luna.get("packet_sha256") == packet.get("packet_sha256"))
            valid_reviews = api_ok and luna_ok
            source = "v116_deepseek_luna_acceptance"
            detail = {"deepseek_packet_and_batch_hashes_match": api_ok,
                      "luna_packet_hash_matches": luna_ok}
        else:
            raise RuntimeError(f"unexpected_v117_train_source:{record_id}")

        passed = bool(exact and valid_reviews)
        counts[source] += 1
        counts["exact_reviewed_content"] += int(exact)
        counts["valid_dual_approval"] += int(passed)
        if not passed:
            raise RuntimeError(f"legacy_review_reconciliation_failed:{record_id}")
        records.append({"record_id": record_id, "source_case_id": case_id,
                        "source_path": match["source_path"], "reviewer_count": 2,
                        "original_content_exact": exact, "valid_dual_approval": passed,
                        "current_content_sha256": current["record_sha256"], **detail})

    report = {
        "version": "v124-v117-legacy-review-reconciliation-v1",
        "inputs": {
            "inventory_manifest_sha256": sha(OUT / "inventory-manifest.json"),
            "lineage_summary_sha256": sha(OUT / "lineage-summary.json"),
            "v117_frozen_manifest_sha256": sha(ROOT / "data/evidence-sft-v117/frozen/manifest.json"),
            "v114_accepted_sha256": sha(ROOT / "data/evidence-sft-v114-review/accepted.jsonl"),
            "v116_train_packet_sha256": packet_file_sha,
        },
        "counts": dict(counts),
        "interpretation": "Historical dual-review votes are reusable only for the exact original v117 states whose full visible messages, action target, evidence and retrieval results matched. Any changed question, query, action, observation, corpus or tokenized input requires fresh review. This reconciliation does not override v124 primary dispositions or authorize training.",
        "records": records,
        "training_allowed": False,
    }
    target = OUT / "v117-legacy-review-reconciliation-private.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": dict(counts), "report": str(target),
                      "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
