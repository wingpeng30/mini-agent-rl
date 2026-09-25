"""Replace evidence-quality failures and reuse only identical blinded packets."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from scripts.prepare_v116_data import OUT, internal_candidates
from scripts.prepare_v116_review import main as rebuild_packets


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review = OUT / "review"
    previous = {r["case_id"]: r for r in rows(review / "internal-packets.jsonl")}
    old_votes = {}
    for path in sorted(review.glob("luna-internal-batch-*.jsonl")):
        for row in rows(path):
            if row["case_id"] in old_votes:
                raise ValueError("duplicate_old_vote")
            old_votes[row["case_id"]] = row
    if len(previous) != 100 or len(old_votes) != 100:
        raise ValueError("initial_internal_review_not_complete")
    excluded = json.loads((OUT / "internal-review-quality-exclusions.json").read_text(encoding="utf-8"))
    private = {r["packet"]["case_id"]: r["source_id"] for r in rows(OUT / "internal-candidate-private.jsonl")}
    rejected_ids = {private[key] for key, vote in old_votes.items() if not vote["approved"]}
    if rejected_ids != set(excluded):
        raise ValueError("quality_exclusion_review_mismatch")
    superseded = OUT / "superseded-internal-v2"
    if superseded.exists():
        raise FileExistsError("revision_already_started")
    superseded.mkdir()
    files = [OUT / name for name in ("internal-candidate-private.jsonl", "internal-source-freeze.json",
                                     "internal-corpus-private.json", "internal-question-quality-exclusions.json")]
    files += list(review.glob("internal*")) + list(review.glob("luna-internal*")) + [review / "manifest.json"]
    for path in files:
        shutil.copy2(path, superseded / ("review-" + path.name if path.parent == review else path.name))
    inventory = json.loads((OUT / "history-private.json").read_text(encoding="utf-8"))
    internal_candidates(set(inventory["ids"]), set(inventory["normalized_questions"]))
    rebuild_packets()
    current = rows(review / "internal-packets.jsonl")
    reused = [old_votes[p["case_id"]] for p in current if p["case_id"] in old_votes
              and previous[p["case_id"]]["packet_sha256"] == p["packet_sha256"]]
    for path in review.glob("luna-internal-batch-*.jsonl"):
        path.unlink()
    (review / "luna-internal-batch-00-reused.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reused), encoding="utf-8")
    reused_ids = {r["case_id"] for r in reused}
    pending = [p for p in current if p["case_id"] not in reused_ids]
    (review / "internal-revision-pending.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pending), encoding="utf-8")
    for number, start in enumerate(range(0, len(pending), 20), 1):
        path = review / f"internal-revision-batch-{number:02}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pending[start:start + 20]),
                        encoding="utf-8")
    revision_batches = sorted(review.glob("internal-revision-batch-*.jsonl"))
    review_manifest = {"version": "v116-internal-revision-review-freeze-v1",
                       "canonical_packets_sha256": sha(review / "internal-packets.jsonl"),
                       "pending_packets_sha256": sha(review / "internal-revision-pending.jsonl"),
                       "same_packet_objects": True, "pending_count": len(pending),
                       "batches": {path.name: {"sha256": sha(path), "rows": len(rows(path))}
                                   for path in revision_batches}}
    (review / "internal-revision-manifest.json").write_text(
        json.dumps(review_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"version": "v116-internal-revision-v1", "superseded": str(superseded.relative_to(OUT)),
              "quality_excluded_questions": len(excluded), "reused_exact_packet_votes": len(reused),
              "pending_independent_review": len(pending), "new_internal_packet_sha256": sha(review / "internal-packets.jsonl"),
              "new_internal_source_sha256": sha(OUT / "internal-candidate-private.jsonl"),
              "selection_uses_model_performance": False}
    (OUT / "internal-revision-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
