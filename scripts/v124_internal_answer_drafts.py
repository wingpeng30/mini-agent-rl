"""Record exact, non-trainable answer corrections for clear v122 errors."""
from scripts.v123_cpu_core import ROOT, digest, readl, writel
from scripts.v124_forensic import OUT


# Only source-supported answer corrections whose paired search state has been reviewed.
CORRECTIONS = {
    138: ("Big Pig", "The Bonk article identifies Big Pig as the band and Breakaway as its Glitch song."),
    174: ("30 March 1930", "Remove the copied '(born' fragment from Rolf Harris's documented birth date."),
    194: ("great-granddaughter", "Alexandra is Victoria's granddaughter and Anastasia is Alexandra's daughter."),
}


def main():
    inventory = {row["record_id"]: row for row in readl(OUT / "inventory-private.jsonl")}
    reviews = {row["record_id"]: row for path in sorted(OUT.glob("primary-batch-*.jsonl"))
               for row in readl(path)}
    source = ROOT / "data/evidence-sft-v122-internal-candidates/review/internal-packets.jsonl"
    packets = readl(source)
    drafts = []
    for number, (answer, reason) in CORRECTIONS.items():
        record_id = f"v122_internal_candidates:{number:04}"
        search_id = f"v122_internal_candidates:{number-1:04}"
        row, review, packet = inventory[record_id], reviews[record_id], packets[number - 1]
        assert row["source_path"] == source.relative_to(ROOT).as_posix()
        assert row["source_line"] == number and digest(packet) == row["source_row_sha256"]
        assert row["proposed_action"]["action"] == packet["proposed_action"]["action"] == "answer"
        assert row["proposed_action"] == packet["proposed_action"]
        assert search_id in reviews and inventory[search_id]["question"] == row["question"]
        assert review["disposition"] in {"repair_candidate", "quarantine_pending_repair"}
        assert answer != row["proposed_action"]["answer"]
        drafts.append({"record_id": record_id, "paired_search_record_id": search_id,
                       "original_packet_sha256": digest(packet),
                       "original_record_sha256": row["record_sha256"],
                       "primary_review_sha256": digest(review),
                       "original_action": row["proposed_action"],
                       "proposed_action": {"action": "answer", "answer": answer},
                       "rationale": reason,
                       "status": "needs_paired_context_token_replay_and_independent_double_review",
                       "approved_training_label": False, "training_allowed": False})
    dest = OUT / "internal-answer-correction-drafts-private.jsonl"
    if dest.exists() and readl(dest) != drafts:
        raise ValueError("existing_answer_drafts_differ")
    if not dest.exists():
        writel(dest, drafts)
    print(f"Saved {len(drafts)} source-bound, non-trainable answer drafts.")


if __name__ == "__main__":
    main()
