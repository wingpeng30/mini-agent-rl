"""Record evidence-bound primary disagreements with v9 batch 02 votes."""
from __future__ import annotations

import json
from collections import defaultdict

from scripts.v123_cpu_core import digest, readl, sha, write
from scripts.v124_forensic import OUT


BASE = OUT / "pending-dual-review-v9"
BATCH = "v9-batch-02"
TARGETS = {
    "v117_train:0183": "USS Elrod query is question/evidence-grounded and retrieves the missing university relation.",
    "v117_train:0285": "USS Elrod query is question/evidence-grounded and retrieves the missing university relation.",
    "v117_train:0286": "USS Elrod query is question/evidence-grounded and retrieves the missing university relation.",
    "v117_train:0287": "Same question/query family is approved despite the same claimed grounding defect.",
    "v117_train:0338": "Carl Sargent appears in the user question and prior evidence; top-3 supplies the cover-artist relation.",
    "v117_train:0067": "Fred Meyer appears in the question and prior evidence; top-3 supplies Rite Aid as occupant.",
    "v117_train:0227": "Fred Meyer appears in the question and prior evidence; top-3 supplies Rite Aid as occupant.",
}


def main() -> None:
    packet_path = BASE / f"{BATCH}-packet-private.json"
    manifest = json.loads((BASE / f"{BATCH}-manifest.json").read_text(encoding="utf-8"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet_hash = packet.pop("packet_sha256")
    assert packet_hash == manifest["packet_payload_sha256"]
    assert sha(packet_path) == manifest["packet_file_sha256"]
    binding = json.loads((BASE / f"{BATCH}-binding-private.json").read_text(encoding="utf-8"))
    source_by_anon = {m["state_id"]: m for m in binding["source_mapping"]}
    review = json.loads((BASE / f"deepseek-responses/{BATCH}/validated-review-private.json").read_text(
        encoding="utf-8"))
    reviewed = {r["state_id"]: r for r in review["reviews"]}
    state_by_anon = {s["state_id"]: s for g in packet["groups"] for s in g["states"]}
    anon_by_source = {m["source_record_id"]: anon for anon, m in source_by_anon.items()}
    if set(TARGETS) - set(anon_by_source):
        raise RuntimeError("disputed_source_ids_not_in_batch")

    findings = []
    for source_id, reason in TARGETS.items():
        anon = anon_by_source[source_id]
        state = state_by_anon[anon]
        vote = reviewed[anon]
        question = next(m["content"] for m in state["visible_messages"] if m["role"] == "user")
        before = state["pre_action_evidence"]
        after = state["post_action_retrieval_results"] or []
        evidence_text = "\n".join(d["text"] for d in before)
        result_text = "\n".join(d["text"] for d in after)
        if source_id.endswith(("0183", "0285", "0286", "0287")):
            assert "USS Elrod" in question
            assert any("Elrod" in d["title"] and "FFG-55" in d["title"] for d in before)
            assert "Yale University" in result_text and "University of Georgia" in result_text
        elif source_id.endswith("0338"):
            assert "Carl Sargent" in question
            assert any("Carl Sargent" in d["title"] for d in before)
            assert "box cover art by Jeff Easley" in result_text
        else:
            assert "Fred Meyer" in question and "Fred Meyer" in evidence_text
            assert "currently occupied by Rite Aid" in result_text
        findings.append({
            "source_record_id": source_id,
            "source_row_sha256": source_by_anon[anon]["original_row_sha256"],
            "revised_row_sha256": source_by_anon[anon]["revised_row_sha256"],
            "state_id_in_blind_packet": anon,
            "deepseek_decision": vote["decision"],
            "deepseek_issues": vote["issues"],
            "evidence_verified": True,
            "primary_disagreement": reason,
            "disagreement_class": "query_grounding_or_relevance_misapplied_to_visible_anchor_and_direct_top3",
            "training_use": "hold_pending_independent_adjudication",
        })

    same_family = [x for x in findings if x["source_record_id"] in {
        "v117_train:0183", "v117_train:0285", "v117_train:0286", "v117_train:0287"}]
    family_votes = {x["deepseek_decision"] for x in same_family}
    assert "approve" in family_votes and "reject" in family_votes
    report = {
        "version": "v124-v9-batch02-primary-disagreement-v1",
        "batch_id": BATCH,
        "packet_sha256": packet_hash,
        "initial_review_sha256": sha(BASE / f"deepseek-responses/{BATCH}/validated-review-private.json"),
        "state_count": len(reviewed),
        "disputed_states": len(findings),
        "same_question_query_family_has_conflicting_decisions": True,
        "source_evidence_rechecked": True,
        "semantic_votes_not_rewritten": True,
        "independent_double_review_pass": False,
        "training_allowed": False,
        "findings": findings,
    }
    write(OUT / "v9-batch02-primary-disagreement-private.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "findings"} |
                     {"source_record_ids": [x["source_record_id"] for x in findings]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
