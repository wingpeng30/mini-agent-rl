"""Prepare a bounded v6 batch of v117 source-scoped question/query drafts.

Only the explicitly listed, unsealed v117 training rows are read. The script
does not alter source rows, run generation/training, submit reviews or authorize
training. Every proposed action/question is bound to source, inventory and
primary-review hashes and remains a draft until independent review.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.v123_cpu_core import ROOT, digest, readl, writel
from scripts.v124_forensic import OUT

TRAIN = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"

SPEC = {
    35: {"question": "How is the film about a tabloid talk show host similar to the talk show that debuted in 1991?"},
    119: {"question": "According to the retrieved United States Navy passage, how many personnel does it report on active duty?"},
    186: {"question": "Who created the box cover art for Night Below, a boxed set written by Carl Sargent?"},
    189: {"question": "Which of these is described as Irish-born: Michael Gambon or the film Viceroy's House?"},
    194: {"question": "According to the supplied Amman article, what population does it report for the city that houses CBRL's British Institute?"},
    200: {"question": "How many votes did Halina Olendzka receive when she ran on the Law and Justice list in the 2005 Sejm election?"},
    201: {"question": "From which years did the duchy of Saxe-Coburg and Gotha, where Gustav Adolf von Wulffen was born, last?"},
    203: {"question": "According to the supplied Pimpinela article, how many records had the Argentine duo sold worldwide?"},
    206: {"question": "According to the supplied Merrimack Valley High School article, approximately how many students does it report?"},
    207: {"question": "According to the supplied Mampong article, what population does it report for the town where Gerald Asamoah was born?"},
    208: {"question": "Which nearly 22-km-high volcano does the supplied account identify as part of the Tharsis dome associated with shifts in Mars's poles?"},
    215: {"question": "Which of these buildings was described as under construction in the source: 599 Lexington Avenue or 30 Hudson Yards?",
          "query": "599 Lexington Avenue 30 Hudson Yards construction Manhattan"},
    257: {"question": "In which years does Jonathan Rea's biography say he was named Irish Motorcyclist of the Year?"},
    299: {"question": "What is Steve Kazee's birth date?", "query": "Steve Kazee birth date"},
    4: {"question": "Which event damaged Southbridge Municipal Airport on June 1, 2011?"},
}

QUERY_OVERRIDES = {
    215: "599 Lexington Avenue 30 Hudson Yards construction Manhattan",
    299: "Steve Kazee birth date",
    343: "Southbridge Municipal Airport June 1 2011 damage event",
}


def target(row: dict) -> dict:
    return json.loads(row["messages"][row["target_assistant_index"]]["content"])


def main() -> None:
    source_rows = readl(TRAIN)
    if len(source_rows) != 345:
        raise RuntimeError("unexpected_v117_train_length")
    by_id = {f"v117_train:{i:04d}": row for i, row in enumerate(source_rows, 1)}
    inventory_rows = readl(OUT / "inventory-private.jsonl")
    inventory = {r["record_id"]: r for r in inventory_rows}
    primary = {r["record_id"]: r for p in sorted(OUT.glob("primary-batch-*.jsonl")) for r in readl(p)}
    if len(inventory) != len(inventory_rows) or len(primary) != len(inventory):
        raise RuntimeError("inventory_or_primary_coverage_mismatch")
    effective = {rid: dict(row) for rid, row in primary.items()}
    amendments = [r for p in sorted(OUT.glob("primary-amendments*-private.jsonl")) for r in readl(p)]
    for change in amendments:
        original = primary[change["record_id"]]
        if change["superseded_primary_review_sha256"] != digest(original):
            raise RuntimeError("primary_amendment_hash_mismatch:" + change["record_id"])
        effective[change["record_id"]]["disposition"] = change["effective_disposition"]
    overrides = readl(OUT / "primary-adjudication-overrides-private.jsonl")
    for change in overrides:
        rid = change["record_id"]
        if effective[rid]["disposition"] != change["supersedes_effective_disposition"]:
            raise RuntimeError("primary_override_chain_mismatch:" + rid)
        effective[rid]["disposition"] = change["effective_disposition"]

    emitted = set()
    drafts = []
    for ordinal, spec in SPEC.items():
        rid = f"v117_train:{ordinal:04d}"
        original = by_id[rid]
        old_q = original["messages"][1]["content"]
        peers = sorted(xid for xid, row in by_id.items()
                       if row["messages"][1]["content"] == old_q)
        if emitted.intersection(peers):
            raise RuntimeError("overlapping_v6_question_groups:" + rid)
        emitted.update(peers)
        for peer_id in peers:
            if effective[peer_id]["disposition"] not in {
                    "repair_candidate", "quarantine_pending_repair", "retain_candidate"}:
                raise RuntimeError(f"nonrepair_peer_requires_manual_resolution:{peer_id}")
        revised_rows = []
        for peer_id in peers:
            revised = copy.deepcopy(by_id[peer_id])
            revised["messages"][1]["content"] = spec["question"]
            proposed_action = target(revised)
            ordinal_peer = int(peer_id.split(":")[1])
            if ordinal_peer in QUERY_OVERRIDES:
                if proposed_action.get("action") != "search":
                    raise RuntimeError("query_override_requires_search_target:" + peer_id)
                proposed_action["query"] = QUERY_OVERRIDES[ordinal_peer]
                revised["messages"][revised["target_assistant_index"]]["content"] = json.dumps(
                    proposed_action, ensure_ascii=False, separators=(",", ":"))
            revised_rows.append({
                "record_id": peer_id,
                "original_row_sha256": digest(by_id[peer_id]),
                "primary_review_sha256": digest(primary[peer_id]),
                "original_question": old_q,
                "same_question_record_ids": peers,
                "same_question_all_states_reviewed": True,
                "revised_row": revised,
                "amendment_version": "v6",
                "status": "draft_only_requires_semantic_double_review",
                "approved_training_label": False,
                "training_allowed": False,
            })
        drafts.extend(revised_rows)

    path = OUT / "repair-drafts-v6-private.jsonl"
    if path.exists() and readl(path) != drafts:
        raise RuntimeError("existing_v6_drafts_differ_refuse_overwrite")
    if not path.exists():
        writel(path, drafts)
    print(json.dumps({"revision_groups": len(SPEC), "states": len(drafts),
                      "states_with_query_change": len(QUERY_OVERRIDES),
                      "status": "draft_only_requires_semantic_double_review",
                      "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
