"""Explicit allowlist and per-record ledger for the full data repair audit.

No API, training, generation, broad data discovery, or automatic label correction.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from scripts.v123_cpu_core import ROOT, digest, readl, sha, write, writel

OUT = ROOT / "data/evidence-sft-v124-forensic"
FROZEN = {
    "v117_train": "data/evidence-sft-v117/frozen/train.jsonl",
    "v117_exposed_internal50": "data/evidence-sft-v117/frozen/internal.jsonl",
}
PACKETS = {
    "v122_internal_candidates": "data/evidence-sft-v122-internal-candidates/review/internal-packets.jsonl",
    "v122_train_candidates": "data/evidence-sft-v122-train-candidates/review/internal-packets.jsonl",
    "v123_internal_candidates": "data/evidence-sft-v123/internal/review/internal-packets.jsonl",
}


def inventory():
    from mini_agent_rl.answer_verifier import visible_documents

    records, sources = [], {}
    for cohort, relative in {**FROZEN, **PACKETS}.items():
        path = ROOT / relative
        rows = readl(path)
        sources[cohort] = {"path": relative, "sha256": sha(path), "records": len(rows)}
        for index, row in enumerate(rows, 1):
            if cohort in FROZEN:
                target = row["target_assistant_index"]
                messages = row["messages"][:target]
                action = json.loads(row["messages"][target]["content"])
                question = next(m["content"] for m in messages if m["role"] == "user")
                before = visible_documents(messages)
                after = None
                original_case = None
            else:
                messages, action = row["visible_messages"], row["proposed_action"]
                question, before = row["question"], row["before_action"]
                after, original_case = row.get("retrieved_after_search"), row["case_id"]
            record = {"record_id": f"{cohort}:{index:04}", "cohort": cohort,
                      "source_path": relative, "source_line": index, "source_row_sha256": digest(row),
                      "original_case_id": original_case, "question": question,
                      "visible_messages": messages, "proposed_action": action,
                      "before_action": before, "retrieved_after_search": after,
                      "search_retrieval_binding": "missing_needs_lineage" if action["action"] == "search" and after is None else "present_or_not_search"}
            record["record_sha256"] = digest(record)
            records.append(record)
    manifest = {"version": "v124-forensic-inventory-v1", "sources": sources,
                "records": len(records), "actions": dict(Counter(r["proposed_action"]["action"] for r in records)),
                "training_allowed": False, "all_rows_semantically_reviewed": False}
    path = OUT / "inventory-private.jsonl"
    if path.exists() and readl(path) != records:
        raise ValueError("inventory_changed_refuse_overwrite")
    if not path.exists():
        writel(path, records)
    manifest["inventory_sha256"] = sha(path)
    write(OUT / "inventory-manifest.json", manifest)
    return manifest


def show(cohort, start, count):
    records = readl(OUT / "inventory-private.jsonl")
    chosen = [r for r in records if r["cohort"] == cohort][start - 1:start - 1 + count]
    for row in chosen:
        print(json.dumps({k: row[k] for k in ("record_id", "record_sha256", "question", "proposed_action", "before_action", "retrieved_after_search")}, ensure_ascii=True))


def lineage():
    index = defaultdict(list)
    paths = ["data/evidence-sft-v114-review/accepted.jsonl", "data/evidence-sft-v116/review/train-packets.jsonl",
             "data/evidence-sft-v117/review/internal-packets.jsonl"]
    hashes = {}
    for relative in paths:
        hashes[relative] = sha(ROOT / relative)
        for row in readl(ROOT / relative):
            if "packet" in row and row.get("split") != "train":
                continue
            packet = row.get("packet", row)
            key = digest([packet["visible_messages"], packet["proposed_action"]])
            index[key].append({"source_path": relative, "source_case_id": packet["case_id"],
                               "source_packet_sha256": digest(packet),
                               "retrieved_after_search": packet.get("retrieved_after_search", packet.get("retrieval_top3")),
                               "before_action": packet.get("before_action", packet.get("visible_evidence"))})
    rows = []
    for row in readl(OUT / "inventory-private.jsonl"):
        if row["cohort"] not in FROZEN:
            continue
        matches = index.get(digest([row["visible_messages"], row["proposed_action"]]), [])
        rows.append({"record_id": row["record_id"], "record_sha256": row["record_sha256"],
                     "matches": matches, "exact_prompt_action_match": bool(matches)})
    writel(OUT / "lineage-private.jsonl", rows)
    result = {"sources": hashes, "records": len(rows), "exact_matched": sum(bool(r["matches"]) for r in rows),
              "missing_record_ids": [r["record_id"] for r in rows if not r["matches"]], "training_allowed": False}
    write(OUT / "lineage-summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "show", "lineage"))
    parser.add_argument("--cohort", default="v117_train", choices=tuple(FROZEN) + tuple(PACKETS))
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    if args.command == "inventory":
        print(json.dumps(inventory(), indent=2))
    elif args.command == "lineage":
        print(json.dumps(lineage(), indent=2))
    else:
        show(args.cohort, args.start, args.count)
