"""Freeze exact final-input blind packets against the immutable v117 corpus."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoProcessor

from scripts.prepare_v116_review import PROCESSOR, packet, sha
from scripts.prepare_v117 import OUT, ROOT, full_rows
from scripts.build_v114_sft_candidates import norm, support_facts


def loadl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def writel(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def structural_issue(search, answer, source):
    if search["proposed_action"]["action"] != "search" or answer["proposed_action"]["action"] != "answer":
        return "incorrect_pair_actions"
    before = search["before_action"]
    after = search["retrieved_after_search"]
    answer_docs = answer["before_action"]
    if not before or not after or not answer_docs:
        return "missing_actual_evidence"
    if not any(doc["id"] not in {d["id"] for d in before} for doc in after):
        return "no_new_document"
    facts = support_facts(source)
    missing = [(title, quote) for title, quote in facts
               if not any(d["title"] == title and quote in d["text"] for d in before)]
    if not missing:
        return "all_annotated_facts_visible_before_search"
    if not any(d["title"] == title and quote in d["text"] and d["id"] not in {b["id"] for b in before}
               for title, quote in missing for d in after):
        return "no_missing_fact_in_new_top3"
    if not all(any(d["title"] == title and quote in d["text"] for d in answer_docs) for title, quote in facts):
        return "answer_facts_lost_after_truncation"
    if norm(answer["proposed_action"]["answer"]) not in norm(" ".join(d["text"] for d in answer_docs)):
        return "answer_string_not_visible"
    query = norm(search["proposed_action"]["query"])
    grounded = norm(search["question"] + " " + " ".join(d["title"] + " " + d["text"] for d in before))
    if not query or not all(token in grounded.split() for token in query.split()):
        return "query_terms_not_visible_before_search"
    return None


def main():
    freeze = json.loads((OUT / "candidate-freeze.json").read_text(encoding="utf-8"))
    if sha(OUT / "candidate-corpus-private.json") != freeze["corpus_sha256"]:
        raise RuntimeError("corpus_changed")
    candidates = loadl(OUT / "candidate-private.jsonl")
    sources = full_rows(freeze["ids"])
    by_source = defaultdict(list)
    for item in candidates:
        by_source[item["source_id"]].append(item)
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    accepted, rejected = [], []
    for qid in freeze["ids"]:
        rows = by_source.get(qid, [])
        if len(rows) != 2:
            rejected.append({"source_id": qid, "reason": "no_pair_in_fixed_corpus"})
            continue
        try:
            records = [packet(row, processor) for row in rows]
            search = next(p for p in records if p["proposed_action"]["action"] == "search")
            answer = next(p for p in records if p["proposed_action"]["action"] == "answer")
            issue = structural_issue(search, answer, sources[qid])
        except (ValueError, StopIteration) as error:
            issue = "packet_error:" + str(error)[:100]
        if issue:
            rejected.append({"source_id": qid, "reason": issue})
            continue
        accepted.append((qid, search, answer))
    work = OUT / "review"
    work.mkdir(parents=True, exist_ok=True)
    for name, rows in (("eligible-private", [{"source_id": qid, "case_ids": [s["case_id"], a["case_id"]]}
                                                      for qid, s, a in accepted]),
                       ("rejected-private", rejected)):
        writel(work / f"{name}.jsonl", rows)
    # Scope risk is a pre-review scheduling hint only. Final quotas use the
    # intersection of the two independent semantic labels.
    priority = []
    scheduled = set()
    for risk in ("scope", "measurement"):
        for item in accepted:
            if risk in item[1]["risk_tags_provisional"] and item[0] not in scheduled:
                priority.append(item)
                scheduled.add(item[0])
    priority.extend(item for item in accepted if item[0] not in scheduled)
    packets = []
    for _, search, answer in priority:
        packets.extend((search, answer))
    if len({p["case_id"] for p in packets}) != len(packets):
        raise RuntimeError("duplicate_candidate_state")
    packet_file = work / "internal-packets.jsonl"
    writel(packet_file, packets)
    batches = {}
    for offset in range(0, len(packets), 20):
        batch = work / f"internal-batch-{offset//20+1:02}.jsonl"
        writel(batch, packets[offset:offset+20])
        batches[batch.name] = {"states": len(packets[offset:offset+20]), "sha256": sha(batch)}
    manifest = {"version": "v117-review-v1", "source_sha256": sha(OUT / "candidate-private.jsonl"),
                "corpus_sha256": freeze["corpus_sha256"], "packet_sha256": sha(packet_file),
                "eligible_questions": len(priority), "states": len(packets), "batches": batches,
                "pre_review_rejections": dict(Counter(x["reason"] for x in rejected)),
                "scheduling": "provisional scope then measurement then remaining frozen source order; no model scores"}
    write(work / "manifest.json", manifest)
    print(json.dumps({"eligible_questions": len(priority), "states": len(packets),
                      "rejected": len(rejected), "batches": len(batches)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
