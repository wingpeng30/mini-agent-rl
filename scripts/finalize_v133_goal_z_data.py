"""Main audit gate for the frozen Goal Z candidate and evidence selections."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import prepare_v133_goal_z_multihop_data as z


OUT = z.OUT


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    target = OUT / "data-review.json"
    if target.exists():
        raise FileExistsError(target)
    curation = json.loads((OUT / "curation.json").read_text(encoding="utf-8"))
    if curation["shortfalls"]:
        raise RuntimeError("Data gate shortfall")
    reviews = [json.loads((OUT / ("replacement-review.json" if n == 1 else f"replacement-review-{n}.json"))
                          .read_text(encoding="utf-8")) for n in range(1, 7)]
    excluded = {split: {v for review in reviews for v in review[split]["excluded_ids"]}
                for split in ("train", "dev", "confirm")}
    ledger = rows(OUT / "train-semantic-rejections.jsonl") + rows(OUT / "eval-semantic-rejections.jsonl")
    if len(ledger) != len({(v["split"], v["id"]) for v in ledger}):
        raise RuntimeError("Duplicate semantic rejection")
    for split in excluded:
        actual = {v["id"] for v in ledger if v["split"] == split and v["reason"].strip()}
        if actual != excluded[split]:
            raise RuntimeError(f"Missing or excess rejection reasons: {split}")
    selected_hashes = {}
    selected = {}
    for split in ("train", "dev", "confirm"):
        path = OUT / ("selected-confirm.jsonl" if split == "confirm" else f"selected-{split}.jsonl")
        selected[split] = rows(path)
        selected_hashes[split] = z.digest(path)
        if selected_hashes[split] != curation["selected_sha256"][split]:
            raise RuntimeError("Curated selection hash mismatch")
        if len(selected[split]) != sum(z.TARGETS[split].values()) or \
           Counter(v["source"] for v in selected[split]) != z.TARGETS[split]:
            raise RuntimeError(f"Split quota mismatch: {split}")
        if excluded[split] & {v["id"] for v in selected[split]}:
            raise RuntimeError(f"Rejected row remains selected: {split}")
        for source in z.TARGETS[split]:
            pool = {v["id"] for v in rows(OUT / f"candidates-{split}-{source}.jsonl")}
            if not {v["id"] for v in selected[split] if v["source"] == source}.issubset(pool):
                raise RuntimeError("Selected ID not in frozen candidate pool")
    corpus = {split: json.loads((OUT / f"corpus-{split}.json").read_text(encoding="utf-8"))
              for split in ("train", "dev", "confirm")}
    titles = {split: {z.norm(v["title"]) for v in value} for split, value in corpus.items()}
    for split in corpus:
        if len(corpus[split]) != 2048 or len(titles[split]) != 2048:
            raise RuntimeError("Corpus count or title uniqueness failed")
    for first, second in (("train", "dev"), ("train", "confirm"), ("dev", "confirm")):
        if titles[first] & titles[second] or \
           {z.norm(v["question"]) for v in selected[first]} & {z.norm(v["question"]) for v in selected[second]} or \
           {x for v in selected[first] for x in v["source_ids"]} & \
           {x for v in selected[second] for x in v["source_ids"]}:
            raise RuntimeError("Cross-split leakage")
    for case in selected["train"]:
        if case["source"] == "old":
            if not case["answer_prompt_token_ids"] or not case["trace"]:
                raise RuntimeError("Old positive lacks actual token record")
            continue
        teacher = case["teacher"]
        first, second = teacher["retrieved"]
        if len(teacher["queries"]) != 2 or \
           not teacher["bridge_visible_in_doc_ids"] or \
           z.norm(teacher["bridge"]) in z.norm(case["question"] + " " + teacher["queries"][0]) or \
           z.norm(teacher["support_titles"][0]) not in {z.norm(v["title"]) for v in first} or \
           z.norm(teacher["support_titles"][1]) in {z.norm(v["title"]) for v in first} or \
           z.norm(teacher["support_titles"][1]) not in {z.norm(v["title"]) for v in second}:
            raise RuntimeError("Two-step teacher evidence invariant failed")
    output = {"status": "approved", "version": "goal-z-data-review-v1",
              "selected_sha256": selected_hashes,
              "corpus_sha256": {split: z.digest(OUT / f"corpus-{split}.json") for split in corpus},
              "source_audit_sha256": z.digest(OUT / "source-audit.json"),
              "curation_sha256": z.digest(OUT / "curation.json"),
              "rejection_ledger_sha256": {
                  "train": z.digest(OUT / "train-semantic-rejections.jsonl"),
                  "eval": z.digest(OUT / "eval-semantic-rejections.jsonl")},
              "rejected_counts": {split: len(ids) for split, ids in excluded.items()},
              "review": "Assistant source audit with main-agent adjudication; source/evidence quality only; no new model outputs were inspected."}
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "approved", "rejected_counts": output["rejected_counts"]}))


if __name__ == "__main__":
    main()
