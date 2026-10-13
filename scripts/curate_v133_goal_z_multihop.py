"""Audit Goal Z retrieval chains against frozen BM25-v2 corpora (CPU only)."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from mini_agent_rl.tools.search import SearchTool

import prepare_v133_goal_z_multihop_data as z


OUT = z.OUT


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def doc_id(split: str, title: str) -> str:
    return hashlib.sha256((split + "\0" + z.norm(title)).encode()).hexdigest()[:24]


def search(tool: SearchTool, query: str) -> list[dict]:
    return tool._search(query)


def teacher(case: dict, tool: SearchTool) -> tuple[dict | None, str | None]:
    supports = [doc for doc in case["docs"] if doc["support"]]
    if case["source"] == "musique":
        chain = case["chain"]
        first = next(d for d in supports if d["idx"] == chain[0]["paragraph_support_idx"])
        second = next(d for d in supports if d["idx"] == chain[1]["paragraph_support_idx"])
        bridge = str(chain[0]["answer"])
        q1 = chain[0]["question"].replace(">>", " ").strip()
        q2 = chain[1]["question"].replace("#1", bridge).replace(">>", " ").strip()
    else:
        first = next(d for d in supports if z.norm(d["title"]) == z.norm(case["chain"][0][0]))
        second = next(d for d in supports if z.norm(d["title"]) == z.norm(case["chain"][1][0]))
        bridge = str(case["chain"][0][2])
        q1 = str(case["chain"][0][0]) + " " + str(case["chain"][0][1]).replace("_", " ")
        q2 = bridge + " " + str(case["chain"][1][1]).replace("_", " ")
    if z.norm(bridge) in z.norm(q1) or z.norm(case["answer"]) in z.norm(q1):
        return None, "hidden_answer_in_first_query"
    first_result = search(tool, q1)
    if not any(z.norm(d["title"]) == z.norm(first["title"]) and z.norm(bridge) in z.norm(d["text"])
               for d in first_result):
        return None, "first_support_not_retrieved"
    if any(z.norm(d["title"]) == z.norm(second["title"]) for d in first_result):
        return None, "second_support_already_retrieved"
    if not any(z.norm(bridge) in z.norm(d["text"]) for d in first_result):
        return None, "bridge_not_visible"
    second_result = search(tool, q2)
    if not any(z.norm(d["title"]) == z.norm(second["title"]) and z.norm(case["answer"]) in z.norm(d["text"])
               for d in second_result):
        return None, "second_support_or_answer_not_retrieved"
    if z.norm(q1) == z.norm(q2):
        return None, "repeated_query"
    return {"queries": [q1, q2], "bridge": bridge,
            "bridge_visible_in_doc_ids": [d["id"] for d in first_result if z.norm(bridge) in z.norm(d["text"])],
            "retrieved": [first_result, second_result],
            "support_titles": [first["title"], second["title"]]}, None


def build_corpus(split: str, cases: list[dict], reserved: set[str], historical: set[str]) -> tuple[list[dict], set[str]]:
    # Support paragraphs come first; natural distractors use fixed hash order.
    support, other = [], []
    for case in cases:
        for doc in case["docs"]:
            (support if doc["support"] else other).append(doc)
    seen = set(reserved)
    corpus = []
    ordered = (sorted(support, key=lambda d: z.stable(split + ":support:" + d["title"])) +
               sorted(other, key=lambda d: z.stable(split + ":other:" + d["title"])))
    for doc in ordered:
        title = z.norm(doc["title"])
        if not title or title in seen or title in historical or not doc["text"].strip():
            continue
        if len(corpus) == 2048:
            break
        seen.add(title)
        corpus.append({"id": doc_id(split, doc["title"]), "title": doc["title"], "text": doc["text"]})
    if len(corpus) < 2048:
        # Draw extra natural paragraphs from the same official source split.
        # Hashing picks a bounded, deterministic set without model feedback.
        extra = []
        for source, loader in (("musique", z.load_musique), ("2wiki", z.load_wiki)):
            raw = loader(split)
            if split != "train":
                side = 0 if split == "dev" else 1
                raw = [v for v in raw if int(z.stable(source + ":" + str(v.get("id", v.get("_id"))))[-1], 16) % 2 == side]
            for case in raw[:2000]:
                if source == "musique":
                    extra.extend({"title": p["title"], "text": p["paragraph_text"]} for p in case["paragraphs"])
                else:
                    extra.extend({"title": title, "text": " ".join(sentences)} for title, sentences in case["context"])
        for doc in sorted(extra, key=lambda d: z.stable(split + ":extra:" + d["title"])):
            title = z.norm(doc["title"])
            if not title or title in seen or title in historical or not doc["text"].strip():
                continue
            seen.add(title)
            corpus.append({"id": doc_id(split, doc["title"]), "title": doc["title"], "text": doc["text"]})
            if len(corpus) == 2048:
                break
    if len(corpus) != 2048:
        raise RuntimeError(f"corpus_shortfall:{split}:{len(corpus)}")
    return corpus, seen - reserved


def main() -> None:
    if (OUT / "curation.json").exists():
        raise FileExistsError(OUT / "curation.json")
    history = z.w.history_exposure()
    historical_titles = history[2]
    source_audit = json.loads((OUT / "source-audit.json").read_text(encoding="utf-8"))
    if source_audit["source_quality_shortfalls"]:
        raise RuntimeError("Source-quality gate failed")
    reasons = {}
    counts = {}
    reserved = set()
    selected_by_split = {}
    for split in ("train", "dev", "confirm"):
        sources = ["musique", "2wiki"]
        cases = [case for source in sources for case in rows(OUT / f"candidates-{split}-{source}.jsonl")
                 if z.audit_case(case, (history[0], history[1], history[2])) is None]
        corpus, titles = build_corpus(split, cases, reserved, historical_titles)
        reserved.update(titles)
        save(OUT / f"corpus-{split}.json", corpus)
        tool = SearchTool(corpus, "bm25-v2")
        selected = []
        split_reasons = Counter()
        # Dev and confirmation use only source/evidence checks; train additionally
        # requires an actually executable two-search demonstration.
        for source in sources:
            got = 0
            for case in cases:
                if case["source"] != source:
                    continue
                support_titles = {z.norm(d["title"]) for d in case["docs"] if d["support"]}
                if not support_titles.issubset(titles):
                    split_reasons["support_missing_from_corpus"] += 1
                    continue
                if split == "train":
                    path, reason = teacher(case, tool)
                    if reason:
                        split_reasons[reason] += 1
                        continue
                    case = {k: v for k, v in case.items() if k != "docs"} | {"teacher": path}
                else:
                    case = {k: v for k, v in case.items() if k != "docs"} | {
                        "support_titles": [d["title"] for d in case["docs"] if d["support"]]}
                selected.append(case)
                got += 1
                if got == z.TARGETS[split][source]:
                    break
            counts[f"{split}-{source}"] = got
        if split == "train":
            old = rows(OUT / "candidates-train-old.jsonl")[:z.TARGETS[split]["old"]]
            selected.extend(old)
            counts["train-old"] = len(old)
        selected_by_split[split] = selected
        filename = "selected-confirm.jsonl" if split == "confirm" else f"selected-{split}.jsonl"
        z.jsonl(OUT / filename, selected)
        reasons[split] = dict(split_reasons)
        print(json.dumps({"split": split, "counts": {k: v for k, v in counts.items() if k.startswith(split)},
                          "reasons": reasons[split]}), flush=True)
    short = {k: z.TARGETS[k.split("-")[0]][k.split("-")[1]] - v for k, v in counts.items()
             if v < z.TARGETS[k.split("-")[0]][k.split("-")[1]]}
    record = {"version": "goal-z-curation-v1", "counts": counts, "shortfalls": short,
              "reasons": reasons, "source_audit_sha256": z.digest(OUT / "source-audit.json"),
              "corpus_sha256": {split: z.digest(OUT / f"corpus-{split}.json")
                                for split in ("train", "dev", "confirm")},
              "selected_sha256": {split: z.digest(OUT / ("selected-confirm.jsonl" if split == "confirm"
                                                  else f"selected-{split}.jsonl"))
                                  for split in ("train", "dev", "confirm")}}
    save(OUT / "curation.json", record)
    if short:
        print(json.dumps({"goal_z_data_gate": "failed", "shortfalls": short}), flush=True)
    else:
        print(json.dumps({"goal_z_data_gate": "passed"}), flush=True)


if __name__ == "__main__":
    main()
