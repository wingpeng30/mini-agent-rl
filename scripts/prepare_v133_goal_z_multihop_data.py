"""Freeze official Goal Z candidate pools and audit two-step evidence chains.

This CPU-only preparation never opens the old sealed evaluation set. Source
archives are read in place and are not extracted or rewritten.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

import prepare_v133_goal_w_data as w


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-z-multihop-sft"
SOURCES = ROOT.parent / "goal-z-sources"
MUSIQUE = SOURCES / "musique_v1.0.zip"
WIKI = SOURCES / "2wiki_data_ids_april7.zip"
LIMITS = {"train": {"musique": 256, "2wiki": 128, "old": 128},
          "dev": {"musique": 128, "2wiki": 128},
          "confirm": {"musique": 128, "2wiki": 128}}
TARGETS = {"train": {"musique": 64, "2wiki": 32, "old": 32},
           "dev": {"musique": 32, "2wiki": 32},
           "confirm": {"musique": 32, "2wiki": 32}}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKD", str(value))
                  .encode("ascii", "ignore").decode("ascii").casefold()).strip()


def stable(value: str) -> str:
    return hashlib.sha256(("goal-z-v1\0" + value).encode()).hexdigest()


def jsonl(path: Path, records) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_musique(split: str) -> list[dict]:
    member = f"data/musique_ans_v1.0_{'train' if split == 'train' else 'dev'}.jsonl"
    with zipfile.ZipFile(MUSIQUE) as archive, archive.open(member) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_wiki(split: str) -> list[dict]:
    with zipfile.ZipFile(WIKI) as archive, archive.open("train.json" if split == "train" else "dev.json") as stream:
        return json.load(stream)


def canonical(source: str, row: dict) -> dict:
    if source == "musique":
        docs = [{"title": p["title"], "text": p["paragraph_text"],
                 "support": bool(p["is_supporting"]), "idx": p["idx"]}
                for p in row["paragraphs"]]
        chain = row["question_decomposition"]
        return {"id": "musique:" + row["id"], "source": source, "question": row["question"],
                "answer": row["answer"], "answer_aliases": row.get("answer_aliases", []),
                "source_ids": [str(x["id"]) for x in chain],
                "chain": chain, "docs": docs, "kind": "2hop"}
    docs = [{"title": title, "text": " ".join(sentences),
             "support": title in {fact[0] for fact in row["supporting_facts"]}}
            for title, sentences in row["context"]]
    return {"id": "2wiki:" + row["_id"], "source": source,
            "question": row["question"], "answer": row["answer"],
            "answer_aliases": [], "source_ids": [row["_id"]],
            "chain": row.get("evidences", []), "docs": docs, "kind": row["type"],
            "supporting_facts": row["supporting_facts"]}


def source_eligible(source: str, row: dict) -> bool:
    if source == "musique":
        return str(row.get("id", "")).startswith("2hop__") and row.get("answerable") is True
    return row.get("type") == "compositional"


def audit_case(case: dict, history: tuple[set, set, set]) -> str | None:
    ids, questions, titles = history
    if any(x in ids for x in case["source_ids"]) or w.qhash(case["question"]) in questions:
        return "historical_question_or_source_id"
    supports = [doc for doc in case["docs"] if doc["support"]]
    if len(supports) != 2 or len({norm(x["title"]) for x in supports}) != 2:
        return "not_two_support_documents"
    if any(norm(x["title"]) in titles for x in supports):
        return "historical_support_title"
    if not case["answer"] or not case["question"] or any(not d["text"].strip() for d in supports):
        return "missing_answer_or_evidence"
    if any(len(d["text"]) > 2200 for d in supports):
        return "long_support_paragraph"
    if case["source"] == "musique":
        chain = case["chain"]
        if len(chain) != 2 or str(chain[0]["answer"]).casefold() not in supports[0]["text"].casefold() + supports[1]["text"].casefold():
            return "broken_composition"
        first = next((d for d in case["docs"] if d.get("idx") == chain[0]["paragraph_support_idx"]), None)
        second = next((d for d in case["docs"] if d.get("idx") == chain[1]["paragraph_support_idx"]), None)
        if not first or not second or not first["support"] or not second["support"]:
            return "support_mapping_mismatch"
        bridge = str(chain[0]["answer"])
        if norm(bridge) in norm(case["question"]):
            return "bridge_given_in_question"
        if norm(bridge) not in norm(first["text"]) or norm(bridge) not in norm(second["title"] + " " + second["text"]):
            return "bridge_not_in_both_paragraphs"
    else:
        chain = case["chain"]
        if len(chain) != 2 or norm(chain[0][2]) != norm(chain[1][0]):
            return "broken_composition"
        bridge = str(chain[0][2])
        if norm(bridge) in norm(case["question"]):
            return "bridge_given_in_question"
        first = next((d for d in supports if norm(d["title"]) == norm(chain[0][0])), None)
        second = next((d for d in supports if norm(d["title"]) == norm(chain[1][0])), None)
        if not first or not second:
            return "support_mapping_mismatch"
        if norm(bridge) not in norm(first["text"]) or norm(case["answer"]) not in norm(second["text"]):
            return "bridge_or_answer_not_in_evidence"
    return None


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    for path in (MUSIQUE, WIKI):
        if not path.is_file():
            raise FileNotFoundError(path)
    history = w.history_exposure()
    historical = (history[0], history[1], history[2])
    OUT.mkdir(parents=True)
    rejection = []
    accepted = {}
    pool_manifest = {}
    for split in ("train", "dev", "confirm"):
        accepted[split] = {}
        for source, loader in (("musique", load_musique), ("2wiki", load_wiki)):
            raw = loader(split)
            if split != "train":
                # Official dev is partitioned before any model output is read.
                side = 0 if split == "dev" else 1
                raw = [v for v in raw if int(stable(source + ":" + str(v.get("id", v.get("_id"))))[-1], 16) % 2 == side]
            ranked = sorted((v for v in raw if source_eligible(source, v)),
                            key=lambda v: stable(source + ":" + str(v.get("id", v.get("_id")))))
            pool = [canonical(source, row) for row in ranked[:LIMITS[split][source]]]
            if len(pool) != LIMITS[split][source]:
                raise RuntimeError(f"candidate_pool_shortfall:{split}:{source}:{len(pool)}")
            jsonl(OUT / f"candidates-{split}-{source}.jsonl", pool)
            approved = []
            for case in pool:
                reason = audit_case(case, historical)
                if reason:
                    rejection.append({"split": split, "source": source, "id": case["id"], "reason": reason})
                else:
                    approved.append(case)
            accepted[split][source] = approved
            pool_manifest[f"{split}-{source}"] = {"candidates": len(pool), "source_quality_pass": len(approved),
                "file_sha256": digest(OUT / f"candidates-{split}-{source}.jsonl")}
            print(json.dumps({"split": split, "source": source, "pool": len(pool),
                              "quality_pass": len(approved)}), flush=True)
        if split == "train":
            old = []
            with (ROOT / "runs/v133-goal-x-audit-and-transfer/train-real-search.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["both_support_full_text_visible"] and row["reason"] == "answer":
                        old.append({"id": "old:" + row["case_id"], "source": "old", "question": row["question"],
                                    "answer": row["official_answer"], "source_ids": [row["source_id"]],
                                    "support_ids": row["support_ids"], "trace": row["trace"],
                                    "answer_prompt_token_ids": row["answer_prompt_token_ids"],
                                    "answer_prompt_attention_mask": row["answer_prompt_attention_mask"]})
            old.sort(key=lambda v: stable(v["id"]))
            pool = old[:LIMITS[split]["old"]]
            if len(pool) != LIMITS[split]["old"]:
                raise RuntimeError("old_candidate_pool_shortfall")
            jsonl(OUT / "candidates-train-old.jsonl", pool)
            accepted[split]["old"] = pool
            pool_manifest["train-old"] = {"candidates": len(pool), "source_quality_pass": len(pool),
                "file_sha256": digest(OUT / "candidates-train-old.jsonl")}
    jsonl(OUT / "source-audit-rejections.jsonl", rejection)
    report = {"version": "goal-z-source-pools-v1", "source_archives": {
        "musique": digest(MUSIQUE), "2wiki": digest(WIKI)},
        "history_counts": [len(v) for v in historical], "pools": pool_manifest,
        "rejection_counts": dict(Counter(v["reason"] for v in rejection)),
        "targets": TARGETS, "source_quality_shortfalls": {
            f"{split}-{source}": TARGETS[split][source] - len(accepted[split][source])
            for split in TARGETS for source in TARGETS[split]
            if len(accepted[split][source]) < TARGETS[split][source]}}
    (OUT / "source-audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"source_quality_shortfalls": report["source_quality_shortfalls"],
                      "rejection_counts": report["rejection_counts"]}), flush=True)


if __name__ == "__main__":
    main()
