"""Build a fixed, performance-blind internal candidate corpus from train only."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from scripts.build_v114_sft_candidates import build_index, norm, public_example, support_facts
from scripts.prepare_v116_data import SOURCE, ROOT, exposed, full_rows, paired_examples, sha, write, writel

OUT = ROOT / "data/evidence-sft-v117"
SEALED = ("final-test", "final_test", "independent", "holdout")


def history():
    ids, questions, scanned, failures, sealed = set(), set(), [], [], []
    prior = json.loads((ROOT / "data/evidence-sft-v116/history-inventory.json").read_text(encoding="utf-8"))
    if not prior.get("ready") or prior.get("failures"):
        raise RuntimeError("v116_history_baseline_invalid")
    prior_private = json.loads((ROOT / "data/evidence-sft-v116/history-private.json").read_text(encoding="utf-8"))
    ids.update(prior_private["ids"])
    questions.update(prior_private["normalized_questions"])
    excluded_scratch = []
    for base in (ROOT / "data", ROOT / "reports", ROOT / "runs", ROOT / "backups"):
        if not base.exists():
            failures.append({"path": str(base), "reason": "missing_root"})
            continue
        def onerror(error):
            failures.append({"path": str(error.filename), "reason": type(error).__name__})
        for directory, subdirs, files in os.walk(base, onerror=onerror):
            path = Path(directory)
            rel = path.relative_to(ROOT)
            if any(any(word in part.lower() for word in SEALED) for part in rel.parts):
                subdirs[:] = []
                sealed.append(str(rel).replace("\\", "/"))
                continue
            retained = []
            for name in subdirs:
                lower = name.lower()
                if any(word in lower for word in SEALED):
                    sealed.append(str(rel / name).replace("\\", "/"))
                elif lower.startswith(("pytest", ".pytest", ".test", "test-", "v113-stage-")):
                    excluded_scratch.append(str(rel / name).replace("\\", "/"))
                else:
                    retained.append(name)
            subdirs[:] = retained
            for name in files:
                file = path / name
                if file.suffix not in {".json", ".jsonl"} or SOURCE in file.parents or OUT in file.parents:
                    continue
                try:
                    if file.suffix == ".jsonl":
                        with file.open(encoding="utf-8") as stream:
                            for line_no, line in enumerate(stream, 1):
                                if line.strip():
                                    try:
                                        exposed(json.loads(line), ids, questions)
                                    except ValueError as error:
                                        raise ValueError(f"line {line_no}: {error}") from error
                    else:
                        exposed(json.loads(file.read_text(encoding="utf-8")), ids, questions)
                    scanned.append({"path": str(file.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(file)})
                except (OSError, UnicodeError, ValueError) as error:
                    failures.append({"path": str(file.relative_to(ROOT)), "reason": str(error)[:180]})
    for filename in ("training-question-freeze-v2.json", "internal-freeze-v3.json"):
        file = ROOT / "data/evidence-sft-v114-review" / filename
        if file.exists():
            ids.update(json.loads(file.read_text(encoding="utf-8")).get("ids", []))
        else:
            failures.append({"path": str(file), "reason": "missing_required_freeze"})
    report = {"version": "v117-history-v1", "ids": len(ids), "normalized_questions": len(questions),
              "files": scanned, "sealed_paths_skipped": sorted(set(sealed)),
              "excluded_scratch_paths": sorted(set(excluded_scratch)),
              "v116_history_baseline_sha256": sha(ROOT / "data/evidence-sft-v116/history-inventory.json"),
              "sealed_content_read": False, "failures": failures, "ready": not failures}
    write(OUT / "history-inventory.json", report)
    write(OUT / "history-private.json", {"ids": sorted(ids), "normalized_questions": sorted(questions)})
    return ids, questions, report


def wording_issue(question):
    q = question.strip()
    if len(q) < 24 or not q.endswith("?") or q.count("?") != 1:
        return "malformed_question"
    if re.search(r"\b(this|that|these|those|it|they|he|she|his|her|their)\b", q, re.I):
        # These forms often have a valid referent; do not silently accept them.
        return "unresolved_pronoun_screen"
    if re.search(r"\b(current|currently|now|today|latest|newest)\b", q, re.I):
        return "unanchored_time_screen"
    return None


def provisional_risks(question):
    q = question.casefold()
    tags = []
    if re.search(r"\b(who|whose|wrote|directed|founded|member|played|parent|spouse|married|born in|sister|brother)\b", q):
        tags.append("relation")
    if re.search(r"\b(year|when|date|century|before|after|during|older|younger|first released|formed in)\b", q):
        tags.append("time")
    if re.search(r"\b(how many|how much|long|length|height|population|area|duration|kilomet|miles?|meters?|years old|percent)\b", q):
        tags.append("measurement")
    if re.search(r"\b(both|among|between|only|all|same|largest|smallest|first|last|in which country|in what country|which state|which city|which county)\b", q):
        tags.append("scope")
    return tags


def source_order(history_ids, history_questions):
    provenance = json.loads((SOURCE / "download-provenance.json").read_text(encoding="utf-8"))
    for item in provenance["files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("source_shard_hash_mismatch")
    seen = set()
    metadata = []
    for file in sorted(SOURCE.glob("train-*-of-*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(columns=["id", "question"], batch_size=4096):
            for item in batch.to_pylist():
                key = norm(item["question"])
                if item["id"] in history_ids or key in history_questions or key in seen:
                    continue
                seen.add(key)
                metadata.append(item)
    metadata.sort(key=lambda r: hashlib.sha256(("v117-internal:" + r["id"]).encode()).hexdigest())
    return metadata[:2000]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids, questions, inventory = history()
    if not inventory["ready"]:
        raise RuntimeError("historical_exposure_scan_incomplete")
    metadata = source_order(ids, questions)
    rows = full_rows([r["id"] for r in metadata])
    chosen, rejects = [], []
    for item in metadata:
        row = rows[item["id"]]
        issue = wording_issue(row["question"])
        if issue:
            rejects.append({"source_id": row["id"], "reason": issue})
            continue
        if len(support_facts(row)) < 2:
            rejects.append({"source_id": row["id"], "reason": "insufficient_annotated_facts"})
            continue
        _, solo = build_index([row])
        if paired_examples(row, solo) is None:
            rejects.append({"source_id": row["id"], "reason": "no_structural_pair_in_source"})
            continue
        chosen.append(row)
        if len(chosen) == 200:
            break
    write(OUT / "preselection-rejections.json", {"screened": len(chosen) + len(rejects), "rejections": rejects,
          "counts": dict(Counter(r["reason"] for r in rejects))})
    if len(chosen) < 200:
        write(OUT / "candidate-gap.json", {"needed": 200, "found": len(chosen), "scanned_limit": 2000})
        raise RuntimeError("insufficient_200_candidates")
    corpus, index = build_index(chosen)
    corpus_path = OUT / "candidate-corpus-private.json"
    write(corpus_path, corpus)
    write(OUT / "candidate-freeze.json", {"version": "v117-candidate-corpus-v1", "ids": [r["id"] for r in chosen],
          "source_count": len(chosen), "documents": len(corpus), "corpus_sha256": sha(corpus_path),
          "retriever": "BM25", "top_k": 3, "source": "official_hotpotqa_distractor_train",
          "selection": "SHA256(v117-internal:id), first 200 wording-and-structure-screened within 2000",
          "model_scores_used": False})
    candidates, no_pair = [], []
    for row in chosen:
        pair = paired_examples(row, index)
        if pair is None:
            no_pair.append(row["id"])
            continue
        for state in pair:
            candidates.append({"source_id": row["id"], "split": "internal",
                               "source": "programmatically_constructed_official_train",
                               "state_kind": state["state_kind"],
                               "risk_tags_provisional": provisional_risks(row["question"]),
                               "packet": public_example(state, 0)})
    writel(OUT / "candidate-private.jsonl", candidates)
    write(OUT / "candidate-construction.json", {"candidate_questions": 200, "pairable_questions": len(candidates)//2,
          "unpairable_ids": no_pair, "corpus_sha256": sha(corpus_path), "pre_review_only": True})
    print(json.dumps({"history_ids": len(ids), "history_questions": len(questions),
                      "preselection_rejected": len(rejects), "fixed_corpus_docs": len(corpus),
                      "pairable_questions": len(candidates)//2}), flush=True)


if __name__ == "__main__":
    main()
