"""Prepare bounded v116 train supplements and new internal paired-state candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from scripts.build_v114_sft_candidates import (build_examples, build_index, documents_for,
                                               norm, public_example, risk_tags)

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/evidence-sft-v114-review"
SOURCE = ROOT / "data/hotpot-full-train-v114"
OUT = ROOT / "data/evidence-sft-v116"
FORBIDDEN = ("final-test", "final_test", "independent", "holdout")
QUALITY_EXCLUSIONS = {
    "5ae7d7a35542993210983f84": "Multiple named department-store chains satisfy the clue.",
    "5ac2c0c8554299218029db14": "Unanchored current role and company size.",
    "5abaa02355429955dce3eea5": "The phrase this county lacks an unambiguous referent.",
    "5ae283e8554299495565da94": "The 21,000-mile Florida premise conflates total network and state scope.",
    "5a7b7bae5542997c3ec971bb": "Question asks both a song name and a number of video albums.",
    "5ae1d77d554299422ee9969f": "Stadium question has no time or team anchor.",
    "5ae63e8a55429908198fa53f": "Incomplete below-Major-League question.",
    "5ae2cc485542996483e64a69": "Incomplete birth-date question.",
    "5a7b36555542992d025e679b": "Flamboyant partner is not a specific relation.",
    "5ac0ba5c5542992a796ded73": "Found Hispaniola and unknown continent are ambiguous premises.",
    "5ac020275542996f0d89cb2a": "Does not identify which Christopher Lloyd film.",
    "5adfaf5855429942ec259af1": "Truncated question ending in slash.",
    "5ab6eefa5542991d322236d4": "Ambiguous voice actor and star referent.",
    "5ab3839455429969a97a816d": "Newest single and chart ranking have no date or chart anchor.",
    "5ac51a0555429924173fb5ac": "You Belong To Me names multiple songs and works without disambiguation.",
    "5abc5c9b55429959677d6a94": "Malformed band-formation and debut-album timing question.",
    "5ab52dbf55429942dd416010": "Three animals and a Hall of Fame manager do not identify a unique movie.",
    "5ae2491b554299492dc91c11": "Seventh episode before is ambiguous as season or relative episode count.",
    "5a8430505542996488c2e501": "Bob Brown and newspaper lack a unique identifying clue.",
}


def loadl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def writel(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False, separators=(",", ":")) + "\n" for v in values), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exposed(value, ids, questions):
    if isinstance(value, dict):
        question = value.get("question")
        if isinstance(question, str) and norm(question):
            questions.add(norm(question))
        for key, item in value.items():
            if key in {"qid", "source_id", "task_id", "id"} and isinstance(item, str):
                raw = item.removeprefix("hotpot-").removeprefix("hotpot:").lower()
                if len(raw) == 24 and all(ch in "0123456789abcdef" for ch in raw):
                    ids.add(raw)
            exposed(item, ids, questions)
    elif isinstance(value, list):
        for item in value:
            exposed(item, ids, questions)


def historical_index():
    ids, questions, scanned, failures, exclusions = set(), set(), [], [], []
    for root in (ROOT / "data", ROOT / "reports", ROOT / "runs", ROOT / "backups"):
        if not root.exists():
            failures.append({"path": str(root.relative_to(ROOT)), "reason": "missing_root"})
            continue
        def walk_error(exc):
            failures.append({"path": str(exc.filename), "reason": type(exc).__name__})
        for directory, subdirs, filenames in os.walk(root, onerror=walk_error):
            relative = Path(directory).relative_to(ROOT)
            if any(any(x in part.lower() for x in FORBIDDEN) for part in relative.parts):
                subdirs[:] = []
                exclusions.append(str(relative).replace("\\", "/"))
                continue
            subdirs[:] = [d for d in subdirs if not any(x in d.lower() for x in FORBIDDEN)
                           and not d.lower().startswith(("pytest", ".pytest", ".test", "test-"))]
            for filename in filenames:
                if not filename.endswith((".json", ".jsonl")):
                    continue
                path = Path(directory) / filename
                if SOURCE in path.parents or WORK_PATH(path):
                    continue  # The source partition itself is not prior exposure.
                try:
                    if path.suffix == ".jsonl":
                        with path.open(encoding="utf-8") as stream:
                            for line_no, line in enumerate(stream, 1):
                                if line.strip():
                                    try:
                                        exposed(json.loads(line), ids, questions)
                                    except ValueError as exc:
                                        raise ValueError(f"line_{line_no}:{exc}") from exc
                    else:
                        exposed(json.loads(path.read_text(encoding="utf-8")), ids, questions)
                    scanned.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path)})
                except (OSError, ValueError, UnicodeError) as exc:
                    failures.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                     "reason": type(exc).__name__ + ":" + str(exc)[:120]})
    # Existing frozen question lists are exposure even if their rows have no question field.
    for filename in ("training-question-freeze-v2.json", "internal-freeze-v3.json"):
        path = OLD / filename
        if path.exists():
            ids.update(json.loads(path.read_text(encoding="utf-8")).get("ids", []))
        else:
            failures.append({"path": str(path), "reason": "missing_required_freeze"})
    report = {"version": "v116-historical-exposure-v1", "ids": len(ids), "normalized_questions": len(questions),
              "files": scanned, "excluded_sealed_paths": sorted(set(exclusions)), "failures": failures,
              "ready": not failures, "sealed_content_read": False}
    write(OUT / "history-inventory.json", report)
    write(OUT / "history-private.json", {"ids": sorted(ids), "normalized_questions": sorted(questions)})
    return ids, questions, report


def WORK_PATH(path: Path) -> bool:
    """Current v116 outputs cannot become their own historical exposure on resume."""
    return (OUT in path.parents or path.name.startswith("v116-"))


def train_supplements():
    approved = [r for r in loadl(OLD / "accepted.jsonl") if r["split"] == "train"]
    by_source = defaultdict(set)
    for row in approved:
        kind = row["packet"]["proposed_action"]["action"]
        by_source[row["source_id"]].add(kind)
    private = {r["case_id"]: r for r in loadl(OLD / "reserve-private.jsonl") if r["split"] == "train"}
    packets = {r["case_id"]: r for r in loadl(OLD / "reserve-a-packets.jsonl")}
    candidates = []
    for case_id, meta in private.items():
        present = by_source.get(meta["source_id"])
        needed = "answer" if present == {"search"} else "search" if present == {"answer"} else None
        if not needed or case_id not in packets:
            continue
        packet = packets[case_id]
        if packet.get("preflight_error") or packet["proposed_action"].get("action") != needed:
            continue
        if needed == "search" and not packet["visible_evidence"]:
            continue
        candidates.append({"case_id": case_id, "source_id": meta["source_id"], "split": "train",
                           "packet": packet, "source": "programmatically_constructed_reserve"})
    candidates.sort(key=lambda r: (hashlib.sha256(("v116-train:" + r["source_id"]).encode()).hexdigest(),
                                   r["case_id"]))
    selected, used = [], set()
    for item in candidates:
        if item["source_id"] in used:
            continue
        selected.append(item)
        used.add(item["source_id"])
        if len(selected) == 60:
            break
    if len(selected) < 30:
        raise RuntimeError("insufficient_train_pair_reserve")
    writel(OUT / "train-supplement-private.jsonl", selected)
    return approved, selected


def full_rows(ordered_ids):
    wanted = set(ordered_ids)
    found = {}
    for path in sorted(SOURCE.glob("train-*-of-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if row["id"] in wanted:
                    found[row["id"]] = row
    if len(found) != len(wanted):
        raise RuntimeError("missing_selected_source_rows")
    return found


def paired_examples(row, index):
    examples = build_examples([row], index)
    answers = [e for e in examples if e["state_kind"] == "supported_answer"]
    searches = [e for e in examples if e["state_kind"] == "nonempty_evidence_search"]
    for answer in sorted(answers, key=lambda e: len(e["messages"])):
        matching = [e for e in searches if answer["messages"][:len(e["messages"])] == e["messages"]]
        if matching:
            return matching[-1], answer
    return None


def internal_candidates(history_ids, history_questions):
    review_quality_path = OUT / "internal-review-quality-exclusions.json"
    review_quality = json.loads(review_quality_path.read_text(encoding="utf-8")) if review_quality_path.exists() else {}
    if not isinstance(review_quality, dict) or any(not isinstance(reason, str) or not reason.strip()
                                                   for reason in review_quality.values()):
        raise ValueError("invalid_review_quality_exclusions")
    excluded_source_ids = QUALITY_EXCLUSIONS.keys() | review_quality.keys()
    provenance = json.loads((SOURCE / "download-provenance.json").read_text(encoding="utf-8"))
    for entry in provenance["files"]:
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError("source_shard_hash_mismatch")
    metadata = []
    seen_questions = set()
    for path in sorted(SOURCE.glob("train-*-of-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=["id", "question"], batch_size=4096):
            for row in batch.to_pylist():
                normalized = norm(row["question"])
                if (row["id"] in history_ids or row["id"] in excluded_source_ids
                        or normalized in history_questions or normalized in seen_questions):
                    continue
                seen_questions.add(normalized)
                metadata.append((row["id"], row["question"]))
    metadata.sort(key=lambda r: hashlib.sha256(("v116-internal:" + r[0]).encode()).hexdigest())
    # Preselect a bounded, performance-blind source pool. No model output is read.
    source = full_rows([qid for qid, _ in metadata[:3000]])
    eligible = []
    for qid, _ in metadata[:3000]:
        row = source[qid]
        _, index = build_index([row])
        if paired_examples(row, index):
            eligible.append(row)
        if len(eligible) >= 400 and sum("measurement" in risk_tags(r["question"]) for r in eligible) >= 15:
            break
    if len(eligible) < 50:
        raise RuntimeError("insufficient_internal_source_pairs")
    measurement = [r for r in eligible if "measurement" in risk_tags(r["question"])]
    if len(measurement) < 5:
        raise RuntimeError("insufficient_internal_measurement_questions")
    priority = measurement[:15]
    selected = priority + [r for r in eligible if r["id"] not in {x["id"] for x in priority}][:50 - len(priority)]
    reserve = [r for r in eligible if r["id"] not in {x["id"] for x in selected}]
    cursor = 0
    for replacement in range(60):
        corpus, index = build_index(selected)
        pairs = {row["id"]: paired_examples(row, index) for row in selected}
        missing = [qid for qid, pair in pairs.items() if pair is None]
        print(json.dumps({"internal_round": replacement, "missing_pairs": len(missing),
                          "replacement_cursor": cursor, "eligible": len(eligible)}), flush=True)
        if not missing:
            break
        if len(reserve) < cursor + len(missing):
            raise RuntimeError("internal_replacement_pool_exhausted")
        selected = [r for r in selected if r["id"] not in missing] + reserve[cursor:cursor + len(missing)]
        cursor += len(missing)
    else:
        raise RuntimeError("internal_corpus_pair_convergence_failed")
    if len(selected) != 50 or any(pairs[r["id"]] is None for r in selected):
        raise RuntimeError("internal_pair_coverage_failed")
    examples = []
    for row in selected:
        for example in pairs[row["id"]]:
            examples.append({"source_id": row["id"], "split": "internal",
                             "source": "programmatically_constructed_official_train",
                             "state_kind": example["state_kind"], "risk_tags_provisional": risk_tags(row["question"]),
                             "packet": public_example(example, 0)})
    writel(OUT / "internal-candidate-private.jsonl", examples)
    write(OUT / "internal-source-freeze.json", {"version": "v116-internal-source-v1",
          "selection": "SHA256(v116-internal:source_id); first quality-eligible paired questions, no model scoring",
          "ids": [r["id"] for r in selected], "questions": len(selected), "candidate_states": len(examples),
          "corpus_documents": len(corpus), "source_partition": "HotpotQA distractor train"})
    write(OUT / "internal-question-quality-exclusions.json", {
        "basis": "Question wording and referent quality only; no model results",
        "source_ids_and_reasons": QUALITY_EXCLUSIONS})
    write(OUT / "internal-corpus-private.json", corpus)
    return examples


def main():
    p = argparse.ArgumentParser();p.add_argument("--history-only", action="store_true");a = p.parse_args()
    ids, questions, report = historical_index()
    print(json.dumps({"history_ready": report["ready"], "failures": len(report["failures"]),
                      "ids": len(ids), "questions": len(questions)}, ensure_ascii=False), flush=True)
    if not report["ready"]:
        raise RuntimeError("historical_exposure_scan_incomplete")
    if a.history_only:
        return
    approved, supplements = train_supplements()
    internal = internal_candidates(ids, questions)
    print(json.dumps({"approved_train": len(approved), "train_supplements": len(supplements),
                      "internal_candidate_states": len(internal)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
