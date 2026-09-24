"""Prepare a deterministic independent holdout from a local official Hotpot dev file.

No downloads, model calls, implicit directory scans, or final-test reads occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

SOURCE_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_question(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in value).split())


def _safe_path(path: Path) -> Path:
    path = path.resolve()
    if any("finaltest" in re.sub(r"[^a-z0-9]", "", part.casefold()) for part in path.parts):
        raise ValueError("final-test paths are forbidden")
    return path


def _rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["context"] = list(zip(row["context"]["title"], row["context"]["sentences"]))
            row["supporting_facts"] = list(zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]))
        return rows
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text) if text.lstrip().startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]


def _identity(row: dict) -> tuple[str, str]:
    metadata = row.get("metadata", {})
    identifier = str(metadata.get("task_id", row.get("_id", row.get("id", ""))))
    question = row.get("question") or next((x["content"] for x in row.get("messages", []) if x["role"] == "user"), "")
    if not identifier or not question:
        raise ValueError("every source/exclusion record must have ID and question")
    return identifier.removeprefix("hotpot-"), normalized_question(question)


def _document(title: str, sentences: list[str]) -> dict:
    # Preserve title/text exactly; only exact title+text matches can be merged.
    # Same title with different text remains distinct, avoiding silent overwrites.
    text = " ".join(sentences)
    key = json.dumps([title, text], ensure_ascii=False, separators=(",", ":"))
    return {"id": "v110-doc-" + hashlib.sha256(key.encode()).hexdigest(), "title": title, "text": text}


def prepare(source: Path, exclusions: list[Path], output: Path, count: int = 100) -> dict:
    source = _safe_path(source)
    exclusions = [_safe_path(p) for p in exclusions]
    output = _safe_path(output)
    if count <= 0 or not exclusions:
        raise ValueError("positive count and explicit train/dev exclusion files are required")
    if any((output / name).exists() for name in ("holdout.jsonl", "tasks.jsonl", "corpus.json", "manifest.json")):
        raise FileExistsError("refusing to overwrite an existing holdout")
    used_ids, used_questions = set(), set()
    for path in exclusions:
        for row in _rows(path):
            identifier, question = _identity(row)
            used_ids.add(identifier)
            used_questions.add(question)
    rows = _rows(source)
    by_id = {}
    excluded = 0
    for row in rows:
        identifier, question = _identity(row)
        if identifier in used_ids or question in used_questions:
            excluded += 1
            continue
        if identifier in by_id and by_id[identifier] != row:
            raise ValueError(f"conflicting source ID: {identifier}")
        by_id[identifier] = row
    ordered = sorted(by_id, key=lambda x: (hashlib.sha256(("v110:" + x).encode()).hexdigest(), x))
    selected, seen_questions = [], set()
    for identifier in ordered:
        row = by_id[identifier]
        question = _identity(row)[1]
        if question not in seen_questions:
            selected.append(row)
            seen_questions.add(question)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} independent questions available; require {count}")
    corpus, records, tasks = {}, [], []
    for row in selected:
        identifier = _identity(row)[0]
        docs = [_document(title, sentences) for title, sentences in row["context"]]
        for doc in docs:
            corpus[doc["id"]] = doc
        metadata = {"task_id": "hotpot-" + identifier, "gold_answer": row["answer"],
                    "source": "HotpotQA/official-dev-distractor", "split": "v110_independent_holdout",
                    "support_titles": list(dict.fromkeys(title for title, _ in row["supporting_facts"])),
                    "corpus_policy": "all_context_including_distractors", "difficulty": row.get("level", "unknown")}
        records.append({"messages": [{"role": "user", "content": row["question"]},
                        {"role": "tool", "content": json.dumps({"ok": True, "result": docs}, ensure_ascii=False)}],
                        "metadata": metadata})
        tasks.append({"id": metadata["task_id"], "question": row["question"], "answer": row["answer"],
                      "metadata": {"source": metadata["source"], "support_titles": metadata["support_titles"]}})
    corpus_list = [corpus[k] for k in sorted(corpus)]
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("holdout.jsonl", records), ("tasks.jsonl", tasks)):
        (output / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in content), encoding="utf-8")
    (output / "corpus.json").write_text(json.dumps(corpus_list, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    manifest = {"version": "v110", "source_url": SOURCE_URL, "source_file": str(source),
                "source_sha256": sha256(source), "source_bytes": source.stat().st_size,
                "source_rows": len(rows), "excluded_rows": excluded,
                "mirror_url": "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet" if source.suffix == ".parquet" else None,
                "exclusion_files": [{"path": str(p), "sha256": sha256(p)} for p in exclusions],
                "selection": "sort SHA256('v110:'+raw_id), deduplicate normalized questions, first count",
                "question_normalization": "NFKC + casefold + non-alphanumeric to space + collapse whitespace",
                "count": count, "task_ids": [t["id"] for t in tasks], "corpus_document_count": len(corpus_list),
                "corpus_policy": "union of ALL selected question contexts; never filter by supporting facts",
                "deduplication": "exact title and joined text; distinct same-title variants retained",
                "files": {n: sha256(output / n) for n in ("holdout.jsonl", "tasks.jsonl", "corpus.json")},
                "final_test_accessed": False,
                "provenance_note": "Local source claimed to be official dev distractor; source hash recorded, no remote attestation."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True, help="Explicit known train/dev file; repeat as needed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.exclude, args.output, args.count), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
