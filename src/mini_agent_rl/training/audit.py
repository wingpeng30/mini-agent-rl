"""SFT JSONL 的只读质量审计。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from .hotpot import _file_sha256, _jsonl


def audit_sft_directory(data_dir: str | Path) -> dict[str, Any]:
    """统计样本、搜索步数、重复问题及 supporting-document 跨 split 泄漏。"""
    root = Path(data_dir)
    summaries: dict[str, Any] = {}
    titles_by_split: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        questions = [item["messages"][1]["content"] for item in records]
        search_counts = [item.get("metadata", {}).get("search_count", 0) for item in records]
        titles = {title for item in records for title in item.get("metadata", {}).get("support_titles", [])}
        titles_by_split[split] = titles
        summaries[split] = {"samples": len(records), "unique_questions": len(set(questions)), "duplicate_questions": len(questions) - len(set(questions)), "search_count_distribution": dict(Counter(search_counts)), "supporting_documents": len(titles), "mean_messages": round(sum(len(item["messages"]) for item in records) / max(len(records), 1), 2)}
    overlaps = {"train_validation": len(titles_by_split["train"] & titles_by_split["validation"]), "train_test": len(titles_by_split["train"] & titles_by_split["test"]), "validation_test": len(titles_by_split["validation"] & titles_by_split["test"])}
    return {"data_dir": str(root), "splits": summaries, "document_overlap": overlaps, "passed_document_isolation": not any(overlaps.values())}


def audit_final_test(source_dir: str | Path, final_dir: str | Path) -> dict[str, Any]:
    """验证最终测试集与已用开发数据的 task/document 隔离及 manifest 哈希。"""
    source, final = Path(source_dir), Path(final_dir)
    source_records = [item for name in ("train", "validation", "test") for item in _jsonl(source / f"{name}.jsonl")]
    final_path = final / "final-test.jsonl"
    final_records = _jsonl(final_path)
    source_ids = {item["metadata"]["task_id"] for item in source_records}
    final_ids = [item["metadata"]["task_id"] for item in final_records]
    source_titles = {title for item in source_records for title in item["metadata"].get("support_titles", [])}
    final_titles = {title for item in final_records for title in item["metadata"].get("support_titles", [])}
    manifest_path = final / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    actual_hash = _file_sha256(final_path)
    return {"source_dir": str(source), "final_dir": str(final), "final_count": len(final_records),
            "duplicate_final_task_ids": len(final_ids) - len(set(final_ids)), "task_id_overlap": len(source_ids & set(final_ids)),
            "supporting_document_overlap": len(source_titles & final_titles), "final_sha256": actual_hash,
            "manifest_sha256_matches": manifest.get("final_test", {}).get("sha256") == actual_hash,
            "passed": not (len(final_ids) - len(set(final_ids))) and not (source_ids & set(final_ids)) and not (source_titles & final_titles) and manifest.get("final_test", {}).get("sha256") == actual_hash}
