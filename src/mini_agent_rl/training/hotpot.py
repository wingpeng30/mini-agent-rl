"""HotpotQA 小规模流式导入：将多跳问答转成项目的离线 Agent 数据。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM = "你是离线检索 Agent。只能输出 JSON：search 需要 query，answer 需要 answer。"


def _split_for_titles(titles: list[str]) -> str | None:
    """按支持文档标题分配 split；只有全部支持文档同属一组才接受，防止文档泄漏。"""
    groups = []
    for title in titles:
        bucket = int(hashlib.sha256(title.encode("utf-8")).hexdigest(), 16) % 100
        groups.append("train" if bucket < 70 else "validation" if bucket < 85 else "test")
    return groups[0] if groups and len(set(groups)) == 1 else None


def _support_docs(example: dict[str, Any]) -> list[dict[str, str]]:
    """仅保留标准答案所需的 supporting documents，不引入 Hotpot distractor 文档。"""
    context = dict(zip(example["context"]["title"], example["context"]["sentences"]))
    titles = list(dict.fromkeys(example["supporting_facts"]["title"]))
    return [{"id": f"hotpot-{example['id']}-{index}", "title": title, "text": " ".join(context[title])} for index, title in enumerate(titles) if title in context]


def _record(example: dict[str, Any], docs: list[dict[str, str]], split: str) -> dict[str, Any]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": example["question"]}]
    for doc in docs:
        messages.append({"role": "assistant", "content": json.dumps({"action": "search", "query": doc["title"]}, ensure_ascii=False)})
        messages.append({"role": "tool", "content": json.dumps({"ok": True, "result": [{**doc, "score": 1}]}, ensure_ascii=False)})
    messages.append({"role": "assistant", "content": json.dumps({"action": "answer", "answer": example["answer"]}, ensure_ascii=False)})
    return {"messages": messages, "metadata": {"task_id": f"hotpot-{example['id']}", "gold_answer": example["answer"], "difficulty": example.get("level", "unknown"), "search_count": len(docs), "source": "HotpotQA/distractor", "license": "CC-BY-SA-4.0", "split": split, "support_titles": [doc["title"] for doc in docs]}}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_final_test(source_dir: str | Path, output_dir: str | Path, count: int = 300, max_scanned: int = 50000) -> dict[str, Any]:
    """从 HotpotQA 流中构造从未使用过的最终测试集。

    当前开发集的 task ID 与 supporting-document 标题均被排除。标准答案仍只存在
    metadata 中，生成给模型的 messages 不包含 gold answer。输出 manifest 固定数据
    哈希和任务顺序，后续评测才能证明使用的是同一份不可见测试集。
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('缺少 datasets，请运行 pip install -e ".[data]"') from exc
    source = Path(source_dir)
    split_files = [source / f"{name}.jsonl" for name in ("train", "validation", "test")]
    if missing := [str(path) for path in split_files if not path.exists()]:
        raise FileNotFoundError(f"缺少现有 split：{missing}")
    existing = [item for path in split_files for item in _jsonl(path)]
    used_task_ids = {item["metadata"]["task_id"] for item in existing}
    used_titles = {title for item in existing for title in item["metadata"].get("support_titles", [])}
    final_records: list[dict[str, Any]] = []
    final_titles: set[str] = set()
    scanned = 0
    stream = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
    for example in stream:
        scanned += 1
        docs = _support_docs(example)
        task_id = f"hotpot-{example['id']}"
        titles = {doc["title"] for doc in docs}
        if task_id in used_task_ids or len(docs) < 2 or titles & used_titles or titles & final_titles:
            if scanned >= max_scanned:
                break
            continue
        final_records.append(_record(example, docs, "final_test"))
        final_titles.update(titles)
        if len(final_records) >= count or scanned >= max_scanned:
            break
    if len(final_records) < count:
        raise RuntimeError(f"最终测试集样本不足：扫描 {scanned} 条，仅得到 {len(final_records)} 条")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "final-test.jsonl"
    final_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in final_records), encoding="utf-8")
    task_ids = [item["metadata"]["task_id"] for item in final_records]
    manifest = {
        "source": "hotpotqa/hotpot_qa", "config": "distractor", "license": "CC-BY-SA-4.0",
        "source_data_dir": str(source), "source_split_sha256": {path.stem: _file_sha256(path) for path in split_files},
        "final_test": {"count": len(final_records), "scanned": scanned, "file": final_path.name,
                       "sha256": _file_sha256(final_path), "task_id_sha256": hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest(),
                       "supporting_documents": len(final_titles)},
        "protocol": "final-test 未用于训练、调参或 checkpoint 选择；只用于模型选择冻结后的最终评测。",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def download_hotpot_subset(output_dir: str | Path, train: int = 600, validation: int = 100, test: int = 100, max_scanned: int = 50000) -> dict[str, Any]:
    """流式下载 HotpotQA 并保存小规模、无 supporting-doc 重叠的数据集。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('缺少 datasets，请运行 pip install -e ".[data]"') from exc
    requested = {"train": train, "validation": validation, "test": test}
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in requested}
    seen_titles: dict[str, set[str]] = {name: set() for name in requested}
    stream = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True)
    scanned = 0
    for example in stream:
        scanned += 1
        docs = _support_docs(example)
        split = _split_for_titles([doc["title"] for doc in docs])
        if split is None or len(docs) < 2 or len(records[split]) >= requested[split]:
            if scanned >= max_scanned:
                break
            continue
        titles = {doc["title"] for doc in docs}
        if titles & seen_titles[split]:
            if scanned >= max_scanned:
                break
            continue
        records[split].append(_record(example, docs, split))
        seen_titles[split].update(titles)
        if all(len(records[name]) >= requested[name] for name in requested):
            break
        if scanned >= max_scanned:
            break
    if any(len(records[name]) < requested[name] for name in requested):
        raise RuntimeError(f"HotpotQA 可用样本不足：扫描 {scanned} 条，得到 { {k: len(v) for k, v in records.items()} }")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, items in records.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
    manifest = {"source": "hotpotqa/hotpot_qa", "config": "distractor", "license": "CC-BY-SA-4.0", "scanned": scanned, "counts": {name: len(items) for name, items in records.items()}, "document_counts": {name: len(titles) for name, titles in seen_titles.items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
