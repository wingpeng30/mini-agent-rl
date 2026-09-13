"""基于离线语料生成可审查的 Agent SFT 轨迹。

这里的目标是教模型遵守动作协议，而不是把标准答案塞进 prompt。
标准答案仅保存在 metadata，训练 messages 中只出现工具返回的证据。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ..tools import SearchTool


QUESTION_TEMPLATES = (
    "{title}是什么？",
    "请根据文档介绍{title}。",
    "关于{title}，文档中有哪些关键信息？",
    "我想了解{title}，请先检索相关资料。",
)


def _record(task_id: str, question: str, doc: dict[str, Any], query: str, difficulty: str, index: int) -> dict[str, Any]:
    observation = {"ok": True, "result": [{"id": doc["id"], "title": doc["title"], "text": doc["text"], "score": 1}]}
    messages = [
        {"role": "system", "content": "你是离线检索 Agent。只能输出 JSON：search 需要 query，answer 需要 answer。"},
        {"role": "user", "content": question},
        {"role": "assistant", "content": json.dumps({"action": "search", "query": query}, ensure_ascii=False)},
        {"role": "tool", "content": json.dumps(observation, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps({"action": "answer", "answer": doc["text"]}, ensure_ascii=False)},
    ]
    return {"messages": messages, "metadata": {"task_id": task_id, "gold_answer": doc["text"], "difficulty": difficulty, "search_count": 1, "source": "synthetic_offline", "template_index": index}}


def generate_split(corpus: list[dict[str, Any]], count: int, split: str, seed: int = 0) -> list[dict[str, Any]]:
    """确定性生成一个 split；同一 seed 和语料会得到完全相同的 JSONL。"""
    if not corpus:
        raise ValueError("语料不能为空")
    rng = random.Random(seed)
    records = []
    for index in range(count):
        doc = corpus[(index + (0 if split == "train" else 1)) % len(corpus)]
        template = QUESTION_TEMPLATES[rng.randrange(len(QUESTION_TEMPLATES))]
        records.append(_record(f"{split}-{index:05d}", template.format(title=doc["title"]), doc, doc["title"], "easy", index))
    return records


def validate_record(record: dict[str, Any]) -> list[str]:
    """检查训练样本结构，返回错误列表而不是静默丢弃坏数据。"""
    errors: list[str] = []
    messages = record.get("messages", [])
    if len(messages) != 5 or [m.get("role") for m in messages] != ["system", "user", "assistant", "tool", "assistant"]:
        errors.append("messages 必须是 system/user/assistant/tool/assistant 五轮")
    if messages:
        for index in (2, 4):
            try:
                action = json.loads(messages[index]["content"])
                if action.get("action") not in {"search", "answer"}:
                    errors.append(f"第 {index} 条 assistant 动作非法")
            except (KeyError, json.JSONDecodeError):
                errors.append(f"第 {index} 条 assistant 不是 JSON")
    if not record.get("metadata", {}).get("gold_answer"):
        errors.append("缺少 gold_answer metadata")
    return errors


def generate_dataset(corpus_path: str | Path, output_dir: str | Path, train: int = 800, validation: int = 100, test: int = 100, seed: int = 42) -> dict[str, Any]:
    """生成 train/validation/test JSONL，并在写入前完成全量质量校验。"""
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"output_dir": str(output), "seed": seed, "counts": {}, "errors": []}
    for split, count in (("train", train), ("validation", validation), ("test", test)):
        records = generate_split(corpus, count, split, seed + len(split))
        errors = [(record["metadata"]["task_id"], validate_record(record)) for record in records]
        result["errors"].extend([item for item in errors if item[1]])
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        result["counts"][split] = len(records)
    if result["errors"]:
        raise ValueError(f"SFT 数据校验失败: {result['errors'][:3]}")
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
