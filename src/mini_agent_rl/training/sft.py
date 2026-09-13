"""Qwen3.5 文本 Agent 的 LoRA-SFT 训练实现。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _ids(processor: Any, messages: list[dict[str, str]]) -> list[int]:
    """用模型自身 chat template 编码，防止手写特殊 token 与 Qwen 模板不一致。"""
    values = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(values, dict):
        values = values["input_ids"]
    return list(values[0] if values and isinstance(values[0], list) else values)


def _find_subsequence(values: list[int], target: list[int], start: int) -> int:
    for index in range(start, len(values) - len(target) + 1):
        if values[index : index + len(target)] == target:
            return index
    return -1


def encode_assistant_only(processor: Any, messages: list[dict[str, str]], max_length: int = 1024) -> dict[str, list[int]]:
    """只为 assistant 消息打 label，tool observation 永远是 -100。

    通过完整 chat template 中的消息内容定位 assistant span，而不是猜测 Qwen 特殊 token。
    """
    input_ids = _ids(processor, messages)
    labels = [-100] * len(input_ids)
    cursor = 0
    for index, message in enumerate(messages):
        # Qwen3.5 的模板会根据后续角色改写前缀，不能用逐轮 token 前缀差。
        # 直接在完整序列中按消息顺序定位原始 content，可精确跳过模板控制 token。
        # Qwen 模板的 ``trim`` 会去掉消息首尾空白；因此按相同规则编码内容。
        # 这不会改变训练目标中的 JSON action，但可避免带尾随空格的问句定位失败。
        template_content = message["content"].strip()
        content_ids = processor.tokenizer(template_content, add_special_tokens=False)["input_ids"]
        start = _find_subsequence(input_ids, content_ids, cursor)
        if start < 0:
            raise ValueError(f"无法在 chat template token 序列中定位第 {index} 条 {message['role']} 内容")
        end = start + len(content_ids)
        if message["role"] == "assistant":
            labels[start:end] = content_ids
        cursor = end
    if len(input_ids) > max_length:
        input_ids, labels = input_ids[-max_length:], labels[-max_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("样本没有可训练的 assistant token")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


@dataclass
class AssistantOnlyCollator:
    """动态 padding；labels 中的 -100 会被 PyTorch CrossEntropy 忽略。"""
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch
        length = max(len(feature["input_ids"]) for feature in features)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = length - len(feature["input_ids"])
            result["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            result["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            result["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit else records


def build_lora_model(model_path: str, dtype: str = "bfloat16"):
    """加载文本+视觉模型，但只在语言模型线性层注入 LoRA；视觉塔保持冻结。"""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(model_path, dtype=getattr(torch, dtype), device_map="cuda", local_files_only=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    return model, processor


def train_sft(model_path: str, train_path: str, validation_path: str, output_dir: str, max_steps: int | None = None, epochs: int = 2, max_length: int = 1024, seed: int = 42) -> dict[str, Any]:
    """执行 LoRA-SFT 并保存 adapter、训练状态和指标。"""
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    model, processor = build_lora_model(model_path)
    train_records = load_jsonl(train_path)
    validation_records = load_jsonl(validation_path)
    train_data = Dataset.from_list([encode_assistant_only(processor, item["messages"], max_length) for item in train_records])
    validation_data = Dataset.from_list([encode_assistant_only(processor, item["messages"], max_length) for item in validation_records])
    arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps is not None else -1,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=20,
        eval_strategy="steps",
        eval_steps=20,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=train_data, eval_dataset=validation_data, data_collator=AssistantOnlyCollator(processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id))
    trainer.train()
    trainer.save_model(output_dir)
    metrics = {"seed": seed, "train_samples": len(train_data), "validation_samples": len(validation_data), "train_metrics": trainer.state.log_history, "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "training_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return metrics
