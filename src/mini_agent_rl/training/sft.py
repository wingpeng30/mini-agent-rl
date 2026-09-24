"""Qwen3.5 文本 Agent 的 LoRA-SFT 训练实现。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _ids(processor: Any, messages: list[dict[str, str]], thinking_enabled: bool = False) -> list[int]:
    """用模型自身 chat template 编码，防止手写特殊 token 与 Qwen 模板不一致。"""
    # Pass the switch for both modes. Omitting it delegates to a template
    # default and can make SFT encode a different prefix from inference.
    kwargs = {"tokenize": True, "add_generation_prompt": False,
              "enable_thinking": bool(thinking_enabled)}
    values = processor.apply_chat_template(messages, **kwargs)
    if isinstance(values, dict):
        values = values["input_ids"]
    return list(values[0] if values and isinstance(values[0], list) else values)


def _find_subsequence(values: list[int], target: list[int], start: int) -> int:
    for index in range(start, len(values) - len(target) + 1):
        if values[index : index + len(target)] == target:
            return index
    return -1


def encode_assistant_only(processor: Any, messages: list[dict[str, str]], max_length: int = 2048,
                          target_assistant_index: int | None = None,
                          thinking_enabled: bool = False) -> dict[str, list[int]]:
    """只为 assistant 消息打 label，tool observation 永远是 -100。

    通过完整 chat template 中的消息内容定位 assistant span，而不是猜测 Qwen 特殊 token。
    """
    input_ids = _ids(processor, messages, thinking_enabled)
    labels = [-100] * len(input_ids)
    cursor = 0
    for index, message in enumerate(messages):
        # Qwen3.5 的模板会根据后续角色改写前缀，不能用逐轮 token 前缀差。
        # 直接在完整序列中按消息顺序定位原始 content，可精确跳过模板控制 token。
        # Qwen 模板的 ``trim`` 会去掉消息首尾空白；因此按相同规则编码内容。
        # 这不会改变训练目标中的 JSON action，但可避免带尾随空格的问句定位失败。
        template_content = message["content"].strip()
        # Qwen's thinking template normalizes the boundary to a blank line
        # between </think> and the final answer.  Locate the actual rendered
        # target rather than the pre-template spelling.
        if thinking_enabled and message["role"] == "assistant" and "</think>\n" in template_content and "</think>\n\n" not in template_content:
            template_content = template_content.replace("</think>\n", "</think>\n\n", 1)
        content_ids = processor.tokenizer(template_content, add_special_tokens=False)["input_ids"]
        start = _find_subsequence(input_ids, content_ids, cursor)
        if start < 0:
            raise ValueError(f"无法在 chat template token 序列中定位第 {index} 条 {message['role']} 内容")
        end = start + len(content_ids)
        if message["role"] == "assistant" and (target_assistant_index is None or index == target_assistant_index):
            labels[start:end] = content_ids
            # The template-owned assistant terminator is part of the target
            # distribution.  Supervise it explicitly without labelling any
            # system/user/tool prefix token.
            convert_token = getattr(processor.tokenizer, "convert_tokens_to_ids", None)
            terminator = convert_token("<|im_end|>") if callable(convert_token) else None
            if isinstance(terminator, int) and end < len(input_ids) and input_ids[end] == terminator:
                labels[end] = terminator
        cursor = end
    if len(input_ids) > max_length:
        raise ValueError(f"样本长度 {len(input_ids)} 超过 max_length={max_length}；拒绝静默截断问题或证据")
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


def build_lora_model(model_path: str, dtype: str = "bfloat16", gradient_checkpointing: bool = True,
                     adapter_path: str | None = None):
    """加载文本+视觉模型，但只在语言模型线性层注入 LoRA；视觉塔保持冻结。"""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(model_path, dtype=getattr(torch, dtype), device_map="cuda", local_files_only=True)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    return model, processor


def train_sft(model_path: str, train_path: str, validation_path: str, output_dir: str, max_steps: int | None = None,
              epochs: int = 2, max_length: int = 2048, seed: int = 42, batch_size: int = 1,
              gradient_accumulation_steps: int = 8, gradient_checkpointing: bool = True,
              adapter_path: str | None = None, enable_thinking: bool = False,
              learning_rate: float = 1e-4, save_steps: int = 50,
              eval_steps: int = 50, warmup_ratio: float = 0.05) -> dict[str, Any]:
    """执行 LoRA-SFT 并保存 adapter、训练状态和指标。"""
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    model, processor = build_lora_model(model_path, gradient_checkpointing=gradient_checkpointing, adapter_path=adapter_path)
    train_records = load_jsonl(train_path)
    validation_records = load_jsonl(validation_path)
    def encode_record(item: dict[str, Any]) -> dict[str, list[int]]:
        return encode_assistant_only(processor, item["messages"], max_length,
                                     item.get("target_assistant_index"),
                                     bool(item.get("thinking_enabled", enable_thinking)))
    train_data = Dataset.from_list([encode_record(item) for item in train_records])
    validation_data = Dataset.from_list([encode_record(item) for item in validation_records])
    # This pinned Transformers build exposes warmup_steps but not warmup_ratio.
    # Calculate the configured fraction explicitly so the schedule remains stable.
    updates_per_epoch = max(1, (len(train_data) + batch_size * gradient_accumulation_steps - 1) // (batch_size * gradient_accumulation_steps))
    planned_steps = max_steps if max_steps is not None else updates_per_epoch * epochs
    warmup_steps = int(planned_steps * warmup_ratio)
    arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps is not None else -1,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=seed,
        data_seed=seed,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=train_data, eval_dataset=validation_data, data_collator=AssistantOnlyCollator(processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id))
    trainer.train()
    trainer.save_model(output_dir)
    metrics = {"seed": seed, "train_samples": len(train_data), "validation_samples": len(validation_data), "max_length": max_length, "batch_size": batch_size, "gradient_accumulation_steps": gradient_accumulation_steps, "gradient_checkpointing": gradient_checkpointing, "adapter_path": adapter_path, "enable_thinking": enable_thinking, "learning_rate": learning_rate, "save_steps": save_steps, "eval_steps": eval_steps, "warmup_ratio": warmup_ratio, "warmup_steps": warmup_steps, "train_metrics": trainer.state.log_history, "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "training_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return metrics
