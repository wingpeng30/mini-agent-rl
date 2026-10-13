"""本地 Transformers 模型适配器；导入本模块不会下载模型或要求 CUDA。"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import hashlib
from pathlib import Path
import time
import threading
import subprocess
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..domain import Message, ModelResponse, ToolCall, ToolDefinition
from .context import budget_messages
from .observation_presentation import LEGACY_JSON, DOCUMENT_BLOCKS_V1, present_observation
from .context_aware_decoding import ContextAwareDecodingConfig


class LocalModelConfig(BaseModel):
    """本地推理配置。模型 ID 只是配置，构造配置对象不会访问网络。"""

    model_id: str = r"D:\qwen_08b"
    adapter_path: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    gradient_checkpointing: bool = True
    max_new_tokens: int = Field(default=128, ge=16, le=2048)
    max_input_tokens: int = Field(default=2048, ge=128, le=32768)
    timeout: float | None = Field(default=120.0, gt=0)
    trust_remote_code: bool = False
    local_files_only: bool = True
    enable_thinking: bool = False
    deterministic: bool = True
    # bf16 是实际采样默认值；fp32 仅用于 v0.6.1 的跨加载数值诊断。
    logprob_mode: Literal["bf16", "fp32"] = "bf16"
    recompute_old_logprobs: bool = True
    # legacy 仅用于历史回放；compact-v1 是 v0.8.6 的可审计证据表达。
    observation_encoding: Literal["legacy", "compact-v1", "compact-v2", "compact-v3"] = "legacy"
    observation_presentation: Literal["legacy-json", "document-blocks-v1", "evidence-addresses-v1"] = "legacy-json"
    action_completion: Literal["eos", "first-json-v1"] = "eos"
    action_constraint: Literal["none", "json-action-v1", "evidence-action-v1"] = "none"
    context_aware_decoding: ContextAwareDecodingConfig | None = None

    @model_validator(mode="after")
    def validate_quantization(self):
        # bitsandbytes 在原生 Windows 上兼容性不稳定，因此量化必须显式开启，不能静默选择。
        if self.load_in_4bit and self.device == "cpu":
            raise ValueError("CPU 模式不支持本项目的 4-bit 加载")
        if self.action_completion == "first-json-v1" and self.enable_thinking:
            raise ValueError("first-json-v1 only supports non-thinking JSON actions")
        if self.action_constraint != "none" and (self.enable_thinking or self.recompute_old_logprobs):
            raise ValueError("action constraints require non-thinking inference without unmasked old logprobs")
        if self.context_aware_decoding is not None and (self.enable_thinking or self.recompute_old_logprobs or self.action_constraint != "none"):
            raise ValueError("CAD requires non-thinking unconstrained inference without raw-policy old logprobs")
        return self


class ActionParseError(ValueError):
    """模型没有遵守 search/answer 动作协议。"""
    def __init__(self, message: str, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text


def parse_agent_action(text: str) -> tuple[str, ToolCall | None]:
    """把模型文本限制为唯一的策略动作，避免把自然语言误当成工具调用。

    observation 由环境产生，不经过此解析器，也不会被记录为策略 action token。
    """

    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"模型输出不是合法 JSON: {exc.msg}", text) from exc
    if not isinstance(payload, dict):
        raise ActionParseError("模型动作必须是 JSON 对象", text)
    action = payload.get("action")
    if action == "search" and isinstance(payload.get("query"), str) and payload["query"].strip():
        return "", ToolCall(name="search", arguments={"query": payload["query"].strip()})
    if action == "answer" and isinstance(payload.get("answer"), str) and payload["answer"].strip():
        return payload["answer"].strip(), None
    raise ActionParseError("动作必须是包含 query 的 search，或包含 answer 的 answer", text)


def split_reasoning_action(text: str, thinking_enabled: bool) -> tuple[str | None, str]:
    """只接受思考段之后的唯一 JSON 动作；思考段里的 JSON 绝不参与解析。"""
    raw = text.strip()
    if not thinking_enabled:
        return None, raw
    if "<think>" not in raw and "</think>" not in raw:
        if raw.startswith("{"):
            return None, raw  # 模型可直接产生动作，仍是可审计的有效退化路径。
        raise ActionParseError("思考模式输出缺少闭合思考段或直接 JSON 动作", text)
    # 模板已把 <think> 放在 prompt 末尾时，生成序列只包含闭合标签。
    normalized = raw if "<think>" in raw else "<think>\n" + raw
    if normalized.count("<think>") != 1 or normalized.count("</think>") != 1:
        raise ActionParseError("思考段必须恰好包含一对 <think>...</think>", text)
    match = re.fullmatch(r"\s*<think>\s*(.*?)\s*</think>\s*(.+?)\s*", normalized, flags=re.DOTALL)
    if not match:
        raise ActionParseError("思考段未闭合或缺少最终动作", text)
    return match.group(1), match.group(2)


def trim_generation_terminators(token_ids: list[int], tokenizer: Any) -> tuple[list[int], dict[str, bool]]:
    """Remove a terminal Qwen message boundary/EOS sequence before parsing."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    message_end = convert("<|im_end|>") if callable(convert) else None
    end_of_text = convert("<|endoftext|>") if callable(convert) else None
    terminal_ids = {value for value in (eos, message_end, end_of_text) if isinstance(value, int) and value >= 0}
    values = list(token_ids)
    seen_terminal = False
    seen_message_end = False
    seen_eos = False
    while values:
        value = values[-1]
        if value in terminal_ids:
            seen_terminal = True
            seen_message_end = seen_message_end or value == message_end
            seen_eos = seen_eos or value == eos or value == end_of_text
            values.pop()
            if value == message_end:
                break
        elif seen_terminal and tokenizer.decode([value], skip_special_tokens=False).strip() == "":
            values.pop()
        else:
            break
    return values, {"generated_message_end": seen_message_end, "generated_eos": seen_eos}


def generation_terminator_ids(tokenizer: Any) -> list[int]:
    """Return the model EOS plus Qwen's assistant-message terminator."""
    values: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        values.append(eos)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    message_end = convert("<|im_end|>") if callable(convert) else None
    special = set(getattr(tokenizer, "all_special_ids", []))
    if isinstance(message_end, int) and message_end in special and message_end not in values:
        values.append(message_end)
    return values


def first_json_object_end(text: str) -> int | None:
    """Return the end offset of the first structurally complete top-level JSON object.

    This deliberately checks only structure, not JSON validity. A malformed first
    object still forms the one attempted action and must never be skipped in favor
    of a later object.
    """
    start = len(text) - len(text.lstrip())
    if start >= len(text) or text[start] != "{":
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack[-1] != pairs[char]:
                # Still stop at a closed outer object when its JSON is malformed.
                if char == "}" and stack == ["{"]:
                    return index + 1
                return None
            stack.pop()
            if not stack:
                return index + 1
    return None


class _FirstJSONStoppingCriteria:
    """Transformers stopping criterion for a single generated action."""

    def __init__(self, tokenizer: Any, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.stop_offset: int | None = None

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        if input_ids.shape[0] != 1:
            raise RuntimeError("first-json-v1 requires batch size 1")
        generated = input_ids[0, self.prompt_length:].tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        self.stop_offset = first_json_object_end(text)
        return torch.tensor([self.stop_offset is not None], dtype=torch.bool, device=input_ids.device)


class _BatchActionStoppingCriteria:
    """Record each row's actual end before generate fills finished rows with PAD."""

    def __init__(self, tokenizer: Any, prompt_length: int, batch_size: int, first_json: bool):
        self.tokenizer, self.prompt_length = tokenizer, prompt_length
        self.first_json = first_json
        self.eos_token_id = generation_terminator_ids(tokenizer)
        self.stop_lengths: list[int | None] = [None] * batch_size
        self.stop_offsets: list[int | None] = [None] * batch_size
        self.stop_reasons: list[str | None] = [None] * batch_size

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        if input_ids.shape[0] != len(self.stop_lengths):
            raise RuntimeError("batch action stopper row count changed")
        for row in range(input_ids.shape[0]):
            if self.stop_lengths[row] is not None:
                continue
            ids = input_ids[row, self.prompt_length:].tolist()
            offset = (first_json_object_end(self.tokenizer.decode(ids, skip_special_tokens=False))
                      if self.first_json else None)
            reason = "first_json" if offset is not None else "eos" if ids and ids[-1] in self.eos_token_id else None
            if reason is not None:
                self.stop_lengths[row], self.stop_offsets[row], self.stop_reasons[row] = len(ids), offset, reason
        return torch.tensor([length is not None for length in self.stop_lengths], dtype=torch.bool,
                            device=input_ids.device)


def local_environment_report() -> dict[str, Any]:
    """只检查依赖和硬件，不加载权重、不访问 Hugging Face。"""

    report: dict[str, Any] = {
        "transformers_installed": importlib.util.find_spec("transformers") is not None,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "cuda_available": False,
        "gpu": None,
        "gpu_memory_gb": None,
        "transformers_version": None,
    }
    if report["transformers_installed"]:
        import transformers
        report["transformers_version"] = transformers.__version__
    if report["torch_installed"]:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = props.name
            report["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
    return report


class TransformersModelClient:
    """以懒加载方式接入 Hugging Face CausalLM。

    首次调用 ``generate`` 才会加载模型；若缓存中没有权重，Transformers 才会下载。
    API key、标准答案和 reward 都不会进入模型加载参数。
    """

    def __init__(self, config: LocalModelConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._load_seconds: float | None = None
        self._generation_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        self._context_records: list[dict[str, Any]] = []

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig
            try:
                from transformers import AutoModelForMultimodalLM as model_cls
            except ImportError:
                # 旧版 Transformers 没有多模态别名；Qwen 纯文本推理可安全回退。
                from transformers import AutoModelForCausalLM as model_cls
        except ImportError as exc:
            raise RuntimeError('缺少本地模型依赖，请运行 pip install -e ".[local-model]"') from exc
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }
        if self.config.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = self.config.device
        else:
            kwargs["dtype"] = getattr(torch, self.config.dtype)
            kwargs["device_map"] = self.config.device
        started = time.perf_counter()
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
        )
        self._model = model_cls.from_pretrained(self.config.model_id, **kwargs)
        if self.config.adapter_path:
            # Adapter 只包含 LoRA 增量参数；先加载冻结的基础模型，再叠加增量，
            # 能保证 base 与 adapter 评测使用完全相同的模型权重和推理配置。
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError('加载 LoRA adapter 需要 peft，请运行 pip install -e ".[train]"') from exc
            self._model = PeftModel.from_pretrained(
                self._model, self.config.adapter_path, local_files_only=self.config.local_files_only,
                # 同一 client 可用于推理或训练；实际是否反传仍由 prepare_for_training 显式控制。
                is_trainable=True,
            )
        self._model.eval()
        self._load_seconds = time.perf_counter() - started

    def _present_messages(self, messages: list[Message]):
        rendered = []
        spans = []
        for index, message in enumerate(messages):
            content = message.content
            if message.role == "tool" and self.config.observation_presentation != LEGACY_JSON:
                if self.config.observation_presentation == "evidence-addresses-v1":
                    from .constrained_evidence import addressed_observation
                    content, records = addressed_observation(content, f"m{index}")
                else:
                    content, records = present_observation(content, self.config.observation_presentation)
                spans.extend({**record, "message_index": index} for record in records)
            rendered.append({"role": message.role, "content": content})
        return rendered, spans

    def _render_chat(self, messages: list[Message], *, return_presentation: bool = False):
        """集中调用 chat template，保证预算估算与最终编码完全一致。"""
        chat, body_spans = self._present_messages(messages)
        # Qwen 模板的 enable_thinking 是直接模板参数；嵌套参数会被忽略。
        template_kwargs = {"enable_thinking": self.config.enable_thinking}
        encoded = self._processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True,
                                                      return_dict=True, return_tensors="pt", **template_kwargs)
        if not return_presentation:
            return encoded
        metadata = {"observation_presentation": self.config.observation_presentation,
                    "document_bodies": body_spans}
        if self.config.observation_presentation != LEGACY_JSON and body_spans:
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            if not getattr(tokenizer, "is_fast", False):
                raise RuntimeError("document-blocks-v1 requires a fast tokenizer for exact citation token offsets")
            rendered_text = self._processor.apply_chat_template(chat, add_generation_prompt=True,
                                                                 tokenize=False, **template_kwargs)
            offset_encoding = tokenizer(rendered_text, add_special_tokens=False,
                                        return_offsets_mapping=True, return_tensors="pt")
            template_ids = offset_encoding["input_ids"][0].tolist()
            actual_ids = encoded["input_ids"][0].tolist()
            if template_ids != actual_ids:
                raise RuntimeError("tokenizer offsets do not reproduce the actual chat-template input")
            offsets = offset_encoding["offset_mapping"][0].tolist()
            cursor = 0
            for record in body_spans:
                message_index = record["message_index"]
                body = chat[message_index]["content"]
                body = body[record["rendered_char_span"][0]:record["rendered_char_span"][1]]
                if not body:
                    record["input_token_span"] = [0, 0]
                    continue
                char_start = rendered_text.find(body, cursor)
                if char_start < 0:
                    raise RuntimeError("presented document body is absent from rendered chat template")
                char_end = char_start + len(body)
                token_indexes = [i for i, pair in enumerate(offsets)
                                 if pair[1] > char_start and pair[0] < char_end]
                if not token_indexes:
                    raise RuntimeError("document body has no corresponding input token span")
                record["rendered_prompt_char_span"] = [char_start, char_end]
                record["input_token_span"] = [token_indexes[0], token_indexes[-1] + 1]
                cursor = char_end
            metadata["rendered_prompt_sha256"] = hashlib.sha256(rendered_text.encode("utf-8")).hexdigest()
        return encoded, metadata

    def _content_length(self, text: str) -> int:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _inputs(self, messages: list[Message], phase: str = "unknown"):
        """由同一 chat template 编码 prompt，避免采集与训练 scorer 的 token 边界漂移。"""
        def render_length(items: list[Message]) -> int:
            return int(self._render_chat(items)["input_ids"].shape[1])
        budgeted = budget_messages(messages, self.config.max_input_tokens, render_length, self._content_length,
                                   self.config.observation_encoding)
        encoded, presentation_metadata = self._render_chat(budgeted.messages, return_presentation=True)
        encoded = encoded.to(self._model.device)
        # 元数据挂在 BatchEncoding 对象上，不能放进字典，否则 generate 会把它
        # 当成模型 forward 参数并报“未使用的 model_kwargs”。
        encoded._context_usage = dict(budgeted.usage)
        encoded._context_usage["phase"] = phase
        encoded._context_usage["prompt_sha256"] = hashlib.sha256(
            bytes(str(encoded["input_ids"][0].tolist()), "utf-8")
        ).hexdigest()
        encoded._context_usage["retained_tool_documents"] = budgeted.retained_tool_documents
        encoded._context_usage["document_changes"] = budgeted.document_changes
        encoded._context_usage["presentation"] = presentation_metadata
        encoded._budgeted_messages = budgeted.messages
        self._context_records.append(dict(encoded._context_usage))
        return encoded

    def _score_sync(self, messages: list[Message], action_token_ids: list[int], with_grad: bool = False,
                    phase: str | None = None, prompt_token_ids: list[int] | None = None,
                    prompt_attention_mask: list[int] | None = None):
        """以原始策略 teacher-forcing 计算 action logprob。

        prompt 与 tool observation 只用于条件化，gather 的位置严格只覆盖 assistant action，
        因而它们永远不会进入 policy loss。
        """
        import torch
        if self._model is None:
            self._load()
        # Generation may have enabled KV cache on the same model instance.
        # Teacher-forced policy scoring and backward must always disable it.
        self._model.config.use_cache = False
        if prompt_token_ids is None:
            inputs = self._inputs(messages, phase or ("current_logprob" if with_grad else "old_logprob"))
        else:
            # 训练重放精确使用采样时的预算后 token，避免重新渲染带来的边界漂移。
            ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=self._model.device)
            mask_values = prompt_attention_mask or [1] * len(prompt_token_ids)
            inputs = {"input_ids": ids, "attention_mask": torch.tensor([mask_values], dtype=torch.long, device=self._model.device)}
            replay_hash = hashlib.sha256(bytes(str(prompt_token_ids), "utf-8")).hexdigest()
            self._context_records.append({"phase": phase or "replay_logprob", "prompt_sha256": replay_hash,
                                          "replayed_prompt": True, "actual_prompt_tokens": len(prompt_token_ids)})
        prompt_length = inputs["input_ids"].shape[1]
        action = torch.tensor([action_token_ids], dtype=torch.long, device=self._model.device)
        full_ids = torch.cat([inputs["input_ids"], action], dim=1)
        action_mask = torch.ones_like(action)
        full_mask = torch.cat([inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])), action_mask], dim=1)
        context = torch.enable_grad() if with_grad else torch.inference_mode()
        forward_kwargs = {}
        if not bool(full_mask.all()):
            # Batch generation numbers semantic tokens from zero, independently
            # of left PAD. Match those positions during exact policy replay.
            positions = full_mask.long().cumsum(-1) - 1
            forward_kwargs["position_ids"] = positions.masked_fill(full_mask == 0, 0)
        with context:
            logits = self._model(input_ids=full_ids, attention_mask=full_mask, **forward_kwargs).logits
            positions = logits[0, prompt_length - 1 : prompt_length - 1 + len(action_token_ids)]
            # 无论 backbone 为 BF16 还是 FP32，softmax 统一在 FP32 中进行，减小数值误差。
            values = torch.log_softmax(positions.float(), dim=-1).gather(1, action[0].unsqueeze(1)).squeeze(1)
        return values

    async def score_action(self, messages: list[Message], action_token_ids: list[int]) -> list[float]:
        """诊断/审计用无梯度 scorer；返回值不用于反向传播。"""
        async with self._generation_lock:
            values = await asyncio.to_thread(self._score_sync, messages, action_token_ids, False, "diagnostic_logprob")
        return values.detach().cpu().tolist()

    def score_action_tensor(self, messages: list[Message], action_token_ids: list[int], with_grad: bool = False,
                            prompt_token_ids: list[int] | None = None, prompt_attention_mask: list[int] | None = None):
        """训练器同步调用的 scorer；with_grad=True 时保留到 LoRA 参数的计算图。"""
        return self._score_sync(messages, action_token_ids, with_grad, "current_logprob" if with_grad else "old_logprob",
                                prompt_token_ids, prompt_attention_mask)

    def reference_logprobs(self, messages: list[Message], action_token_ids: list[int], reference_state: dict[str, Any] | None = None,
                            prompt_token_ids: list[int] | None = None, prompt_attention_mask: list[int] | None = None):
        """计算冻结 reference。

        传入起点 LoRA 快照时，临时恢复快照再前向，避免额外加载一份完整模型；
        未传快照则保留兼容行为，使用不带 adapter 的 base policy。
        """
        if self._model is None:
            self._load()
        if reference_state is not None:
            import torch
            saved = {name: parameter.detach().clone() for name, parameter in self._model.named_parameters() if name in reference_state}
            try:
                with torch.no_grad():
                    for name, parameter in self._model.named_parameters():
                        if name in reference_state:
                            parameter.copy_(reference_state[name])
                return self._score_sync(messages, action_token_ids, False, "reference_logprob", prompt_token_ids, prompt_attention_mask)
            finally:
                with torch.no_grad():
                    for name, parameter in self._model.named_parameters():
                        if name in saved:
                            parameter.copy_(saved[name])
        if not hasattr(self._model, "disable_adapter"):
            raise RuntimeError("GRPO reference 需要已加载 PEFT LoRA adapter")
        with self._model.disable_adapter():
            return self._score_sync(messages, action_token_ids, False, "reference_logprob", prompt_token_ids, prompt_attention_mask)

    def prepare_for_training(self):
        """只开启 LoRA 训练，基础模型/视觉塔始终冻结。"""
        if self._model is None:
            self._load()
        self._model.config.use_cache = False
        # Qwen3.5 的完整语言模型被冻结，checkpoint 只保存 LoRA 反向所需的中间状态，
        # 并让输入 embedding 保留梯度路径；这是单张 8GB 卡运行 GRPO 的关键显存开关。
        if self.config.gradient_checkpointing and hasattr(self._model, "gradient_checkpointing_enable"):
            self._model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        elif hasattr(self._model, "gradient_checkpointing_disable"):
            self._model.gradient_checkpointing_disable()
        if hasattr(self._model, "enable_input_require_grads"):
            self._model.enable_input_require_grads()
        # GRPO needs autograd, not stochastic train-mode behavior.  ``eval``
        # keeps dropout/other stochastic modules identical to rollout sampling;
        # requires_grad remains enabled and backward still works normally.
        self._model.eval()
        trainable = [(name, p) for name, p in self._model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in name for name, _ in trainable):
            raise RuntimeError("训练安全检查失败：只能让 LoRA 参数产生梯度")
        return trainable

    def named_trainable_parameters(self):
        """返回已通过 LoRA 安全检查的参数，供 optimizer 与 reference 快照共享。"""
        if self._model is None:
            self._load()
        values = [(name, parameter) for name, parameter in self._model.named_parameters() if parameter.requires_grad]
        if not values or any("lora_" not in name for name, _ in values):
            raise RuntimeError("训练安全检查失败：只能让 LoRA 参数产生梯度")
        return values

    def set_training_forward_mode(self) -> None:
        """Enable gradients while keeping the sampled policy deterministic.

        ``eval`` does not disable autograd.  It does disable every stochastic
        dropout path in the base model; merely disabling LoRA dropout left the
        first actual GRPO backward ratio different from the frozen old policy.
        """
        if self._model is None:
            self._load()
        self._model.eval()
        # SFT adapter 可能以 dropout=0.05 保存。不能改写 adapter 配置或权重；只在
        # 当前 GRPO 进程把 LoRA dropout module 切到 eval，仍保持主模型训练前向和梯度 checkpoint。
        disabled = []
        for name, module in self._model.named_modules():
            if "lora_dropout" in name:
                module.eval(); disabled.append(name)
        self._effective_lora_dropout_disabled = disabled

    def training_mode_report(self) -> dict[str, Any]:
        """供实验报告确认实际训练模式，而不是只记录 adapter 文件中的静态配置。"""
        return {"gradient_checkpointing": bool(getattr(self._model, "is_gradient_checkpointing", False)),
                "model_training": bool(self._model.training),
                "lora_dropout_modules_forced_eval": list(getattr(self, "_effective_lora_dropout_disabled", []))}

    def policy_fingerprint(self) -> dict[str, Any]:
        """可 JSON 序列化的运行指纹；不包含私密 token 或任何训练样本内容。"""
        import torch
        try:
            import transformers, peft
            peft_version = peft.__version__
        except ImportError:
            import transformers
            peft_version = None
        def digest(path: Path) -> str | None:
            if not path.exists(): return None
            hasher = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""): hasher.update(block)
            return hasher.hexdigest()
        base = Path(self.config.model_id)
        adapter = Path(self.config.adapter_path) if self.config.adapter_path else None
        config_file = base / "config.json"
        tokenizer_file = base / "tokenizer.json"
        adapter_weights = next(iter(adapter.glob("adapter_model.*")), None) if adapter and adapter.exists() else None
        return {"base_model_path": str(base), "base_config_sha256": digest(config_file),
                "adapter_path": str(adapter) if adapter else None, "adapter_config_sha256": digest(adapter / "adapter_config.json") if adapter else None,
                "adapter_weight_sha256": digest(adapter_weights) if adapter_weights else None,
                "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft_version,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "dtype": self.config.dtype, "tokenizer_sha256": digest(tokenizer_file),
                "generation": {"max_new_tokens": self.config.max_new_tokens, "max_input_tokens": self.config.max_input_tokens,
                               "load_in_4bit": self.config.load_in_4bit, "thinking": self.config.enable_thinking,
                               "deterministic": self.config.deterministic, "logprob_mode": self.config.logprob_mode,
                               "observation_encoding": self.config.observation_encoding,
                               "action_completion": self.config.action_completion,
                               "context_aware_decoding": self.config.context_aware_decoding.model_dump() if self.config.context_aware_decoding else None}}

    def _generate_sync(self, messages: list[Message], temperature: float, seed: int | None) -> ModelResponse:
        import torch

        if self._model is None:
            self._load()
        if seed is not None:
            torch.manual_seed(seed)
        if self.config.deterministic:
            # CUDA 上的 kernel 选择会影响相同 seed 的采样与 logprob；优先稳定复现，
            # 速度不是本阶段的首要目标。warn_only 防止第三方算子直接中断采集。
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
        inputs = self._inputs(messages, "rollout_sampling")
        do_sample = temperature > 0
        cad_processor = None
        cad = self.config.context_aware_decoding
        if cad is not None and do_sample:
            raise ValueError("CAD currently supports greedy single-sample inference only")
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        terminators = generation_terminator_ids(tokenizer)
        generation = {"max_new_tokens": self.config.max_new_tokens, "do_sample": do_sample}
        if terminators:
            generation["eos_token_id"] = terminators
        first_json_stopper = None
        if self.config.action_completion == "first-json-v1":
            if inputs["input_ids"].shape[0] != 1:
                raise RuntimeError("first-json-v1 requires batch size 1")
            from transformers import StoppingCriteriaList
            first_json_stopper = _FirstJSONStoppingCriteria(tokenizer, inputs["input_ids"].shape[1])
            generation["stopping_criteria"] = StoppingCriteriaList([first_json_stopper])
        if do_sample:
            generation["temperature"] = temperature
        if self.config.action_constraint != "none":
            from .constrained_evidence import make_processor
            generation["logits_processor"] = [make_processor(self, inputs)]
        if cad is not None:
            from .context_aware_decoding import ContextAwareLogitsProcessor, reference_messages, search_count
            if cad.alpha > 0 and search_count(messages) in cad.after_searches:
                if inputs["input_ids"].shape[0] != 1:
                    raise RuntimeError("CAD requires batch size 1")
                if any(inputs._context_usage.get(k) for k in ("lossy_body_trimmed", "documents_deleted", "history_dropped", "prompt_truncated", "context_budget_error")):
                    raise RuntimeError("CAD experiment refuses clipped evidence")
                reference = self._render_chat(reference_messages(inputs._budgeted_messages)).to(self._model.device)
                if reference["input_ids"].shape[1] > self.config.max_input_tokens:
                    raise RuntimeError("CAD reference exceeds input budget")
                cad_processor = ContextAwareLogitsProcessor(self._model, inputs["input_ids"].shape[1], reference, cad.alpha)
                generation["logits_processor"] = [cad_processor]
        # Generation uses KV cache; training scorers disable it again in
        # prepare_for_training(), preventing phase-specific settings from leaking.
        self._model.config.use_cache = True
        with torch.inference_mode():
            output = self._model.generate(**inputs, **generation)
        prompt_length = inputs["input_ids"].shape[1]
        action_ids = output[0, prompt_length:]
        # Preserve generated special tokens for audit.  Parsing removes only a
        # terminal EOS token, never a think boundary or an embedded JSON.
        raw = self._processor.decode(action_ids, skip_special_tokens=False)
        parse_ids, terminal_usage = trim_generation_terminators(action_ids.tolist(), tokenizer)
        parse_text = self._processor.decode(parse_ids, skip_special_tokens=False)
        first_json_end = (first_json_object_end(parse_text)
                          if self.config.action_completion == "first-json-v1" else None)
        first_json_tail = parse_text[first_json_end:] if first_json_end is not None else ""
        if first_json_end is not None:
            parse_text = parse_text[:first_json_end]
        # generate 的 score 可能已被 temperature 或 logits processor 改写；RL 的
        # old logprob 必须来自原始策略，因此对 prompt+action 做一次 teacher-forced 前向。
        # 采样后的 old logprob 精确重放本次生成使用的 token，不重新渲染消息。
        logprobs = self._score_sync(
            messages, action_ids.tolist(), False, "sampling_old_logprob",
            inputs["input_ids"][0].tolist(),
            inputs.get("attention_mask")[0].tolist() if inputs.get("attention_mask") is not None else None,
        ).detach().cpu().tolist() if self.config.recompute_old_logprobs else []
        try:
            reasoning, action_text = split_reasoning_action(parse_text, self.config.enable_thinking)
            content, tool_call = parse_agent_action(action_text)
            parse_error, parse_status = None, "valid"
        except ActionParseError as exc:
            reasoning, content, tool_call, parse_error = None, "", None, str(exc)
            parse_status = ("unclosed_think" if self.config.enable_thinking and "</think>" not in raw
                            else "invalid_json_action")
        usage = dict(getattr(inputs, "_context_usage", {}))
        if cad is not None:
            usage["context_aware_decoding"] = cad_processor.audit() if cad_processor else dict(enabled=False, alpha=cad.alpha, reference_sequences=0)
        reasoning_tokens = len(tokenizer.encode(reasoning, add_special_tokens=False)) if reasoning else 0
        action_text = raw if parse_error else (action_text if 'action_text' in locals() else raw)
        # A malformed completion remains in the policy sequence, but it has no JSON action segment.
        json_tokens = 0 if parse_error else (len(tokenizer.encode(action_text, add_special_tokens=False)) if action_text else 0)
        usage.update({"prompt_tokens": prompt_length, "completion_tokens": len(action_ids),
                      "thinking_enabled": self.config.enable_thinking,
                      "reasoning_tokens": reasoning_tokens, "json_action_tokens": json_tokens,
                      "action_completion": self.config.action_completion,
                      "first_json_stop_offset": first_json_end,
                      "first_json_tail_chars": len(first_json_tail),
                      "output_truncated": len(action_ids) >= self.config.max_new_tokens and not any(terminal_usage.values()) and first_json_end is None,
                      "generation_stop_reason": ("first_json" if first_json_end is not None else
                                                 "eos" if any(terminal_usage.values()) else
                                                 "length" if len(action_ids) >= self.config.max_new_tokens else "other"),
                      "thinking_parse_status": parse_status,
                      **terminal_usage,
                      "action_protocol": "think-json-v1" if self.config.enable_thinking else "json-v1"})
        return ModelResponse(
            content=content,
            tool_calls=[tool_call] if tool_call else [],
            prompt_token_ids=inputs["input_ids"][0].tolist(),
            prompt_attention_mask=inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))[0].tolist(),
            action_token_ids=action_ids.tolist(),
            old_logprobs=logprobs,
            action_mask=[1] * len(action_ids),
            parse_error=parse_error,
            usage=usage,
            raw_text=raw,
            reasoning_text=reasoning,
            reasoning_tokens=reasoning_tokens,
            json_action_tokens=json_tokens,
            action_protocol=usage["action_protocol"],
        )

    def _generate_batch_sync(self, requests: list[dict[str, Any]], batch_seed: int,
                             policy_version: str) -> list[ModelResponse]:
        """A single GPU generate call; row order and one batch RNG seed are explicit."""
        import math
        import torch
        from transformers import StoppingCriteriaList

        if not requests:
            raise ValueError("generate_batch requires at least one request")
        modes = []
        for request in requests:
            temperature = float(request.get("temperature", 0.0))
            sample = bool(request.get("sample", temperature > 0))
            if not math.isfinite(temperature) or (sample and temperature <= 0) or (not sample and temperature != 0):
                raise ValueError("batch sampling requires positive temperature; greedy requires temperature 0")
            if not isinstance(request.get("messages"), list) or not request["messages"]:
                raise ValueError("each batch request requires messages")
            modes.append((sample, temperature))
        if len(set(modes)) != 1:
            raise ValueError("one batch must use the same sampling mode and temperature")
        if self.config.enable_thinking or self.config.action_constraint != "none" or self.config.context_aware_decoding is not None:
            raise ValueError("generate_batch supports direct unconstrained actions without CAD")
        if self._model is None:
            self._load()
        torch.manual_seed(batch_seed)
        if self.config.deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        terminators = generation_terminator_ids(tokenizer)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = terminators[0] if terminators else None
        if pad_id is None or not terminators:
            raise ValueError("batch generation requires PAD and an EOS/message terminator")
        encoded = [self._inputs(request["messages"], "rollout_batch_sampling") for request in requests]
        width = max(item["input_ids"].shape[1] for item in encoded)
        ids = torch.full((len(encoded), width), pad_id, dtype=torch.long, device=self._model.device)
        masks = torch.zeros_like(ids)
        contexts = []
        for row, item in enumerate(encoded):
            length = item["input_ids"].shape[1]
            ids[row, -length:] = item["input_ids"][0]
            masks[row, -length:] = item.get("attention_mask", torch.ones_like(item["input_ids"]))[0]
            context = dict(getattr(item, "_context_usage", {}))
            context.update(unpadded_prompt_sha256=context.get("prompt_sha256"), left_padding_tokens=width - length,
                           padded_prompt_tokens=width, actual_prompt_tokens=int(masks[row].sum()),
                           prompt_sha256=hashlib.sha256(bytes(str(ids[row].tolist()), "utf-8")).hexdigest())
            contexts.append(context)
        sample, temperature = modes[0]
        stopper = _BatchActionStoppingCriteria(tokenizer, width, len(encoded),
                                               self.config.action_completion == "first-json-v1")
        generation = dict(max_new_tokens=self.config.max_new_tokens, do_sample=sample,
                          top_p=1.0, top_k=0, repetition_penalty=1.0,
                          eos_token_id=terminators, pad_token_id=pad_id,
                          stopping_criteria=StoppingCriteriaList([stopper]))
        if sample:
            generation["temperature"] = temperature
        positions = (masks.long().cumsum(-1) - 1).masked_fill(masks == 0, 0)
        self._model.config.use_cache = True
        with torch.inference_mode():
            output = self._model.generate(input_ids=ids, attention_mask=masks, position_ids=positions, **generation)
        if output.shape[0] != len(encoded) or not torch.equal(output[:, :width], ids):
            raise RuntimeError("batch generation changed the row count or prompt tokens")
        responses = []
        for row, request in enumerate(requests):
            all_actions = output[row, width:].tolist()
            stop_length = stopper.stop_lengths[row]
            if stop_length is None:
                # EOS is retained if it really ended the row; later batch PAD is not.
                eos_at = next((index for index, token in enumerate(all_actions) if token in terminators), None)
                stop_length = eos_at + 1 if eos_at is not None else len(all_actions)
            action_ids = all_actions[:stop_length]
            if not action_ids:
                raise RuntimeError("batch generation returned an empty action")
            raw = self._processor.decode(action_ids, skip_special_tokens=False)
            parse_ids, terminal_usage = trim_generation_terminators(action_ids, tokenizer)
            parse_text = self._processor.decode(parse_ids, skip_special_tokens=False)
            first_json_end = first_json_object_end(parse_text) if stopper.first_json else None
            tail = parse_text[first_json_end:] if first_json_end is not None else ""
            if first_json_end is not None:
                parse_text = parse_text[:first_json_end]
            try:
                _, action_text = split_reasoning_action(parse_text, False)
                content, tool_call = parse_agent_action(action_text)
                parse_error = None
            except ActionParseError as exc:
                content, tool_call, parse_error = "", None, str(exc)
            prompt_ids, prompt_mask = ids[row].tolist(), masks[row].tolist()
            logprobs = (self._score_sync(request["messages"], action_ids, False, "sampling_old_logprob",
                                         prompt_ids, prompt_mask).detach().cpu().tolist()
                        if self.config.recompute_old_logprobs else [])
            stop_reason = ("first_json" if first_json_end is not None else "eos" if any(terminal_usage.values())
                           else "length" if len(action_ids) >= self.config.max_new_tokens else "other")
            context = contexts[row]
            usage = dict(context)
            usage.update(context_usage=dict(context), prompt_tokens=int(masks[row].sum()),
                         completion_tokens=len(action_ids), thinking_enabled=False, reasoning_tokens=0,
                         json_action_tokens=0 if parse_error else len(tokenizer.encode(action_text, add_special_tokens=False)),
                         action_completion=self.config.action_completion, first_json_stop_offset=first_json_end,
                         first_json_tail_chars=len(tail), output_truncated=stop_reason == "length",
                         generation_stop_reason=stop_reason, thinking_parse_status="invalid_json_action" if parse_error else "valid",
                         action_protocol="json-v1", batch_seed=batch_seed, batch_row=row, batch_size=len(encoded),
                         policy_version=policy_version, sampling_temperature=temperature, sampling_top_p=1.0,
                         sampling_top_k=0, sampling_repetition_penalty=1.0, rng_mode="fixed-batch-row-order",
                         position_ids_mode="attention-mask-cumsum", prompt_position_ids=positions[row].tolist(),
                         **terminal_usage)
            responses.append(ModelResponse(content=content, tool_calls=[tool_call] if tool_call else [],
                prompt_token_ids=prompt_ids, prompt_attention_mask=prompt_mask, action_token_ids=action_ids,
                action_mask=[1] * len(action_ids), old_logprobs=logprobs, parse_error=parse_error,
                usage=usage, raw_text=raw, json_action_tokens=usage["json_action_tokens"], action_protocol="json-v1"))
        return responses

    async def generate_batch(self, requests: list[dict[str, Any]], batch_seed: int,
                             policy_version: str) -> list[ModelResponse]:
        """Serialize one real batched GPU call against single-request generation."""
        async with self._generation_lock:
            generation = asyncio.to_thread(self._generate_batch_sync, requests, batch_seed, policy_version)
            if self.config.timeout is None:
                return await generation
            return await asyncio.wait_for(generation, timeout=self.config.timeout)

    async def generate(self, messages: list[Message], tools: list[ToolDefinition], *, temperature: float = 0.0, seed: int | None = None) -> ModelResponse:
        # 同步 GPU 推理由线程承载，使现有异步 Agent runtime 不被阻塞。
        async with self._generation_lock:
            generation = asyncio.to_thread(self._generate_sync, messages, temperature, seed)
            if self.config.timeout is None:
                return await generation
            return await asyncio.wait_for(generation, timeout=self.config.timeout)

    @property
    def load_seconds(self) -> float | None:
        return self._load_seconds

    def gpu_memory_peak_gb(self) -> float | None:
        """返回当前进程记录的 CUDA 峰值显存，未使用 CUDA 时返回 None。"""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            return round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        except Exception:
            return None

    def gpu_memory_stats(self) -> dict[str, float | None]:
        """同时记录 allocated、reserved 与设备进程显存，避免单一指标误判。"""
        result = {"peak_allocated_gb": None, "peak_reserved_gb": None, "device_process_gb": None,
                  "device_memory_source": None}
        try:
            import torch
            if not torch.cuda.is_available(): return result
            result["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            result["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
            query = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            entries = []
            for line in query.stdout.splitlines():
                fields = [item.strip() for item in line.split(",")]
                if len(fields) == 2:
                    entries.append((fields[0], float(fields[1]) / 1024))
            own = next((memory for pid, memory in entries if pid == str(os.getpid())), None)
            if own is not None:
                result["device_process_gb"] = round(own, 3)
                result["device_memory_source"] = "pid_match"
            elif len(entries) == 1:
                # Container PID namespaces can hide the host PID. With exactly
                # one CUDA compute process, its device memory is unambiguous.
                result["device_process_gb"] = round(entries[0][1], 3)
                result["device_memory_source"] = "single_compute_process_fallback"
        except Exception:
            pass
        return result

    def reset_memory_stats(self) -> None:
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def reset_context_records(self) -> None:
        self._context_records.clear()

    def context_usage_summary(self) -> dict[str, Any]:
        """按调用阶段汇总结构化预算结果。"""
        phases: dict[str, dict[str, Any]] = {}
        for record in self._context_records:
            phase = str(record.get("phase", "unknown"))
            row = phases.setdefault(phase, {"calls": 0, "truncated": 0, "dropped_observation_tokens": 0, "observation_compacted": 0, "history_dropped": 0, "context_budget_errors": 0, "retained_tool_documents": 0, "lossless_expression_changed": 0, "lossy_body_trimmed": 0, "documents_deleted": 0})
            row["calls"] += 1
            row["truncated"] += int(bool(record.get("prompt_truncated")))
            row["dropped_observation_tokens"] += int(record.get("dropped_observation_tokens", 0) or 0)
            row["observation_compacted"] += int(bool(record.get("observation_compacted")))
            row["history_dropped"] += int(bool(record.get("history_dropped")))
            row["context_budget_errors"] += int(bool(record.get("context_budget_error")))
            row["retained_tool_documents"] += len(record.get("retained_tool_documents") or [])
            changes = record.get("document_changes") or []
            if isinstance(changes, dict):
                changes = [changes]
            row["lossless_expression_changed"] += int(bool(record.get("lossless_expression_changed")))
            row["lossy_body_trimmed"] += int(bool(record.get("lossy_body_trimmed")))
            row["documents_deleted"] += int(record.get("documents_deleted", 0) or 0)
        for row in phases.values():
            row["truncation_rate"] = row["truncated"] / row["calls"] if row["calls"] else 0.0
        return {"phases": phases, "records": list(self._context_records)}
