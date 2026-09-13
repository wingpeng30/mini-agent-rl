"""本地 Transformers 模型适配器；导入本模块不会下载模型或要求 CUDA。"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import hashlib
from pathlib import Path
import time
import threading
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..domain import Message, ModelResponse, ToolCall, ToolDefinition


class LocalModelConfig(BaseModel):
    """本地推理配置。模型 ID 只是配置，构造配置对象不会访问网络。"""

    model_id: str = r"D:\qwen_08b"
    adapter_path: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    max_new_tokens: int = Field(default=128, ge=16, le=2048)
    max_input_tokens: int = Field(default=2048, ge=128, le=32768)
    timeout: float = Field(default=120.0, gt=0)
    trust_remote_code: bool = False
    local_files_only: bool = True
    enable_thinking: bool = False
    deterministic: bool = True
    # bf16 是实际采样默认值；fp32 仅用于 v0.6.1 的跨加载数值诊断。
    logprob_mode: Literal["bf16", "fp32"] = "bf16"
    recompute_old_logprobs: bool = True

    @model_validator(mode="after")
    def validate_quantization(self):
        # bitsandbytes 在原生 Windows 上兼容性不稳定，因此量化必须显式开启，不能静默选择。
        if self.load_in_4bit and self.device == "cpu":
            raise ValueError("CPU 模式不支持本项目的 4-bit 加载")
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

    def _inputs(self, messages: list[Message]):
        """由同一 chat template 编码 prompt，避免采集与训练 scorer 的 token 边界漂移。"""
        chat = [{"role": m.role, "content": m.content} for m in messages]
        template_kwargs = {"chat_template_kwargs": {"enable_thinking": True}} if self.config.enable_thinking else {}
        encoded = self._processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True,
                                                       return_dict=True, return_tensors="pt", **template_kwargs)
        # 直接在渲染完成后保留末尾 token，避免把 truncation 作为未知参数传给 Qwen processor；
        # 末尾包含 generation prompt，且训练/推理两条路径使用同一截断规则。
        original_length = int(encoded["input_ids"].shape[1])
        truncated = original_length > self.config.max_input_tokens
        if truncated:
            encoded["input_ids"] = encoded["input_ids"][:, -self.config.max_input_tokens:]
            if "attention_mask" in encoded:
                encoded["attention_mask"] = encoded["attention_mask"][:, -self.config.max_input_tokens:]
        encoded = encoded.to(self._model.device)
        # 元数据挂在 BatchEncoding 对象上，不能放进字典，否则 generate 会把它
        # 当成模型 forward 参数并报“未使用的 model_kwargs”。
        encoded._original_prompt_tokens = original_length
        encoded._prompt_truncated = truncated
        return encoded

    def _score_sync(self, messages: list[Message], action_token_ids: list[int], with_grad: bool = False):
        """以原始策略 teacher-forcing 计算 action logprob。

        prompt 与 tool observation 只用于条件化，gather 的位置严格只覆盖 assistant action，
        因而它们永远不会进入 policy loss。
        """
        import torch
        if self._model is None:
            self._load()
        inputs = self._inputs(messages)
        prompt_length = inputs["input_ids"].shape[1]
        action = torch.tensor([action_token_ids], dtype=torch.long, device=self._model.device)
        full_ids = torch.cat([inputs["input_ids"], action], dim=1)
        full_mask = torch.ones_like(full_ids)
        context = torch.enable_grad() if with_grad else torch.inference_mode()
        with context:
            logits = self._model(input_ids=full_ids, attention_mask=full_mask).logits
            positions = logits[0, prompt_length - 1 : prompt_length - 1 + len(action_token_ids)]
            # 无论 backbone 为 BF16 还是 FP32，softmax 统一在 FP32 中进行，减小数值误差。
            values = torch.log_softmax(positions.float(), dim=-1).gather(1, action[0].unsqueeze(1)).squeeze(1)
        return values

    async def score_action(self, messages: list[Message], action_token_ids: list[int]) -> list[float]:
        """诊断/审计用无梯度 scorer；返回值不用于反向传播。"""
        async with self._generation_lock:
            values = await asyncio.to_thread(self._score_sync, messages, action_token_ids, False)
        return values.detach().cpu().tolist()

    def score_action_tensor(self, messages: list[Message], action_token_ids: list[int], with_grad: bool = False):
        """训练器同步调用的 scorer；with_grad=True 时保留到 LoRA 参数的计算图。"""
        return self._score_sync(messages, action_token_ids, with_grad)

    def reference_logprobs(self, messages: list[Message], action_token_ids: list[int], reference_state: dict[str, Any] | None = None):
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
                return self._score_sync(messages, action_token_ids, False)
            finally:
                with torch.no_grad():
                    for name, parameter in self._model.named_parameters():
                        if name in saved:
                            parameter.copy_(saved[name])
        if not hasattr(self._model, "disable_adapter"):
            raise RuntimeError("GRPO reference 需要已加载 PEFT LoRA adapter")
        with self._model.disable_adapter():
            return self._score_sync(messages, action_token_ids, False)

    def prepare_for_training(self):
        """只开启 LoRA 训练，基础模型/视觉塔始终冻结。"""
        if self._model is None:
            self._load()
        self._model.config.use_cache = False
        # Qwen3.5 的完整语言模型被冻结，checkpoint 只保存 LoRA 反向所需的中间状态，
        # 并让输入 embedding 保留梯度路径；这是单张 8GB 卡运行 GRPO 的关键显存开关。
        if hasattr(self._model, "gradient_checkpointing_enable"):
            self._model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(self._model, "enable_input_require_grads"):
            self._model.enable_input_require_grads()
        self._model.train()
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
                "generation": {"max_new_tokens": self.config.max_new_tokens, "thinking": self.config.enable_thinking,
                               "deterministic": self.config.deterministic, "logprob_mode": self.config.logprob_mode}}

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
        inputs = self._inputs(messages)
        do_sample = temperature > 0
        generation = {"max_new_tokens": self.config.max_new_tokens, "do_sample": do_sample}
        if do_sample:
            generation["temperature"] = temperature
        with torch.inference_mode():
            output = self._model.generate(**inputs, **generation)
        prompt_length = inputs["input_ids"].shape[1]
        action_ids = output[0, prompt_length:]
        raw = self._processor.decode(action_ids, skip_special_tokens=True)
        # generate 的 score 可能已被 temperature 或 logits processor 改写；RL 的
        # old logprob 必须来自原始策略，因此对 prompt+action 做一次 teacher-forced 前向。
        logprobs = self._score_sync(messages, action_ids.tolist(), False).detach().cpu().tolist() if self.config.recompute_old_logprobs else []
        try:
            content, tool_call = parse_agent_action(raw)
            parse_error = None
        except ActionParseError as exc:
            content, tool_call, parse_error = "", None, str(exc)
        return ModelResponse(
            content=content,
            tool_calls=[tool_call] if tool_call else [],
            prompt_token_ids=inputs["input_ids"][0].tolist(),
            action_token_ids=action_ids.tolist(),
            old_logprobs=logprobs,
            action_mask=[1] * len(action_ids),
            parse_error=parse_error,
            usage={"prompt_tokens": prompt_length, "completion_tokens": len(action_ids),
                   "original_prompt_tokens": int(getattr(inputs, "_original_prompt_tokens", prompt_length)),
                   "prompt_truncated": bool(getattr(inputs, "_prompt_truncated", False))},
            raw_text=raw,
        )

    async def generate(self, messages: list[Message], tools: list[ToolDefinition], *, temperature: float = 0.0, seed: int | None = None) -> ModelResponse:
        # 同步 GPU 推理由线程承载，使现有异步 Agent runtime 不被阻塞。
        async with self._generation_lock:
            return await asyncio.wait_for(asyncio.to_thread(self._generate_sync, messages, temperature, seed), timeout=self.config.timeout)

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
