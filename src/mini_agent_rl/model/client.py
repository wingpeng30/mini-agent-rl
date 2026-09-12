"""模型适配层：Agent 不依赖具体云模型或本地模型。"""

from __future__ import annotations
import asyncio
import ast
import json
import os
from typing import Awaitable, Callable, Protocol
import httpx
from ..domain import Message, ModelResponse, ToolCall, ToolDefinition

class ModelClient(Protocol):
    async def generate(self, messages: list[Message], tools: list[ToolDefinition], *, temperature: float = 0.0, seed: int | None = None) -> ModelResponse: ...

class FakeModelClient:
    """离线确定性模型：根据问题和已检索内容生成搜索或回答动作。"""
    def __init__(self, answer: str | None = None): self.answer = answer
    async def generate(self, messages, tools, *, temperature=0.0, seed=None):
        question = next((m.content for m in messages if m.role == "user"), "")
        observations = [m.content for m in messages if m.role == "tool"]
        if not observations:
            return ModelResponse(tool_call=ToolCall(name="search", arguments={"query": question}))
        if self.answer:
            answer = self.answer
        else:
            # 工具返回的是包装后的 observation；演示模型应提取检索到的
            # 文档文本作为最终回答，而不是把整个包装字典原样输出。
            raw = observations[-1]
            if isinstance(raw, str):
                # Runtime 为了写入标准 Message，会把 observation 序列化成字符串；
                # FakeModel 在离线演示中将其恢复为结构化对象再提取文档文本。
                try:
                    raw = ast.literal_eval(raw)
                except (SyntaxError, ValueError):
                    pass
            answer = raw
            if isinstance(raw, dict):
                result = raw.get("result", raw)
                if isinstance(result, str):
                    try:
                        result = ast.literal_eval(result)
                    except (SyntaxError, ValueError):
                        pass
                if isinstance(result, list) and result:
                    answer = result[0].get("text", str(result[0]))
                else:
                    answer = str(result)
        return ModelResponse(content=answer)

class OpenAICompatibleClient:
    """OpenAI-compatible 客户端；token/logprob 为空时仍可运行 RL 数据采集。"""
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60.0, retries: int = 2):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout, self.retries = timeout, retries
    async def generate(self, messages, tools, *, temperature=0.0, seed=None):
        payload = {"model": self.model, "messages": [m.model_dump(exclude_none=True) for m in messages], "temperature": temperature}
        if seed is not None: payload["seed"] = seed
        if tools: payload["tools"] = [{"type": "function", "function": t.model_dump()} for t in tools]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        last = None
        for attempt in range(self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                    response.raise_for_status(); data = response.json()
                choice = data["choices"][0]["message"]
                calls = choice.get("tool_calls") or []
                call = calls[0].get("function") if calls else None
                return ModelResponse(content=choice.get("content") or "", tool_call=ToolCall(name=call["name"], arguments=json.loads(call.get("arguments", "{}"))) if call else None, usage=data.get("usage", {}))
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last = exc
                if attempt < self.retries: await asyncio.sleep(0.1 * (attempt + 1))
        raise RuntimeError(f"模型请求失败: {last}")
