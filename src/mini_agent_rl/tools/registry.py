"""工具注册表：schema 是模型和环境之间的安全边界。"""
from __future__ import annotations
import asyncio, inspect, time
from typing import Any, Awaitable, Callable
from pydantic import ValidationError
from ..domain import ToolDefinition

class Tool:
    def __init__(self, definition: ToolDefinition, handler: Callable[..., Any], timeout: float = 10.0, max_output: int = 4000): self.definition, self.handler, self.timeout, self.max_output = definition, handler, timeout, max_output
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            # Tool schema 是模型和环境之间的边界。首版实现 JSON Schema 的
            # 必需字段与基本类型校验，避免模型任意参数直接进入工具函数。
            schema = self.definition.parameters
            if schema.get("type") == "object":
                if not isinstance(arguments, dict):
                    raise ValueError("工具参数必须是 JSON object")
                for key in schema.get("required", []):
                    if key not in arguments:
                        raise ValueError(f"缺少必需参数: {key}")
                for key, value in arguments.items():
                    expected = schema.get("properties", {}).get(key, {}).get("type")
                    if expected == "string" and not isinstance(value, str):
                        raise ValueError(f"参数 {key} 必须是字符串")
                    if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                        raise ValueError(f"参数 {key} 必须是数字")
            result = self.handler(**arguments)
            if inspect.isawaitable(result): result = await asyncio.wait_for(result, timeout=self.timeout)
            # JSON 结构的工具输出必须原样保留，供后续模型消息、轨迹回放和
            # evidence reward 使用；只有普通文本才执行长度截断。
            normalized = result if isinstance(result, (dict, list, int, float, bool, type(None))) else str(result)[:self.max_output]
            return {"ok": True, "result": normalized, "latency_ms": round((time.perf_counter()-start)*1000, 2)}
        except Exception as exc: return {"ok": False, "error": str(exc), "latency_ms": round((time.perf_counter()-start)*1000, 2)}

class ToolRegistry:
    def __init__(self): self._tools: dict[str, Tool] = {}
    def register(self, tool: Tool):
        if tool.definition.name in self._tools: raise ValueError(f"重复工具: {tool.definition.name}")
        self._tools[tool.definition.name] = tool
    def definitions(self) -> list[ToolDefinition]: return [t.definition for t in self._tools.values()]
    async def execute(self, name: str, arguments: dict[str, Any]):
        if name not in self._tools: return {"ok": False, "error": f"未知工具: {name}"}
        return await self._tools[name].execute(arguments)
