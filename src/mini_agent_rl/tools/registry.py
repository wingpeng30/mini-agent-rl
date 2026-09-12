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
            result = self.handler(**arguments)
            if inspect.isawaitable(result): result = await asyncio.wait_for(result, timeout=self.timeout)
            text = str(result); return {"ok": True, "result": text[:self.max_output], "latency_ms": round((time.perf_counter()-start)*1000, 2)}
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
