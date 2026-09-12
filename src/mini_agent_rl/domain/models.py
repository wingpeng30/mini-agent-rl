"""领域模型：这些对象是 Agent 运行时和未来 RL trainer 的稳定边界。"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field

def now() -> datetime:
    return datetime.now(timezone.utc)

class RolloutStatus(str, Enum):
    QUEUED = "QUEUED"; RUNNING = "RUNNING"; SUCCEEDED = "SUCCEEDED"; FAILED = "FAILED"; CANCELLED = "CANCELLED"

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class Message(BaseModel):
    role: str
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)

class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid4().hex}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class ModelResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prompt_token_ids: list[int] | None = None
    action_token_ids: list[int] | None = None
    old_logprobs: list[float] | None = None
    usage: dict[str, int | float] = Field(default_factory=dict)

    @property
    def tool_call(self) -> ToolCall | None:
        """第一版运行时每一步只执行一个工具；保留便捷访问避免调用方误用列表。"""
        return self.tool_calls[0] if self.tool_calls else None

class Transition(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    rollout_id: str
    step: int
    messages: list[Message]
    response: ModelResponse
    observation: dict[str, Any] | None = None
    reward: float = 0.0
    terminated: bool = False

class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    rollout_id: str
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now)

class Rollout(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    group_id: str
    task_id: str
    seed: int
    status: RolloutStatus = RolloutStatus.QUEUED
    reward: float = 0.0
    reward_components: dict[str, float] = Field(default_factory=dict)
    final_answer: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

class RolloutGroup(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    rollout_ids: list[str] = Field(default_factory=list)
    rewards: list[float] = Field(default_factory=list)
    mean_reward: float = 0.0
    std_reward: float = 0.0
    success_rate: float = 0.0
