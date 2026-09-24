"""领域模型：这些对象是 Agent 运行时和未来 RL trainer 的稳定边界。"""

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
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
    # 预算后的真实输入。训练时重放这组 token，而不是重新渲染消息后碰运气得到相同输入。
    prompt_attention_mask: list[int] | None = None
    action_token_ids: list[int] | None = None
    old_logprobs: list[float] | None = None
    action_mask: list[int] | None = None
    parse_error: str | None = None
    # 上下文预算包含数值、布尔和阶段标签，统一保留为可 JSON 序列化字典。
    usage: dict[str, Any] = Field(default_factory=dict)
    raw_text: str | None = None
    # 思考与最终动作分开审计；完整 action_token_ids 仍覆盖二者，供策略重放使用。
    reasoning_text: str | None = None
    reasoning_tokens: int = 0
    json_action_tokens: int = 0
    action_protocol: str = "json-v1"

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
    # 即时行为标签用于区分“策略在本步做错了什么”，不能只依赖最终终止原因。
    reward_tags: list[str] = Field(default_factory=list)
    terminated: bool = False
    termination_reason: str | None = None

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
    # Observability values are kept separate so they cannot affect reward sum or advantage.
    reward_diagnostics: dict[str, float] = Field(default_factory=dict)
    # 记录产生该 reward 的配置，供未来独立审计；不含 gold answer 或私密内容。
    reward_config: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = None
    error: str | None = None
    termination_reason: Literal["answer", "invalid_action", "repeated_action", "max_steps", "cancelled", "model_error", "tool_error"] | None = None
    advantage: float | None = None
    eligible_for_rl: bool = True
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
    zero_variance: bool = False
    # 指纹绑定采集时实际使用的策略，训练前可拒绝“数据与权重不属于同一策略”的误用。
    policy_fingerprint: dict[str, Any] = Field(default_factory=dict)
    # 环境指纹绑定任务顺序、固定 corpus 和检索器版本，防止训练与回放使用不同环境。
    environment_fingerprint: dict[str, Any] = Field(default_factory=dict)
