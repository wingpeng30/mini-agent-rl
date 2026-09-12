"""奖励函数：同时保留终局正确性、证据质量和行为成本。"""
from __future__ import annotations
import re
from typing import Protocol
from ..domain import Task, Transition

class RewardFunction(Protocol):
    def evaluate(self, task: Task, transitions: list[Transition], final_answer: str | None) -> dict[str, float]: ...

def norm(text: str | None) -> str: return re.sub(r"\s+", " ", (text or "").strip().lower())
class ExactMatchReward:
    def evaluate(self, task, transitions, final_answer): return {"exact_match": 1.0 if norm(task.answer) == norm(final_answer) else 0.0}
class EvidenceReward:
    def evaluate(self, task, transitions, final_answer):
        answer = norm(final_answer); return {"evidence_supported": 0.2 if any(answer and answer in norm(str(t.observation)) for t in transitions) else 0.0}
class SearchCostPenalty:
    def evaluate(self, task, transitions, final_answer): return {"search_cost": -0.03 * sum(1 for t in transitions if t.response.tool_call and t.response.tool_call.name == "search")}
class RepeatedSearchPenalty:
    def evaluate(self, task, transitions, final_answer):
        queries = [t.response.tool_call.arguments.get("query", "") for t in transitions if t.response.tool_call and t.response.tool_call.name == "search"]
        return {"repeated_search": -0.02 * (len(queries) - len(set(queries)))}
class InvalidActionPenalty:
    def evaluate(self, task, transitions, final_answer): return {"invalid_action": -0.1 * sum(1 for t in transitions if t.observation and t.observation.get("ok") is False)}
class CompositeReward:
    def __init__(self, functions: list[RewardFunction]): self.functions = functions
    def evaluate(self, task, transitions, final_answer):
        parts = {}
        for fn in self.functions: parts.update(fn.evaluate(task, transitions, final_answer))
        return parts
