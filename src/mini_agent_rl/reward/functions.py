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
class SupportCoverageReward:
    """支持文档标题只用于环境后的奖励，绝不进入模型上下文。"""
    def evaluate(self, task, transitions, final_answer):
        wanted = {norm(x) for x in task.metadata.get("support_titles", [])}
        found = {
            norm(doc.get("title")) for transition in transitions
            for doc in ((transition.observation or {}).get("result") or []) if isinstance(doc, dict)
        }
        return {"support_coverage": 0.2 * len(wanted & found) / len(wanted) if wanted else 0.0}
class SearchCostPenalty:
    def evaluate(self, task, transitions, final_answer): return {"search_cost": -0.03 * sum(1 for t in transitions if t.response.tool_call and t.response.tool_call.name == "search")}
class RepeatedSearchPenalty:
    def evaluate(self, task, transitions, final_answer):
        return {"repeated_search": -0.5 if any(t.termination_reason == "repeated_action" for t in transitions) else 0.0}
class InvalidActionPenalty:
    def evaluate(self, task, transitions, final_answer): return {"invalid_action": -0.5 if any(t.termination_reason == "invalid_action" for t in transitions) else 0.0}
class MaxStepsPenalty:
    def evaluate(self, task, transitions, final_answer): return {"max_steps": -0.25 if any(t.termination_reason == "max_steps" for t in transitions) else 0.0}
class UngroundedAnswerPenalty:
    def evaluate(self, task, transitions, final_answer):
        searched = any(t.response.tool_call and t.response.tool_call.name == "search" for t in transitions)
        return {"ungrounded_answer": -0.1 if not searched and norm(final_answer) != norm(task.answer) else 0.0}
class CompositeReward:
    def __init__(self, functions: list[RewardFunction]): self.functions = functions
    def evaluate(self, task, transitions, final_answer):
        parts = {}
        for fn in self.functions: parts.update(fn.evaluate(task, transitions, final_answer))
        return parts
