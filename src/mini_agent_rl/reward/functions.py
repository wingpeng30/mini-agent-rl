"""奖励函数：同时保留终局正确性、证据质量和行为成本。"""
from __future__ import annotations
import re
from typing import Protocol
from ..domain import Task, Transition
from ..answer_scoring import normalize_answer, reward_token_f1

class RewardFunction(Protocol):
    def evaluate(self, task: Task, transitions: list[Transition], final_answer: str | None) -> dict[str, float]: ...

def norm(text: str | None) -> str: return re.sub(r"\s+", " ", (text or "").strip().lower())

def _support_is_complete(task: Task, transitions: list[Transition]) -> bool:
    """仅在奖励侧比较 gold supporting titles，绝不把它们泄漏给模型或检索工具。"""
    wanted = {norm(x) for x in task.metadata.get("support_titles", [])}
    found = {
        norm(doc.get("title")) for transition in transitions
        for doc in ((transition.observation or {}).get("result") or []) if isinstance(doc, dict)
    }
    return bool(wanted) and wanted <= found
class ExactMatchReward:
    def __init__(self, weight: float = 1.0): self.weight = weight
    def evaluate(self, task, transitions, final_answer): return {"exact_match": self.weight if norm(task.answer) == norm(final_answer) else 0.0}
class EvidenceReward:
    def evaluate(self, task, transitions, final_answer):
        answer = norm(final_answer); return {"evidence_supported": 0.2 if any(answer and answer in norm(str(t.observation)) for t in transitions) else 0.0}


class AnswerTokenF1Reward:
    """Bounded partial-answer reward for a legal terminal answer only."""
    def __init__(self, weight: float = 0.20): self.weight = weight
    def evaluate(self, task, transitions, final_answer):
        answered = any(t.termination_reason == "answer" for t in transitions)
        score = reward_token_f1(final_answer, task.answer) if answered and norm(final_answer) else 0.0
        return {"answer_token_f1": self.weight * score}
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
    def __init__(self, penalty: float = -0.5): self.penalty = penalty
    def evaluate(self, task, transitions, final_answer):
        return {"repeated_search": self.penalty if any(t.termination_reason == "repeated_action" for t in transitions) else 0.0}
class InvalidActionPenalty:
    def evaluate(self, task, transitions, final_answer): return {"invalid_action": -0.5 if any(t.termination_reason == "invalid_action" for t in transitions) else 0.0}
class MaxStepsPenalty:
    def evaluate(self, task, transitions, final_answer): return {"max_steps": -0.25 if any(t.termination_reason == "max_steps" for t in transitions) else 0.0}
class UngroundedAnswerPenalty:
    def evaluate(self, task, transitions, final_answer):
        searched = any(t.response.tool_call and t.response.tool_call.name == "search" for t in transitions)
        return {"ungrounded_answer": -0.1 if not searched and norm(final_answer) != norm(task.answer) else 0.0}
class OverSearchAfterEvidencePenalty:
    """把“证据已齐仍搜索”作为即时策略错误，既保留轨迹也纳入终局总奖励。"""
    def __init__(self, penalty: float = -0.2): self.penalty = penalty
    def evaluate(self, task, transitions, final_answer):
        count = sum("over_search_after_evidence" in transition.reward_tags for transition in transitions)
        return {"over_search_after_evidence": self.penalty * count}
class EvidenceStopReward:
    """仅在答案正确且已检索到支持文档时奖励停止，避免奖励凭空抢答。"""
    def __init__(self, weight: float = 0.3): self.weight = weight
    def evaluate(self, task, transitions, final_answer):
        correct = norm(final_answer) == norm(task.answer)
        return {"evidence_stop": self.weight if correct and _support_is_complete(task, transitions) else 0.0}

class IncompleteEvidenceAnswerPenalty:
    """惩罚已检索但证据尚不完整时的最终抢答。

    supporting titles only appear in the post-rollout reward calculation.  They
    are never appended to the policy context, tool result, or retrieval query.
    """
    def __init__(self, penalty: float = -0.15): self.penalty = penalty
    def evaluate(self, task, transitions, final_answer):
        searched = any(t.response.tool_call and t.response.tool_call.name == "search" for t in transitions)
        answered = bool(final_answer and norm(final_answer))
        return {"incomplete_evidence_answer": self.penalty if searched and answered and not _support_is_complete(task, transitions) else 0.0}

class EmptyEvidenceAnswerPenalty:
    """Discourage answering after every executed search returned no document."""
    def __init__(self, penalty: float = -0.10): self.penalty = penalty
    def evaluate(self, task, transitions, final_answer):
        searched = [t for t in transitions if t.response.tool_call and t.response.tool_call.name == "search"]
        all_empty = bool(searched) and all(not ((t.observation or {}).get("result") or []) for t in searched)
        answered = bool(final_answer and norm(final_answer))
        return {"empty_evidence_answer": self.penalty if answered and all_empty else 0.0}
class CompositeReward:
    def __init__(self, functions: list[RewardFunction], version: str = "v096"):
        self.functions, self.version = functions, version
    def evaluate(self, task, transitions, final_answer):
        parts = {}
        for fn in self.functions: parts.update(fn.evaluate(task, transitions, final_answer))
        return parts

    def diagnostics(self, task, transitions, final_answer) -> dict[str, float]:
        """Non-reward observability; values never enter advantage calculation."""
        answer = norm(final_answer)
        hit = any(answer and answer in norm(str(t.observation)) for t in transitions)
        return {"answer_string_hit": float(hit)}
