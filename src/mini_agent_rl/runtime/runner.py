"""Agent loop、策略终止语义和 rollout group 编排。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
from pathlib import Path

from ..domain import Event, Message, Rollout, RolloutGroup, RolloutStatus, Task, Transition, now
from ..model import ModelClient
from ..reward import RewardFunction
from ..storage import SQLiteStore
from ..tools import ToolRegistry
from ..reward.functions import _support_is_complete
from ..prompts import PROMPT_VERSION, system_prompt


def _query_key(query: str) -> str:
    """搜索判重使用稳定规范化，不因大小写或连续空白绕过环境边界。"""
    return " ".join(query.strip().lower().split())


class AgentRunner:
    def __init__(self, model: ModelClient, tools: ToolRegistry, reward: RewardFunction, store: SQLiteStore,
                 max_steps: int = 5, max_concurrency: int = 4, temperature: float = 0.0,
                 anti_repeat_prompt: bool = True, base_seed: int = 0,
                 environment_fingerprint: dict[str, object] | None = None,
                 enable_thinking: bool = False, prompt_version: str = PROMPT_VERSION,
                 prompt_variant: str = "direct-v090"):
        self.model, self.tools, self.reward, self.store = model, tools, reward, store
        self.max_steps, self.max_concurrency = max_steps, max_concurrency
        self.temperature, self.anti_repeat_prompt = temperature, anti_repeat_prompt
        self.base_seed = base_seed
        self.environment_fingerprint = environment_fingerprint or {}
        self.enable_thinking, self.prompt_version, self.prompt_variant = enable_thinking, prompt_version, prompt_variant
        system_prompt(enable_thinking, prompt_variant)
        over_search = next((item for item in getattr(reward, "functions", [])
                            if type(item).__name__ == "OverSearchAfterEvidencePenalty"), None)
        self.over_search_penalty = float(getattr(over_search, "penalty", -0.2))

    def _reward_config(self) -> dict[str, object]:
        """把可组合 reward 的公开权重落库，历史没有该字段的轨迹由审计标为不可验证。"""
        functions = getattr(self.reward, "functions", [self.reward])
        result = {type(item).__name__: {key: value for key, value in vars(item).items() if isinstance(value, (int, float, str, bool))}
                  for item in functions}
        implementation = Path(__file__).resolve().parents[1] / "reward" / "functions.py"
        result["_meta"] = {"reward_version": getattr(self.reward, "version", "v096"),
                           "implementation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest()}
        return result

    def _messages(self, task: Task) -> list[Message]:
        prompt = system_prompt(getattr(self, "enable_thinking", False), getattr(self, "prompt_variant", "direct-v090"))
        if self.anti_repeat_prompt:
            prompt += ' Do not repeat an executed query.'
        return [Message(role="system", content=prompt), Message(role="user", content=task.question)]

    def protocol_fingerprint(self) -> dict[str, object]:
        """Bind reports to the rendered system protocol, not only its label."""
        variant = getattr(self, "prompt_variant", "direct-v090")
        prompt = system_prompt(self.enable_thinking, variant)
        if self.anti_repeat_prompt:
            prompt += ' Do not repeat an executed query.'
        return {
            "prompt_version": self.prompt_version,
            "prompt_variant": variant,
            "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "enable_thinking": self.enable_thinking,
            "anti_repeat_prompt": self.anti_repeat_prompt,
            "max_steps": self.max_steps,
        }

    async def run_rollout(self, task: Task, group_id: str, seed: int = 0) -> Rollout:
        rollout = Rollout(group_id=group_id, task_id=task.id, seed=seed, status=RolloutStatus.RUNNING)
        rollout.started_at = now(); self.store.save_rollout(rollout)
        self.store.append_event(Event(rollout_id=rollout.id, type="rollout_started"))
        messages, transitions, seen = self._messages(task), [], set()
        try:
            for step in range(self.max_steps):
                response = await self.model.generate(messages, self.tools.definitions(), temperature=self.temperature, seed=seed + step)
                self.store.append_event(Event(rollout_id=rollout.id, type="model_request", payload={"step": step, "response": response.model_dump(mode="json")}))
                if response.parse_error:
                    transition = Transition(rollout_id=rollout.id, step=step, messages=messages.copy(), response=response,
                                            reward=-0.5, terminated=True, termination_reason="invalid_action")
                    transitions.append(transition); self.store.save_transition(transition)
                    rollout.termination_reason = "invalid_action"; break
                if response.tool_call:
                    key = (response.tool_call.name, _query_key(str(response.tool_call.arguments.get("query", ""))))
                    if key in seen:
                        transition = Transition(rollout_id=rollout.id, step=step, messages=messages.copy(), response=response,
                                                reward=-0.5, terminated=True, termination_reason="repeated_action")
                        transitions.append(transition); self.store.save_transition(transition)
                        rollout.termination_reason = "repeated_action"; break
                    seen.add(key)
                    observation = await self.tools.execute(response.tool_call.name, response.tool_call.arguments)
                    # 将已搜索列表放入 tool observation，而非在中途插入 system message；
                    # Qwen chat template 要求 system 只能位于消息开头。
                    observation["searched_queries"] = sorted(item[1] for item in seen)
                    # 支持证据已完整出现时继续搜索并非基础设施故障；记录为可学习的即时负行为。
                    # 判断发生在执行前，确保本步的 action 才是被归因的对象。
                    over_search = response.tool_call.name == "search" and _support_is_complete(task, transitions)
                    transition = Transition(rollout_id=rollout.id, step=step, messages=messages.copy(), response=response, observation=observation,
                                            reward=self.over_search_penalty if over_search else 0.0,
                                            reward_tags=["over_search_after_evidence"] if over_search else [])
                    transitions.append(transition); self.store.save_transition(transition)
                    self.store.append_event(Event(rollout_id=rollout.id, type="tool_call", payload={"name": response.tool_call.name, "arguments": response.tool_call.arguments}))
                    if not observation.get("ok"):
                        transition.terminated = True; transition.termination_reason = "tool_error"; self.store.save_transition(transition)
                        rollout.status, rollout.eligible_for_rl, rollout.termination_reason = RolloutStatus.FAILED, False, "tool_error"; break
                    # tool call 也属于策略动作，必须以真实 JSON 文本进入下一轮上下文。
                    # 历史仅回填可执行 JSON，不让思考文字逐轮消耗上下文预算。
                    action_content = json.dumps(
                        {"action": "search", "query": response.tool_call.arguments.get("query", "")}, ensure_ascii=False
                    )
                    # latency_ms 是本机执行噪声，不是任务证据；保留在 Transition.observation 供审计，
                    # 但不让它改变下一轮模型输入长度或同 seed 的离散轨迹。
                    model_observation = {key: value for key, value in observation.items() if key != "latency_ms"}
                    messages += [Message(role="assistant", content=action_content, tool_calls=[response.tool_call]),
                                 Message(role="tool", content=json.dumps(model_observation, ensure_ascii=False), tool_call_id=response.tool_call.id)]
                    continue
                rollout.final_answer = response.content
                transition = Transition(rollout_id=rollout.id, step=step, messages=messages.copy(), response=response, terminated=True, termination_reason="answer")
                transitions.append(transition); self.store.save_transition(transition)
                rollout.termination_reason = "answer"; break
            else:
                if transitions:
                    transitions[-1].terminated, transitions[-1].termination_reason, transitions[-1].reward = True, "max_steps", -0.25
                    self.store.save_transition(transitions[-1])
                rollout.termination_reason = "max_steps"
            if rollout.status == RolloutStatus.RUNNING:
                parts = self.reward.evaluate(task, transitions, rollout.final_answer)
                rollout.reward_components, rollout.reward, rollout.status = parts, sum(parts.values()), RolloutStatus.SUCCEEDED
                diagnostics = getattr(self.reward, "diagnostics", None)
                rollout.reward_diagnostics = diagnostics(task, transitions, rollout.final_answer) if callable(diagnostics) else {}
                rollout.reward_config = self._reward_config()
                self.store.append_event(Event(rollout_id=rollout.id, type="reward", payload={"total": rollout.reward, "components": parts}))
        except asyncio.CancelledError:
            rollout.status, rollout.termination_reason, rollout.eligible_for_rl = RolloutStatus.CANCELLED, "cancelled", False
        except Exception as exc:
            rollout.status, rollout.termination_reason, rollout.eligible_for_rl, rollout.error = RolloutStatus.FAILED, "model_error", False, str(exc)
            self.store.append_event(Event(rollout_id=rollout.id, type="error", payload={"error": str(exc)}))
        finally:
            rollout.finished_at = now(); self.store.save_rollout(rollout)
            self.store.append_event(Event(rollout_id=rollout.id, type="rollout_finished", payload={"status": rollout.status.value, "reason": rollout.termination_reason, "error": rollout.error}))
        return rollout

    async def run_group(self, task: Task, group_size: int = 4) -> RolloutGroup:
        group = RolloutGroup(task_id=task.id); self.store.save_task(task); self.store.save_group(group)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        async def bounded(seed: int):
            async with semaphore: return await self.run_rollout(task, group.id, seed)
        results = await asyncio.gather(*(bounded(self.base_seed + seed) for seed in range(group_size)))
        eligible = [item for item in results if item.eligible_for_rl]
        rewards = [item.reward for item in eligible]
        group.rollout_ids, group.rewards = [item.id for item in results], rewards
        group.mean_reward = statistics.mean(rewards) if rewards else 0.0
        group.std_reward = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        group.zero_variance = group.std_reward == 0.0
        for item in results:
            item.advantage = 0.0 if not item.eligible_for_rl or group.zero_variance else (item.reward - group.mean_reward) / (group.std_reward + 1e-8)
            self.store.save_rollout(item)
        group.success_rate = sum(item.status == RolloutStatus.SUCCEEDED for item in results) / len(results) if results else 0.0
        # 模型在第一次 generate 后才懒加载；因此在组结束时读取指纹才能如实记录环境。
        fingerprint = getattr(self.model, "policy_fingerprint", None)
        group.policy_fingerprint = fingerprint() if callable(fingerprint) else {}
        group.environment_fingerprint = dict(self.environment_fingerprint)
        self.store.save_group(group); return group
