"""Agent loop 和 rollout 编排。"""
from __future__ import annotations
import asyncio, json, statistics
from datetime import datetime, timezone
from ..domain import *
from ..model import ModelClient
from ..tools import ToolRegistry
from ..reward import RewardFunction
from ..storage import SQLiteStore

class AgentRunner:
    def __init__(self, model: ModelClient, tools: ToolRegistry, reward: RewardFunction, store: SQLiteStore, max_steps: int = 5, max_concurrency: int = 4): self.model, self.tools, self.reward, self.store, self.max_steps, self.max_concurrency = model, tools, reward, store, max_steps, max_concurrency
    async def run_rollout(self, task: Task, group_id: str, seed: int = 0) -> Rollout:
        rollout = Rollout(group_id=group_id, task_id=task.id, seed=seed); rollout.status = RolloutStatus.RUNNING; rollout.started_at = now(); self.store.save_rollout(rollout); self.store.append_event(Event(rollout_id=rollout.id, type="rollout_started"))
        messages = [Message(role="system", content="你是一个只使用已提供工具的检索 Agent。"), Message(role="user", content=task.question)]; transitions=[]; seen=set()
        try:
            for step in range(self.max_steps):
                response = await self.model.generate(messages, self.tools.definitions(), seed=seed)
                self.store.append_event(Event(rollout_id=rollout.id, type="model_request", payload={"step": step, "response": response.model_dump(mode="json")}))
                if response.tool_call:
                    key = (response.tool_call.name, tuple(sorted(response.tool_call.arguments.items())))
                    if key in seen: raise RuntimeError("检测到连续重复工具动作")
                    seen.add(key); observation = await self.tools.execute(response.tool_call.name, response.tool_call.arguments)
                    transition = Transition(rollout_id=rollout.id, step=step, messages=messages.copy(), response=response, observation=observation)
                    # 真实 OpenAI-compatible API 要求 tool message 关联上一次
                    # assistant tool_call 的 ID；JSON 观察结果不经过不安全的 repr/ast 往返。
                    messages += [Message(role="assistant", content=response.content, tool_calls=[response.tool_call]), Message(role="tool", content=json.dumps(observation, ensure_ascii=False), tool_call_id=response.tool_call.id)]
                    self.store.append_event(Event(rollout_id=rollout.id, type="tool_call", payload={"name": response.tool_call.name, "arguments": response.tool_call.arguments})); transitions.append(transition); self.store.save_transition(transition); continue
                rollout.final_answer=response.content; transition=Transition(rollout_id=rollout.id, step=step, messages=messages.copy(), response=response, terminated=True); transitions.append(transition); self.store.save_transition(transition); break
            else: raise RuntimeError("超过最大步数")
            parts=self.reward.evaluate(task, transitions, rollout.final_answer); rollout.reward_components=parts; rollout.reward=sum(parts.values()); rollout.status=RolloutStatus.SUCCEEDED
            self.store.append_event(Event(rollout_id=rollout.id, type="reward", payload={"total": rollout.reward, "components": parts}))
        except asyncio.CancelledError:
            rollout.status=RolloutStatus.CANCELLED; rollout.error="任务被取消"
        except Exception as exc: rollout.status=RolloutStatus.FAILED; rollout.error=str(exc); self.store.append_event(Event(rollout_id=rollout.id, type="error", payload={"error": str(exc)}))
        finally:
            # 无论模型、工具、取消哪一环出错，rollout 都必须进入终局状态并落库，
            # 否则恢复程序无法区分仍在执行和已丢失的任务。
            rollout.finished_at=now(); self.store.save_rollout(rollout)
            self.store.append_event(Event(rollout_id=rollout.id, type="rollout_finished", payload={"status": rollout.status.value, "error": rollout.error}))
        return rollout

    async def run_group(self, task: Task, group_size: int = 4) -> RolloutGroup:
        group=RolloutGroup(task_id=task.id); self.store.save_task(task); self.store.save_group(group)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        async def bounded(seed: int):
            async with semaphore:
                return await self.run_rollout(task, group.id, seed)
        results=await asyncio.gather(*(bounded(i) for i in range(group_size)), return_exceptions=False); group.rollout_ids=[r.id for r in results]; group.rewards=[r.reward for r in results]; group.mean_reward=statistics.mean(group.rewards) if group.rewards else 0.0; group.std_reward=statistics.pstdev(group.rewards) if len(group.rewards)>1 else 0.0; group.success_rate=sum(r.status == RolloutStatus.SUCCEEDED for r in results)/len(results) if results else 0.0; self.store.save_group(group); return group
