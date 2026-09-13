import json

import pytest

from mini_agent_rl.domain import ModelResponse, Task, ToolCall
from mini_agent_rl.reward import (CompositeReward, EvidenceReward, ExactMatchReward, InvalidActionPenalty,
                                  MaxStepsPenalty, RepeatedSearchPenalty, SearchCostPenalty, SupportCoverageReward,
                                  UngroundedAnswerPenalty)
from mini_agent_rl.runtime import AgentRunner
from mini_agent_rl.storage import SQLiteStore
from mini_agent_rl.tools import SearchTool, ToolRegistry


def make_runner(tmp_path, client, max_steps=3):
    registry = ToolRegistry(); registry.register(SearchTool([{"id": "d", "title": "Doc", "text": "answer"}]))
    reward = CompositeReward([ExactMatchReward(), EvidenceReward(), SupportCoverageReward(), SearchCostPenalty(), RepeatedSearchPenalty(), InvalidActionPenalty(), MaxStepsPenalty(), UngroundedAnswerPenalty()])
    return AgentRunner(client, registry, reward, SQLiteStore(tmp_path / "rl.db"), max_steps=max_steps)


class RepeatingClient:
    async def generate(self, messages, tools, *, temperature=0.0, seed=None):
        calls = sum(message.role == "tool" for message in messages)
        return ModelResponse(tool_calls=[ToolCall(name="search", arguments={"query": " Doc " if calls == 0 else "doc"})], action_token_ids=[1], old_logprobs=[-0.1], action_mask=[1])


class InvalidClient:
    async def generate(self, messages, tools, *, temperature=0.0, seed=None):
        return ModelResponse(raw_text="not json", action_token_ids=[1, 2], old_logprobs=[-0.1, -0.2], action_mask=[1, 1], parse_error="模型输出不是合法 JSON")


class SeedClient:
    async def generate(self, messages, tools, *, temperature=0.0, seed=None):
        answer = "answer" if seed % 2 == 0 else "wrong"
        return ModelResponse(content=answer, action_token_ids=[1], old_logprobs=[-0.1], action_mask=[1])


@pytest.mark.asyncio
async def test_repeated_search_is_saved_as_negative_policy_trajectory(tmp_path):
    runner = make_runner(tmp_path, RepeatingClient())
    group = await runner.run_group(Task(id="t", question="q", answer="answer"), 1)
    rollout = runner.store.get_rollout(group.rollout_ids[0]); transitions = runner.store.get_transitions(rollout.id)
    assert rollout.termination_reason == "repeated_action" and rollout.eligible_for_rl
    assert len(transitions) == 2 and transitions[-1].terminated
    assert transitions[-1].reward == -0.5 and rollout.reward_components["repeated_search"] == -0.5


@pytest.mark.asyncio
async def test_invalid_action_preserves_tokens_and_logprobs(tmp_path):
    runner = make_runner(tmp_path, InvalidClient())
    group = await runner.run_group(Task(id="t", question="q", answer="answer"), 1)
    rollout = runner.store.get_rollout(group.rollout_ids[0]); transition = runner.store.get_transitions(rollout.id)[0]
    assert rollout.termination_reason == "invalid_action" and rollout.reward < 0
    assert transition.response.action_token_ids == [1, 2]
    assert len(transition.response.old_logprobs) == len(transition.response.action_mask) == 2


@pytest.mark.asyncio
async def test_group_advantage_and_export_include_group(tmp_path):
    runner = make_runner(tmp_path, SeedClient())
    group = await runner.run_group(Task(id="t", question="q", answer="answer"), 2)
    rollouts = [runner.store.get_rollout(item) for item in group.rollout_ids]
    assert not group.zero_variance and {round(item.advantage) for item in rollouts} == {-1, 1}
    output = tmp_path / "rl.jsonl"; runner.store.export_jsonl(str(output))
    payload = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert payload["group"]["id"] == group.id and "advantage" in payload["rollout"]
