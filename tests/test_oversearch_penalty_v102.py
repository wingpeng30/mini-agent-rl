import asyncio

from mini_agent_rl.cli import default_reward
from mini_agent_rl.model import FakeModelClient
from mini_agent_rl.runtime import AgentRunner
from mini_agent_rl.storage import SQLiteStore
from mini_agent_rl.tools import ToolRegistry


def test_v102_reward_records_zero_oversearch_penalty(tmp_path):
    reward = default_reward(reward_version="v102-oversearch-zero", over_search_penalty=0.0)
    runner = AgentRunner(FakeModelClient(), ToolRegistry(), reward, SQLiteStore(tmp_path / "run.db"))
    assert runner.over_search_penalty == 0.0
    assert runner._reward_config()["OverSearchAfterEvidencePenalty"]["penalty"] == 0.0
    assert runner._reward_config()["_meta"]["reward_version"] == "v102-oversearch-zero"


def test_old_reward_keeps_original_oversearch_penalty(tmp_path):
    reward = default_reward(reward_version="v098-answer-v1")
    runner = AgentRunner(FakeModelClient(), ToolRegistry(), reward, SQLiteStore(tmp_path / "run.db"))
    assert runner.over_search_penalty == -0.2
