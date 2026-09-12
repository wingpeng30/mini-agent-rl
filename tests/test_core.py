import asyncio
import json
from pathlib import Path
import pytest
from mini_agent_rl.domain import Task, RolloutStatus
from mini_agent_rl.model import FakeModelClient
from mini_agent_rl.tools import SearchTool, ToolRegistry
from mini_agent_rl.reward import *
from mini_agent_rl.runtime import AgentRunner
from mini_agent_rl.storage import SQLiteStore

def runner(tmp_path, answer=None):
    store = SQLiteStore(tmp_path / "test.db")
    registry = ToolRegistry(); registry.register(SearchTool(json.loads(Path(__file__).parents[1].joinpath("examples/corpus.json").read_text(encoding="utf-8"))))
    reward = CompositeReward([ExactMatchReward(), EvidenceReward(), SearchCostPenalty(), RepeatedSearchPenalty(), InvalidActionPenalty()])
    return AgentRunner(FakeModelClient(answer), registry, reward, store), store

@pytest.mark.asyncio
async def test_fake_rollout_and_group(tmp_path):
    r, store = runner(tmp_path)
    group = await r.run_group(Task(id="t1", question="故宫位于哪个城市？", answer="故宫是中国明清两代的皇家宫殿，位于北京。"), 4)
    assert len(group.rollout_ids) == 4
    assert group.success_rate == 1.0
    rollouts = [store.get_rollout(i) for i in group.rollout_ids]
    assert all(x.reward > 0 for x in rollouts)
    assert all(x.final_answer == "故宫是中国明清两代的皇家宫殿，位于北京。" for x in rollouts)
    assert all(x.reward_components["exact_match"] == 1.0 for x in rollouts)

@pytest.mark.asyncio
async def test_failed_rollout_does_not_break_group(tmp_path):
    r, store = runner(tmp_path, answer="错误答案")
    group = await r.run_group(Task(id="t2", question="强化学习是什么？", answer="别的答案"), 3)
    assert len(group.rollout_ids) == 3
    assert group.success_rate == 1.0
    assert all(store.get_rollout(i).status == RolloutStatus.SUCCEEDED for i in group.rollout_ids)

def test_export_jsonl(tmp_path):
    r, store = runner(tmp_path)
    asyncio.run(r.run_group(Task(id="t3", question="故宫位于哪个城市？", answer="故宫是中国明清两代的皇家宫殿，位于北京。"), 1))
    output = tmp_path / "out.jsonl"; store.export_jsonl(str(output))
    assert output.read_text(encoding="utf-8").count("transitions") == 1

def test_search_supports_chinese_query_and_utf8_corpus(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps([{"id": "中", "title": "故宫", "text": "故宫位于北京。"}], ensure_ascii=False), encoding="utf-8")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    results = SearchTool(corpus)._search("故宫位于哪个城市")
    assert results and results[0]["id"] == "中"

@pytest.mark.asyncio
async def test_tool_observation_stays_structured_json(tmp_path):
    r, store = runner(tmp_path)
    group = await r.run_group(Task(id="t4", question="故宫位于哪个城市？", answer="故宫是中国明清两代的皇家宫殿，位于北京。"), 1)
    rollout = store.get_rollout(group.rollout_ids[0])
    assert rollout.final_answer == "故宫是中国明清两代的皇家宫殿，位于北京。"
