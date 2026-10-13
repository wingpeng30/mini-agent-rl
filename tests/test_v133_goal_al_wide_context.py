import asyncio
from types import SimpleNamespace

import pytest


def test_single_configuration_reaches_factory(monkeypatch):
    import run_v133_goal_q_search as q
    import run_v133_goal_al_wide_context as al
    import mini_agent_rl.model as model
    import mini_agent_rl.tools as tools
    captured = {}
    class Client:
        def __init__(self, config): self.config = config; captured["config"] = config
    class Registry:
        def register(self, tool): captured["tool"] = tool
    monkeypatch.setattr(model, "TransformersModelClient", Client)
    monkeypatch.setattr(tools, "ToolRegistry", Registry)
    monkeypatch.setattr(tools, "SearchTool", lambda corpus, version: (corpus, version))
    monkeypatch.setattr(q, "read", lambda path: [{"id": "frozen"}])
    monkeypatch.setattr(q, "load_config", lambda: pytest.fail("explicit config read global product"))
    for pc in al.ARMS.values():
        client, _, _ = q.make_client("adapter", deterministic=True, product_config=pc, model_path="base", corpus="corpus")
        assert client.config.max_input_tokens == pc["max_input_tokens"]
        assert client.config.max_new_tokens == pc["max_new_tokens"]
        assert client.config.action_completion == pc["action_completion"]
        assert client.config.timeout is None


def test_explicit_rollout_rejects_drift_before_generation():
    import run_v133_goal_q_search as q
    import run_v133_goal_al_wide_context as al
    bad = SimpleNamespace(base=SimpleNamespace(config=al.config(al.ARMS["A"])))
    with pytest.raises(RuntimeError, match="max_input_tokens"):
        asyncio.run(q.rollout("q", bad, None, sample=False, seed=42, product_config=al.ARMS["B"]))


def test_budget_difference_is_only_two_fields():
    import run_v133_goal_al_wide_context as al
    assert {k for k in al.ARMS["A"] if al.ARMS["A"][k] != al.ARMS["B"][k]} == {"max_input_tokens", "max_new_tokens"}


def test_source_reconstruction_preserves_tool_serialization_order():
    import json
    import run_v133_goal_al_wide_context as al
    observation = dict(ok=True, result=[dict(id="d", title="T", text="Evidence.", score=1)], searched_queries=["q"])
    messages = al.extend(al.initial("question"), dict(query="q", observation=observation))
    assert messages[-1].content.index('"title"') < messages[-1].content.index('"text"') < messages[-1].content.index('"score"')
    sorted_archive = json.loads(json.dumps(observation, sort_keys=True))
    archived_messages = al.extend(al.initial("question"), dict(query="q", observation=sorted_archive))
    assert json.loads(messages[-1].content) == json.loads(archived_messages[-1].content)
    assert messages[-1].content != archived_messages[-1].content


def test_source_gold_field_is_answer():
    import run_v133_goal_al_wide_context as al
    result = al.finalize_case(dict(answer="Hyde Park"), dict(case_id="test", answer="Hyde Park"), "B")
    assert result["official_answer"] == "Hyde Park"
    assert result["em"] == 1
