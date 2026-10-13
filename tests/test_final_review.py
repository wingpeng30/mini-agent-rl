import asyncio
import json
from types import SimpleNamespace
import pytest
from mini_agent_rl.domain import Message, ModelResponse, ToolCall
from mini_agent_rl.model.final_review import FinalReview, validate_review


def response(value):
    raw = json.dumps(value)
    return ModelResponse(raw_text=raw, content=value.get("answer", ""),
        tool_calls=[ToolCall(name="search", arguments={"query": value["query"]})] if value["action"] == "search" else [],
        prompt_token_ids=[1], action_token_ids=[2])


def grounded(answer="Dad", quote="Actor is the son of Dad."):
    return dict(links=[dict(subject="Actor", relation="father", object=answer, document_id="d", quote=quote)],
        missing_relation="", action="answer", answer=answer)


def messages():
    return [Message(role="system", content="original"), Message(role="user", content="Who is Actor's father?"),
        Message(role="tool", content=json.dumps(dict(result=[dict(id="d", text="Actor is the son of Dad.")], searched_queries=["Actor father"]))) ]


def test_terminal_object_and_hidden_quote_rejected():
    docs = [dict(document_id="d", text="Actor is the son of Dad.")]
    assert validate_review(response(grounded()), docs, "relation-review-v1")["valid"]
    bad = grounded(); bad["answer"] = "Actor"
    assert not validate_review(response(bad), docs, "relation-review-v1")["valid"]
    assert not validate_review(response(grounded(quote="Hidden fact")), docs, "relation-review-v1")["valid"]


def test_review_once_and_budget_counts():
    calls = []
    async def generate(ms, tools, **kwargs):
        calls.append(ms)
        return response(grounded()) if len(calls) == 2 else response(dict(action="answer", answer="Actor"))
    pipe = FinalReview("relation-review-v1", generate)
    result = asyncio.run(pipe(messages(), [], temperature=0, seed=42))
    assert result.content == "Dad" and result.usage["final_review"]["accepted"]
    for _ in range(3):
        asyncio.run(pipe(messages(), [], temperature=0, seed=42))
    assert pipe.model_calls == 5 and pipe.auxiliary_calls == 1
    with pytest.raises(RuntimeError, match="budget"):
        asyncio.run(pipe(messages(), [], temperature=0, seed=42))
    assert "unverified_candidate" in calls[1][1].content
    assert "support_titles" not in calls[1][1].content


def test_invalid_review_falls_back_without_retry():
    count = 0
    async def generate(*args, **kwargs):
        nonlocal count
        count += 1
        return response(dict(action="answer", answer="Actor")) if count == 1 else response(grounded(quote="invented"))
    pipe = FinalReview("relation-review-v1", generate)
    result = asyncio.run(pipe(messages(), [], temperature=0, seed=42))
    assert result.content == "Actor" and not result.usage["final_review"]["accepted"]
    assert count == 2


def test_no_evidence_skips_review_and_clipped_quote_is_invalid():
    async def generate(*args, **kwargs):
        return response(dict(action="answer", answer="Actor"))
    pipe = FinalReview("generic-review-v1", generate)
    asyncio.run(pipe(messages()[:2], [], temperature=0, seed=42))
    assert pipe.model_calls == 1 and pipe.auxiliary_calls == 0
    count = 0
    class Budgeter:
        def _inputs(self, ms, phase):
            return SimpleNamespace(_budgeted_messages=ms[:2] if phase == "final_review_quote_audit" else ms)
    async def with_quote(*args, **kwargs):
        nonlocal count
        count += 1
        return response(dict(action="answer", answer="Actor")) if count == 1 else response(grounded())
    pipe = FinalReview("relation-review-v1", with_quote, Budgeter())
    result = asyncio.run(pipe(messages(), [], temperature=0, seed=42))
    assert not result.usage["final_review"]["accepted"]


def test_missing_relation_at_last_decision_is_insufficient_not_search():
    from mini_agent_rl.model import LocalModelConfig
    import run_v133_goal_q_search as q
    from mini_agent_rl.web.server import ROOT, read
    pc = read(ROOT / "configs/demo-web.json")["product"]
    values = [response(dict(action="search", query="one")), response(dict(action="search", query="two")),
        response(dict(action="search", query="three")), response(dict(action="answer", answer="Actor")),
        response(dict(links=[], missing_relation="father", action="search", query="four"))]
    class Client:
        config = LocalModelConfig(**{k: pc[k] for k in ("max_input_tokens", "max_new_tokens", "observation_encoding", "observation_presentation", "action_completion", "action_constraint")}, enable_thinking=False)
        _context_records = []
        async def generate(self, *args, **kwargs): return values.pop(0)
    class Registry:
        count = 0
        def definitions(self): return []
        async def execute(self, *args):
            self.count += 1
            return dict(ok=True, result=[dict(id="d", text="Actor is the son of Dad.")])
    client, registry = Client(), Registry()
    wrapped = SimpleNamespace(base=client, generate=client.generate)
    pipeline = FinalReview("relation-review-v1", client.generate)
    result = asyncio.run(q.rollout("question", wrapped, registry, sample=False, seed=42, product_config=pc, decision_pipeline=pipeline))
    assert result["reason"] == "insufficient_evidence" and result["answer"] is None
    assert result["model_calls"] == 5 and result["auxiliary_calls"] == 1
    assert registry.count == result["searches"] == 3


def test_api_reasoning_defaults_off():
    from mini_agent_rl.web.server import RunRequest
    assert RunRequest(question="question").reasoning_mode == "off"
    with pytest.raises(ValueError): RunRequest(question="question", reasoning_mode="unknown")


def test_missing_relation_with_remaining_budget_executes_search_once():
    import run_v133_goal_q_search as q
    from mini_agent_rl.model import LocalModelConfig
    from mini_agent_rl.web.server import ROOT, read
    pc = read(ROOT / "configs/demo-web.json")["product"]
    values = [response(dict(action="search", query="one")), response(dict(action="answer", answer="Actor")),
        response(dict(links=[], missing_relation="father", action="search", query="father")),
        response(dict(action="answer", answer="Dad"))]
    class Client:
        config = LocalModelConfig(**{k: pc[k] for k in ("max_input_tokens", "max_new_tokens", "observation_encoding", "observation_presentation", "action_completion", "action_constraint")}, enable_thinking=False)
        _context_records = []
        async def generate(self, *args, **kwargs): return values.pop(0)
    class Registry:
        def definitions(self): return []
        async def execute(self, *args): return dict(ok=True, result=[dict(id="d", text="Actor is the son of Dad.")])
    client = Client()
    result = asyncio.run(q.rollout("question", SimpleNamespace(base=client, generate=client.generate), Registry(),
        sample=False, seed=42, product_config=pc, decision_pipeline=FinalReview("relation-review-v1", client.generate)))
    assert result["answer"] == "Dad" and result["searches"] == 2
    assert result["model_calls"] == 4 and result["auxiliary_calls"] == 1


def test_malformed_auxiliary_completion_keeps_original():
    count = 0
    async def generate(*args, **kwargs):
        nonlocal count
        count += 1
        return response(dict(action="answer", answer="Actor")) if count == 1 else ModelResponse(raw_text="broken", parse_error="invalid JSON")
    result = asyncio.run(FinalReview("relation-review-v1", generate)(messages(), [], temperature=0, seed=42))
    assert result.content == "Actor" and not result.usage["final_review"]["accepted"]


from test_local_retrieval import web_service


def test_web_review_records_actual_calls_and_quotes(web_service):
    from mini_agent_rl.model import LocalModelConfig
    from run_v133_goal_p_variant import SearchTextAliasClient
    from mini_agent_rl.web.server import create_app
    from fastapi.testclient import TestClient
    from test_final_web import wait
    pc = web_service.config["product"]
    document = web_service.retrieval.documents[0]
    answer = document["text"].split()[0]
    class Client:
        config = LocalModelConfig(**{k: pc[k] for k in ("max_input_tokens", "max_new_tokens", "observation_encoding", "observation_presentation", "action_completion", "action_constraint")}, enable_thinking=False)
        _context_records = []
        count = 0
        async def generate(self, *args, **kwargs):
            self.count += 1
            if self.count == 1: return response(dict(action="search", query="query"))
            if self.count == 2: return response(dict(action="answer", answer="bridge"))
            return response(dict(links=[dict(subject=document["title"], relation="mentions", object=answer,
                document_id=document["id"], quote=document["text"])], missing_relation="", action="answer", answer=answer))
    web_service.client = Client()
    web_service.wrapped = SearchTextAliasClient(web_service.client)
    web_service.state, web_service.model_id = "ready", "goal-z"
    web_service.release_model = lambda: None
    with TestClient(create_app(service=web_service, autoload=False)) as client:
        started = client.post("/api/runs", json=dict(question="query", retriever="hybrid-rrf-v1", reasoning_mode="relation-review-v1"))
        job = wait(client, started.json()["id"])
        assert job["state"] == "complete", job
        result = job["result"]
        assert result["answer"] == answer and result["reasoning_mode"] == "relation-review-v1"
        assert result["model_calls"] == 3 and result["auxiliary_calls"] == 1
        assert result["steps"][-1]["final_review"]["audit"]["valid"]
