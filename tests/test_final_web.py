"""Local UI contract tests. Fake actions exercise the real Goal Q rollout."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import time

from fastapi.testclient import TestClient
import pytest

from mini_agent_rl.domain import ModelResponse, ToolCall
from mini_agent_rl.web.server import ROOT, DemoService, create_app, read, validate_documents


@pytest.fixture
def service(tmp_path, monkeypatch):
    instance = DemoService(read(ROOT / "configs/demo-web.json"), runtime_dir=tmp_path)
    monkeypatch.setattr(instance, "release_model", lambda: None)
    return instance


def wait(client, job_id):
    for _ in range(100):
        result = client.get("/api/runs/" + job_id).json()
        if result["state"] in {"complete", "error"}:
            return result
        time.sleep(.01)
    pytest.fail("Fake rollout did not complete")


def test_status_and_six_replays_bind_identity(service):
    with TestClient(create_app(autoload=False, service=service)) as client:
        status = client.get("/api/status").json()
        assert status["product"]["max_searches_per_case"] == 3
        examples = client.get("/api/examples").json()
        assert len(examples) == 6
        for example in examples:
            result = client.get(f"/api/replay/{example['case_id']}", params={"model_id": example["model_id"]}).json()
            assert result["mode"] == "replay" and result["historical_latency"]
            assert result["adapter_sha256"] == service.config["models"][example["model_id"]]["sha256"]
        assert client.get("/api/replay/unknown").status_code == 404
        assert client.get("/api/replay/unknown", params={"model_id": "../private"}).status_code == 422


def test_autoload_and_failure_are_visible_without_fallback(service, monkeypatch):
    called = []
    monkeypatch.setattr(service, "load_sync", lambda model: called.append(model))
    with TestClient(create_app(service=service)) as client:
        for _ in range(50):
            if client.get("/api/status").json()["state"] == "ready":
                break
            time.sleep(.01)
        default_model = service.config.get("default_model", "goal-z")
        assert called == [default_model]
        assert service.model_id == default_model
        def fail(model):
            raise FileNotFoundError("adapter missing")
        monkeypatch.setattr(service, "load_sync", fail)
        other_model = "goal-z" if default_model == "am-research" else "am-research"
        client.post("/api/model/load", json={"model_id": other_model})
        for _ in range(50):
            if service.state == "error":
                break
            time.sleep(.01)
        assert service.state == "error" and service.model_id is None
        assert "adapter missing" in service.error


def test_live_runs_existing_loop_and_isolates_records(service):
    from mini_agent_rl.model import LocalModelConfig
    from run_v133_goal_p_variant import SearchTextAliasClient
    pc = service.config["product"]
    class Actions:
        config = LocalModelConfig(**{k: pc[k] for k in ("max_input_tokens", "max_new_tokens", "observation_encoding", "observation_presentation", "action_completion", "action_constraint")}, enable_thinking=False)
        _context_records = []
        count = 0
        async def generate(self, messages, tools, *, temperature, seed):
            self.count += 1
            if self.count == 1:
                return ModelResponse(tool_calls=[ToolCall(name="search", arguments={"query": "Rapana"})], raw_text='{"action":"search","query":"Rapana"}', prompt_token_ids=[1], action_token_ids=[2])
            assert messages[-1].role == "tool" and "Rapana" in messages[-1].content
            return ModelResponse(content="Shell", raw_text='{"action":"answer","answer":"Shell"}', prompt_token_ids=[1, 2], action_token_ids=[3])
    service.client = Actions()
    service.wrapped = SearchTextAliasClient(service.client)
    service.state, service.model_id = "ready", "goal-z"
    with TestClient(create_app(autoload=False, service=service)) as client:
        created = client.post("/api/runs", json={"question": "Who owned Rapana?"})
        assert created.status_code == 202
        job = wait(client, created.json()["id"])
        assert job["state"] == "complete", job
        result = job["result"]
        assert result["mode"] == "live" and result["searches"] == 1
        assert result["model_calls"] == 2 and result["steps"][0]["documents"]
        assert result["adapter_sha256"] == service.config["models"]["goal-z"]["sha256"]
        assert len(list(service.runtime_dir.glob("*.json"))) == 1
        assert client.post("/api/runs", json={"question": "   "}).status_code == 422
        assert client.post("/api/runs", json={"question": "q", "adapter_path": "private"}).status_code == 422


def test_busy_and_unknown_models_rejected(service):
    service.busy = True
    with TestClient(create_app(autoload=False, service=service)) as client:
        assert client.post("/api/model/load", json={"model_id": "goal-z"}).status_code == 409
        assert client.post("/api/runs", json={"question": "q"}).status_code == 409
        assert client.post("/api/model/load", json={"model_id": "unknown"}).status_code == 422


def test_custom_corpus_validation_and_identity(service):
    docs = [dict(id="a", title="Hello", text="Example body")]
    with TestClient(create_app(autoload=False, service=service)) as client:
        result = client.post("/api/corpora", json={"documents": docs}).json()
        assert result["id"].startswith("custom-") and result["documents"] == 1
        spec = service.corpus_spec(result["id"])
        assert read(spec["path"]) == docs
        assert client.post("/api/corpora", json={"documents": docs + docs}).status_code == 422
        assert client.post("/api/corpora", json={"documents": [dict(id="x", title="x", text=3)]}).status_code == 422


def test_cross_origin_and_arbitrary_host_rejected(service):
    with TestClient(create_app(autoload=False, service=service)) as client:
        assert client.post("/api/model/load", headers={"Origin": "https://attacker.invalid"}, json={"model_id": "goal-z"}).status_code == 403
        assert client.get("/api/status", headers={"Host": "attacker.invalid"}).status_code == 403
        assert client.get("/").status_code == 200


def test_missing_base_fails_before_model_load(service, monkeypatch):
    # Torch is replaced here: this guard test performs no CUDA work.
    import sys
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", torch)
    service.config["base_model"] = str(service.runtime_dir / "missing-base")
    with pytest.raises(RuntimeError, match="基础模型"):
        service.load_sync("goal-z")
