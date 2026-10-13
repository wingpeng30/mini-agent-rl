import json
from pathlib import Path
from types import SimpleNamespace
import pytest

np = pytest.importorskip("numpy")
from mini_agent_rl.retrieval import local
from mini_agent_rl.tools import SearchTool
from mini_agent_rl.web.server import DemoService, ROOT, create_app, read
from fastapi.testclient import TestClient


def doc(key, text="evidence"):
    return dict(id=key, title=key, text=text)


def test_rrf_adds_both_ranks_and_stable_ties():
    result = local.fuse([doc("a"), doc("b")], [doc("b"), doc("a")])
    assert [r["id"] for r in result] == ["a", "b"]
    assert result[0]["score"] == round(1 / 61 + 1 / 62, 8)
    assert len(local.fuse([doc("a"), doc("a")], [])) == 1


def test_chunking_covers_tail_and_respects_encoder_limit():
    class Tokenizer:
        cls_token_id, sep_token_id = 101, 102
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text)))
    cfg = dict(title_tokens=32, chunk_tokens=220, overlap_tokens=32, max_length=256)
    inputs, mapping = local.token_chunks(Tokenizer(), [dict(id="a", title="x" * 80, text="x" * 768)], cfg)
    assert all(len(r["input_ids"]) <= 256 for r in inputs)
    covered = {n for r in mapping for n in range(r["start_token"], r["end_token"])}
    assert covered == set(range(768))
    assert mapping[-1]["end_token"] == 768
    assert inputs[0]["input_ids"][0] == 101 and inputs[0]["input_ids"][-1] == 102


def indexed(tmp_path):
    docs = [doc("a"), doc("b")]
    cfg = dict(index_dir=str(tmp_path), top_k=3, candidate_k=20, rrf_k=60)
    fingerprint = dict(test=True)
    vectors = np.zeros((3, 384), dtype=np.float32)
    vectors[0, 1] = 1
    vectors[1:, 0] = 1
    np.save(tmp_path / "vectors.npy", vectors)
    (tmp_path / "chunks.json").write_text(json.dumps([dict(document_id="a"), dict(document_id="a"), dict(document_id="b")]))
    manifest = dict(identity=fingerprint, chunks=3, files={n: local.sha(tmp_path / n) for n in ("vectors.npy", "chunks.json")})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return docs, cfg, fingerprint


def test_index_reload_deduplicates_and_returns_original_body(tmp_path, monkeypatch):
    docs, cfg, fingerprint = indexed(tmp_path)
    monkeypatch.setattr(local, "identity", lambda cfg: fingerprint)
    class Encoder:
        def __init__(self, cfg): pass
        def query(self, query):
            vector = np.zeros(384, dtype=np.float32)
            vector[0] = 1
            return vector
    monkeypatch.setattr(local, "Encoder", Encoder)
    first = local.DenseIndex(cfg, docs)
    result = first.search("query")
    assert [r["id"] for r in result] == ["a", "b"]
    assert result[0]["text"] == "evidence"
    assert first.search("  ") == []
    assert local.DenseIndex(cfg, docs).search("query") == result


def test_index_rejects_identity_and_file_drift(tmp_path, monkeypatch):
    docs, cfg, fingerprint = indexed(tmp_path)
    monkeypatch.setattr(local, "identity", lambda cfg: dict(changed=True))
    with pytest.raises(ValueError, match="身份"):
        local.DenseIndex(cfg, docs)
    monkeypatch.setattr(local, "identity", lambda cfg: fingerprint)
    with (tmp_path / "chunks.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="哈希"):
        local.DenseIndex(cfg, docs)


def test_search_tool_rejects_different_corpus():
    engine = SimpleNamespace(documents=[doc("a")])
    with pytest.raises(ValueError, match="匹配"):
        SearchTool([doc("b")], "dense-v1", retrieval=engine)


def test_corpus_rejects_duplicate_ids_and_hash_drift(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([doc("a"), doc("a")]), encoding="utf-8")
    cfg = dict(corpus=str(path), corpus_sha256=local.sha(path), document_count=2)
    with pytest.raises(ValueError, match="文档ID"):
        local.corpus(cfg)
    cfg["corpus_sha256"] = "invalid"
    with pytest.raises(ValueError, match="哈希"):
        local.corpus(cfg)


def test_legacy_run_request_defaults_to_bm25():
    from mini_agent_rl.web.server import RunRequest
    assert RunRequest(question="query").retriever == "bm25-v2"


@pytest.fixture
def web_service(tmp_path):
    service = DemoService(read(ROOT / "configs/demo-web.json"), runtime_dir=tmp_path)
    documents = read(ROOT / service.config["corpora"]["goal-z-dev"]["path"])
    class Engine:
        def __init__(self): self.documents = documents
        def dense_index(self): return SimpleNamespace(manifest_sha256="test")
        def search(self, query, mode): return [documents[0] | dict(score=0.9)]
        def metadata(self, mode): return dict(retriever=mode, index_sha256="test", corpus_sha256="test")
        def status(self): return dict(ready=True, index_sha256="test")
    service.retrieval = Engine()
    return service


def test_web_search_without_agent_and_custom_corpus_rejected(web_service):
    with TestClient(create_app(service=web_service, autoload=False)) as client:
        result = client.post("/api/search", json=dict(query="query")).json()
        assert result["mode"] == "search" and result["documents"]
        assert result["retrieval"]["retriever"] == "hybrid-rrf-v1"
        assert web_service.state == "idle"
        custom = client.post("/api/corpora", json=dict(documents=[doc("custom")])).json()
        assert client.post("/api/search", json=dict(query="query", corpus_id=custom["id"])).status_code == 422
        assert client.post("/api/search", json=dict(query="evidence", corpus_id=custom["id"], retriever="bm25-v2")).status_code == 200
        assert client.post("/api/search", json=dict(query="query", retriever="bad")).status_code == 422


def test_missing_index_fails_explicitly_without_bm25_fallback(web_service):
    def missing():
        raise ValueError("索引缺失，请运行 mini-agent-search index")
    web_service.retrieval.dense_index = missing
    with TestClient(create_app(service=web_service, autoload=False)) as client:
        for mode in ("dense-v1", "hybrid-rrf-v1"):
            response = client.post("/api/search", json=dict(query="query", retriever=mode))
            assert response.status_code == 409
            assert "mini-agent-search index" in response.json()["detail"]
        response = client.post("/api/search", json=dict(query="evidence", retriever="bm25-v2"))
        assert response.status_code == 200
        assert response.json()["retrieval"]["retriever"] == "bm25-v2"


def test_hybrid_results_enter_real_agent_loop(web_service):
    from mini_agent_rl.domain import ModelResponse, ToolCall
    from mini_agent_rl.model import LocalModelConfig
    from run_v133_goal_p_variant import SearchTextAliasClient
    pc = web_service.config["product"]
    class Actions:
        config = LocalModelConfig(**{k: pc[k] for k in ("max_input_tokens", "max_new_tokens", "observation_encoding", "observation_presentation", "action_completion", "action_constraint")}, enable_thinking=False)
        _context_records = []
        count = 0
        async def generate(self, messages, tools, *, temperature, seed):
            self.count += 1
            if self.count == 1:
                return ModelResponse(tool_calls=[ToolCall(name="search", arguments=dict(query="query"))], prompt_token_ids=[1], action_token_ids=[2])
            assert web_service.retrieval.documents[0]["id"] in messages[-1].content
            return ModelResponse(content="answer", prompt_token_ids=[1], action_token_ids=[2])
    web_service.client = Actions()
    web_service.wrapped = SearchTextAliasClient(web_service.client)
    web_service.state, web_service.model_id = "ready", "goal-z"
    web_service.release_model = lambda: None
    with TestClient(create_app(service=web_service, autoload=False)) as client:
        result = client.post("/api/runs", json=dict(question="query", retriever="hybrid-rrf-v1"))
        assert result.status_code == 202
        from test_final_web import wait
        job = wait(client, result.json()["id"])
        assert job["state"] == "complete", job
        assert job["result"]["searches"] == 1
        assert job["result"]["retrieval"]["retriever"] == "hybrid-rrf-v1"
