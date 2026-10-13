"""Local browser service. No training, external search, or teacher API calls."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import threading
from uuid import uuid4
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).parent / "static"
BD = ROOT / "runs/v133-goal-bd-final-regression-v1"
sys.path.insert(0, str(ROOT / "scripts"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def public_result(row, model_id, corpus, mode):
    steps = []
    for step in row["trace"]:
        steps.append(dict(step=step["step"], action=step.get("action"), query=step.get("query"),
            answer=step.get("answer"), raw=step.get("raw"), parse_error=step.get("parse_error"),
            repeated=step.get("repeated", False), documents=step.get("observation", {}).get("result", []),
            input_tokens=len(step.get("prompt_token_ids") or []),
            output_tokens=len(step.get("action_token_ids") or []),
            generation_stop_reason=step.get("usage", {}).get("generation_stop_reason"),
            final_review=step.get("usage", {}).get("final_review"),
            context_usage=step.get("context_usage", {})))
    return dict(mode=mode, model_id=model_id, corpus=corpus, question=row["question"],
        answer=row.get("answer"), reason=row["reason"], steps=steps,
        model_calls=row.get("model_calls", len(steps)), searches=row["searches"],
        auxiliary_calls=row.get("auxiliary_calls", 0),
        elapsed_seconds=row["elapsed_seconds"], historical_latency=mode == "replay")


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    model_id: str = "goal-z"
    corpus_id: str = "goal-z-dev"
    retriever: Literal["bm25-v2", "dense-v1", "hybrid-rrf-v1"] = "bm25-v2"
    reasoning_mode: Literal["off", "generic-review-v1", "relation-review-v1"] = "off"


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    corpus_id: str = "goal-z-dev"
    retriever: Literal["bm25-v2", "dense-v1", "hybrid-rrf-v1"] = "hybrid-rrf-v1"


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = "goal-z"


class CorpusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(default="自定义语料", min_length=1, max_length=100)
    documents: list[dict] = Field(min_length=1, max_length=10000)


def validate_documents(documents):
    ids = set()
    for document in documents:
        for key in ("id", "title", "text"):
            if not isinstance(document.get(key), str):
                raise ValueError(f"每篇文档需要字符串字段：{key}")
        if not document["id"] or document["id"] in ids:
            raise ValueError("文档id必须非空且唯一")
        ids.add(document["id"])
    if len(json.dumps(documents, ensure_ascii=False).encode("utf-8")) > 20 * 1024 * 1024:
        raise ValueError("自定义语料超过20MB")
    return documents


class DemoService:
    def __init__(self, config, runtime_dir=None):
        self.config = config
        self.runtime_dir = runtime_dir or Path("E:/AgentLightningAssets/runs/web")
        self.retrieval = None
        self._retrieval_lock = threading.Lock()
        self.state = "idle"
        self.model_id = None
        self.error = None
        self.loading_seconds = None
        self.client = self.wrapped = None
        self.jobs = {}
        self.corpora = dict(config["corpora"])
        self.busy = False
        self.tasks = set()
        self.replays = [json.loads(line) for line in (BD / "demo-replays.jsonl").read_text(encoding="utf-8").splitlines()] if (BD / "demo-replays.jsonl").is_file() else []

    def schedule(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def model_spec(self, model_id):
        if model_id not in self.config["models"]:
            raise HTTPException(422, "未知模型ID")
        return self.config["models"][model_id]

    def corpus_spec(self, corpus_id):
        if corpus_id not in self.corpora:
            raise HTTPException(422, "未知语料ID，请先导入语料")
        return self.corpora[corpus_id]

    def retrieval_engine(self):
        with self._retrieval_lock:
            if self.retrieval is None:
                from mini_agent_rl.retrieval.local import LocalRetrieval
                self.retrieval = LocalRetrieval()
            return self.retrieval

    def check_retriever(self, corpus_id, retriever):
        self.corpus_spec(corpus_id)
        if retriever != "bm25-v2":
            if corpus_id != "goal-z-dev":
                raise HTTPException(422, "首版向量及混合检索仅支持内置2048篇语料；自定义语料请选择BM25")
            try:
                self.retrieval_engine().dense_index()
            except (ValueError, OSError, KeyError) as exc:
                raise HTTPException(409, str(exc)) from exc

    def search_tool(self, corpus_id, retriever):
        from mini_agent_rl.tools import SearchTool
        spec = self.corpus_spec(corpus_id)
        path = ROOT / spec["path"]
        if sha(path) != spec["sha256"]:
            raise ValueError("语料SHA-256漂移")
        documents = validate_documents(read(path))
        engine = self.retrieval_engine() if retriever != "bm25-v2" else None
        return SearchTool(documents, retriever, retrieval=engine)

    def retrieval_metadata(self, retriever, corpus_id="goal-z-dev"):
        if retriever == "bm25-v2":
            return dict(retriever=retriever, corpus_sha256=self.corpus_spec(corpus_id)["sha256"], index_sha256=None)
        return self.retrieval_engine().metadata(retriever)

    def search_sync(self, request):
        self.check_retriever(request.corpus_id, request.retriever)
        started = time.perf_counter()
        documents = self.search_tool(request.corpus_id, request.retriever)._search(request.query.strip()) if request.query.strip() else []
        result = dict(mode="search", query=request.query, documents=documents,
            retrieval=self.retrieval_metadata(request.retriever, request.corpus_id),
            elapsed_seconds=time.perf_counter() - started)
        folder = self.runtime_dir / "search"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (uuid4().hex + ".json")).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result

    def release_model(self):
        # Explicitly break references before loading another adapter/model.
        self.wrapped = None
        self.client = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_sync(self, model_id):
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("需要支持BF16的本地CUDA环境；不会静默切换CPU或量化")
        model = self.model_spec(model_id)
        adapter = ROOT / model["adapter"]
        base = Path(self.config["base_model"])
        weights = list(base.glob("*.safetensors"))
        if len(weights) != 1 or sha(weights[0]) != self.config["base_weights_sha256"]:
            raise RuntimeError("基础模型缺失或权重SHA-256不符")
        if sha(adapter / "adapter_model.safetensors") != model["sha256"]:
            raise RuntimeError("LoRA权重SHA-256不符")
        from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
        from run_v133_goal_p_variant import SearchTextAliasClient
        pc = self.config["product"]
        self.release_model()
        self.client = TransformersModelClient(LocalModelConfig(model_id=str(base), adapter_path=str(adapter),
            max_input_tokens=pc["max_input_tokens"], max_new_tokens=pc["max_new_tokens"],
            enable_thinking=pc["thinking"], deterministic=True, recompute_old_logprobs=False,
            **{key: pc[key] for key in ("device", "dtype", "load_in_4bit", "timeout", "action_completion",
                "observation_encoding", "observation_presentation", "action_constraint", "context_aware_decoding")}))
        # Warm-up loads weights/tokenizer only, without generating an action.
        self.client._load()
        for parameter in self.client._model.parameters():
            parameter.requires_grad_(False)
        self.wrapped = SearchTextAliasClient(self.client)

    async def load(self, model_id):
        self.state, self.error = "loading", None
        started = time.monotonic()
        try:
            await asyncio.to_thread(self.load_sync, model_id)
            self.model_id = model_id
            self.loading_seconds = time.monotonic() - started
            self.state = "ready"
        except Exception as exc:
            self.error, self.state = str(exc), "error"
            self.model_id = None
            await asyncio.to_thread(self.release_model)
        finally:
            self.busy = False

    def begin_load(self, model_id):
        self.model_spec(model_id)
        if self.busy:
            raise HTTPException(409, "正在加载或运行，请等待当前任务完成")
        if self.state == "ready" and self.model_id == model_id:
            return
        self.busy = True
        self.state = "loading"
        self.schedule(self.load(model_id))

    async def run(self, job_id, request):
        job = self.jobs[job_id]
        job["state"] = "running"
        started = time.monotonic()
        try:
            from mini_agent_rl.tools import SearchTool, ToolRegistry
            import run_v133_goal_q_search as q
            corpus = self.corpus_spec(request.corpus_id)
            path = ROOT / corpus["path"]
            if sha(path) != corpus["sha256"]:
                raise RuntimeError("语料SHA-256漂移")
            registry = ToolRegistry()
            registry.register(self.search_tool(request.corpus_id, request.retriever))
            wrapped = self.wrapped

            class ObservedClient:
                base = wrapped.base

                async def generate(self, messages, tools, *, temperature, seed):
                    response = await wrapped.generate(messages, tools, temperature=temperature, seed=seed)
                    job["events"].append(dict(type="action", raw=response.raw_text,
                        query=response.tool_call.arguments.get("query") if response.tool_call else None,
                        answer=response.content if not response.tool_call else None))
                    return response

            class ObservedRegistry:
                def definitions(self):
                    return registry.definitions()

                async def execute(self, name, arguments):
                    result = await registry.execute(name, arguments)
                    job["events"].append(dict(type="search_results", query=arguments.get("query"),
                        documents=result.get("result", [])))
                    return result

            observed = ObservedClient()
            pipeline = None
            if request.reasoning_mode != "off":
                from mini_agent_rl.model.final_review import FinalReview
                pipeline = FinalReview(request.reasoning_mode, observed.generate, observed.base)
            row = await q.rollout(request.question.strip(), observed, ObservedRegistry(),
                sample=False, seed=42, product_config=self.config["product"], decision_pipeline=pipeline)
            result = public_result(row, request.model_id,
                dict(id=request.corpus_id, label=corpus["label"], sha256=corpus["sha256"]), "live")
            result["adapter_sha256"] = self.model_spec(request.model_id)["sha256"]
            result["retrieval"] = self.retrieval_metadata(request.retriever, request.corpus_id)
            result["reasoning_mode"] = request.reasoning_mode
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            (self.runtime_dir / f"{job_id}.json").write_text(json.dumps(dict(result=result, raw_trace=row), ensure_ascii=False), encoding="utf-8")
            job.update(state="complete", result=result)
        except Exception as exc:
            job.update(state="error", error=str(exc))
        finally:
            job["wall_seconds"] = time.monotonic() - started
            self.busy = False

    def begin_run(self, request):
        self.model_spec(request.model_id)
        self.corpus_spec(request.corpus_id)
        self.check_retriever(request.corpus_id, request.retriever)
        if not request.question.strip():
            raise HTTPException(422, "问题不能为空")
        if self.busy:
            raise HTTPException(409, "仅支持一个运行任务，请等待")
        if self.state != "ready" or self.model_id != request.model_id:
            raise HTTPException(409, "请先加载所选模型")
        if len(self.jobs) >= 32:
            self.jobs.pop(next(iter(self.jobs)))
        job_id = uuid4().hex
        self.jobs[job_id] = dict(id=job_id, state="queued", events=[])
        self.busy = True
        self.schedule(self.run(job_id, request))
        return self.jobs[job_id]

    def status(self):
        try:
            retrieval_status = self.retrieval_engine().status()
        except (ValueError, OSError, KeyError) as exc:
            retrieval_status = dict(ready=False, error=str(exc))
        return dict(application="mini-agent-final-web-v1", state=self.state, model_id=self.model_id,
            busy=self.busy, error=self.error, loading_seconds=self.loading_seconds,
            models=[dict(id=k, **v) for k, v in self.config["models"].items()],
            corpora=[dict(id=k, label=v["label"], sha256=v["sha256"]) for k, v in self.corpora.items()],
            product=self.config["product"], replay_available=bool(self.replays),
            retrieval=retrieval_status,
            note="本地封闭语料搜索；宽窗演示配置；旧产品默认值不变")


def create_app(config_path=None, *, autoload=True, service=None):
    service = service or DemoService(read(config_path or ROOT / "configs/demo-web.json"))

    @asynccontextmanager
    async def lifespan(app):
        if autoload:
            service.begin_load(service.config.get("default_model", "goal-z"))
        yield
        if service.tasks:
            await asyncio.gather(*service.tasks, return_exceptions=True)
        if service.client is not None:
            await asyncio.to_thread(service.release_model)

    app = FastAPI(title="Mini Agent · Local Search", lifespan=lifespan)
    app.state.service = service

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        host = request.headers.get("host", "").split(":")[0]
        origin = request.headers.get("origin")
        if host not in {"127.0.0.1", "localhost", "testserver"} or (origin and origin not in {"http://127.0.0.1:8765", "http://localhost:8765", "http://testserver"}):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "只允许本机同源访问"}, status_code=403)
        content_length = request.headers.get("content-length", "0")
        if content_length.isdigit() and int(content_length) > 21 * 1024 * 1024:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "请求超过21MB"}, status_code=413)
        return await call_next(request)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/status")
    async def status():
        return service.status()

    @app.post("/api/model/load")
    async def load_model(request: LoadRequest):
        service.begin_load(request.model_id)
        return service.status()

    @app.get("/api/examples")
    async def examples():
        return [dict(case_id=r["case_id"], question=r["question"], model_id="goal-z" if r["arm"] == "A" else "am-research") for r in service.replays]

    @app.get("/api/replay/{case_id}")
    async def replay(case_id: str, model_id: str = "goal-z"):
        model = service.model_spec(model_id)
        row = next((r for r in service.replays if r["case_id"] == case_id and r["arm"] == ("A" if model_id == "goal-z" else "B")), None)
        if row is None:
            raise HTTPException(404, "回放案例不存在")
        result = public_result(row, model_id, service.corpus_spec("goal-z-dev"), "replay")
        result["adapter_sha256"] = model["sha256"]
        result["retrieval"] = dict(retriever="bm25-v2", index_sha256=None, historical=True)
        return result

    @app.post("/api/search")
    async def search(request: SearchRequest):
        try:
            return await asyncio.to_thread(service.search_sync, request)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/corpora")
    async def upload_corpus(request: CorpusRequest):
        try:
            documents = validate_documents(request.documents)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        encoded = json.dumps(documents, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        corpus_id = "custom-" + digest[:16]
        target = service.runtime_dir / "corpora" / f"{digest}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        service.corpora[corpus_id] = dict(label=request.label, path=str(target), sha256=digest)
        return dict(id=corpus_id, label=request.label, sha256=digest, documents=len(documents))

    @app.post("/api/runs", status_code=202)
    async def run(request: RunRequest):
        return service.begin_run(request)

    @app.get("/api/runs/{job_id}")
    async def run_status(job_id: str):
        if job_id not in service.jobs:
            raise HTTPException(404, "运行记录不存在或已过期")
        return service.jobs[job_id]

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/demo-web.json")
    parser.add_argument("--no-autoload", action="store_true", help="Only for offline tests/replay")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.config, autoload=not args.no_autoload), host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
