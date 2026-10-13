"""Offline MiniLM index and document-level reciprocal rank fusion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/retrieval-local.json"
MODES = ("bm25-v2", "dense-v1", "hybrid-rrf-v1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config(path=None):
    value = read(path or DEFAULT_CONFIG)
    value["corpus"] = str(ROOT / value["corpus"])
    if not 0 <= value["overlap_tokens"] < value["chunk_tokens"]:
        raise ValueError("Invalid chunk overlap")
    if value["title_tokens"] + value["chunk_tokens"] + 2 > value["max_length"]:
        raise ValueError("Chunk exceeds encoder length")
    if value["candidate_k"] < value["top_k"] or value["rrf_k"] <= 0:
        raise ValueError("Invalid retrieval limits")
    return value


def corpus(cfg):
    if sha(cfg["corpus"]) != cfg["corpus_sha256"]:
        raise ValueError("语料哈希不匹配")
    documents = read(cfg["corpus"])
    ids = [row["id"] for row in documents]
    if len(documents) != cfg["document_count"] or len(set(ids)) != len(ids):
        raise ValueError("语料数量或文档ID不匹配")
    if any(not isinstance(row.get(key), str) for row in documents for key in ("id", "title", "text")):
        raise ValueError("语料需要字符串id/title/text")
    return documents


def identity(cfg):
    folder = Path(cfg["model_path"])
    names = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
    return dict(version="local-minilm-v1", corpus_sha256=cfg["corpus_sha256"],
        document_count=cfg["document_count"], model_revision=cfg["model_revision"],
        model_hashes={name: sha(folder / name) for name in names},
        dimension=384, pooling="attention-mask-mean", normalized=True, dtype="float32",
        **{key: cfg[key] for key in ("title_tokens", "chunk_tokens", "overlap_tokens", "max_length")})


def token_chunks(tokenizer, documents, cfg):
    """Build token IDs directly: every body token is covered, without re-tokenization."""
    inputs, mapping = [], []
    stride = cfg["chunk_tokens"] - cfg["overlap_tokens"]
    for row in documents:
        title = tokenizer.encode(row["title"], add_special_tokens=False)[:cfg["title_tokens"]]
        body = tokenizer.encode(row["text"], add_special_tokens=False)
        for start in range(0, max(1, len(body)), stride):
            end = min(len(body), start + cfg["chunk_tokens"])
            # Transformers 5 removed the legacy build_inputs_with_special_tokens
            # method. MiniLM uses BERT's single-sequence CLS/SEP envelope.
            if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
                raise ValueError("Expected BERT CLS/SEP tokens")
            ids = [tokenizer.cls_token_id] + title + body[start:end] + [tokenizer.sep_token_id]
            if len(ids) > cfg["max_length"]:
                raise ValueError("Chunk exceeds encoder budget")
            inputs.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
            mapping.append(dict(document_id=row["id"], start_token=start, end_token=end))
            if end == len(body):
                break
    return inputs, mapping


class Encoder:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(cfg["cpu_threads"])
        self.cfg, self.torch = cfg, torch
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True)
        self.model = AutoModel.from_pretrained(cfg["model_path"], local_files_only=True).to("cpu").eval()
        self.lock = threading.Lock()

    def encode_inputs(self, inputs):
        import numpy as np
        torch = self.torch
        output = []
        with self.lock, torch.inference_mode():
            for start in range(0, len(inputs), self.cfg["batch_size"]):
                batch = self.tokenizer.pad(inputs[start:start + self.cfg["batch_size"]], padding=True, return_tensors="pt")
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                vectors = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), p=2, dim=1)
                output.append(vectors.float().cpu().numpy())
        result = np.concatenate(output).astype("float32")
        if not np.isfinite(result).all() or result.shape[1] != 384:
            raise ValueError("Invalid encoder output")
        return result

    def query(self, value):
        encoded = self.tokenizer(value, truncation=True, max_length=self.cfg["max_length"], add_special_tokens=True)
        return self.encode_inputs([encoded])[0]


def build_index(cfg):
    import numpy as np
    documents, fingerprint = corpus(cfg), identity(cfg)
    folder = Path(cfg["index_dir"])
    if folder.exists():
        DenseIndex(cfg, documents)
        return read(folder / "manifest.json")
    encoder = Encoder(cfg)
    inputs, mapping = token_chunks(encoder.tokenizer, documents, cfg)
    started = time.perf_counter()
    vectors = encoder.encode_inputs(inputs)
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = folder.with_name(folder.name + ".building-" + uuid4().hex)
    staging.mkdir()
    np.save(staging / "vectors.npy", vectors, allow_pickle=False)
    (staging / "chunks.json").write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    manifest = dict(identity=fingerprint, chunks=len(mapping), build_seconds=time.perf_counter() - started,
        files={name: sha(staging / name) for name in ("vectors.npy", "chunks.json")})
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    staging.rename(folder)
    return manifest


class DenseIndex:
    def __init__(self, cfg, documents):
        import numpy as np
        folder = Path(cfg["index_dir"])
        if not (folder / "manifest.json").is_file():
            raise ValueError("向量索引未构建，请运行 mini-agent-search index")
        manifest = read(folder / "manifest.json")
        if manifest["identity"] != identity(cfg):
            raise ValueError("索引身份不匹配，请使用匹配配置重新建索引（不要覆盖历史索引）")
        for name, expected in manifest["files"].items():
            if name not in {"vectors.npy", "chunks.json"} or sha(folder / name) != expected:
                raise ValueError("索引文件哈希不匹配")
        self.vectors = np.load(folder / "vectors.npy", allow_pickle=False)
        self.mapping = read(folder / "chunks.json")
        self.documents = {d["id"]: d for d in documents}
        if (self.vectors.shape != (manifest["chunks"], 384) or self.vectors.dtype != np.float32
            or len(self.mapping) != manifest["chunks"] or not np.isfinite(self.vectors).all()
            or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-4)
            or set(r["document_id"] for r in self.mapping) != set(self.documents)):
            raise ValueError("索引向量或文档映射不匹配")
        self.cfg, self.encoder = cfg, None
        self.manifest = manifest
        self.manifest_sha256 = sha(folder / "manifest.json")
        self.lock = threading.Lock()

    def search(self, query, top_k=3):
        if not query.strip():
            return []
        with self.lock:
            if self.encoder is None:
                self.encoder = Encoder(self.cfg)
            vector = self.encoder.query(query)
        scores = self.vectors @ vector
        by_id = {}
        for row, score in zip(self.mapping, scores):
            doc_id = row["document_id"]
            by_id[doc_id] = max(by_id.get(doc_id, float("-inf")), float(score))
        ranked = sorted(by_id, key=lambda doc_id: (-by_id[doc_id], doc_id))[:top_k]
        return [dict(id=doc_id, title=self.documents[doc_id]["title"], text=self.documents[doc_id]["text"], score=round(by_id[doc_id], 6)) for doc_id in ranked]


def fuse(sparse, dense, top_k=3, rrf_k=60):
    scores, documents = {}, {}
    for results in (sparse, dense):
        seen = set()
        for rank, row in enumerate(results, 1):
            doc_id = row["id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            documents[doc_id] = row
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank)
    ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
    return [dict(id=doc_id, title=documents[doc_id]["title"], text=documents[doc_id]["text"], score=round(scores[doc_id], 8)) for doc_id in ranked]


class LocalRetrieval:
    def __init__(self, cfg=None):
        from ..tools.search import _BM25Index
        self.cfg = cfg or config()
        self.documents = corpus(self.cfg)
        self.bm25 = _BM25Index(self.documents)
        self.dense = None
        self.lock = threading.Lock()

    def dense_index(self):
        with self.lock:
            if self.dense is None:
                self.dense = DenseIndex(self.cfg, self.documents)
            return self.dense

    def search(self, query, mode="hybrid-rrf-v1", top_k=None):
        if mode not in MODES:
            raise ValueError("Unknown retrieval mode")
        if not query.strip():
            return []
        top_k = self.cfg["top_k"] if top_k is None else top_k
        if mode == "bm25-v2":
            return self.bm25.search(query, top_k)
        dense = self.dense_index().search(query, max(top_k, self.cfg["candidate_k"]))
        if mode == "dense-v1":
            return dense[:top_k]
        sparse = self.bm25.search(query, max(top_k, self.cfg["candidate_k"]))
        return fuse(sparse, dense, top_k, self.cfg["rrf_k"])

    def metadata(self, mode):
        return dict(retriever=mode, corpus_sha256=self.cfg["corpus_sha256"],
            index_sha256=self.dense_index().manifest_sha256 if mode != "bm25-v2" else None,
            candidate_k=self.cfg["candidate_k"], rrf_k=self.cfg["rrf_k"] if mode == "hybrid-rrf-v1" else None)

    def status(self):
        try:
            index = self.dense_index()
            return dict(ready=True, index_sha256=index.manifest_sha256, chunks=index.manifest["chunks"], error=None)
        except (ValueError, OSError, KeyError) as exc:
            return dict(ready=False, index_sha256=None, error=str(exc))
