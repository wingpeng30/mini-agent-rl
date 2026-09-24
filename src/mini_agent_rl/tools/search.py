"""离线检索工具：不把标准答案暴露给 Agent，只返回可观察文档。"""
from __future__ import annotations
import math
import re
from collections import Counter
from .registry import Tool
from ..domain import ToolDefinition

_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "with",
    "from", "by", "is", "was", "were", "are", "what", "which", "who", "where",
    "when", "did", "does", "do", "this", "that", "it", "its", "also", "called",
}


def _v2_tokens(value: str) -> list[str]:
    """BM25-v2 的稳定分词；英文单字符被过滤，中文按字匹配。"""
    tokens = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[a-z0-9]+", value.casefold())
    return [token for token in tokens if token not in _STOPWORDS and (len(token) >= 2 or not token.isascii())]


def _v2_norm(value: str) -> str:
    return " ".join(_v2_tokens(value))


class _BM25Index:
    """进程内、无外部依赖的 BM25 风格索引。"""

    def __init__(self, corpus: list[dict]):
        self.corpus = list(corpus)
        self.docs: list[tuple[dict, Counter[str], int]] = []
        document_frequency: Counter[str] = Counter()
        for document in self.corpus:
            title = _v2_tokens(str(document.get("title", "")))
            text = _v2_tokens(str(document.get("text", "")))
            counts = Counter(title * 3 + text)
            self.docs.append((document, counts, sum(counts.values())))
            document_frequency.update(counts.keys())
        self.document_frequency = document_frequency
        self.average_length = sum(length for _, _, length in self.docs) / max(len(self.docs), 1)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_tokens = list(dict.fromkeys(_v2_tokens(query)))
        normalized_query = _v2_norm(query)
        scored: list[tuple[float, dict]] = []
        for document, counts, length in self.docs:
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                df = self.document_frequency[token]
                idf = math.log(1.0 + (len(self.docs) - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (1.0 - 0.75 + 0.75 * length / max(self.average_length, 1.0))
                score += idf * (frequency * 2.5 / denominator)
            title = str(document.get("title", ""))
            normalized_title = _v2_norm(title)
            if normalized_query and normalized_query == normalized_title:
                score += 10.0
            elif normalized_title and normalized_title in normalized_query:
                score += 4.0
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("id", ""))))
        return [{"id": document.get("id"), "title": document.get("title"), "text": document.get("text"), "score": round(score, 6)} for score, document in scored[:top_k]]


class SearchTool(Tool):
    def __init__(self, corpus: list[dict], retriever_version: str = "keyword-v1"):
        if retriever_version not in {"keyword-v1", "bm25-v2"}:
            raise ValueError(f"未知检索器版本: {retriever_version}")
        self.corpus = corpus
        self.retriever_version = retriever_version
        self._bm25_index = _BM25Index(corpus) if retriever_version == "bm25-v2" else None
        definition = ToolDefinition(name="search", description="在离线文档中检索与问题相关的证据", parameters={"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})
        super().__init__(definition, self._search)
    def _search(self, query: str):
        if self.retriever_version == "bm25-v2":
            return self._bm25_index.search(query)
        terms = self._terms(query)
        scored = []
        for doc in self.corpus:
            text = f"{doc.get('title','')} {doc.get('text','')}".lower()
            score = sum(text.count(term) for term in terms)
            if score: scored.append((score, doc))
        scored.sort(key=lambda x: (-x[0], x[1].get("id", "")))
        return [{"id": d.get("id"), "title": d.get("title"), "text": d.get("text"), "score": s} for s, d in scored[:3]]

    @staticmethod
    def _terms(value: str) -> set[str]:
        """Split Latin words and CJK characters so Chinese queries are searchable.

        Python's ``\\w+`` treats a contiguous Chinese sentence as one token.  A
        document almost never contains the exact same sentence as the query, so
        that made ordinary Chinese lookups return no results.  Keep Latin words
        intact while matching each Han character (and the other CJK ranges)
        individually.  ``casefold`` also handles non-ASCII Latin text safely.
        """
        normalized = value.casefold()
        return {
            token
            for token in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[a-z0-9]+", normalized)
            if token.strip()
        }
