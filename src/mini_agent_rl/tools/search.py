"""离线检索工具：不把标准答案暴露给 Agent，只返回可观察文档。"""
from __future__ import annotations
import re
from .registry import Tool
from ..domain import ToolDefinition

class SearchTool(Tool):
    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        definition = ToolDefinition(name="search", description="在离线文档中检索与问题相关的证据", parameters={"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})
        super().__init__(definition, self._search)
    def _search(self, query: str):
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
