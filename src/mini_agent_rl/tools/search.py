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


class _ParagraphBM25Index(_BM25Index):
    """BM25 over natural paragraphs, returning at most one passage per page."""

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

        # The corpus can contain many passages per page. Keep the best passage
        # for each normalized title, with stable paragraph order for exact ties.
        scored.sort(key=lambda item: (
            -item[0],
            _v2_norm(str(item[1].get("title", ""))),
            int(item[1].get("paragraph_index", 0)),
            str(item[1].get("id", "")),
        ))
        distinct: list[tuple[float, dict]] = []
        seen_titles: set[str] = set()
        for score, document in scored:
            title_key = _v2_norm(str(document.get("title", "")))
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            distinct.append((score, document))
            if len(distinct) >= top_k:
                break
        return [{"id": document.get("id"), "title": document.get("title"), "text": document.get("text"),
                 "score": round(score, 6)} for score, document in distinct]


class _PageParagraphBM25Index:
    """Rank complete pages, then choose a body passage without title boosting."""

    _SECTION_HEADINGS = {
        "bibliography", "references", "external links", "see also", "notes", "further reading",
    }

    def __init__(self, corpus: list[dict], tokenizer):
        if tokenizer is None:
            raise ValueError("bm25-v4-page-paragraph requires the product tokenizer")
        self.tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        self.pages: dict[str, list[dict]] = {}
        for row in corpus:
            key = _v2_norm(str(row.get("title", ""))) or f"__untokenized_title__:{row.get('id', '')}"
            self.pages.setdefault(key, []).append(dict(row))
        for values in self.pages.values():
            values.sort(key=lambda row: (int(row.get("paragraph_index", 0)), str(row.get("id", ""))))

        page_docs = []
        paragraph_terms: list[tuple[dict, Counter[str], int]] = []
        self.provenance_by_id: dict[str, dict] = {}
        for key, values in self.pages.items():
            title = str(values[0]["title"])
            page_docs.append({"id": key, "title": title,
                              "text": "\n".join(str(row["text"]) for row in values)})
            for row in values:
                terms = _v2_tokens(str(row.get("text", "")))
                if not terms or _v2_norm(str(row.get("text", ""))) in self._SECTION_HEADINGS:
                    continue
                paragraph_terms.append((row, Counter(terms), len(terms)))
                self.provenance_by_id[str(row["id"])] = {
                    "source_id": row.get("source_id"), "source_url": row.get("source_url"),
                    "paragraph_index": row.get("paragraph_index"), "text": str(row["text"]),
                }
        self.page_index = _BM25Index(page_docs)
        self.paragraph_terms = paragraph_terms
        self.paragraph_count = len(paragraph_terms)
        self.paragraph_df: Counter[str] = Counter()
        for _, counts, _ in paragraph_terms:
            self.paragraph_df.update(counts.keys())
        self.paragraph_avg_length = sum(length for _, _, length in paragraph_terms) / max(1, len(paragraph_terms))
        self.last_sources: dict[str, dict] = {}
        self.source_history: list[dict[str, dict]] = []

    def _token_spans(self, text: str) -> list[tuple[int, int]]:
        result = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = result["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
            offsets = offsets[0]
        return [(int(a), int(b)) for a, b in offsets if int(b) > int(a)]

    def _excerpt(self, text: str, limit: int, query_tokens: set[str]) -> tuple[str, tuple[int, int]]:
        spans = self._token_spans(text)
        if not spans:
            return "", (0, 0)
        if len(spans) <= limit:
            return text[spans[0][0]:spans[-1][1]], (spans[0][0], spans[-1][1])
        best_start, best_key = 0, (-1, -1)
        # Fixed-size windows make the truncation deterministic. Prefer coverage
        # of distinct query terms, then total matched terms, then earliest span.
        token_values = [_v2_tokens(text[a:b]) for a, b in spans]
        for start in range(len(spans) - limit + 1):
            window = [term for values in token_values[start:start + limit] for term in values]
            matched = [term for term in window if term in query_tokens]
            key = (len(set(matched)), len(matched))
            if key > best_key:
                best_key, best_start = key, start
        start_char, end_char = spans[best_start][0], spans[best_start + limit - 1][1]
        return text[start_char:end_char], (start_char, end_char)

    def _body_score(self, query_tokens: list[str], counts: Counter[str], length: int) -> float:
        score = 0.0
        for token in query_tokens:
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            df = self.paragraph_df[token]
            idf = math.log(1.0 + (self.paragraph_count - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (
                1.0 - 0.75 + 0.75 * length / max(self.paragraph_avg_length, 1.0)
            )
            score += idf * (frequency * 2.5 / denominator)
        return score

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_tokens = list(dict.fromkeys(_v2_tokens(query)))
        query_set = set(query_tokens)
        ranked_pages = self.page_index.search(query, top_k=max(1, top_k))
        self.last_sources = {}
        output = []
        for page in ranked_pages:
            key = str(page["id"])
            values = self.pages[key]
            eligible = [row for row in values if str(row.get("id")) in self.provenance_by_id]
            if not eligible:
                continue
            scored = []
            for row in eligible:
                terms = Counter(_v2_tokens(str(row.get("text", ""))))
                score = self._body_score(query_tokens, terms, sum(terms.values()))
                scored.append((score, int(row.get("paragraph_index", 0)), str(row.get("id", "")), row))
            scored.sort(key=lambda item: (-item[0], item[1], item[2]))
            matching = [item for item in scored if item[0] > 0]
            best = matching[0][3] if matching else None
            lead = eligible[0]
            lead_text = str(lead.get("text", ""))
            if best is None:
                text, span = self._excerpt(lead_text, 384, query_set)
                source_rows = [(lead, span)]
            elif str(best.get("id")) == str(lead.get("id")):
                text, span = self._excerpt(str(best["text"]), 288, query_set)
                source_rows = [(best, span)]
            else:
                intro, intro_span = self._excerpt(lead_text, 96, query_set)
                body, body_span = self._excerpt(str(best["text"]), 288, query_set)
                text = intro if not body else f"{intro}\n{body}"
                source_rows = [(lead, intro_span), (best, body_span)]
            if not text.strip():
                continue
            doc_id = str(best.get("id")) if best is not None else str(lead.get("id"))
            self.last_sources[doc_id] = {
                "segments": [{"paragraph_id": row.get("id"), "paragraph_index": row.get("paragraph_index"),
                              "source_url": row.get("source_url"), "char_span": [span[0], span[1]],
                              "text": str(row.get("text", ""))[span[0]:span[1]]}
                             for row, span in source_rows],
                "query": query,
            }
            output.append({"id": doc_id, "title": str(page["title"]), "text": text,
                           "score": page["score"]})
        self.source_history.append(dict(self.last_sources))
        return output


class SearchTool(Tool):
    def __init__(self, corpus: list[dict], retriever_version: str = "keyword-v1", *, tokenizer=None, retrieval=None):
        if retriever_version not in {"keyword-v1", "bm25-v2", "bm25-v3-paragraph", "bm25-v4-page-paragraph", "dense-v1", "hybrid-rrf-v1"}:
            raise ValueError(f"未知检索器版本: {retriever_version}")
        self.corpus = corpus
        self.retriever_version = retriever_version
        self._retrieval = retrieval
        if retriever_version in {"dense-v1", "hybrid-rrf-v1"}:
            if self._retrieval is None:
                from ..retrieval.local import LocalRetrieval
                self._retrieval = LocalRetrieval()
            expected = [(d["id"], d["title"], d["text"]) for d in self._retrieval.documents]
            if [(d["id"], d["title"], d["text"]) for d in corpus] != expected:
                raise ValueError("向量检索仅支持与索引匹配的内置语料")
        if retriever_version == "bm25-v2":
            self._bm25_index = _BM25Index(corpus)
        elif retriever_version == "bm25-v3-paragraph":
            self._bm25_index = _ParagraphBM25Index(corpus)
        elif retriever_version == "bm25-v4-page-paragraph":
            self._bm25_index = _PageParagraphBM25Index(corpus, tokenizer)
        else:
            self._bm25_index = None
        definition = ToolDefinition(name="search", description="在离线文档中检索与问题相关的证据", parameters={"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})
        super().__init__(definition, self._async_search if retriever_version in {"dense-v1", "hybrid-rrf-v1"} else self._search)
    async def _async_search(self, query: str):
        import asyncio
        return await asyncio.to_thread(self._search, query)
    def _search(self, query: str):
        if self.retriever_version in {"dense-v1", "hybrid-rrf-v1"}:
            return self._retrieval.search(query, self.retriever_version)
        if self.retriever_version in {"bm25-v2", "bm25-v3-paragraph", "bm25-v4-page-paragraph"}:
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
