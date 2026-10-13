"""Protocol-only query guard for a future v123 inference comparison.

Lexical anchor findings are diagnostics. They cannot establish whether the
query denotes the correct entity or relation and never block a search.
"""
from __future__ import annotations

import re
import unicodedata


def query_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def query_check(query: str, previous: set[str], question: str, visible_text: list[str]) -> dict:
    if not isinstance(query, str) or not query.strip():
        return {"allow": False, "reason": "empty_query", "lexical_anchor": None}
    if query_key(query) in {query_key(item) for item in previous}:
        return {"allow": False, "reason": "repeated_query", "lexical_anchor": None}
    # Token-level surface overlap is deliberately weak; even an overlap of
    # several names can refer to the wrong person, city, or relation.
    visible = unicodedata.normalize("NFKC", question + " " + " ".join(visible_text)).casefold()
    tokens = [token for token in re.findall(r"[\w!]+", unicodedata.normalize("NFKC", query).casefold())
              if len(token) > 2]
    diagnostic = any(token in visible for token in tokens)
    return {"allow": True, "reason": None, "lexical_anchor": diagnostic,
            "semantic_grounding": "unverified"}
