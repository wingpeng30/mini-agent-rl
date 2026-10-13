"""Lossless presentation for retrieved document observations."""
from __future__ import annotations

import hashlib
import json
from typing import Any


LEGACY_JSON = "legacy-json"
DOCUMENT_BLOCKS_V1 = "document-blocks-v1"


def present_observation(raw: str, presentation: str) -> tuple[str, list[dict[str, Any]]]:
    """Render one tool JSON payload and retain exact body offsets.

    The document-block mode changes only the textual representation. All
    metadata and bodies are preserved in order, including duplicate documents.
    """
    if presentation == LEGACY_JSON:
        return raw, []
    if presentation != DOCUMENT_BLOCKS_V1:
        raise ValueError(f"unknown observation presentation: {presentation}")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("result", []), list):
        raise ValueError("document-blocks-v1 requires an observation object with a result list")

    if any(not isinstance(item, dict) or not isinstance(item.get("text"), str) for item in payload.get("result", [])):
        raise ValueError("document-blocks-v1 requires every search result to have a string body")
    docs = payload.get("result", [])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    nonce = digest
    suffix = 0
    while True:
        begin = f"<<<BEGIN-SEARCH-DOCUMENT-{nonce}>>>"
        end = f"<<<END-SEARCH-DOCUMENT-{nonce}>>>"
        if all(begin not in str(doc.get("text", "")) and end not in str(doc.get("text", "")) for doc in docs):
            break
        suffix += 1
        nonce = f"{digest}-{suffix}"

    metadata = {key: value for key, value in payload.items() if key != "result"}
    chunks = ["Search result metadata (JSON):\n", json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), "\n"]
    cursor = sum(map(len, chunks))
    spans: list[dict[str, Any]] = []
    for rank, doc in enumerate(docs, 1):
        doc_id = str(doc.get("id", ""))
        title = str(doc.get("title", ""))
        body = doc["text"]
        other = {key: value for key, value in doc.items() if key != "text"}
        prefix = (f"\n{begin}\nDocument rank: {rank}\n"
                  f"Document metadata (JSON):\n{json.dumps(other, ensure_ascii=False, separators=(',', ':'))}\n"
                  "Full document text:\n")
        chunks.append(prefix)
        cursor += len(prefix)
        body_start = cursor
        chunks.append(body)
        cursor += len(body)
        body_end = cursor
        chunks.append(f"\n{end}\n")
        cursor += len(f"\n{end}\n")
        spans.append({"rank": rank, "document_id": doc_id, "title": title,
                      "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                      "body_chars": len(body), "rendered_char_span": [body_start, body_end]})
    return "".join(chunks), spans


def parse_document_blocks(rendered: str) -> dict[str, Any]:
    """Inverse document-blocks-v1 renderer, used for losslessness validation."""
    header = "Search result metadata (JSON):\n"
    if not rendered.startswith(header):
        raise ValueError("missing document-block metadata header")
    first_end = rendered.find("\n", len(header))
    if first_end < 0:
        raise ValueError("missing document-block metadata terminator")
    metadata = json.loads(rendered[len(header):first_end])
    rest = rendered[first_end + 1:]
    marker_start = "<<<BEGIN-SEARCH-DOCUMENT-"
    documents: list[dict[str, Any]] = []
    while rest:
        if not rest.startswith("\n" + marker_start):
            raise ValueError("unexpected text between document blocks")
        first_line_end = rest.find("\n", 1)
        begin = rest[1:first_line_end]
        nonce = begin.removeprefix("<<<BEGIN-SEARCH-DOCUMENT-").removesuffix(">>>")
        if not nonce or not begin.endswith(">>>"):
            raise ValueError("invalid begin marker")
        prefix = (f"\n{begin}\nDocument rank: ")
        if not rest.startswith(prefix):
            raise ValueError("invalid document block prefix")
        rank_end = rest.find("\n", len(prefix))
        rank = int(rest[len(prefix):rank_end])
        metadata_header = "Document metadata (JSON):\n"
        if not rest.startswith(metadata_header, rank_end + 1):
            raise ValueError("missing document metadata")
        meta_start = rank_end + 1 + len(metadata_header)
        meta_end = rest.find("\n", meta_start)
        document_metadata = json.loads(rest[meta_start:meta_end])
        text_header = "Full document text:\n"
        if not rest.startswith(text_header, meta_end + 1):
            raise ValueError("missing document body")
        body_start = meta_end + 1 + len(text_header)
        end_marker = f"\n<<<END-SEARCH-DOCUMENT-{nonce}>>>\n"
        body_end = rest.find(end_marker, body_start)
        if body_end < 0:
            raise ValueError("missing document end marker")
        body = rest[body_start:body_end]
        if rank != len(documents) + 1:
            raise ValueError("document ranks are not sequential")
        documents.append({**document_metadata, "text": body})
        rest = rest[body_end + len(end_marker):]
    return {**metadata, "result": documents}
