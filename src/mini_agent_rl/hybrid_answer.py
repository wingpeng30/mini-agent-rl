"""Schemas and validation for the v109 evidence-grounded hybrid answerer."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .answer_scoring import normalize_answer, token_f1


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    quote: str


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    citations: list[Citation] = Field(min_length=1)


class CandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[Candidate] = Field(max_length=4)


class DirectResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    citations: list[Citation] = Field(min_length=1)


class SelectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["select", "abstain"]
    candidate_id: str | None = None
    citations: list[Citation] = Field(default_factory=list)


def evidence_hash(documents: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(documents, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_citations(citations: list[Citation], documents: list[dict[str, str]]) -> bool:
    by_id = {str(item["id"]): str(item["text"]) for item in documents}
    return bool(citations) and all(ref.document_id in by_id and ref.quote and ref.quote in by_id[ref.document_id]
                                   for ref in citations)


def parse_candidates(raw: str, documents: list[dict[str, str]]) -> tuple[list[dict[str, Any]], str | None]:
    try:
        parsed = CandidateResponse.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        return [], f"invalid_schema: {exc}"
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for candidate in parsed.candidates:
        key = normalize_answer(candidate.answer)
        if not key or key in seen or not validate_citations(candidate.citations, documents):
            continue
        seen.add(key)
        output.append(candidate.model_dump(mode="json"))
    return output, None


def merge_candidates(original: str, passes: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged = [{"answer": original, "origin": "original", "citations": []}]
    seen = {normalize_answer(original)}
    for pass_index, rows in enumerate(passes, 1):
        for row in rows:
            key = normalize_answer(row["answer"])
            if key and key not in seen and len(merged) <= 8:
                seen.add(key); merged.append({**row, "origin": f"pass_{pass_index}"})
    return merged


def anonymize_candidates(task_id: str, candidates: list[dict[str, Any]], order_variant: int = 0) -> list[dict[str, Any]]:
    values = [dict(item) for item in candidates]
    seed = int(hashlib.sha256(f"42:{order_variant}:{task_id}".encode()).hexdigest(), 16)
    random.Random(seed).shuffle(values)
    return [{"candidate_id": f"C{index + 1}", **item} for index, item in enumerate(values)]


def apply_selection(raw: str, candidates: list[dict[str, Any]], original: str,
                    documents: list[dict[str, str]]) -> tuple[str, str, str | None]:
    try:
        parsed = SelectionResponse.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        return original, "fallback_invalid_schema", str(exc)
    if parsed.decision == "abstain":
        return original, "abstain", None
    by_id = {item["candidate_id"]: item for item in candidates}
    if parsed.candidate_id not in by_id or not validate_citations(parsed.citations, documents):
        return original, "fallback_invalid_selection", "missing candidate or invalid citation"
    return str(by_id[parsed.candidate_id]["answer"]), "selected", None


def conservative_boundary_guard(selected: str, original: str,
                                candidates: list[dict[str, Any]]) -> tuple[str, str]:
    """Remove explanatory answer tails and preserve alias-like baseline answers."""
    selected_key = normalize_answer(selected)
    embedded = [str(item["answer"]) for item in candidates
                if normalize_answer(str(item["answer"])) and
                normalize_answer(str(item["answer"])) in selected_key]
    compressed = min(embedded, key=lambda value: len(normalize_answer(value))) if embedded else selected
    original_key, compressed_key = normalize_answer(original), normalize_answer(compressed)
    if original_key and compressed_key and (original_key in compressed_key or compressed_key in original_key):
        return original, "preserve_alias_boundary"
    return compressed, "compressed_selection" if compressed != selected else "unchanged_selection"


def answer_metrics(prediction: str, gold: str) -> dict[str, float]:
    return {"strict_em": float(prediction.strip() == gold.strip()),
            "normalized_em": float(normalize_answer(prediction) == normalize_answer(gold)),
            "f1": token_f1(prediction, gold)}
