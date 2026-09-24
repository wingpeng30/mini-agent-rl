"""Evidence-bounded candidate generation and discriminative answer selection."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .answer_scoring import normalize_answer
from .domain import Message

CANDIDATE_SYSTEM = """Propose up to four alternative short answers to the question using only exact contiguous spans from the supplied evidence. Return exactly {\"candidates\":[\"...\"]}. Do not include the existing answer. Do not explain, infer, combine spans, or use outside knowledge."""
SELECTOR_SYSTEM = """Select the answer best supported by the supplied evidence. Reply with exactly one option label. Select NONE when no listed answer is supported. Use only the supplied question and evidence."""
OPTION_LABELS = ("A", "B", "C", "D", "E", "NONE")

class CandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[str] = Field(default_factory=list, max_length=4)

def evidence_sha256(documents: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(documents, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()

def candidate_messages(question: str, documents: list[dict[str, str]], original: str) -> list[Message]:
    payload = {"question": question, "evidence": documents, "existing_answer": original}
    return [Message(role="system", content=CANDIDATE_SYSTEM),
            Message(role="user", content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))]

def _span_location(value: str, documents: list[dict[str, str]]) -> dict[str, Any] | None:
    if not value or value != value.strip():
        return None
    for document in documents:
        start = document["text"].find(value)
        if start >= 0:
            return {"document_id": document["id"], "start": start, "end": start + len(value)}
    return None

def parse_candidates(raw: str, original: str, documents: list[dict[str, str]]) -> tuple[list[dict[str, Any]], str | None]:
    try:
        payload = CandidatePayload.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        return [], f"invalid_json: {exc}"
    seen = {normalize_answer(original)}
    accepted: list[dict[str, Any]] = []
    for value in payload.candidates:
        normalized = normalize_answer(value)
        location = _span_location(value, documents)
        if not normalized or normalized in seen or location is None:
            continue
        seen.add(normalized)
        accepted.append({"answer": value, **location})
    return accepted, None

def ordered_options(task_id: str, original: str, alternatives: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    values = [{"answer": original, "origin": "original"}] + [
        {"answer": item["answer"], "origin": "evidence_span", "location": {
            key: item[key] for key in ("document_id", "start", "end")}}
        for item in alternatives[:4]
    ]
    stable = int(hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(), 16)
    random.Random(stable).shuffle(values)
    return values

def selector_messages(question: str, documents: list[dict[str, str]], options: list[dict[str, Any]]) -> list[Message]:
    rendered = [{"label": OPTION_LABELS[index], "answer": item["answer"]}
                for index, item in enumerate(options)]
    rendered.append({"label": "NONE", "answer": "No listed answer is supported"})
    payload = {"question": question, "evidence": documents, "options": rendered}
    return [Message(role="system", content=SELECTOR_SYSTEM),
            Message(role="user", content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))]

def correct_option_labels(options: list[dict[str, Any]], gold: str) -> list[str]:
    labels = [OPTION_LABELS[index] for index, item in enumerate(options)
              if normalize_answer(item["answer"]) == normalize_answer(gold)]
    return labels or ["NONE"]

def dedupe_key(task_id: str, evidence_hash: str, candidate: str, target: str) -> tuple[str, str, str, str]:
    return task_id, evidence_hash, normalize_answer(candidate), target

@dataclass(frozen=True)
class Selection:
    answer: str
    label: str
    status: str

def apply_label(label: str, options: list[dict[str, Any]], original: str) -> Selection:
    if label == "NONE":
        return Selection(original, label, "abstain_preserve")
    try:
        index = OPTION_LABELS.index(label)
    except ValueError:
        return Selection(original, label, "fallback_invalid_label")
    if index >= len(options):
        return Selection(original, label, "fallback_invalid_label")
    answer = options[index]["answer"]
    return Selection(answer, label, "preserve" if answer == original else "replace")
