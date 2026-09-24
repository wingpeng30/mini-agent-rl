"""Evidence-bounded post-answer verification for frozen agent trajectories."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .domain import Message


VERIFIER_SYSTEM = """You verify one candidate answer using only the supplied question and evidence. Return exactly one JSON object with decision, answer, and evidence_refs. decision is keep, replace, or insufficient. Keep preserves the candidate exactly. Replace is allowed only when quoted visible evidence directly supports the corrected answer. Insufficient means the evidence cannot justify a correction and also preserves the candidate. evidence_refs is a list of {id, quote}; every quote must occur verbatim in the referenced visible document. Do not use outside knowledge."""


class EvidenceRef(BaseModel):
    id: str
    quote: str


class VerifierAction(BaseModel):
    decision: Literal["keep", "replace", "insufficient"]
    answer: str
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


def visible_documents(messages: list[dict[str, Any]] | list[Message]) -> list[dict[str, str]]:
    """Recover only document text present in model-visible tool messages."""
    documents: dict[str, dict[str, str]] = {}
    for raw in messages:
        role = raw.role if isinstance(raw, Message) else raw.get("role")
        content = raw.content if isinstance(raw, Message) else raw.get("content", "")
        if role != "tool":
            continue
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            continue
        for item in payload.get("result") or []:
            if not isinstance(item, dict) or not item.get("id") or not item.get("text"):
                continue
            documents[str(item["id"])] = {"id": str(item["id"]), "title": str(item.get("title", "")), "text": str(item["text"])}
    return list(documents.values())


def verifier_messages(question: str, documents: list[dict[str, str]], candidate: str) -> list[Message]:
    payload = {"question": question, "evidence": documents, "candidate_answer": candidate}
    return [Message(role="system", content=VERIFIER_SYSTEM),
            Message(role="user", content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))]


@dataclass
class AppliedVerification:
    answer: str
    status: str
    action: VerifierAction | None
    error: str | None = None


def parse_and_apply(raw: str, candidate: str, documents: list[dict[str, str]]) -> AppliedVerification:
    """Validate protocol and citations; every failure safely preserves candidate."""
    try:
        action = VerifierAction.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        return AppliedVerification(candidate, "fallback_invalid_json", None, str(exc))
    by_id = {item["id"]: item["text"] for item in documents}
    for ref in action.evidence_refs:
        if ref.id not in by_id or not ref.quote or ref.quote not in by_id[ref.id]:
            return AppliedVerification(candidate, "fallback_invalid_reference", action, f"invalid reference: {ref.id}")
    if action.decision in {"keep", "insufficient"}:
        if action.answer != candidate:
            return AppliedVerification(candidate, "fallback_changed_preserved_answer", action, "keep/insufficient must preserve candidate")
        return AppliedVerification(candidate, action.decision, action)
    if not action.answer.strip() or not action.evidence_refs:
        return AppliedVerification(candidate, "fallback_unsupported_replace", action, "replace requires answer and evidence")
    return AppliedVerification(action.answer.strip(), "replace", action)
