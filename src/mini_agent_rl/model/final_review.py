"""One bounded final review. Quote checks are structural, not semantic proof."""
import json
import time
from ..domain import Message
from .grounded_action_audit import visible_documents, audit_grounded_action

MODES = ("off", "generic-review-v1", "relation-review-v1")
GENERIC = ('Check whether the unverified candidate answers the ORIGINAL QUESTION using only visible document bodies. '
    'Return exactly one JSON action: {"action":"answer","answer":"..."} or '
    '{"action":"search","query":"..."} if a necessary fact is missing. '
    'Preserve roles, relation direction, time and scope. Do not repeat a previous query.')
RELATION = ('Check whether the unverified candidate answers the ORIGINAL QUESTION using only visible document bodies. '
    'Resolve the bridge entity, then the FINAL requested relation and answer role. Do not answer with an intermediate entity. '
    'Return exactly one JSON object, putting links and missing_relation BEFORE action. '
    'links contains zero to two objects with string subject, relation, object, document_id and quote. '
    'Each quote must be an EXACT substring of that document body proving the stated relation. '
    'For an answer, give one or two necessary links, set missing_relation to "", and set answer equal to the last link object. '
    'For a missing link, describe it in missing_relation and output action search with query for a known entity and missing relation. '
    'Do not mistake shared names or co-occurrence for a relation. Do not repeat a previous query. '
    'Example shape only: {"links":[],"missing_relation":"...","action":"search","query":"..."}.')


def actual_messages(base, messages, phase):
    """Use the same renderer/budgeter as generation, never the unbudgeted input."""
    if base is not None and hasattr(base, "_inputs"):
        return base._inputs(messages, phase)._budgeted_messages
    return messages  # Deterministic test clients have no token budgeter.


def review_messages(base, messages, candidate, mode):
    actual = actual_messages(base, messages, "final_review_source")
    docs = visible_documents([m.model_dump() if hasattr(m, "model_dump") else m for m in actual])
    question = next(m.content for m in messages if m.role == "user")
    queries = []
    for message in actual:
        if message.role == "tool":
            payload = json.loads(message.content)
            queries.extend(payload.get("searched_queries", []))
    unique = {(d["document_id"], d["text"]): d for d in docs}
    prepared = [Message(role="system", content=RELATION if mode == "relation-review-v1" else GENERIC),
        Message(role="user", content=json.dumps(dict(original_question=question,
            unverified_candidate=candidate.content, previous_queries=sorted(set(queries))), ensure_ascii=False)),
        Message(role="tool", content=json.dumps(dict(result=[dict(id=d["document_id"], text=d["text"])
            for d in unique.values()]), ensure_ascii=False))]
    return prepared, docs


def validate_review(response, documents, mode):
    if response.parse_error:
        return dict(valid=False, errors=[response.parse_error])
    raw = response.raw_text or ""
    # Production raw may end in EOS. Parse only the first complete JSON object.
    from .local import first_json_object_end
    end = first_json_object_end(raw)
    try:
        value = json.loads(raw[:end] if end is not None else raw)
    except (ValueError, TypeError):
        return dict(valid=False, errors=["invalid_review_json"])
    if not isinstance(value, dict):
        return dict(valid=False, errors=["review_must_be_json_object"])
    if mode == "generic-review-v1":
        key = "query" if value.get("action") == "search" else "answer"
        valid = value.get("action") in ("search", "answer") and isinstance(value.get(key), str) and bool(value[key].strip())
        return dict(valid=valid, errors=[] if valid else ["invalid_review_action"], semantic_support_not_automatically_proven=True)
    audit = audit_grounded_action(json.dumps(value, ensure_ascii=False), documents)
    errors = list(audit["errors"])
    if not audit["exact_quotes_valid"]:
        errors.append("quote_not_in_actual_visible_body")
    links = value.get("links", [])
    if value.get("action") == "answer":
        if not links or value.get("missing_relation"):
            errors.append("answer_requires_links_and_no_missing_relation")
        elif not isinstance(links[-1], dict) or value.get("answer", "").strip() != links[-1].get("object", "").strip():
            errors.append("answer_must_equal_terminal_object")
    elif not value.get("missing_relation", "").strip():
        errors.append("search_requires_missing_relation")
    return dict(valid=not errors and audit["schema_valid"] and audit["exact_quotes_valid"],
        errors=errors, links=audit["links"], semantic_support_not_automatically_proven=True)


class FinalReview:
    def __init__(self, mode, generate, base=None):
        if mode not in MODES or mode == "off":
            raise ValueError("FinalReview requires an enabled mode")
        self.mode, self.generate, self.base = mode, generate, base
        self.model_calls = self.auxiliary_calls = self.decisions = 0

    async def __call__(self, messages, tools, **kwargs):
        if self.decisions >= 4 or self.model_calls >= 5:
            raise RuntimeError("Final review call budget exceeded")
        self.decisions += 1
        self.model_calls += 1
        candidate = await self.generate(messages, tools, **kwargs)
        if candidate.parse_error or candidate.tool_call or self.auxiliary_calls:
            return candidate
        prepared, documents = review_messages(self.base, messages, candidate, self.mode)
        if not documents:
            return candidate
        if self.model_calls >= 5:
            raise RuntimeError("Final review auxiliary budget exceeded")
        self.model_calls += 1
        self.auxiliary_calls += 1
        before = time.monotonic()
        reviewed = await self.generate(prepared, [], **kwargs)
        elapsed = time.monotonic() - before
        actual = actual_messages(self.base, prepared, "final_review_quote_audit")
        visible = visible_documents([m.model_dump() if hasattr(m, "model_dump") else m for m in actual])
        audit = validate_review(reviewed, visible, self.mode)
        record = dict(mode=self.mode, candidate=candidate.model_dump(), response=reviewed.model_dump(), audit=audit,
            accepted=audit["valid"], elapsed_seconds=elapsed, model_calls=self.model_calls,
            auxiliary_calls=self.auxiliary_calls, auxiliary_input_tokens=len(reviewed.prompt_token_ids or []),
            auxiliary_output_tokens=len(reviewed.action_token_ids or []))
        result = reviewed if audit["valid"] else candidate
        return result.model_copy(update={"usage": dict(result.usage, final_review=record)})
