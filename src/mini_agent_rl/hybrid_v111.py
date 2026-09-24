"""Pure v111 evidence registry, strict protocol validation and preservation policy.

All public functions return JSON-serializable dictionaries. Validators accept
decoded JSON objects (not JSON strings). Citation validity is provenance only;
it never establishes semantic support. ``decide`` consumes assessments remapped
by the caller from anonymous IDs to ``original`` and ``selected``.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

MAX_TEXT = 4000
MAX_REFS = 64


def build_evidence(documents: list[dict[str, str]]) -> dict[str, Any]:
    """Return snapshot_hash, registry and full annotated evidence.

    Offsets count Python Unicode characters, end-exclusive. Paragraph boundaries
    are preserved; paragraphs longer than 800 characters are split losslessly.
    IDs are bound to the entire ordered snapshot, document and span offsets.
    """
    if not isinstance(documents, list):
        raise ValueError('documents_list_required')
    seen = set()
    for doc in documents:
        if not isinstance(doc, dict) or not isinstance(doc.get('id'), str) or not doc['id']:
            raise ValueError('document_id_required')
        if doc['id'] in seen:
            raise ValueError('duplicate_document_id')
        seen.add(doc['id'])
        if not isinstance(doc.get('text'), str) or not isinstance(doc.get('title', ''), str):
            raise ValueError('document_text_title_required')
    canonical = [{'id': d['id'], 'title': d.get('title', ''), 'text': d['text']} for d in documents]
    snapshot = hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                                         separators=(',', ':')).encode('utf-8')).hexdigest()
    registry, evidence = {}, []
    for index, doc in enumerate(canonical):
        spans = []
        start = 0
        boundaries = [m.end() for m in re.finditer(r'(?:\r?\n)[ \t]*(?:\r?\n)', doc['text'])]
        boundaries.append(len(doc['text']))
        for boundary in boundaries:
            while start < boundary:
                end = min(start + 800, boundary)
                identifier = f'E:{snapshot}:{index}:{start}:{end}'
                span = {'evidence_id': identifier, 'start': start, 'end': end, 'text': doc['text'][start:end]}
                spans.append(span)
                registry[identifier] = {'document_id': doc['id'], 'snapshot_hash': snapshot,
                                        'start': start, 'end': end, 'text': span['text']}
                start = end
        evidence.append({**doc, 'spans': spans})
    return {'snapshot_hash': snapshot, 'registry': registry, 'evidence': evidence}


def validate_refs(ids: Any, registry: dict[str, Any]) -> dict[str, Any]:
    """Check a bounded list of unique IDs against the supplied snapshot registry."""
    errors = []
    if not isinstance(ids, list) or len(ids) > MAX_REFS:
        return {'valid': False, 'errors': ['evidence_ids_list_required_or_too_long'], 'evidence_ids': []}
    seen = set()
    for i, identifier in enumerate(ids):
        if not isinstance(identifier, str):
            errors.append(f'evidence_{i}:string_required')
        elif identifier in seen:
            errors.append(f'evidence_{i}:duplicate')
        else:
            seen.add(identifier)
            if identifier not in registry:
                errors.append(f'evidence_{i}:unknown_or_stale')
    return {'valid': not errors, 'errors': errors, 'evidence_ids': ids}


def _object(raw: Any, required: set[str], optional: set[str] | None = None) -> list[str]:
    if not isinstance(raw, dict):
        return ['object_required']
    return (['missing_fields'] if required - raw.keys() else []) + (
        ['unexpected_fields'] if raw.keys() - required - (optional or set()) else [])


def _text(value: Any, allow_empty: bool = False) -> bool:
    return isinstance(value, str) and len(value) <= MAX_TEXT and (allow_empty or bool(value.strip()))


def validate_candidates(raw: Any, registry: dict[str, Any]) -> dict[str, Any]:
    """Validate {candidates:[{answer,evidence_ids}]} (at most four)."""
    errors = _object(raw, {'candidates'})
    values = raw.get('candidates') if isinstance(raw, dict) else None
    if not isinstance(values, list) or len(values) > 4:
        return {'valid': False, 'errors': errors + ['candidates_list_required_or_too_long'], 'candidates': []}
    for i, candidate in enumerate(values):
        local = _object(candidate, {'answer', 'evidence_ids'})
        if isinstance(candidate, dict):
            if not _text(candidate.get('answer')):
                local.append('answer_invalid')
            local.extend(validate_refs(candidate.get('evidence_ids'), registry)['errors'])
            if candidate.get('evidence_ids') == []:
                local.append('missing_evidence')
        errors.extend(f'candidate_{i}:{e}' for e in local)
    return {'valid': not errors, 'errors': errors, 'candidates': values if not errors else []}


def validate_selection(raw: Any, candidates: list[dict[str, Any]], registry: dict[str, Any]) -> dict[str, Any]:
    """Validate {decision,candidate_id,evidence_ids}; candidates have candidate_id."""
    errors = _object(raw, {'decision', 'candidate_id', 'evidence_ids'})
    if not isinstance(raw, dict):
        return {'valid': False, 'errors': errors, 'decision': 'abstain', 'candidate_id': None, 'evidence_ids': []}
    decision, identifier = raw.get('decision'), raw.get('candidate_id')
    errors.extend(validate_refs(raw.get('evidence_ids'), registry)['errors'])
    ids = [c.get('candidate_id') for c in candidates if isinstance(c, dict)]
    if len(ids) != len(candidates) or any(not isinstance(i, str) for i in ids) or len(set(i for i in ids if isinstance(i, str))) != len(ids):
        errors.append('candidate_ids_invalid')
    if decision == 'select':
        if not isinstance(identifier, str) or identifier not in ids:
            errors.append('unknown_candidate')
        if raw.get('evidence_ids') == []:
            errors.append('missing_evidence')
    elif decision == 'abstain':
        if identifier is not None or raw.get('evidence_ids') != []:
            errors.append('abstain_requires_null_id_empty_evidence')
    else:
        errors.append('decision_invalid')
    return {'valid': not errors, 'errors': errors, 'decision': decision,
            'candidate_id': identifier, 'evidence_ids': raw.get('evidence_ids', [])}


def validate_review(raw: Any, answer_ids: list[str], registry: dict[str, Any]) -> dict[str, Any]:
    """Validate independent assessments, retaining validity and errors per answer.

    Required top-level fields: assessments, information_gap, queries (<=2).
    Each assessment: id,status,reason,evidence_ids. Supported and contradicted
    require references. An invalid sibling does not erase a valid assessment.
    """
    if not isinstance(answer_ids, list) or not answer_ids or any(not isinstance(i, str) or not i for i in answer_ids) or len(set(answer_ids)) != len(answer_ids):
        raise ValueError('answer_ids_invalid')
    errors = _object(raw, {'assessments', 'information_gap', 'queries'})
    raw = raw if isinstance(raw, dict) else {}
    values = raw.get('assessments')
    if not isinstance(values, list) or len(values) > 4:
        errors.append('assessments_list_required_or_too_long')
        values = []
    if not _text(raw.get('information_gap'), allow_empty=True):
        errors.append('information_gap_invalid')
    queries = raw.get('queries')
    if not isinstance(queries, list) or len(queries) > 2 or any(not _text(q) for q in queries):
        errors.append('queries_invalid')
        queries = []
    mapped = {}
    for identifier in answer_ids:
        matching = [a for a in values if isinstance(a, dict) and a.get('id') == identifier]
        if len(matching) != 1:
            mapped[identifier] = {'id': identifier, 'status': 'insufficient', 'reason': '',
                                  'evidence_ids': [], 'valid': False, 'errors': ['missing_or_duplicate_assessment']}
            continue
        a = matching[0]
        local = _object(a, {'id', 'status', 'reason', 'evidence_ids'})
        if a.get('status') not in ('supported', 'contradicted', 'insufficient'):
            local.append('status_invalid')
        if not _text(a.get('reason')):
            local.append('reason_invalid')
        local.extend(validate_refs(a.get('evidence_ids'), registry)['errors'])
        if a.get('status') in ('supported', 'contradicted') and a.get('evidence_ids') == []:
            local.append('missing_evidence')
        mapped[identifier] = {**a, 'valid': not local, 'errors': local}
    if any(not isinstance(a, dict) or a.get('id') not in answer_ids for a in values):
        errors.append('unexpected_assessment')
    errors.extend(f'{identifier}:{e}' for identifier, a in mapped.items() for e in a['errors'])
    return {'valid': not errors, 'errors': errors, 'assessments': mapped,
            'information_gap': raw.get('information_gap', ''), 'queries': queries}


def decide(original: str, selected: str, review: dict[str, Any], protect: bool = True) -> dict[str, Any]:
    """Preserve a valid supported original first; otherwise accept supported selection.

    Invalid overall protocol is retained even if one assessment permits preserving
    the original. Replacing an answer requires the complete review to be valid.
    """
    assessments = review.get('assessments', {})
    def supported(key: str) -> bool:
        a = assessments.get(key, {})
        return a.get('valid') is True and a.get('status') == 'supported'
    answer, status = original, 'fallback_original'
    if protect and supported('original'):
        status = 'preserve_supported_original'
    elif review.get('valid') is True and supported('selected'):
        answer, status = selected, 'selected_supported'
    return {'answer': answer, 'status': status, 'protocol_valid': review.get('valid') is True,
            'errors': review.get('errors', [])}
