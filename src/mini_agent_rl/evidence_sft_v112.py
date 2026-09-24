"""Fail-closed, offline acceptance checks for reviewed v112 evidence SFT rows."""
from __future__ import annotations

import hashlib
import json
import posixpath
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

RISK_LABELS = frozenset({'relation', 'measurement', 'time', 'scope'})


def load_budget_policy(root: Path) -> dict:
    path=root/'configs/v112-budget-policy.json'
    if not path.exists():
        return {'mode':'capped','cumulative_cap_cny':39.600696,'new_cap_cny':12,'sha256':None}
    raw=path.read_bytes();policy=json.loads(raw)
    if policy.get('mode')!='uncapped' or policy.get('version')!='v112' or not policy.get('authorization'):
        raise ValueError('invalid_budget_authorization')
    if policy.get('cumulative_cap_cny') is not None or policy.get('new_cap_cny') is not None:
        raise ValueError('inconsistent_uncapped_policy')
    return {**policy,'sha256':hashlib.sha256(raw).hexdigest()}


def normalize_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError('question_required')
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in
                           unicodedata.normalize('NFKC', question).casefold()).split())


def compute_evidence_hash(evidence: list[dict]) -> str:
    payload = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def compute_state_hash(question: str, messages: list[dict]) -> str:
    """Targets and reviewer metadata deliberately do not affect state identity."""
    visible = [{'role': m['role'], 'content': m['content']} for m in messages[:-1]]
    payload = {'question': normalize_question(question), 'messages': visible}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def compute_candidate_hash(row: dict) -> str:
    """Bind a review to its target while keeping state deduplication target-free."""
    value = [row['state_hash'], row['target'], row['evidence_hash']]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def validate_source_path(path: str | Path) -> Path:
    """Reject protected sources before opening, including encoded/alias spellings."""
    raw = unicodedata.normalize('NFKC', str(path))
    for _ in range(8):
        decoded = unquote(raw)
        if decoded == raw:
            break
        raw = decoded
    lexical = raw.replace('\\', '/')
    resolved = str(Path(raw).resolve()).replace('\\', '/')
    for candidate in (lexical, posixpath.normpath(lexical), resolved):
        for component in candidate.split('/'):
            compact = re.sub(r'[^a-z0-9]', '', component.casefold())
            if any(token in compact for token in ('finaltest', 'independent', 'holdout')):
                raise ValueError('protected_source_path')
    return Path(raw)


def validate_row(row: dict) -> dict:
    """Validate explicit provenance and review metadata; never infer missing values."""
    if not isinstance(row, dict):
        raise ValueError('row_object_required')
    required = {'question', 'task_id', 'messages', 'target', 'step', 'risk_labels',
                'evidence', 'evidence_hash', 'reviews', 'state_hash', 'target_assistant_index'}
    if required - row.keys():
        raise ValueError('missing_fields:' + ','.join(sorted(required - row.keys())))
    normalize_question(row['question'])
    if not isinstance(row['task_id'], str) or not row['task_id'].strip():
        raise ValueError('task_id_required')
    messages = row['messages']
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages_required')
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                or message['role'] not in {'system', 'user', 'assistant', 'tool'}
                or not isinstance(message['content'], str)):
            raise ValueError('invalid_visible_message')
    if (type(row['target_assistant_index']) is not int or
            row['target_assistant_index'] != len(messages) - 1 or messages[-1]['role'] != 'assistant'):
        raise ValueError('final_target_assistant_required')
    target = row['target']
    if (not isinstance(target, dict) or set(target) != {'action', 'value'}
            or target['action'] not in {'answer', 'search'}
            or not isinstance(target['value'], str) or not target['value'].strip()):
        raise ValueError('invalid_target')
    action = json.loads(messages[-1]['content'])
    value_key = 'answer' if target['action'] == 'answer' else 'query'
    if action != {'action': target['action'], value_key: target['value']}:
        raise ValueError('target_message_mismatch')
    if type(row['step']) is not int or row['step'] < 0:
        raise ValueError('invalid_step')
    risks = row['risk_labels']
    if (not isinstance(risks, list) or not all(isinstance(x, str) for x in risks)
            or len(set(risks)) != len(risks) or not set(risks) <= RISK_LABELS):
        raise ValueError('invalid_risk_labels')
    evidence = row['evidence']
    if (not isinstance(evidence, list)
            or any(not isinstance(d, dict) or set(d) != {'id', 'title', 'text'}
                   or any(not isinstance(d[k], str) for k in ('id', 'title', 'text'))
                   or not d['id'].strip() or not d['text'].strip() for d in evidence)
            or len({d['id'] for d in evidence}) != len(evidence)):
        raise ValueError('evidence_required')
    if not evidence and (target['action'] != 'search' or row['step'] != 0):
        raise ValueError('empty_evidence_only_initial_search')
    from .answer_verifier import visible_documents
    try:
        visible = visible_documents(messages[:-1])
    except (AttributeError, TypeError) as exc:
        raise ValueError('malformed_tool_evidence') from exc
    if evidence != visible:
        raise ValueError('visible_evidence_mismatch')
    if row['evidence_hash'] != compute_evidence_hash(evidence):
        raise ValueError('evidence_hash_mismatch')
    if any(re.search(r'(?i)["\']?\b(?:gold(?:en)?[ _-]?(?:answer|label)?|ground[ _-]?truth|support(?:ing)?[ _-]?(?:label|facts)|answer[ _-]?key)["\']?\s*[:=]',
                     m['content']) for m in messages):
        raise ValueError('gold_metadata_in_messages')
    reviews = row['reviews']
    if (not isinstance(reviews, list) or len(reviews) != 2
            or any(not isinstance(r, dict) or r.get('approved') is not True
                   or r.get('kind') not in {'api', 'task_agent'}
                   or not isinstance(r.get('reason'), str) or not r['reason'].strip()
                   or r.get('state_hash') != row['state_hash']
                   or r.get('candidate_hash') != compute_candidate_hash(row)
                   or not isinstance(r.get('reviewer'), str) or not r['reviewer'].strip()
                   for r in reviews)
            or {r['kind'] for r in reviews} != {'api', 'task_agent'}
            or len({r['reviewer'].strip().casefold() for r in reviews}) != 2):
        raise ValueError('dual_approved_reviews_required')
    if row['state_hash'] != compute_state_hash(row['question'], messages):
        raise ValueError('state_hash_mismatch')
    for review in reviews:
        docs = evidence if target['action']=='answer' else evidence + row.get('retrieval_results',[])[:3]
        texts = {d['id']:d['text'] for d in docs}
        citations = review.get('citations',[])
        if not citations or any(c.get('document_id') not in texts or not isinstance(c.get('quote'),str)
                               or not c['quote'] or c['quote'] not in texts[c['document_id']] for c in citations):
            raise ValueError('review_citations_required')
        if target['action']=='search':
            if not review.get('missing_fact'):
                raise ValueError('search_gap_required')
            current_texts = {d['text'] for d in evidence}
            retrieved = {d['id']:d['text'] for d in row.get('retrieval_results',[])[:3]}
            if not any(c['document_id'] in retrieved and c['quote'] in retrieved[c['document_id']]
                       and retrieved[c['document_id']] not in current_texts for c in citations):
                raise ValueError('search_citation_must_add_evidence')
    if target['action']=='search':
        prior=[]
        for message in messages[:-1]:
            if message['role']=='assistant':
                previous=json.loads(message['content'])
                if previous.get('action')=='search':prior.append(normalize_question(previous['query']))
        if normalize_question(target['value']) in prior:
            raise ValueError('repeated_search_target')
    return row


def _validate_dataset(train: list[dict], internal: list[dict]) -> dict:
    """Raise ValueError on any failed acceptance gate; return auditable counts."""
    groups: dict[str, set[str]] = {}
    seen: set[str] = set()
    report = {}
    for split, rows, minimum, maximum, tasks, risk_min in (
        ('train', train, 300, 400, 150, 20),
        ('internal', internal, 100, None, 50, 5),
    ):
        if not isinstance(rows, list) or len(rows) < minimum or (maximum and len(rows) > maximum):
            raise ValueError(f'{split}:state_count')
        for row in rows:
            validate_row(row)
            if row['state_hash'] in seen:
                raise ValueError('duplicate_state')
            seen.add(row['state_hash'])
        groups[split] = {normalize_question(r['question']) for r in rows}
        # Count question groups as tasks too, so aliases cannot inflate coverage.
        if len({r['task_id'] for r in rows}) < tasks or len(groups[split]) < tasks:
            raise ValueError(f'{split}:task_count')
        actions = Counter(r['target']['action'] for r in rows)
        if any(actions[action] * 10 < len(rows) * 3 for action in ('answer', 'search')):
            raise ValueError(f'{split}:action_balance')
        if sum(r['target']['action']=='search' and bool(r['evidence']) for r in rows)*10 < len(rows)*3:
            raise ValueError(f'{split}:nonempty_search_balance')
        if sum(r['step'] == 0 for r in rows) * 5 > len(rows):
            raise ValueError(f'{split}:step0_fraction')
        risks = Counter(label for row in rows for label in row['risk_labels'])
        if any(risks[risk] < risk_min for risk in RISK_LABELS):
            raise ValueError(f'{split}:risk_coverage')
        report[split] = {'states': len(rows), 'tasks': len(groups[split]),
                         'actions': dict(actions), 'risks': dict(risks)}
    if groups['train'] & groups['internal']:
        raise ValueError('question_group_overlap')
    if {r['task_id'] for r in train} & {r['task_id'] for r in internal}:
        raise ValueError('task_id_overlap')
    return {'passed': True, 'errors': [], 'stats': report}


def validate_dataset(train: list[dict], internal: list[dict]) -> dict:
    """Return a fail-closed acceptance report without reading or writing files."""
    try:
        return _validate_dataset(train, internal)
    except (ValueError, TypeError, KeyError) as exc:
        return {'passed': False, 'errors': [str(exc)], 'stats': {}}
