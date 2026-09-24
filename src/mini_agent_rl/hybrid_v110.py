"""Auditable v110 selection, official Hotpot answer scoring, and durable API budgets."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re
import sqlite3
import string
from pathlib import Path
from typing import Any

from .hybrid_answer import SelectionResponse


def normalize_answer(text: str) -> str:
    # Official evaluator order matters: remove punctuation before articles.
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def token_f1(prediction: str, reference: str) -> float:
    pred, gold = normalize_answer(prediction), normalize_answer(reference)
    if pred != gold and ({pred, gold} & {'yes', 'no', 'noanswer'}):
        return 0.0
    a, b = pred.split(), gold.split()
    common = sum((Counter(a) & Counter(b)).values())
    if not common:
        return 0.0
    precision, recall = common / len(a), common / len(b)
    return 2 * precision * recall / (precision + recall)


def answer_metrics(prediction: str, gold: str) -> dict[str, float]:
    return {'strict_em': float(prediction.strip() == gold.strip()),
            'normalized_em': float(normalize_answer(prediction) == normalize_answer(gold)),
            'f1': token_f1(prediction, gold)}


def validate_citations_detailed(citations: list[Any], documents: list[dict[str, str]]) -> dict[str, Any]:
    errors: list[str] = []
    by_id: dict[str, str] = {}
    for document in documents:
        doc_id = str(document['id'])
        if doc_id in by_id:
            errors.append(f'duplicate_document_id:{doc_id}')
        by_id[doc_id] = document['text']
    if not citations:
        errors.append('missing_citations')
    for index, citation in enumerate(citations):
        ref = citation.model_dump() if hasattr(citation, 'model_dump') else citation
        if not isinstance(ref, dict):
            errors.append(f'citation_{index}:invalid_schema')
            continue
        doc_id, quote = ref.get('document_id'), ref.get('quote')
        if doc_id not in by_id:
            errors.append(f'citation_{index}:unknown_document:{doc_id}')
        if not isinstance(quote, str) or not quote.strip():
            errors.append(f'citation_{index}:empty_quote')
        elif doc_id in by_id and quote not in by_id[doc_id]:
            errors.append(f'citation_{index}:quote_not_verbatim')
    return {'valid': not errors, 'errors': errors}


def safe_selection(selected: str, original: str) -> tuple[str, str]:
    if normalize_answer(selected) == normalize_answer(original):
        return original, 'preserve_normalized_equal'
    return selected, 'selected'


def apply_selection(raw: str, candidates: list[dict[str, Any]], original: str,
                    documents: list[dict[str, str]]) -> tuple[str, str, str | None]:
    try:
        parsed = SelectionResponse.model_validate_json(raw)
    except ValueError as exc:
        return original, 'fallback_invalid_schema', str(exc)
    if parsed.decision == 'abstain':
        return original, 'abstain', None
    ids = [row['candidate_id'] for row in candidates]
    if len(ids) != len(set(ids)) or parsed.candidate_id not in ids:
        return original, 'fallback_invalid_selection', 'duplicate or unknown candidate ID'
    report = validate_citations_detailed(parsed.citations, documents)
    if not report['valid']:
        return original, 'fallback_invalid_citation', ';'.join(report['errors'])
    selected = next(row['answer'] for row in candidates if row['candidate_id'] == parsed.candidate_id)
    answer, status = safe_selection(selected, original)
    return answer, status, None


def validate_manifest_results(manifest: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Join by task_id; conflicting duplicate results cannot silently overwrite a row."""
    ids = [str(row['task_id']) for row in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate manifest task_id')
    grouped: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    for result in results:
        task_id = str(result['task_id'])
        if task_id not in ids:
            errors.append(f'unknown_result_id:{task_id}')
        else:
            grouped.setdefault(task_id, []).append(result)
    output, missing = [], []
    for original in manifest:
        task_id = str(original['task_id'])
        found = grouped.get(task_id, [])
        conflict = len({json.dumps(row, sort_keys=True, ensure_ascii=False) for row in found}) > 1
        if not found or conflict:
            reason = 'conflicting_duplicate' if conflict else 'missing_result'
            errors.append(f'{reason}:{task_id}')
            if not found:
                missing.append(task_id)
            output.append({**original, 'answer': original['original_answer'], 'status': f'fallback_{reason}'})
        else:
            output.append({**original, **found[0]})
    return {'rows': output, 'errors': errors, 'missing_ids': missing}


class BudgetExceeded(RuntimeError):
    pass


class SQLiteAPILedger:
    """Reserve before dispatch. Only a newly reserved call may be sent to the API.

    A repeated key returns cached, pending, or unknown and must never be dispatched
    again. Unknown outcomes retain the entire reservation until reconciled.
    Costs and caps use the same currency (CNY in the v110 runner).
    """

    def __init__(self, path: str | Path, total_cap: float = 60.0,
                 stage_caps: dict[str, float] | None = None):
        self.total_cap = self._amount(total_cap)
        self.stage_caps = {key: self._amount(value) for key, value in (stage_caps or {}).items()}
        self.db = sqlite3.connect(str(path), timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS calls (
            key TEXT PRIMARY KEY, payload TEXT NOT NULL, config TEXT NOT NULL,
            stage TEXT NOT NULL, reserved REAL NOT NULL, charged REAL NOT NULL,
            status TEXT NOT NULL, raw_response TEXT, usage TEXT, error TEXT)''')
        self.db.commit()

    @staticmethod
    def _amount(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('budget/cost must be finite and nonnegative')
        return value

    def reserve(self, payload: Any, config: Any, stage: str, upper_bound: float) -> dict[str, Any]:
        amount = self._amount(upper_bound)
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        config_json = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        key = hashlib.sha256((payload_json + '\n' + config_json).encode()).hexdigest()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            old = self.db.execute('SELECT * FROM calls WHERE key=?', (key,)).fetchone()
            if old:
                self.db.commit()
                return {'key': key, 'status': 'cached' if old['status'] == 'complete' else old['status'],
                        'cached_response': old['raw_response'], 'usage': json.loads(old['usage']) if old['usage'] else None}
            total = self.db.execute('SELECT COALESCE(SUM(charged),0) FROM calls').fetchone()[0]
            staged = self.db.execute('SELECT COALESCE(SUM(charged),0) FROM calls WHERE stage=?', (stage,)).fetchone()[0]
            if total + amount > self.total_cap or staged + amount > self.stage_caps.get(stage, math.inf):
                raise BudgetExceeded(f'cannot reserve {amount}: total={total}, stage={stage}:{staged}')
            self.db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?,NULL,NULL,NULL)',
                            (key, payload_json, config_json, stage, amount, amount, 'pending'))
            self.db.commit()
            return {'key': key, 'status': 'reserved', 'cached_response': None}
        except Exception:
            self.db.rollback()
            raise

    def complete(self, key: str, raw_response: str, usage: Any, actual_cost: float) -> None:
        amount = self._amount(actual_cost)
        with self.db:
            row = self.db.execute('SELECT * FROM calls WHERE key=?', (key,)).fetchone()
            if row is None:
                raise KeyError(key)
            serialized = json.dumps(usage, sort_keys=True)
            if row['status'] == 'complete':
                if (row['raw_response'], row['usage'], row['charged']) != (raw_response, serialized, amount):
                    raise ValueError('conflicting completion for cached call')
                return
            # Keep the real charge even if the caller underestimated its reservation.
            self.db.execute('UPDATE calls SET status=?, raw_response=?, usage=?, charged=?, error=? WHERE key=?',
                            ('complete', raw_response, serialized, amount,
                             'reservation_exceeded' if amount > row['reserved'] else None, key))
        if amount > row['reserved']:
            raise BudgetExceeded('actual cost exceeded reservation; recorded actual charge, stop API calls')

    def mark_unknown(self, key: str, error: str) -> None:
        with self.db:
            row = self.db.execute('SELECT status FROM calls WHERE key=?', (key,)).fetchone()
            if row is None:
                raise KeyError(key)
            if row['status'] != 'complete':
                self.db.execute('UPDATE calls SET status=?, error=? WHERE key=?', ('unknown', error, key))

    def summary(self) -> dict[str, Any]:
        rows = [dict(row) for row in self.db.execute('SELECT * FROM calls ORDER BY rowid')]
        return {'total_charged': sum(row['charged'] for row in rows), 'total_cap': self.total_cap,
                'calls': rows}

    def close(self) -> None:
        self.db.close()
