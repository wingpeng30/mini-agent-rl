"""Offline protocol and retrieval tests: no external API calls."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

from mini_agent_rl.hybrid_answer import evidence_hash
from mini_agent_rl.hybrid_v110 import SQLiteAPILedger

SPEC = importlib.util.spec_from_file_location('run_v110_test', Path(__file__).parents[1] / 'scripts/run_v110.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

DOCS = [{'id': 'd', 'title': 'York', 'text': 'York is a city.'}]
ROW = {'task_id': 'test', 'question': 'Which city?', 'original_answer': 'York',
       'provenance_valid': True, 'prompt_replay_exact': True, 'e0': DOCS, 'e0_sha256': evidence_hash(DOCS),
       'prompt_token_ids': [10, 11], 'replayed_prompt_token_ids': [10, 11],
       'prompt_attention_mask': [1, 1], 'replayed_prompt_attention_mask': [1, 1],
       'saved_prompt_sha256': hashlib.sha256(str([10, 11]).encode()).hexdigest(),
       'replayed_prompt_sha256': hashlib.sha256(str([10, 11]).encode()).hexdigest(),
       'replayed_messages': [{'role': 'tool', 'content': json.dumps({'result': DOCS})}]}


class FakeAPI:
    config = {'supplemental_rounds': 2, 'retrieval_top_k': 5}

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def call(self, kind, system, payload, retry=0):
        self.calls.append((kind, payload, retry))
        return self.answer(kind, payload, retry)


def test_candidate_protocol_retries_once_and_invalid_citations_fall_back():
    api = FakeAPI(lambda *_: {'candidates': [{'answer': 'New York', 'citations': [
        {'document_id': 'd', 'quote': 'invented'}]}]})
    candidates, protocol = runner.generate(api, ROW, DOCS)
    assert not protocol['valid']
    assert candidates == [{'answer': 'York', 'citations': []}]
    assert [call[2] for call in api.calls] == [0, 1]


def test_verify_requires_both_ids_and_valid_supported_citations():
    api = FakeAPI(lambda *_: {'assessments': [{'id': 'A1', 'status': 'supported'}]})
    with pytest.raises(runner.ProtocolFailure, match='review_ids'):
        runner.verify(api, ROW, DOCS, 'New York')
    api = FakeAPI(lambda *_: {'assessments': [
        {'id': key, 'status': 'supported', 'citations': []} for key in ['A1', 'A2']]})
    with pytest.raises(runner.ProtocolFailure, match='review_invalid_citation'):
        runner.verify(api, ROW, DOCS, 'New York')


def test_supplemental_retrieval_is_bounded_and_deduplicates_documents():
    class Index:
        def __init__(self):
            self.calls = []

        def search(self, query, top_k):
            self.calls.append((query, top_k))
            return DOCS + [{'id': 'new', 'title': 'New York', 'text': 'New York is correct.'}]

    def response(kind, payload, retry):
        if kind == 'candidates':
            return {'candidates': []}
        if kind == 'select':
            return {'decision': 'abstain', 'citations': []}
        assert kind == 'verify'
        return {'assessments': [{'id': item['id'], 'status': 'insufficient', 'citations': []}
                                 for item in payload['answers']],
                'information_gap': 'Missing target city', 'queries': [' target ', 'target', 'second', 'ignored']}

    index = Index()
    result = runner.mechanism(FakeAPI(response), ROW, index)
    assert result['B'] == result['C'] == ROW['original_answer']
    assert index.calls == [('target', 5), ('second', 5)] * 2
    assert [doc['id'] for doc in result['final_evidence']] == ['d', 'new']
    assert len(result['retrieval']) == 4


def test_contradiction_does_not_trigger_retrieval():
    def response(kind, payload, retry):
        if kind == 'candidates':
            return {'candidates': []}
        if kind == 'select':
            return {'decision': 'abstain', 'citations': []}
        return {'assessments': [{'id': item['id'], 'status': 'contradicted', 'citations': []}
                                for item in payload['answers']], 'queries': ['unused'], 'information_gap': 'gap'}

    result = runner.mechanism(FakeAPI(response), ROW, None)
    assert result['C'] == 'York' and result['retrieval'] == []


def test_paid_call_is_cached_and_finish_reason_rejected(tmp_path, monkeypatch):
    config = json.loads((Path(__file__).parents[1] / 'configs/v110-hybrid.json').read_text())
    ledger = SQLiteAPILedger(tmp_path / 'calls.sqlite')
    api = runner.API(config, ledger, 'fake', 'train', 20)
    calls = []

    class Response:
        @property
        def text(self):
            return json.dumps(self.json())

        def raise_for_status(self):
            pass

        def json(self):
            return {'usage': {'prompt_tokens': 10, 'completion_tokens': 20},
                    'choices': [{'finish_reason': 'length', 'message': {'content': '{"ok":true}'}}]}

    monkeypatch.setattr(api.client, 'post', lambda *args, **kwargs: calls.append(kwargs) or Response())
    for _ in range(2):
        with pytest.raises(runner.ProtocolFailure, match='finish_reason'):
            api.call('test', 'system', {'question': 'x'})
    assert len(calls) == 1
    assert len(ledger.summary()['calls']) == 1
    api.client.close()
    ledger.close()
