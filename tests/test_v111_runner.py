"""Offline runner contracts; all model responses and retrieval are fakes."""
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('v111_runner_tests', SCRIPTS / 'run_v111.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

DOCS = [{'id': 'e0', 'title': 'Source', 'text': 'Original source.\n\n' + '甲🙂' * 450}]
ROW = {'task_id': 'fake-task', 'question': 'Who won?', 'original_answer': 'Original',
       'e0': DOCS, 'gold': 'SECRET_GOLD', 'supporting_facts': ['SECRET_SUPPORT']}


class FakeAPI:
    config = {'supplemental_rounds': 2, 'retrieval_top_k': 5}

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def call(self, kind, prompt, payload, retry=0):
        self.calls.append({'kind': kind, 'prompt': prompt, 'payload': payload, 'retry': retry})
        return self.respond(kind, payload, retry)


def ref(payload):
    return payload['evidence'][0]['spans'][0]['evidence_id']


def standard(kind, payload, retry=0, insufficient=False):
    evidence_ids = [ref(payload)]
    if kind.endswith(':candidates'):
        return {'candidates': [{'answer': 'Alternative', 'evidence_ids': evidence_ids}]}
    if kind.endswith(':select'):
        candidate = next(c for c in payload['candidates'] if c['answer'] == 'Alternative')
        return {'decision': 'select', 'candidate_id': candidate['candidate_id'], 'evidence_ids': evidence_ids}
    if kind == 'direct':
        return {'decision': 'answer', 'answer': 'Independent', 'evidence_ids': evidence_ids}
    assert kind.endswith(':verify')
    return {'assessments': [{'id': a['id'], 'status': 'insufficient' if insufficient else 'supported',
             'reason': 'Missing fact.' if insufficient else 'Source supports the answer.',
             'evidence_ids': [] if insufficient else evidence_ids} for a in payload['answers']],
            'information_gap': 'Winner missing' if insufficient else '',
            'queries': ['winner'] if insufficient else []}


@pytest.fixture(autouse=True)
def mechanism_only(monkeypatch):
    # Input provenance has its own replay checks; these tests isolate orchestration.
    monkeypatch.setattr(runner, 'validate_input', lambda row: None)


class FakeIndex:
    def __init__(self):
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return [{'id': 'new' + str(len(self.calls)), 'title': 'New evidence', 'text': 'New source.'}]


def test_two_attempt_bound_feedback_and_no_semantic_retry():
    api = FakeAPI(lambda *_: {'bad': True})
    result = runner.request(api, 'test', 'prompt', {}, lambda raw: {'valid': False, 'errors': ['bad_schema']})
    assert result['call_count'] == 2 and not result['valid']
    assert [c['retry'] for c in api.calls] == [0, 1]
    assert api.calls[1]['payload']['protocol_feedback']['errors'] == ['bad_schema']
    api = FakeAPI(lambda *_: {'status': 'insufficient'})
    result = runner.request(api, 'test', 'prompt', {}, lambda raw: {'valid': True, **raw})
    assert result['call_count'] == 1


def test_length_retry_uses_retry_budget_without_schema_feedback():
    def respond(kind, payload, retry):
        if retry == 0:
            raise runner.ProtocolFailure('finish_reason:length')
        return {'ok': True}
    api = FakeAPI(respond)
    result = runner.request(api, 'verify', 'prompt', {}, lambda raw: {'valid': True})
    assert result['valid'] and result['call_count'] == 2
    assert api.calls[1]['retry'] == 1
    assert 'protocol_feedback' not in api.calls[1]['payload']


def test_budget_exhaustion_is_not_retried():
    from mini_agent_rl.hybrid_v110 import BudgetExceeded
    def fail(*args):
        raise BudgetExceeded('cap')
    api = FakeAPI(fail)
    with pytest.raises(BudgetExceeded):
        runner.request(api, 'verify', 'prompt', {}, lambda raw: {'valid': True})
    assert len(api.calls) == 1


def test_original_precedence_direct_independence_and_no_gold_leak():
    api = FakeAPI(standard)
    index = FakeIndex()
    result = runner.mechanism(api, ROW, index)
    assert result['answers']['U'] == 'Alternative'
    assert result['answers']['P'] == result['answers']['R'] == 'Original'
    assert result['answers']['D'] == 'Independent'
    assert not index.calls
    direct = next(c for c in api.calls if c['kind'] == 'direct')
    assert set(direct['payload']) == {'question', 'evidence'}
    for call in api.calls:
        assert 'SECRET_GOLD' not in str(call) and 'SECRET_SUPPORT' not in str(call)
        for doc in call['payload']['evidence']:
            assert ''.join(s['text'] for s in doc['spans']) == DOCS[0]['text']


def test_matched_generation_uses_e0_and_actual_retrieval_round_count():
    api = FakeAPI(lambda kind, payload, retry: standard(kind, payload, retry, insufficient=True))
    result = runner.mechanism(api, ROW, FakeIndex())
    assert len(result['retrieval_rounds']) == len(result['matched_rounds']) == 2
    matched = [c for c in api.calls if c['kind'].startswith('matched:')]
    assert len(matched) == 6
    assert all([d['document_id'] for d in c['payload']['evidence']] == ['e0'] for c in matched)
    assert len(result['evidence_by_arm']['R']) == 3
    assert result['evidence_by_arm']['M'] == DOCS
    assert all(c['retry'] == 0 for c in api.calls)  # insufficient is a valid semantic outcome
    assert result['protocol']['M']['call_count'] == result['protocol']['R']['call_count']


def test_no_new_documents_means_no_matched_regeneration():
    class ExistingIndex:
        def search(self, query, top_k):
            return DOCS
    api = FakeAPI(lambda kind, payload, retry: standard(kind, payload, retry, insufficient=True))
    result = runner.mechanism(api, ROW, ExistingIndex())
    assert result['retrieval_rounds'] == result['matched_rounds'] == []
    assert not any(c['kind'].startswith('matched:') for c in api.calls)


def test_malformed_assessment_types_get_bounded_protocol_retry():
    api = FakeAPI(lambda *_: {'assessments': [None, 3], 'information_gap': '', 'queries': []})
    value, call = runner.verify(api, ROW, 'Alternative', runner.build_evidence(DOCS))
    assert not value['valid'] and call['call_count'] == 2
    assert not value['assessments']['original']['valid']
    assert not value['assessments']['selected']['valid']


def test_direct_only_never_generates_candidates_or_retrieves():
    api = FakeAPI(standard)
    index = FakeIndex()
    result = runner.mechanism(api, ROW, index, candidate='D')
    assert set(result['answers']) == {'D'}
    assert [c['kind'] for c in api.calls] == ['direct']
    assert not index.calls
