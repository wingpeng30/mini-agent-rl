import copy
import json

import pytest

from mini_agent_rl.evidence_sft_v112 import (
    compute_evidence_hash, compute_state_hash, compute_candidate_hash, normalize_question,
    validate_dataset, validate_row, validate_source_path,
)


def row(index=0, split='train'):
    question = f'{split} question {index // 2}'
    evidence = [{'id': 'doc', 'title': 'Source', 'text': f'visible evidence {index}'}]
    action = 'answer' if index % 2 else 'search'
    messages = [{'role': 'user', 'content': question},
                {'role': 'tool', 'content': json.dumps({'result': evidence})},
                {'role': 'assistant', 'content': json.dumps({'action': action, 'answer' if action == 'answer' else 'query': 'supported'})}]
    state = compute_state_hash(question, messages)
    evidence = [{'id': 'doc', 'title': 'Source', 'text': f'visible evidence {index}'}]
    result = {'question': question, 'task_id': f'{split}-{index // 2}', 'messages': messages,
            'target': {'action': 'answer' if index % 2 else 'search', 'value': 'supported'},
            'target_assistant_index': 2, 'step': 1,
            'risk_labels': ['relation', 'measurement', 'time', 'scope'],
            'evidence': evidence, 'evidence_hash': compute_evidence_hash(evidence),
            'state_hash': state,
            'retrieval_results': [{'id':'new','title':'New','text':'new missing fact'}],
            'reviews': [{'kind': kind, 'reviewer': kind, 'approved': True,
                         'reason': 'checked', 'state_hash': state,
                         'missing_fact':'missing fact' if action=='search' else '',
                         'citations':[{'document_id':'new' if action=='search' else 'doc',
                                       'quote':'new missing fact' if action=='search' else evidence[0]['text']}]}
                        for kind in ('api', 'task_agent')]}
    for review in result['reviews']:
        review['candidate_hash'] = compute_candidate_hash(result)
    return result


def dataset():
    return [row(i) for i in range(300)], [row(i, 'internal') for i in range(100)]


def test_accepts_complete_dataset():
    report = validate_dataset(*dataset())
    assert report['passed']
    assert report['stats']['train']['tasks'] == 150


@pytest.mark.parametrize('path', [
    'data/final-test.jsonl', 'data/FINAL_test.jsonl', 'data/independent_holdout/data.jsonl',
    r'data\final test\x.jsonl', 'data/%66inal%2Dtest.jsonl',
    'data/final-test/../train.jsonl', 'data/ｆｉｎａｌ－ｔｅｓｔ.jsonl',
    'data/hotpot-v110-independent/train.jsonl', 'data/independent100.jsonl',
    'data/holdout/train.jsonl',
])
def test_protected_sources_rejected(path):
    with pytest.raises(ValueError, match='protected'):
        validate_source_path(path)


def test_safe_path_and_normalization():
    assert str(validate_source_path('data/train.jsonl'))
    assert str(validate_source_path('reports/final/test-fixture.json'))
    assert normalize_question('  Ａ  Question\n') == 'a question'
    assert normalize_question('When? First-year!') == 'when first year'


def test_targets_do_not_change_state_identity():
    first = row()
    changed = copy.deepcopy(first)
    changed['messages'][-1]['content'] = json.dumps({'action': 'search', 'query': 'another target'})
    changed['target']['value'] = 'another target'
    for review in changed['reviews']:
        review['candidate_hash'] = compute_candidate_hash(changed)
    assert compute_state_hash(first['question'], first['messages']) == compute_state_hash(
        changed['question'], changed['messages'])
    train, internal = dataset()
    train[1] = changed
    assert validate_dataset(train, internal)['errors'] == ['duplicate_state']


@pytest.mark.parametrize('field', ['reviews', 'step', 'risk_labels', 'evidence_hash',
                                  'state_hash', 'target_assistant_index'])
def test_missing_metadata_rejected(field):
    sample = row()
    del sample[field]
    with pytest.raises(ValueError, match='missing_fields'):
        validate_row(sample)


def test_tampered_evidence_and_stale_reviews_rejected():
    sample = row()
    sample['evidence'][0]['text'] = 'tampered'
    with pytest.raises(ValueError, match='visible_evidence'):
        validate_row(sample)
    sample = row()
    sample['reviews'][0]['state_hash'] = 'stale'
    with pytest.raises(ValueError, match='dual_approved'):
        validate_row(sample)


def test_explicit_gold_leak_rejected_but_natural_answer_allowed():
    sample = row()
    sample['gold'] = 'visible evidence'
    validate_row(sample)
    sample['messages'][0]['content'] = '{"gold_answer": "x"}'
    with pytest.raises(ValueError, match='gold_metadata'):
        validate_row(sample)


@pytest.mark.parametrize('failure', ['states', 'tasks', 'actions', 'step0', 'risks', 'overlap'])
def test_dataset_constraints(failure):
    train, internal = dataset()
    if failure == 'states':
        train.pop()
    elif failure == 'tasks':
        for sample in train:
            sample['task_id'] = 'same'
    elif failure == 'actions':
        for sample in train:
            sample['target']['action'] = 'answer'
            sample['messages'][-1]['content'] = json.dumps({'action': 'answer', 'answer': 'supported'})
    elif failure == 'step0':
        for sample in train[:61]:
            sample['step'] = 0
    elif failure == 'risks':
        for sample in train[19:]:
            sample['risk_labels'].remove('scope')
    else:
        sample = internal[0]
        sample['question'] = '  TRAIN QUESTION 0  '
        sample['state_hash'] = compute_state_hash(sample['question'], sample['messages'])
        for review in sample['reviews']:
            review['state_hash'] = sample['state_hash']
    assert not validate_dataset(train, internal)['passed']


def test_target_and_visible_evidence_binding():
    sample = row()
    sample['messages'][-1]['content'] = json.dumps({'action': 'search', 'query': 'wrong'})
    with pytest.raises(ValueError, match='target_message_mismatch'):
        validate_row(sample)
    sample = row()
    sample['evidence'][0]['text'] = 'detached'
    sample['evidence_hash'] = compute_evidence_hash(sample['evidence'])
    with pytest.raises(ValueError, match='visible_evidence_mismatch'):
        validate_row(sample)
