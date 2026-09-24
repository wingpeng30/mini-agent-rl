import pytest

from mini_agent_rl.hybrid_v111 import (
    build_evidence, validate_refs, validate_candidates, validate_selection,
    validate_review, decide,
)


def fixture():
    data = build_evidence([{'id': 'one', 'text': 'Alice won in 2001.'}])
    return data['registry'], next(iter(data['registry']))


def assessment(identifier, status, refs):
    return {'id': identifier, 'status': status, 'reason': 'Evidence assessment.', 'evidence_ids': refs}


def review(a, b, registry):
    return validate_review({'assessments': [a, b], 'information_gap': '', 'queries': []},
                           ['original', 'selected'], registry)


def test_lossless_unicode_paragraphs_and_empty_documents():
    text = '甲🙂' * 500 + '\r\n\r\n  next\n\nlast  '
    docs = [{'id': 'd', 'text': text}, {'id': 'empty', 'text': ''}]
    result = build_evidence(docs)
    spans = result['evidence'][0]['spans']
    assert ''.join(s['text'] for s in spans) == text
    assert all(s['text'] == text[s['start']:s['end']] and len(s['text']) <= 800 for s in spans)
    assert result['evidence'][1]['text'] == ''
    assert result == build_evidence(docs)


def test_snapshot_and_document_identity():
    a = build_evidence([{'id': 'a', 'text': 'same'}, {'id': 'b', 'text': 'same'}])
    assert len(a['registry']) == 2
    b = build_evidence([{'id': 'a', 'text': 'same!'}, {'id': 'b', 'text': 'same'}])
    assert not validate_refs(list(a['registry']), b['registry'])['valid']
    assert not validate_refs(['unknown'], a['registry'])['valid']
    with pytest.raises(ValueError, match='duplicate'):
        build_evidence([{'id': 'a', 'text': 'x'}, {'id': 'a', 'text': 'y'}])


def test_original_priority_and_incorrect_original_replacement():
    registry, ref = fixture()
    both = review(assessment('original', 'supported', [ref]), assessment('selected', 'supported', [ref]), registry)
    assert decide('Alice', 'Alice (winner)', both)['answer'] == 'Alice'
    wrong = review(assessment('original', 'contradicted', [ref]), assessment('selected', 'supported', [ref]), registry)
    assert decide('Bob', 'Alice', wrong)['answer'] == 'Alice'
    assert decide('Alice', 'Alice (winner)', both, protect=False)['answer'] == 'Alice (winner)'


def test_invalid_sibling_preserves_valid_original_but_failure_remains():
    registry, ref = fixture()
    value = review(assessment('original', 'supported', [ref]), assessment('selected', 'supported', []), registry)
    assert value['assessments']['original']['valid']
    result = decide('Alice', 'Bob', value)
    assert result['status'] == 'preserve_supported_original'
    assert not result['protocol_valid'] and result['errors']


@pytest.mark.parametrize('bad', [None, 1, True, 'json', [], {'assessments': [None]},
                                        {'assessments': {}, 'information_gap': [], 'queries': {}}])
def test_malformed_objects_are_structured_failures(bad):
    registry, _ = fixture()
    assert not validate_review(bad, ['A1', 'A2'], registry)['valid']
    assert not validate_candidates(bad, registry)['valid']
    assert not validate_selection(bad, [], registry)['valid']


def test_unknown_duplicate_and_bad_reference_types():
    registry, ref = fixture()
    for refs in (None, {}, 'x', [None], [[ref]], [ref, ref], ['unknown']):
        assert not validate_refs(refs, registry)['valid']


def test_duplicate_review_ids_cannot_overwrite():
    registry, ref = fixture()
    a = assessment('A1', 'supported', [ref])
    value = validate_review({'assessments': [a, a], 'information_gap': '', 'queries': []}, ['A1', 'A2'], registry)
    assert not value['valid']
    assert not value['assessments']['A1']['valid']
    assert not value['assessments']['A2']['valid']


def test_candidate_selection_bounds_and_abstain():
    registry, ref = fixture()
    c = {'answer': 'Alice', 'evidence_ids': [ref]}
    assert validate_candidates({'candidates': [c]}, registry)['valid']
    assert not validate_candidates({'candidates': [c] * 5}, registry)['valid']
    assert not validate_candidates({'candidates': [{**c, 'quote': 'invented'}]}, registry)['valid']
    candidates = [{'candidate_id': 'C1', **c}]
    assert validate_selection({'decision': 'select', 'candidate_id': 'C1', 'evidence_ids': [ref]}, candidates, registry)['valid']
    assert validate_selection({'decision': 'abstain', 'candidate_id': None, 'evidence_ids': []}, candidates, registry)['valid']
    assert not validate_selection({'decision': 'select', 'candidate_id': 'C2', 'evidence_ids': [ref]}, candidates, registry)['valid']


def test_reference_validity_does_not_infer_support():
    registry, ref = fixture()
    value = review(assessment('original', 'insufficient', []), assessment('selected', 'insufficient', [ref]), registry)
    assert value['valid']
    assert decide('Bob', 'Charlie', value)['answer'] == 'Bob'
