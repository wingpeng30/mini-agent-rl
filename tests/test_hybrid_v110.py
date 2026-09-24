import json

import pytest

from mini_agent_rl.hybrid_v110 import (
    BudgetExceeded, SQLiteAPILedger, answer_metrics, apply_selection,
    normalize_answer, safe_selection, token_f1, validate_citations_detailed,
    validate_manifest_results,
)


def test_official_hotpot_order_special_answers_and_empty():
    assert normalize_answer('"The", Beatles!') == 'beatles'
    assert token_f1('yes', 'yes indeed') == 0
    assert token_f1('no answer given', 'noanswer') == 0
    assert token_f1('yes', 'YES') == 1
    assert token_f1('', '') == 0
    assert token_f1('red red blue', 'red blue') == pytest.approx(.8)
    assert answer_metrics('The Beatles', 'Beatles') == {'strict_em': 0, 'normalized_em': 1, 'f1': 1}


def test_safe_selection_does_not_restore_or_compress_substrings():
    assert safe_selection('New York', 'York') == ('New York', 'selected')
    assert safe_selection('York', 'New York') == ('York', 'selected')
    assert safe_selection('12 June 1980', '1980') == ('12 June 1980', 'selected')
    assert safe_selection('1980', '12 June 1980') == ('1980', 'selected')
    assert safe_selection('The Beatles', 'Beatles') == ('Beatles', 'preserve_normalized_equal')
    docs = [{'id': 'd', 'text': 'New York is here.'}]
    raw = json.dumps({'decision': 'select', 'candidate_id': 'C1',
                      'citations': [{'document_id': 'd', 'quote': 'New York'}]})
    assert apply_selection(raw, [{'candidate_id': 'C1', 'answer': 'New York'}], 'York', docs) == ('New York', 'selected', None)
    assert apply_selection(raw, [{'candidate_id': 'C1', 'answer': 'New York'}], 'York', [])[:2] == ('York', 'fallback_invalid_citation')
    assert apply_selection('{"decision":"abstain","citations":[]}', [], 'York', []) == ('York', 'abstain', None)


def test_bm25_optional_top_k_preserves_default_order():
    from mini_agent_rl.tools.search import _BM25Index

    corpus = [{'id': str(i), 'title': f'City {i}', 'text': 'New York city travel.'}
              for i in range(6)]
    index = _BM25Index(corpus)
    default = index.search('New York')
    expanded = index.search('New York', top_k=5)
    assert len(default) == 3 and len(expanded) == 5
    assert default == expanded[:3] == index.search('New York', top_k=3)


def test_citation_report_distinguishes_invalid_references():
    report = validate_citations_detailed([
        {'document_id': 'missing', 'quote': 'word'},
        {'document_id': 'd', 'quote': ''},
        {'document_id': 'd', 'quote': 'Word'},
    ], [{'id': 'd', 'text': 'word'}])
    assert not report['valid']
    assert report['errors'] == ['citation_0:unknown_document:missing',
                                'citation_1:empty_quote', 'citation_2:quote_not_verbatim']
    assert not validate_citations_detailed([], [])['valid']


def test_manifest_preserves_order_and_falls_back_for_conflicts_or_missing():
    manifest = [{'task_id': key, 'original_answer': key} for key in ['a', 'b', 'c']]
    results = [{'task_id': 'b', 'answer': 'new'}, {'task_id': 'a', 'answer': 'x'},
               {'task_id': 'a', 'answer': 'y'}, {'task_id': 'outside', 'answer': 'z'}]
    report = validate_manifest_results(manifest, results)
    assert [row['answer'] for row in report['rows']] == ['a', 'new', 'c']
    assert report['missing_ids'] == ['c']
    assert 'conflicting_duplicate:a' in report['errors']
    assert 'unknown_result_id:outside' in report['errors']
    with pytest.raises(ValueError, match='duplicate manifest'):
        validate_manifest_results(manifest + manifest, [])


def test_ledger_persists_unknown_reservations_and_cached_results(tmp_path):
    path = tmp_path / 'ledger.sqlite'
    ledger = SQLiteAPILedger(path, total_cap=10)
    first = ledger.reserve({'prompt': 'a'}, {'model': 'm'}, 'candidates', 4)
    assert first['status'] == 'reserved'
    assert ledger.reserve({'prompt': 'a'}, {'model': 'm'}, 'candidates', 4)['status'] == 'pending'
    ledger.mark_unknown(first['key'], 'timeout')
    ledger.close()
    ledger = SQLiteAPILedger(path, total_cap=10)
    assert ledger.reserve({'prompt': 'a'}, {'model': 'm'}, 'candidates', 4)['status'] == 'unknown'
    second = ledger.reserve({'prompt': 'b'}, {'model': 'm'}, 'selection', 6)
    with pytest.raises(BudgetExceeded):
        ledger.reserve({'prompt': 'c'}, {}, 'selection', .01)
    ledger.complete(second['key'], 'raw', {'tokens': 12}, 2)
    ledger.complete(second['key'], 'raw', {'tokens': 12}, 2)
    cached = ledger.reserve({'prompt': 'b'}, {'model': 'm'}, 'selection', 6)
    assert cached['status'] == 'cached' and cached['cached_response'] == 'raw'
    assert ledger.summary()['total_charged'] == 6
    assert len(ledger.summary()['calls']) == 2
    ledger.close()


def test_ledger_stage_cap_and_actual_overrun_are_audited(tmp_path):
    ledger = SQLiteAPILedger(tmp_path / 'ledger.sqlite', stage_caps={'a': 2})
    row = ledger.reserve('p', 'config', 'a', 2)
    with pytest.raises(BudgetExceeded):
        ledger.reserve('different', 'config', 'a', .01)
    with pytest.raises(BudgetExceeded, match='actual cost'):
        ledger.complete(row['key'], 'raw', {'tokens': 99}, 3)
    assert ledger.summary()['total_charged'] == 3
    assert ledger.summary()['calls'][0]['error'] == 'reservation_exceeded'
    with pytest.raises(ValueError):
        ledger.reserve('p2', {}, 'b', float('nan'))
    ledger.close()
