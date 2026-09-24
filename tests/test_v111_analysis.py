"""Manifest integrity, gates, and paired controls without model calls."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('v111_analysis', Path(__file__).parents[1] / 'scripts/analyze_v111.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def source(task='a', original='right'):
    return {'task_id': task, 'original_answer': original, 'gold': 'right'}


def row(task='a', answer='right'):
    return {'task_id': task, 'answers': {a: answer for a in analysis.ARMS},
            'protocol': {a: {'valid': True, 'first_pass_valid': True} for a in analysis.ARMS},
            'status': 'ok', 'retrieval': []}


def run(inputs, rows):
    return analysis.analyze(inputs, rows, resamples=20, expected_tasks=len(inputs))


def test_missing_row_falls_back_but_blocks_all_gates():
    report, details = run([source('a', 'wrong'), source('b')], [row()])
    assert report['missing_tasks'] == 1
    assert report['metrics']['A']['normalized_em'] == .5
    assert report['metrics']['P']['normalized_em'] == 1
    assert details[1]['P_answer'] == 'right'
    assert not any(report['numerical_gate'].values())


@pytest.mark.parametrize('inputs,rows', [([source()], [row('unknown')]),
                                       ([source()], [row(), row()]),
                                       ([source(), source()], [row()])])
def test_conflicting_identity_rejected(inputs, rows):
    with pytest.raises(ValueError, match='identity_conflict'):
        run(inputs, rows)


def test_default_requires_100_and_semantic_pending_even_after_pass():
    report, _ = analysis.analyze([source()], [row()], resamples=20)
    assert not report['manifest_complete']
    report, _ = run([source()], [row()])
    assert all(report['numerical_gate'].values())
    assert report['selected_numerical_candidate'] == 'D'
    assert report['semantic_review_status'] == 'pending'
    assert not report['adoption_approved']


def test_missing_answer_and_error_cannot_pass_by_fallback():
    broken = row()
    del broken['answers']['U']
    report, _ = run([source()], [broken])
    assert report['missing_answers']['U'] == 1
    assert not any(report['numerical_gate'].values())
    broken = row()
    broken['status'] = 'error'
    report, _ = run([source()], [broken])
    assert report['infrastructure_errors'] == 1
    assert not any(report['numerical_gate'].values())


def test_protocol_arm_specific_and_first_pass_reported():
    result = row()
    result['protocol']['R']['valid'] = False
    result['protocol']['P']['first_pass_valid'] = False
    report, _ = run([source()], [result])
    assert report['numerical_gate'] == {'P': True, 'R': False, 'D': True}
    assert report['protocol_first_pass_valid_rate']['P'] == 0


def test_paired_controls_and_conditional_matched_cohort():
    result = row('a')
    result['answers'].update(U='wrong', P='wrong', M='wrong')
    result['retrieval'] = [{'query': 'x'}]
    report, _ = run([source('a', 'wrong'), source('b')], [result, row('b')])
    assert report['paired_R_minus_P']['normalized_em']['mean'] == .5
    assert report['paired_vs_D']['P']['normalized_em']['mean'] == -.5
    assert report['matched_regeneration']['R_minus_M']['normalized_em']['mean'] == 1
    assert report['matched_regeneration']['observed_tasks'] == 1
    assert report['matched_regeneration']['complete_trigger_cohort']


def test_retention_and_calls_ranking():
    result = row()
    result['answers']['R'] = 'wrong'
    result['calls_by_arm'] = {'P': 1, 'D': 2, 'R': 3, 'U': 1}
    report, _ = run([source()], [result])
    assert not report['numerical_gate']['R']
    assert report['selected_numerical_candidate'] == 'P'
    del result['calls_by_arm']['D']
    report, _ = run([source()], [result])
    assert report['selected_numerical_candidate'] == 'D'


def test_bootstrap_deterministic_and_empty_safe():
    assert analysis.bootstrap([0, 1, -1], 20) == analysis.bootstrap([0, 1, -1], 20)
    assert analysis.bootstrap([], 20)['ci95'] is None


def test_explicit_empty_answer_is_scored_without_fallback():
    result = row()
    result['answers']['P'] = ''
    report, details = run([source()], [result])
    assert report['missing_answers']['P'] == 0
    assert details[0]['P_answer'] == ''
    assert report['metrics']['P']['normalized_em'] == 0
    assert report['manifest_complete']
    assert not report['numerical_gate']['P']


def test_independent_subset_omits_unrun_arms():
    result = row()
    result['answers'] = {'P': 'right', 'D': 'right'}
    result['protocol'] = {a: result['protocol'][a] for a in ('P', 'D')}
    report, details = analysis.analyze([source()], [result], resamples=20,
                                       expected_tasks=1, evaluated_arms=('P', 'D'))
    assert report['manifest_complete']
    assert report['numerical_gate'] == {'P': True, 'D': True}
    assert set(report['metrics']) == {'A', 'P', 'D'}
    assert set(report['paired_vs_A']) == {'P', 'D'}
    assert set(report['paired_vs_D']) == {'P'}
    assert 'paired_P_minus_U' not in report
    assert 'paired_R_minus_P' not in report
    assert 'R_minus_M' not in report['matched_regeneration']
    assert 'U' not in details[0] and 'R_answer' not in details[0]
    assert analysis.summarize([source()], [result], resamples=20,
                              expected_tasks=1, evaluated_arms=('P', 'D')) == report


@pytest.mark.parametrize('arms', [(), ('X',), ('P', 'P')])
def test_invalid_arm_selection_rejected(arms):
    with pytest.raises(ValueError, match='invalid_evaluated_arms'):
        analysis.analyze([source()], [row()], resamples=20, evaluated_arms=arms)
