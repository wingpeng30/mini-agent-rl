"""Offline checks of paired denominators and the training gate."""
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location('analyze_v110_test', SCRIPTS / 'analyze_v110.py')
    analysis = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(analysis)
finally:
    sys.path.remove(str(SCRIPTS))


def result(task_id, answer):
    return {'task_id': task_id, 'B': answer, 'C': answer, 'status': 'ok',
            'protocol': {'valid': True}, 'selection': {'error': None}, 'checks': [],
            'initial_candidates': [{'answer': answer}]}


def test_missing_result_keeps_manifest_denominator_and_fails_gate():
    inputs = [{'task_id': 'a', 'original_answer': 'wrong', 'gold': 'right'},
              {'task_id': 'b', 'original_answer': 'right', 'gold': 'right'}]
    report, paired = analysis.analyze(inputs, [result('a', 'right')])
    assert report['tasks'] == len(paired) == 2
    assert report['missing_tasks'] == 1
    assert report['metrics']['A']['normalized_em'] == .5
    assert report['metrics']['B']['normalized_em'] == 1
    assert report['paired']['B']['normalized_em']['mean'] == .5
    assert report['numerical_gate'] == {'B': False, 'C': False}
    assert report['candidate_recall_on_baseline_wrong'] == {'numerator': 1, 'denominator': 1}


def test_manifest_identity_conflicts_rejected_before_gate():
    inputs = [{'task_id': 'a', 'original_answer': 'right', 'gold': 'right'}]
    for rows in ([result('unknown', 'right')], [result('a', 'right')] * 2):
        with pytest.raises(ValueError, match='manifest_result_identity_conflict'):
            analysis.analyze(inputs, rows)


def test_arm_specific_verification_protocol_failure_blocks_c_only():
    inputs = [{'task_id': 'a', 'original_answer': 'right', 'gold': 'right'}]
    row = result('a', 'right')
    row['checks'] = [{'error': 'invalid_review'}]
    report, _ = analysis.analyze(inputs, [row])
    assert report['protocol_valid_rate'] == {'B': 1, 'C': 0}
    assert report['numerical_gate'] == {'B': True, 'C': False}
    assert report['semantic_review_required_before_adoption'] is True
