"""Offline baseline invariants; never load weights or call CUDA/API."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('v111_baseline', Path(__file__).parents[1] / 'scripts/run_v111_baseline.py')
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def rollout(task='a', status='FAILED', error=None, reason='invalid_action'):
    return SimpleNamespace(task_id=task, status=SimpleNamespace(value=status), error=error, termination_reason=reason)


def test_fixed_bf16_config():
    config = baseline.model_config('model', 'adapter')
    assert config.dtype == 'bfloat16' and not config.load_in_4bit
    assert (config.max_input_tokens, config.max_new_tokens) == (2048, 128)
    assert not config.enable_thinking and config.observation_encoding == 'compact-v3'
    assert baseline.RUN_CONFIG['base_seed'] == 42
    assert baseline.RUN_CONFIG['prompt_variant'] == 'direct-v090'


def test_failed_task_stays_done_and_duplicate_foreign_incomplete_rejected():
    assert baseline.validate_resume([rollout()], ['a']) == {'a'}
    for rows in ([rollout(), rollout()], [rollout('b')], [rollout(status='RUNNING')]):
        with pytest.raises(RuntimeError):
            baseline.validate_resume(rows, ['a'])


@pytest.mark.parametrize('error', ['CUDA out of memory', 'generation timed out', 'non-finite logits', 'CUDA error'])
def test_infrastructure_failure_stops_resume(error):
    with pytest.raises(RuntimeError, match='infrastructure_failure'):
        baseline.validate_resume([rollout(error=error)], ['a'])


def test_unknown_model_exception_is_infrastructure():
    with pytest.raises(RuntimeError, match='infrastructure_failure'):
        baseline.check_infrastructure_error(rollout(error='missing model shard', reason='model_error'))
    with pytest.raises(RuntimeError, match='infrastructure_failure'):
        baseline.check_infrastructure_error(rollout(reason='tool_error'))


@pytest.mark.parametrize('value', [None, float('nan'), 21.01])
def test_peak_memory_guard(value):
    with pytest.raises(RuntimeError, match='gpu_peak_memory_gate'):
        baseline.check_peak_memory({'peak_allocated_gb': 20, 'peak_reserved_gb': value})
    baseline.check_peak_memory({'peak_allocated_gb': 21, 'peak_reserved_gb': 21})


def test_memory_gate(monkeypatch):
    monkeypatch.setattr(baseline.os, 'getpid', lambda: 7)
    monkeypatch.setattr(baseline.subprocess, 'check_output', lambda *a, **k: '7, 21504\n')
    assert baseline.device_memory_gib() == 21
    monkeypatch.setattr(baseline.subprocess, 'check_output', lambda *a, **k: '7, 21505\n')
    with pytest.raises(RuntimeError, match='device_memory_gate'):
        baseline.device_memory_gib()


def test_fingerprint_checks_adapter_before_loading_any_model(monkeypatch):
    monkeypatch.setattr(baseline, 'sha256', lambda _: 'actual')
    with pytest.raises(RuntimeError, match='adapter_mismatch'):
        baseline.fingerprint(None, baseline.model_config('m', 'a'), 'different')


def test_fingerprint_requires_full_corpus(monkeypatch):
    monkeypatch.setattr(baseline, 'sha256', lambda _: 'actual')
    env = SimpleNamespace(tasks=list(range(100)), task_ids=list(range(100)), corpus_document_count=200)
    with pytest.raises(RuntimeError, match='986_document'):
        baseline.fingerprint(env, baseline.model_config('m', 'a'), 'actual')


def test_resume_fingerprint_tracks_code_model_config_and_corpus(monkeypatch):
    state = {'code': 'code1', 'model': 'weights1', 'corpus': 'corpus1'}
    monkeypatch.setattr(baseline, 'sha256', lambda path: 'adapter' if Path(path).name == 'adapter_model.safetensors' else state['code'])
    monkeypatch.setattr(baseline, 'tree_fingerprint', lambda path: {'weights': state['model']})
    env = SimpleNamespace(tasks=list(range(100)), task_ids=list(range(100)), corpus_document_count=986,
                          fingerprint=lambda *args: {'corpus_sha256': state['corpus']})
    config = baseline.model_config('model', 'adapter')
    original = baseline.fingerprint(env, config, 'adapter')
    assert original == baseline.fingerprint(env, config, 'adapter')
    for key in state:
        old = state[key]
        state[key] = 'changed'
        assert original != baseline.fingerprint(env, config, 'adapter')
        state[key] = old
    config.max_new_tokens = 256
    assert original != baseline.fingerprint(env, config, 'adapter')
