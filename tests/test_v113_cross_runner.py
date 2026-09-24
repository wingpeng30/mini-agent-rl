import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('v113_cross', Path(__file__).resolve().parents[1] / 'scripts/run_v113_cross.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def row(arm='baseline'):
    ids = [0, 2, 3]
    return {'input_hash': runner.token_hash(ids), 'messages': [{'role': 'user', 'content': 'Q'}],
            'prompt_token_ids': ids, 'prompt_token_sha256': runner.token_hash(ids), 'step': 0,
            'observed': {arm: {'raw_text': '{}', 'prompt_token_ids': ids}}}


def test_dedup_preserves_source_outputs_and_only_generates_missing_arm():
    left = row()
    assert len(runner.missing_pairs(runner.deduplicate_inputs([left, left]))) == 1
    combined = runner.deduplicate_inputs([left, row('candidate')])
    assert len(combined) == 1
    assert runner.missing_pairs(combined) == []
    assert combined[0]['source_outputs']['baseline'] == left['observed']['baseline']


def test_ids_canonical_and_bound_to_tokens_not_transport_ids():
    first = row()
    second = row()
    second['messages'][0]['tool_call_id'] = 'unrendered-uuid'
    second['prompt_token_ids'] = [[0, 2, 3]]
    assert len(runner.deduplicate_inputs([first, second])) == 1
    first['input_hash'] = 'tampered'
    with pytest.raises(ValueError, match='noncanonical_input_id'):
        runner.deduplicate_inputs([first])


def test_tokenizer_replay_mismatch_aborts_before_inference():
    frozen = runner.deduplicate_inputs([row()])[0]
    with pytest.raises(ValueError, match='prompt_token_replay_mismatch'):
        runner.verify_replay(frozen, [0, 2, 4])
    for bad in ([True], [-1], [], [[[1]]]):
        with pytest.raises(ValueError):
            runner.tokens(bad)
