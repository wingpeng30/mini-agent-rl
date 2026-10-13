from mini_agent_rl.training.sft import _evaluation_arguments


def test_eval_is_disabled_without_validation_data():
    assert _evaluation_arguments(False, 50) == {"eval_strategy": "no", "eval_steps": None}


def test_existing_validation_schedule_is_preserved():
    assert _evaluation_arguments(True, 37) == {"eval_strategy": "steps", "eval_steps": 37}
