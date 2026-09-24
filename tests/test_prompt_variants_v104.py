import pytest

from mini_agent_rl.prompts import DIRECT_SYSTEM, system_prompt


def test_default_prompt_is_backward_compatible():
    assert system_prompt(False, "direct-v090") == DIRECT_SYSTEM


def test_precise_variants_are_distinct_and_require_one_action():
    first=system_prompt(False,"precise-answer-v1")
    second=system_prompt(False,"precise-answer-v2")
    assert first != second
    assert "shortest" in first and "one JSON" in first
    assert "shortest" in second and "one JSON" in second


def test_precise_variant_rejected_for_thinking_mode():
    with pytest.raises(ValueError):
        system_prompt(True,"precise-answer-v1")


def test_v106_fewshot_is_fixed_and_contains_no_development_examples():
    prompt = system_prompt(False, "fewshot-v106")
    assert prompt.count('Action: {"action":"answer"') == 3
    assert "Arthur's Magazine" in prompt
    assert "February 5, 1953" in prompt
    assert "the Desert Fox" in prompt
    assert "Hate to Feel" not in prompt
