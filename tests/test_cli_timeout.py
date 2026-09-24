import inspect

from mini_agent_rl.cli import _hard_timeout_expired, train_grpo


def test_zero_timeout_is_unlimited() -> None:
    assert not _hard_timeout_expired(10.0, 0, now=10_000.0)


def test_positive_timeout_expires_only_after_limit() -> None:
    assert not _hard_timeout_expired(10.0, 2, now=130.0)
    assert _hard_timeout_expired(10.0, 2, now=130.01)


def test_grpo_temperature_is_explicit() -> None:
    assert inspect.signature(train_grpo).parameters["temperature"].default == 0.7
