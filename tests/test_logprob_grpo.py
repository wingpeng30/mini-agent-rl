"""v0.6.1/v0.7.0 的 CPU 级纯函数和边界测试；不要求下载本地模型。"""
import pytest

from mini_agent_rl.model.local import LocalModelConfig
from mini_agent_rl.training.grpo import (GRPOConfig, aggregate_rollout_token_values,
                                         clipped_grpo_loss, guarded_optimizer_step,
                                         ratio_statistics, require_initial_ratio)
from mini_agent_rl.training.logprob import compare_logprobs, stable_same_instance
from mini_agent_rl.training.gates import resolve_memory_limit_gb


def test_logprob_same_instance_strict_gate_and_cross_load_statistics():
    same = stable_same_instance([[-1.0, -2.0], [-1.0, -2.0]])
    assert same["passed"] is True
    cross = compare_logprobs([-1.0, -2.0], [-1.0001, -2.0])
    assert cross["valid"] and cross["absolute_delta"]["p95"] > 0
    assert compare_logprobs([-1.0], [-1.0, -2.0])["valid"] is False


def test_local_logprob_defaults_are_explicit():
    config = LocalModelConfig()
    assert config.logprob_mode == "bf16"
    assert config.recompute_old_logprobs and config.deterministic


def test_clipped_grpo_loss_is_finite_and_backpropagates():
    torch = pytest.importorskip("torch")
    current = torch.tensor([-1.0, -1.2], requires_grad=True)
    old = torch.tensor([-1.0, -1.2])
    reference = torch.tensor([-1.1, -1.1])
    loss, kl, clipped = clipped_grpo_loss(current, old, reference, 1.0, .2, .02)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(kl) and torch.isfinite(clipped)
    assert current.grad is not None


def test_rollout_aggregation_weights_all_action_tokens_equally():
    torch = pytest.importorskip("torch")
    short = torch.tensor(1.0, requires_grad=True)
    long = torch.tensor(3.0, requires_grad=True)
    combined = aggregate_rollout_token_values([(short, 1), (long, 3)])
    assert float(combined.detach()) == pytest.approx(2.5)
    combined.backward()
    assert float(short.grad) == pytest.approx(0.25)
    assert float(long.grad) == pytest.approx(0.75)


def test_grpo_memory_gate_defaults_to_safe_8gb_budget():
    assert GRPOConfig().max_reserved_memory_gb == pytest.approx(7.4)


def test_cloud_memory_budget_is_parameterized_from_visible_card():
    assert resolve_memory_limit_gb(None, 24.0) == pytest.approx(21.0)
    assert resolve_memory_limit_gb(19.5, 24.0) == pytest.approx(19.5)


def test_kl_is_zero_at_reference_and_increases_away_from_reference():
    torch = pytest.importorskip("torch")
    old = torch.tensor([-1.0]); reference = torch.tensor([-1.0])
    same, kl_same, _ = clipped_grpo_loss(torch.tensor([-1.0], requires_grad=True), old, reference, 0.0, .2, 1.0)
    different, kl_different, _ = clipped_grpo_loss(torch.tensor([-0.2], requires_grad=True), old, reference, 0.0, .2, 1.0)
    assert float(kl_same.detach()) == pytest.approx(0.0)
    assert float(kl_different.detach()) > 0 and float(different.detach()) > float(same.detach())


def test_streaming_weight_matches_rollout_equal_gradient():
    """逐步加权 backward 与保留全部图的 rollout 等权目标必须数学等价。"""
    torch = pytest.importorskip("torch")
    left = torch.tensor(2.0, requires_grad=True); right = torch.tensor(4.0, requires_grad=True)
    aggregate_rollout_token_values([(left, 1), (right, 3)]).backward()
    aggregate_grad = (float(left.grad), float(right.grad))
    left = torch.tensor(2.0, requires_grad=True); right = torch.tensor(4.0, requires_grad=True)
    ((left * 1 / 4) + (right * 3 / 4)).backward()
    assert (float(left.grad), float(right.grad)) == pytest.approx(aggregate_grad)


def test_optimizer_is_not_called_when_pre_step_gate_fails():
    torch = pytest.importorskip("torch")
    parameter = torch.tensor([1.0], requires_grad=True)
    parameter.grad = torch.tensor([float("nan")])
    class Optimizer:
        calls = 0
        def step(self): self.calls += 1
    optimizer = Optimizer()
    with pytest.raises(RuntimeError, match="梯度"):
        guarded_optimizer_step(optimizer, [parameter], {"peak_reserved_gb": 1.0}, 21.0)
    assert optimizer.calls == 0


def test_optimizer_is_not_called_when_memory_is_unobserved_or_exceeded():
    torch = pytest.importorskip("torch")
    parameter = torch.tensor([1.0], requires_grad=True)
    parameter.grad = torch.tensor([1.0])
    class Optimizer:
        calls = 0
        def step(self): self.calls += 1
    for stats in ({}, {"peak_reserved_gb": 21.1}):
        optimizer = Optimizer()
        with pytest.raises(RuntimeError, match="显存"):
            guarded_optimizer_step(optimizer, [parameter], stats, 21.0)
        assert optimizer.calls == 0


def test_full_ratio_gate_rejects_any_outlier_before_optimizer_step():
    torch = pytest.importorskip("torch")
    stats = ratio_statistics([torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.11])])
    assert stats["count"] == 4
    with pytest.raises(RuntimeError, match="首次实际反传"):
        require_initial_ratio(stats, label="首次实际反传 current")


def test_ratio_statistics_rejects_nonfinite_or_empty_values():
    torch = pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="空值"):
        ratio_statistics([])
    with pytest.raises(RuntimeError, match="非有限"):
        ratio_statistics([torch.tensor([float("nan")])])
