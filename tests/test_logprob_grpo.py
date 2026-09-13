"""v0.6.1/v0.7.0 的 CPU 级纯函数和边界测试；不要求下载本地模型。"""
import pytest

from mini_agent_rl.model.local import LocalModelConfig
from mini_agent_rl.training.grpo import clipped_grpo_loss
from mini_agent_rl.training.logprob import compare_logprobs, stable_same_instance


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
