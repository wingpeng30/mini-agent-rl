"""v0.7.0：适合单卡冒烟实验的最小 LoRA-GRPO 更新器。"""
from __future__ import annotations

import json, math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain import RolloutGroup, Task


@dataclass
class GRPOConfig:
    learning_rate: float = 5e-6
    clip_epsilon: float = .2
    kl_beta: float = .02
    policy_epochs: int = 2
    max_grad_norm: float = 1.0

@dataclass
class RewardConfig:
    """受控消融的奖励权重；只改变明确指定的单项。"""
    exact_match: float = 1.0
    repeated_search: float = -0.5

def rollout_equal_grpo_loss(current, old, reference, advantage: float, clip_epsilon: float, kl_beta: float):
    """单条 rollout 内聚合 token，调用方对 rollout 等权平均。"""
    return clipped_grpo_loss(current, old, reference, advantage, clip_epsilon, kl_beta)


def clipped_grpo_loss(current, old, reference, advantage: float, clip_epsilon: float, kl_beta: float):
    """逐 action-token 广播 rollout advantage 的 clipped GRPO + reference KL。"""
    import torch
    advantage_tensor = torch.full_like(current, float(advantage))
    ratio = torch.exp(current - old)
    objective = torch.minimum(ratio * advantage_tensor, torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage_tensor)
    # 非负的近似 KL；reference 不参与梯度，只有当前 LoRA 策略得到更新。
    kl = torch.exp(current - reference) - (current - reference) - 1
    return -objective.mean() + kl_beta * kl.mean(), kl.mean().detach(), ((ratio - 1).abs() > clip_epsilon).float().mean().detach()


class MinimalGRPOTrainer:
    """每个 rollout group 进行小批量更新，刻意优先可审计和显存安全而非吞吐。"""
    def __init__(self, client, runner, store, config: GRPOConfig):
        self.client, self.runner, self.store, self.config = client, runner, store, config

    def trainable_names(self) -> list[str]:
        return [name for name, _ in self.client.prepare_for_training()]

    def update_group(self, group: RolloutGroup, optimizer, params, reference_state) -> dict[str, Any]:
        import torch
        if group.zero_variance:
            return {"group_id": group.id, "skipped": "zero_variance"}
        rollouts = [self.store.get_rollout(item) for item in group.rollout_ids]
        rollouts = [item for item in rollouts if item and item.eligible_for_rl and item.advantage is not None]
        samples = [(rollout, transition) for rollout in rollouts for transition in self.store.get_transitions(rollout.id)
                   if transition.response.action_token_ids and transition.response.action_mask and any(transition.response.action_mask)]
        if not samples:
            return {"group_id": group.id, "skipped": "no_action_tokens"}
        # 训练前重算 old/reference，绝不盲信跨进程 JSONL 中的 BF16 数字。
        # 此处必须 eval：LoRA dropout 若在 train mode 会把 old policy 本身变成随机变量。
        self.client._model.eval()
        frozen = []
        for rollout, transition in samples:
            ids = transition.response.action_token_ids or []
            old = self.client.score_action_tensor(transition.messages, ids, False).detach()
            reference = self.client.reference_logprobs(transition.messages, ids, reference_state).detach()
            if old.numel() != len(ids) or reference.numel() != len(ids) or not torch.isfinite(old).all() or not torch.isfinite(reference).all():
                raise RuntimeError("old/reference logprob 长度不一致或出现非有限值")
            frozen.append((rollout, transition, old, reference))
        epoch_metrics = []
        for _ in range(self.config.policy_epochs):
            optimizer.zero_grad(set_to_none=True); losses=[]; kls=[]; clips=[]
            by_rollout = {}
            for sample in frozen: by_rollout.setdefault(sample[0].id, []).append(sample)
            rollout_losses = []
            # 逐 rollout 反传后立即释放计算图：保持 rollout 等权，避免 8GB 显存同时保存整组图。
            rollout_count = len(by_rollout)
            for samples_for_rollout in by_rollout.values():
                token_losses=[]; token_kls=[]; token_clips=[]
                for rollout, transition, old, reference in samples_for_rollout:
                    current = self.client.score_action_tensor(transition.messages, transition.response.action_token_ids or [], True)
                    loss, kl, clipped = clipped_grpo_loss(current, old, reference, rollout.advantage or 0.0, self.config.clip_epsilon, self.config.kl_beta)
                    if not torch.isfinite(loss): raise RuntimeError("GRPO loss 出现 NaN/Inf")
                    # 每条 rollout 的 transition 平均后再按 rollout 数平均，等价于原目标。
                    (loss / (rollout_count * len(samples_for_rollout))).backward()
                    token_losses.append(loss.detach()); token_kls.append(kl); token_clips.append(clipped)
                rollout_losses.append(torch.stack(token_losses).mean()); losses.append(float(rollout_losses[-1])); kls.append(float(torch.stack(token_kls).mean())); clips.append(float(torch.stack(token_clips).mean()))
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm))
            if not math.isfinite(grad_norm): raise RuntimeError("梯度范数出现 NaN/Inf")
            optimizer.step(); epoch_metrics.append({"loss": sum(losses)/len(losses), "kl": sum(kls)/len(kls), "clip_fraction": sum(clips)/len(clips), "grad_norm": grad_norm})
        self.client._model.eval()
        return {"group_id": group.id, "skipped": None, "samples": len(frozen), "reward_mean": group.mean_reward,
                "reward_std": group.std_reward, "metrics": epoch_metrics, "peak_gpu_memory_gb": self.client.gpu_memory_peak_gb()}

    async def train(self, tasks: list[Task], group_size: int, output_dir: str, expected_fingerprint: dict[str, Any]) -> dict[str, Any]:
        """采样与更新必须共享一个已加载实例，避免 BF16 跨加载比例漂移。"""
        import torch
        if self.client._model is None:
            self.client._load()
        actual = self.client.policy_fingerprint()
        # 权重哈希/config 哈希是训练边界；运行期显存等易变字段不参与匹配。
        keys = ("base_config_sha256", "adapter_config_sha256", "adapter_weight_sha256", "tokenizer_sha256")
        if any(expected_fingerprint.get(k) != actual.get(k) for k in keys):
            raise RuntimeError("策略指纹不匹配：拒绝将不属于当前 adapter 的轨迹用于 GRPO")
        before = {name: value.detach().float().cpu().clone() for name, value in self.client.prepare_for_training()}
        # 快照只包含 LoRA 参数，作为整个 run 的冻结 reference policy。
        reference_state = {name: value.detach().clone() for name, value in self.client.named_trainable_parameters()}
        params = [value for _, value in self.client.named_trainable_parameters()]
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)
        records=[]; started=time.perf_counter()
        for index, task in enumerate(tasks, 1):
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                # 只清零统计计数，不释放模型缓存；这样每组峰值可比较且不会破坏权重。
                torch.cuda.reset_peak_memory_stats()
            self.client._model.eval()
            group = await self.runner.run_group(task, group_size)
            metrics = self.update_group(group, optimizer, params, reference_state); metrics["task_id"] = task.id; metrics["index"] = index; records.append(metrics)
        after = {name: value.detach().float().cpu() for name, value in self.client.prepare_for_training()}
        changed = [name for name in before if not torch_equal(before[name], after[name])]
        if not changed: raise RuntimeError("GRPO 安全检查失败：没有任何 LoRA 参数发生变化")
        output=Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        self.client._model.save_pretrained(output)
        result={"version":"v0.7.0", "elapsed_seconds": time.perf_counter()-started, "policy_fingerprint": actual,
                "trainable_parameter_names": list(before), "changed_lora_parameters": changed, "reference_mode": "initial_adapter", "groups": records}
        (output / "grpo_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def torch_equal(left, right) -> bool:
    import torch
    return torch.equal(left, right)
