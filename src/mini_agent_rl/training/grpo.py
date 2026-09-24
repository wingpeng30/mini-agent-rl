"""v0.7.0：适合单卡冒烟实验的最小 LoRA-GRPO 更新器。"""
from __future__ import annotations

import hashlib, json, math, time
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
    # CLI resolves this from the actual card (24GB -> 21GB); direct unit-test use
    # remains intentionally conservative.
    max_reserved_memory_gb: float = 7.4
    # A diagnostic pilot may deliberately require fewer than half its tasks to
    # update.  `None` retains the conservative historical half-of-tasks rule.
    min_effective_groups: int | None = None

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
    # 对 current 相对 reference 的非负近似 KL：exp(ref-current) + current-ref - 1。
    # 旧版反向使用 delta，会惩罚相反方向；版本报告必须保留这一训练语义变更。
    delta = current - reference
    kl = torch.exp(-delta) + delta - 1
    return -objective.mean() + kl_beta * kl.mean(), kl.mean().detach(), ((ratio - 1).abs() > clip_epsilon).float().mean().detach()


def aggregate_rollout_token_values(values):
    """把同一 rollout 的各 transition 按 action token 数聚合，避免短动作权重过大。"""
    import torch
    if not values:
        raise ValueError("rollout 至少需要一个 assistant action")
    total_tokens = sum(int(token_count) for _, token_count in values)
    if total_tokens <= 0:
        raise ValueError("assistant action token 数必须大于 0")
    return sum(value * int(token_count) for value, token_count in values) / total_tokens


def _pearson(left: list[float], right: list[float]) -> float | None:
    """小样本诊断相关性；零方差时返回 None，避免制造虚假数值。"""
    if len(left) != len(right) or len(left) < 2: return None
    mean_left, mean_right = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - mean_left) ** 2 for x in left) * sum((y - mean_right) ** 2 for y in right))
    return numerator / denominator if denominator else None


def guarded_optimizer_step(optimizer, params, memory_stats: dict[str, Any], memory_limit_gb: float) -> None:
    """Apply an update only after every irreversible numerical/resource gate."""
    import torch
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in params):
        raise RuntimeError("optimizer 更新前梯度出现 NaN/Inf")
    if any(not torch.isfinite(parameter).all() for parameter in params):
        raise RuntimeError("optimizer 更新前参数出现 NaN/Inf")
    reserved = memory_stats.get("peak_reserved_gb")
    if reserved is None or not math.isfinite(float(reserved)) or float(reserved) > memory_limit_gb:
        raise RuntimeError(f"optimizer 更新前显存门槛失败：{reserved}")
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in params):
        raise RuntimeError("optimizer 更新后参数出现 NaN/Inf")


def ratio_statistics(values) -> dict[str, float]:
    """Summarize a nonempty finite ratio tensor for a pre-update gate."""
    import torch
    if not values:
        raise RuntimeError("ratio 含空值")
    joined = torch.cat([value.detach().float().cpu() for value in values])
    if not joined.numel() or not torch.isfinite(joined).all():
        raise RuntimeError("ratio 含空值或非有限值")
    return {"min": float(joined.min()), "max": float(joined.max()),
            "p95": float(torch.quantile(joined, .95)), "count": int(joined.numel())}


def require_initial_ratio(stats: dict[str, float], *, label: str) -> None:
    """Reject a group before its irreversible optimizer update."""
    if not (.99 <= stats["p95"] <= 1.01 and .90 <= stats["min"] and stats["max"] <= 1.10):
        raise RuntimeError(f"{label} ratio 门槛失败：{stats}")


class MinimalGRPOTrainer:
    """每个 rollout group 进行小批量更新，刻意优先可审计和显存安全而非吞吐。"""
    def __init__(self, client, runner, store, config: GRPOConfig):
        self.client, self.runner, self.store, self.config = client, runner, store, config

    def trainable_names(self) -> list[str]:
        return [name for name, _ in self.client.prepare_for_training()]

    def update_group(self, task: Task, group: RolloutGroup, optimizer, params, reference_state) -> dict[str, Any]:
        import torch
        if self.config.policy_epochs < 1:
            raise ValueError("policy_epochs 必须至少为 1，才能验证首次实际反传")
        if group.zero_variance:
            return {"group_id": group.id, "skipped": "zero_variance"}
        rollouts = [self.store.get_rollout(item) for item in group.rollout_ids]
        if any(item is None for item in rollouts):
            raise RuntimeError("RL group 含无法读取的 rollout；拒绝静默过滤")
        rollouts = [item for item in rollouts if item is not None]
        if any(not item.eligible_for_rl or item.advantage is None for item in rollouts):
            raise RuntimeError("RL group 含不具资格或缺少 advantage 的 rollout；拒绝静默过滤")
        for rollout in rollouts:
            transitions = self.store.get_transitions(rollout.id)
            recomputed = self.runner.reward.evaluate(task, transitions, rollout.final_answer)
            if set(recomputed) != set(rollout.reward_components) or any(
                not math.isclose(float(value), float(rollout.reward_components[name]), abs_tol=1e-9)
                for name, value in recomputed.items()
            ) or not math.isclose(sum(recomputed.values()), float(rollout.reward), abs_tol=1e-9):
                raise RuntimeError("optimizer 更新前奖励重算不一致")
        all_transitions = [(rollout, transition) for rollout in rollouts for transition in self.store.get_transitions(rollout.id)]
        samples = [(rollout, transition) for rollout, transition in all_transitions
                   if transition.response.action_token_ids and transition.response.action_mask and any(transition.response.action_mask)]
        if len(samples) != len(all_transitions):
            raise RuntimeError("RL group 含缺少 action token/mask 的 transition；拒绝静默过滤")
        if not samples:
            return {"group_id": group.id, "skipped": "no_action_tokens"}
        # 训练前重算 old/reference，绝不盲信跨进程 JSONL 中的 BF16 数字。
        # 此处必须 eval：LoRA dropout 若在 train mode 会把 old policy 本身变成随机变量。
        self.client._model.eval()
        frozen = []
        for rollout, transition in samples:
            ids = transition.response.action_token_ids or []
            prompt_ids = transition.response.prompt_token_ids
            prompt_mask = transition.response.prompt_attention_mask
            if not prompt_ids:
                raise RuntimeError("RL 轨迹缺少预算后 prompt_token_ids，拒绝不精确重放")
            if prompt_mask is None or len(prompt_mask) != len(prompt_ids) or any(v not in (0, 1) for v in prompt_mask) or not any(prompt_mask):
                raise RuntimeError("prompt attention mask 无效")
            if transition.response.action_mask != [1] * len(ids):
                raise RuntimeError("当前 scorer 只支持完整 assistant action；拒绝不匹配的 action mask")
            expected_prompt_hash = transition.response.usage.get("prompt_sha256")
            if not expected_prompt_hash:
                raise RuntimeError("RL 轨迹缺少 prompt_sha256，拒绝不精确重放")
            old = self.client.score_action_tensor(transition.messages, ids, False, prompt_ids, prompt_mask).detach()
            old_prompt_hash = self.client._context_records[-1].get("prompt_sha256")
            reference = self.client.reference_logprobs(transition.messages, ids, reference_state, prompt_ids, prompt_mask).detach()
            reference_prompt_hash = self.client._context_records[-1].get("prompt_sha256")
            if old.numel() != len(ids) or reference.numel() != len(ids) or not torch.isfinite(old).all() or not torch.isfinite(reference).all():
                raise RuntimeError("old/reference logprob 长度不一致或出现非有限值")
            if old_prompt_hash != expected_prompt_hash or reference_prompt_hash != expected_prompt_hash:
                raise RuntimeError("采样与 old/reference logprob 使用了不同的上下文 token")
            recorded = transition.response.old_logprobs
            if recorded is None or len(recorded) != len(ids) or not all(math.isfinite(float(value)) for value in recorded):
                raise RuntimeError("采样时 old_logprobs 缺失、长度不一致或非有限")
            raw_delta = float((old.detach().cpu() - torch.tensor(recorded)).abs().max())
            frozen.append((rollout, transition, old, reference, raw_delta))
        # Test the exact training-mode forward before any backward/update. LoRA
        # dropout is forced to eval, while the model and checkpointing stay in the
        # same mode used by the actual optimizer pass.
        self.client.set_training_forward_mode()
        training_mode = self.client.training_mode_report()
        initial_ratio_tensors = []
        for _, transition, old, _, _ in frozen:
            current = self.client.score_action_tensor(
                transition.messages, transition.response.action_token_ids or [], True,
                transition.response.prompt_token_ids, transition.response.prompt_attention_mask,
            )
            if current.numel() != old.numel() or not torch.isfinite(current).all():
                raise RuntimeError("initial current logprob 长度不一致或出现非有限值")
            initial_ratio_tensors.append(torch.exp(current.detach() - old).float().cpu())
        ratio_stats = ratio_statistics(initial_ratio_tensors)
        require_initial_ratio(ratio_stats, label="initial")
        epoch_metrics = []
        first_backward_ratio_tensors = []
        for epoch_index in range(self.config.policy_epochs):
            optimizer.zero_grad(set_to_none=True); losses=[]; kls=[]; clips=[]
            by_rollout = {}
            for sample in frozen: by_rollout.setdefault(sample[0].id, []).append(sample)
            rollout_losses = []
            # 逐 rollout 反传后立即释放计算图：保持 rollout 等权，避免 8GB 显存同时保存整组图。
            rollout_count = len(by_rollout)
            for samples_for_rollout in by_rollout.values():
                # 先取得 rollout 的总 token 数，再逐 transition 加权反传。这样目标仍是
                # rollout 等权，却不用同时保留多个 forward graph。
                rollout_tokens = sum(len(item[1].response.action_token_ids or []) for item in samples_for_rollout)
                if rollout_tokens <= 0:
                    raise RuntimeError("rollout 没有有效 action token")
                token_kls=[]; token_clips=[]; weighted_loss = 0.0
                for rollout, transition, old, reference, _ in samples_for_rollout:
                    current = self.client.score_action_tensor(
                        transition.messages, transition.response.action_token_ids or [], True,
                        transition.response.prompt_token_ids, transition.response.prompt_attention_mask,
                    )
                    expected_prompt_hash = transition.response.usage.get("prompt_sha256")
                    if not expected_prompt_hash or self.client._context_records[-1].get("prompt_sha256") != expected_prompt_hash:
                        raise RuntimeError("采样与 current logprob 使用了不同的上下文 token")
                    if epoch_index == 0:
                        first_backward_ratio_tensors.append(torch.exp(current.detach() - old).float().cpu())
                    loss, kl, clipped = clipped_grpo_loss(current, old, reference, rollout.advantage or 0.0, self.config.clip_epsilon, self.config.kl_beta)
                    if not torch.isfinite(loss): raise RuntimeError("GRPO loss 出现 NaN/Inf")
                    token_count = int(current.numel())
                    weight = token_count / rollout_tokens
                    # 此处立即 backward，Python 引用离开循环后可释放当前计算图。
                    (loss * weight / rollout_count).backward()
                    weighted_loss += float(loss.detach()) * weight
                    token_kls.append((kl, token_count)); token_clips.append((clipped, token_count))
                rollout_losses.append(weighted_loss); losses.append(weighted_loss)
                kls.append(float(aggregate_rollout_token_values(token_kls)))
                clips.append(float(aggregate_rollout_token_values(token_clips)))
            # The first actual backward covers every transition in the first
            # policy epoch.  Its ratio is a safety gate, so check it before the
            # first irreversible optimizer step.
            if epoch_index == 0:
                first_backward_stats = ratio_statistics(first_backward_ratio_tensors)
                require_initial_ratio(first_backward_stats, label="首次实际反传 current")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm))
            if not math.isfinite(grad_norm): raise RuntimeError("梯度范数出现 NaN/Inf")
            pre_step_memory = self.client.gpu_memory_stats()
            guarded_optimizer_step(optimizer, params, pre_step_memory, self.config.max_reserved_memory_gb)
            epoch_metrics.append({"loss": sum(losses)/len(losses), "kl": sum(kls)/len(kls), "clip_fraction": sum(clips)/len(clips), "grad_norm": grad_norm,
                                  "pre_step_gpu_memory": pre_step_memory})
        self.client._model.eval()
        action_lengths = [len(transition.response.action_token_ids or []) for _, transition, _, _, _ in frozen]
        advantages = [float(rollout.advantage or 0.0) for rollout, _, _, _, _ in frozen]
        rewards = [float(rollout.reward) for rollout, _, _, _, _ in frozen]
        return {"group_id": group.id, "skipped": None, "samples": len(frozen), "reward_mean": group.mean_reward,
                "reward_std": group.std_reward, "metrics": epoch_metrics, "peak_gpu_memory_gb": self.client.gpu_memory_peak_gb(),
                "initial_ratio": ratio_stats, "first_backward_ratio": first_backward_stats,
                "raw_vs_recomputed_logprob_max_abs": max((item[4] or 0.0 for item in frozen), default=0.0),
                "gpu_memory": self.client.gpu_memory_stats(), "mean_action_tokens": sum(action_lengths) / len(action_lengths),
                "action_length_advantage_correlation": _pearson([float(x) for x in action_lengths], advantages),
                "action_length_reward_correlation": _pearson([float(x) for x in action_lengths], rewards)}

    async def train(self, tasks: list[Task], group_size: int, output_dir: str, expected_fingerprint: dict[str, Any]) -> dict[str, Any]:
        """采样与更新必须共享一个已加载实例，避免 BF16 跨加载比例漂移。"""
        import torch
        if self.client._model is None:
            self.client._load()
        self.client.reset_context_records()
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
        records=[]; started=time.perf_counter(); stopped_reason: str | None = None
        training_mode: dict[str, Any] = {}
        previous_sampling = {"calls": 0, "truncated": 0, "compacted": 0, "history_dropped": 0, "budget_errors": 0, "retained_docs": 0, "lossless": 0, "lossy": 0, "documents_deleted": 0}
        for index, task in enumerate(tasks, 1):
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                # 只清零统计计数，不释放模型缓存；这样每组峰值可比较且不会破坏权重。
                torch.cuda.reset_peak_memory_stats()
            self.client._model.eval()
            group = await self.runner.run_group(task, group_size)
            usage = self.client.context_usage_summary().get("phases", {}).get("rollout_sampling", {})
            current_sampling = {
                "calls": int(usage.get("calls", 0)),
                "truncated": int(usage.get("truncated", 0)),
                "compacted": int(usage.get("observation_compacted", 0)),
                "history_dropped": int(usage.get("history_dropped", 0)),
                "budget_errors": int(usage.get("context_budget_errors", 0)),
                "retained_docs": int(usage.get("retained_tool_documents", 0)),
                "lossless": int(usage.get("lossless_expression_changed", 0)),
                "lossy": int(usage.get("lossy_body_trimmed", 0)),
                "documents_deleted": int(usage.get("documents_deleted", 0)),
            }
            delta_calls = max(0, current_sampling["calls"] - previous_sampling["calls"])
            delta = {key: max(0, current_sampling[key] - previous_sampling[key]) for key in current_sampling}
            pre_update_failures = []
            if delta_calls == 0: pre_update_failures.append("no_sampling_context_record")
            if delta["budget_errors"]: pre_update_failures.append("context_budget_error")
            if delta["history_dropped"]: pre_update_failures.append("history_dropped")
            if delta["documents_deleted"]: pre_update_failures.append("lossy_document_deletion")
            if delta_calls and delta["lossy"] / delta_calls > .05: pre_update_failures.append("lossy_body_trim_rate_exceeded")
            if any(not rollout.eligible_for_rl for rollout in (self.store.get_rollout(rid) for rid in group.rollout_ids)):
                pre_update_failures.append("infrastructure_failure")
            if pre_update_failures:
                metrics = {"group_id": group.id, "skipped": "pre_update_gate", "pre_update_failures": pre_update_failures}
            else:
                metrics = self.update_group(task, group, optimizer, params, reference_state)
            metrics.update({
                "task_id": task.id,
                "index": index,
                "group_index_one_based": index,
                "task_sha256": hashlib.sha256(task.model_dump_json().encode("utf-8")).hexdigest(),
                "environment_fingerprint": dict(self.runner.environment_fingerprint),
                "sampling_calls": delta_calls,
                "legacy_prompt_truncation_rate": delta["truncated"] / delta_calls if delta_calls else 0.0,
                "observation_compaction_rate": delta["compacted"] / delta_calls if delta_calls else 0.0,
                "history_drop_rate": delta["history_dropped"] / delta_calls if delta_calls else 0.0,
                "context_budget_errors": delta["budget_errors"],
                "retained_tool_documents": delta["retained_docs"],
                "lossless_expression_changed": delta["lossless"],
                "lossy_body_trimmed": delta["lossy"],
                "documents_deleted": delta["documents_deleted"],
            })
            records.append(metrics)
            previous_sampling = current_sampling
            training_mode = self.client.training_mode_report()
            if pre_update_failures:
                stopped_reason = f"group_{index}_pre_update_gate:{','.join(pre_update_failures)}"; break
            # Zero-variance groups are a valid diagnostic outcome; final effective
            # group gates decide whether there was sufficient learning signal.
            if metrics.get("skipped"):
                continue
            # 每组结束后立即检查，不让已知不安全的实验继续占用显存或产生混杂轨迹。
            memory_stats = metrics.get("gpu_memory", {})
            memory_value = memory_stats.get("peak_reserved_gb")
            memory = float(memory_value) if memory_value is not None else None
            required_memory = ("peak_allocated_gb", "peak_reserved_gb", "device_process_gb")
            if any(memory_stats.get(key) is None for key in required_memory):
                stopped_reason = f"group_{index}_memory_telemetry_unobserved"; break
            if not math.isfinite(memory) or memory > self.config.max_reserved_memory_gb:
                stopped_reason = f"group_{index}_reserved_memory_exceeded:{memory:.3f}GB"; break
        after = {name: value.detach().float().cpu() for name, value in self.client.prepare_for_training()}
        changed = [name for name in before if not torch_equal(before[name], after[name])]
        if not changed and stopped_reason is None:
            stopped_reason = "no_lora_parameter_changed"
        output=Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        reward_coverage: dict[str, int] = {}
        trained_rollouts = [self.store.get_rollout(row[0]) for row in self.store.conn.execute("SELECT id FROM rollouts")]
        for rollout in (item for item in trained_rollouts if item):
            for name, value in rollout.reward_components.items():
                if value != 0: reward_coverage[name] = reward_coverage.get(name, 0) + 1
        result={"version":"v0.8.3", "status": "GATE_FAILED" if stopped_reason else "COMPLETED",
                "stopped_reason": stopped_reason, "elapsed_seconds": time.perf_counter()-started, "policy_fingerprint": actual,
                "trainable_parameter_names": list(before), "changed_lora_parameters": changed, "reference_mode": "initial_adapter", "groups": records}
        result["reference_snapshot_unchanged"] = all(
            torch_equal(reference_state[name].detach().float().cpu(), before[name]) for name in reference_state
        )
        if not result["reference_snapshot_unchanged"] and stopped_reason is None:
            stopped_reason = "reference_snapshot_changed"
            result["stopped_reason"] = stopped_reason
        result["training_mode"] = training_mode
        result["context_usage"] = self.client.context_usage_summary()
        result["reward_component_nonzero_counts"] = reward_coverage
        phases = result["context_usage"]["phases"]
        sampling = phases.get("rollout_sampling", {})
        sampling_rate = float(sampling.get("truncation_rate", 0.0))
        lossy_rate = int(sampling.get("lossy_body_trimmed", 0)) / int(sampling.get("calls", 1) or 1)
        effective_groups = sum(not row.get("skipped") for row in records)
        required_effective_groups = self.config.min_effective_groups if self.config.min_effective_groups is not None else math.ceil(len(tasks) / 2)
        infrastructure_failures = sum(item is not None and not item.eligible_for_rl for item in trained_rollouts)
        reserved_values = [row.get("gpu_memory", {}).get("peak_reserved_gb") for row in records if not row.get("skipped")]
        memory_observed = bool(reserved_values) and all(value is not None for value in reserved_values)
        device_observed = all(row.get("gpu_memory", {}).get("device_process_gb") is not None for row in records if not row.get("skipped"))
        peak_reserved = max((float(value) for value in reserved_values if value is not None), default=None)
        result["gates"] = {
            "sampling_truncation_rate": sampling_rate,
            "lossy_body_trim_rate": lossy_rate,
            "sampling_truncation_passed": lossy_rate <= 0.05 and int(sampling.get("history_dropped", 0)) == 0 and int(sampling.get("context_budget_errors", 0)) == 0 and int(sampling.get("documents_deleted", 0)) == 0,
            "effective_groups": effective_groups,
            "required_effective_groups": required_effective_groups,
            "reward_variance_passed": effective_groups >= required_effective_groups,
            "infrastructure_failures": infrastructure_failures,
            "infrastructure_passed": infrastructure_failures == 0,
            "peak_reserved_memory_gb": peak_reserved,
            "memory_observed": memory_observed,
            "device_memory_observed": device_observed,
            "reserved_memory_limit_gb": self.config.max_reserved_memory_gb,
            "memory_passed": memory_observed and device_observed and peak_reserved is not None and peak_reserved <= self.config.max_reserved_memory_gb,
        }
        result["gates"]["passed"] = stopped_reason is None and all((result["gates"]["sampling_truncation_passed"],
                                           result["gates"]["reward_variance_passed"],
                                           result["gates"]["infrastructure_passed"],
                                           result["gates"]["memory_passed"]))
        # A checkpoint becomes an experimental artifact only after every final gate.
        result["status"] = "COMPLETED" if result["gates"]["passed"] else "GATE_FAILED"
        if not result["gates"]["passed"] and result["stopped_reason"] is None:
            result["stopped_reason"] = "final_gate_failed"
        if result["gates"]["passed"]:
            self.client._model.save_pretrained(output)
        (output / "grpo_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def torch_equal(left, right) -> bool:
    import torch
    return torch.equal(left, right)
