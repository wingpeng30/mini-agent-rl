"""v095 GRPO correctness gate with exact token replay and an external reload."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient
from mini_agent_rl.prompts import DIRECT_SYSTEM, PROMPT_VERSION
from mini_agent_rl.storage.sqlite import SQLiteStore
from mini_agent_rl.training.grpo import (clipped_grpo_loss, guarded_optimizer_step,
                                         ratio_statistics, require_initial_ratio)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def named_tensor_digests(model, predicate) -> dict[str, str]:
    return {name: tensor_digest(value) for name, value in model.named_parameters() if predicate(name)}


def adapter_file_digests(path: Path) -> dict[str, str]:
    from safetensors.torch import load_file
    return {name: tensor_digest(value) for name, value in load_file(str(path / "adapter_model.safetensors")).items()}


def score_rows(client, transitions, *, grad: bool) -> list[torch.Tensor]:
    rows = []
    for _, transition in transitions:
        response = transition.response
        ids, prompt, mask = response.action_token_ids, response.prompt_token_ids, response.prompt_attention_mask
        if not ids or not prompt or mask is None or len(prompt) != len(mask) or response.action_mask != [1] * len(ids):
            raise RuntimeError("验收轨迹缺少完整 token/mask")
        if not response.usage.get("prompt_sha256"):
            raise RuntimeError("验收轨迹缺少 prompt hash")
        value = client.score_action_tensor(transition.messages, ids, grad, prompt, mask)
        if value.numel() != len(ids) or not torch.isfinite(value).all():
            raise RuntimeError("评分长度不一致或非有限")
        if client._context_records[-1].get("prompt_sha256") != response.usage["prompt_sha256"]:
            raise RuntimeError("验收时输入 token 未精确重放")
        rows.append(value)
    return rows


def flatten(rows: list[torch.Tensor]) -> list[float]:
    return [float(item) for row in rows for item in row.detach().float().cpu().tolist()]


def abs_delta(left: list[torch.Tensor], right: list[torch.Tensor]) -> dict[str, float]:
    values = [torch.abs(a.detach().float().cpu() - b.detach().float().cpu()) for a, b in zip(left, right, strict=True)]
    joined = torch.cat(values)
    return {"max": float(joined.max()), "p95": float(torch.quantile(joined, .95)), "count": int(joined.numel())}


def required_memory(memory: dict, limit: float) -> tuple[bool, dict[str, float | None]]:
    required = {key: memory.get(key) for key in ("peak_allocated_gb", "peak_reserved_gb", "device_process_gb")}
    valid = all(value is not None and math.isfinite(float(value)) and float(value) <= limit for value in required.values())
    return valid, required


def select_transitions(store: SQLiteStore):
    selected, found = [], set()
    for rollout in store.list_rollouts():
        for transition in store.get_transitions(rollout.id):
            if not transition.response.action_token_ids:
                continue
            kind = "invalid" if rollout.termination_reason == "invalid_action" else None
            if kind and kind not in found:
                selected.append((kind, transition)); found.add(kind)
            if transition.response.tool_call and "search" not in found:
                selected.append(("search", transition)); found.add("search")
            if transition.termination_reason == "answer" and "answer" not in found:
                selected.append(("answer", transition)); found.add("answer")
    missing = {"search", "answer", "invalid"} - found
    if missing:
        raise RuntimeError(f"验收源缺少轨迹类型：{sorted(missing)}")
    return selected


def verifier(args) -> None:
    store = SQLiteStore(args.source_db)
    selected = select_transitions(store)
    client = TransformersModelClient(LocalModelConfig(model_id=args.model, adapter_path=args.adapter,
        deterministic=True, dtype="bfloat16", max_input_tokens=2048, max_new_tokens=128,
        gradient_checkpointing=False, observation_encoding="compact-v3", enable_thinking=False))
    client._load(); client._model.eval()
    rows = score_rows(client, selected, grad=False)
    payload = {"status": "PASSED", "policy_fingerprint": client.policy_fingerprint(),
               "adapter_file_digests": adapter_file_digests(Path(args.adapter)),
               "scores": flatten(rows), "score_token_count": sum(row.numel() for row in rows)}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def acceptance(args) -> None:
    output, probe = Path(args.output), Path(args.probe_adapter)
    if output.exists() or probe.exists():
        raise FileExistsError("v095 验收输出已存在，拒绝混入旧运行")
    report = {"version": "v0.9.5", "status": "RUNNING", "checkpoint_usable": False,
              "started_unix": time.time(), "configuration": {"model": args.model, "adapter": args.adapter,
              "source_db": args.source_db, "retriever_version": args.retriever_version,
              "prompt_version": PROMPT_VERSION, "system_prompt_sha256": hashlib.sha256(DIRECT_SYSTEM.encode()).hexdigest(),
              "max_input_tokens": 2048, "max_new_tokens": 128, "memory_limit_gb": 21.0}}
    try:
        store = SQLiteStore(args.source_db); selected = select_transitions(store)
        report["source_db_sha256"] = file_sha256(Path(args.source_db))
        report["trajectory_kinds"] = [kind for kind, _ in selected]
        client = TransformersModelClient(LocalModelConfig(model_id=args.model, adapter_path=args.adapter,
            deterministic=True, dtype="bfloat16", max_input_tokens=2048, max_new_tokens=128,
            gradient_checkpointing=False, observation_encoding="compact-v3", enable_thinking=False))
        client._load(); client.reset_memory_stats(); client._model.eval()
        report["policy_fingerprint"] = client.policy_fingerprint()
        old = [row.detach() for row in score_rows(client, selected, grad=False)]
        raw_deltas = []
        for (_, transition), value in zip(selected, old, strict=True):
            raw = transition.response.old_logprobs
            if raw is None or len(raw) != value.numel() or not all(math.isfinite(float(item)) for item in raw):
                raise RuntimeError("验收源的采样 old logprob 不完整")
            raw_deltas.append(float((value.cpu() - torch.tensor(raw)).abs().max()))
        report["exact_replay"] = {"raw_vs_recomputed_logprob_max_abs": max(raw_deltas),
                                  "token_count": sum(value.numel() for value in old)}
        base_before = named_tensor_digests(client._model, lambda name: "lora" not in name.lower())
        params = [value for _, value in client.prepare_for_training()]
        client.set_training_forward_mode()
        trainable_before = {name: tensor_digest(value) for name, value in client.named_trainable_parameters()}
        reference = {name: value.detach().clone() for name, value in client.named_trainable_parameters()}
        reference_before = {name: tensor_digest(value) for name, value in reference.items()}
        reference_scores_before = [client.reference_logprobs(t.messages, t.response.action_token_ids or [], reference,
                                  t.response.prompt_token_ids, t.response.prompt_attention_mask).detach() for _, t in selected]
        # This is deliberately grad-enabled to match training mode, but its
        # graph is not part of the optimizer pass and must be released before
        # reference scoring temporarily swaps LoRA tensors.
        initial = score_rows(client, selected, grad=True)
        initial_stats = ratio_statistics([torch.exp(current.detach() - prior) for current, prior in zip(initial, old, strict=True)])
        del initial
        require_initial_ratio(initial_stats, label="initial")
        optimizer = torch.optim.AdamW(params, lr=5e-6); optimizer.zero_grad(set_to_none=True)
        actual_ratios = []
        for (_, transition), prior in zip(selected, old, strict=True):
            reference_lp = client.reference_logprobs(transition.messages, transition.response.action_token_ids or [], reference,
                                                     transition.response.prompt_token_ids, transition.response.prompt_attention_mask).detach()
            current = client.score_action_tensor(transition.messages, transition.response.action_token_ids or [], True,
                                                 transition.response.prompt_token_ids, transition.response.prompt_attention_mask)
            actual_ratios.append(torch.exp(current.detach() - prior))
            loss, _, _ = clipped_grpo_loss(current, prior, reference_lp, 1.0, .2, .02)
            if not torch.isfinite(loss):
                raise RuntimeError("验收 loss 非有限")
            (loss / len(selected)).backward()
        actual_stats = ratio_statistics(actual_ratios); require_initial_ratio(actual_stats, label="首次实际反传 current")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if not math.isfinite(grad_norm):
            raise RuntimeError("验收梯度非有限")
        pre_step_memory = client.gpu_memory_stats()
        guarded_optimizer_step(optimizer, params, pre_step_memory, 21.0)
        base_after = named_tensor_digests(client._model, lambda name: "lora" not in name.lower())
        trainable_after = {name: tensor_digest(value) for name, value in client.named_trainable_parameters()}
        reference_after = {name: tensor_digest(value) for name, value in reference.items()}
        reference_scores_after = [client.reference_logprobs(t.messages, t.response.action_token_ids or [], reference,
                                 t.response.prompt_token_ids, t.response.prompt_attention_mask).detach() for _, t in selected]
        post_scores = [row.detach() for row in score_rows(client, selected, grad=False)]
        changed = [name for name in trainable_before if trainable_before[name] != trainable_after[name]]
        client._model.save_pretrained(probe)
        file_digests = adapter_file_digests(probe)
        memory_adapter_values = sorted(trainable_after.values())
        file_adapter_values = sorted(file_digests.values())
        verify_out = probe / "external-reload.json"
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--verify-adapter", "--model", args.model,
                                    "--adapter", str(probe), "--source-db", args.source_db, "--output", str(verify_out)],
                                   capture_output=True, text=True, timeout=600, check=False)
        if completed.returncode != 0 or not verify_out.exists():
            raise RuntimeError(f"独立重载进程失败：{completed.stderr[-800:]}")
        external = json.loads(verify_out.read_text(encoding="utf-8"))
        reload_delta = abs_delta([torch.tensor(flatten(post_scores))], [torch.tensor(external["scores"])])
        pressure = TransformersModelClient(LocalModelConfig(model_id=args.model, adapter_path=args.adapter, deterministic=True,
            dtype="bfloat16", max_input_tokens=2048, max_new_tokens=128, gradient_checkpointing=False, observation_encoding="compact-v3"))
        pressure._load(); pressure.reset_memory_stats(); pressure_params = [value for _, value in pressure.prepare_for_training()]
        tokenizer = getattr(pressure._processor, "tokenizer", pressure._processor)
        token = tokenizer.encode("evidence retrieval diagnostic token ", add_special_tokens=False)
        prompt = (token * (2048 // len(token) + 1))[:2048]; action = (token * (128 // len(token) + 1))[:128]
        old_pressure = pressure.score_action_tensor([], action, False, prompt, [1] * len(prompt)).detach()
        pressure.set_training_forward_mode(); current_pressure = pressure.score_action_tensor([], action, True, prompt, [1] * len(prompt))
        pressure_loss, _, _ = clipped_grpo_loss(current_pressure, old_pressure, old_pressure, 1.0, .2, .02)
        pressure_loss.backward(); pressure_optimizer = torch.optim.AdamW(pressure_params, lr=5e-6)
        pressure_memory = pressure.gpu_memory_stats(); guarded_optimizer_step(pressure_optimizer, pressure_params, pressure_memory, 21.0)
        memory_ok, memory_values = required_memory(pressure_memory, 21.0)
        report.update({"initial_ratio": initial_stats, "first_actual_backward_ratio": actual_stats,
                       "optimizer_probe": {"gradient_norm": grad_norm, "changed_lora_parameters": changed,
                                           "all_changed_are_lora": all("lora" in name.lower() for name in changed),
                                           "base_parameters_unchanged": base_before == base_after,
                                           "reference_clone_unchanged": reference_before == reference_after,
                                           "reference_score_delta": abs_delta(reference_scores_before, reference_scores_after)},
                       "reload": {"saved_file_tensor_digests": file_digests, "external_file_tensor_digests": external["adapter_file_digests"],
                                  "memory_to_saved_tensor_exact": memory_adapter_values == file_adapter_values,
                                  "adapter_tensor_exact": file_digests == external["adapter_file_digests"], "score_delta": reload_delta,
                                  "external_policy_fingerprint": external["policy_fingerprint"]},
                       "pressure_2048_128": {"loss_finite": bool(torch.isfinite(pressure_loss)), "memory": memory_values}})
        gates = {"initial_ratio": True, "first_actual_backward_ratio": True,
                 "exact_replay": math.isfinite(report["exact_replay"]["raw_vs_recomputed_logprob_max_abs"]),
                 "lora_only_nonzero_update": bool(changed) and report["optimizer_probe"]["all_changed_are_lora"],
                 "base_unchanged": report["optimizer_probe"]["base_parameters_unchanged"],
                 "reference_unchanged": report["optimizer_probe"]["reference_clone_unchanged"] and report["optimizer_probe"]["reference_score_delta"]["max"] <= 1e-6,
                 "reload": report["reload"]["memory_to_saved_tensor_exact"] and report["reload"]["adapter_tensor_exact"] and reload_delta["p95"] <= 1e-3,
                 "pressure": bool(torch.isfinite(pressure_loss)) and memory_ok}
        report["gates"] = gates; report["checkpoint_usable"] = all(gates.values()); report["status"] = "PASSED" if report["checkpoint_usable"] else "FAILED"
    except Exception as exc:
        report["status"] = "FAILED"; report["error"] = repr(exc)
    finally:
        report["finished_unix"] = time.time(); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(json.dumps({"status": report["status"], "checkpoint_usable": report["checkpoint_usable"], "output": str(output)}), flush=True)
    if report["status"] != "PASSED":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-adapter", action="store_true")
    parser.add_argument("--model", default="/models/qwen_08b")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-adapter")
    parser.add_argument("--retriever-version", default="bm25-v2")
    args = parser.parse_args()
    if args.verify_adapter:
        verifier(args)
    elif not args.probe_adapter:
        parser.error("--probe-adapter is required for acceptance")
    else:
        acceptance(args)


if __name__ == "__main__":
    main()
