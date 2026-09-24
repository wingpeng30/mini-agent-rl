"""Synthetic full-length optimizer benchmark; never an effectiveness evaluation.

Run each configuration in a fresh process. No checkpoint is published.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient
from mini_agent_rl.training.grpo import clipped_grpo_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/models/qwen_08b')
    parser.add_argument('--adapter', default='checkpoints/qwen35-08b-sft-v052')
    parser.add_argument('--four-bit', action='store_true')
    parser.add_argument('--checkpointing', action='store_true')
    parser.add_argument('--max-input-tokens', type=int, default=2048)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    client = TransformersModelClient(LocalModelConfig(
        model_id=args.model, adapter_path=args.adapter, load_in_4bit=args.four_bit,
        gradient_checkpointing=args.checkpointing, max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens))
    started = time.perf_counter()
    client._load()
    params = [p for _, p in client.prepare_for_training()]
    reference = {n: p.detach().clone() for n, p in client.named_trainable_parameters()}
    optimizer = torch.optim.AdamW(params, lr=5e-6)
    # Fixed synthetic token sequence deliberately fills the input and output budgets.
    tokenizer = getattr(client._processor, 'tokenizer', client._processor)
    tokens = tokenizer.encode('The evidence describes a city and its history. ', add_special_tokens=False)
    prompt = (tokens * (args.max_input_tokens // len(tokens) + 1))[:args.max_input_tokens]
    action = (tokens * (args.max_new_tokens // len(tokens) + 1))[:args.max_new_tokens]
    mask = [1] * len(prompt)
    client.reset_memory_stats()
    client._model.eval()
    old = client.score_action_tensor([], action, False, prompt, mask).detach().clone()
    ref = client.reference_logprobs([], action, reference, prompt, mask).detach().clone()
    assert torch.isfinite(old).all() and torch.isfinite(ref).all()
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    metrics = []
    client.set_training_forward_mode()
    for epoch in range(2):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(2):
            current = client.score_action_tensor([], action, True, prompt, mask)
            loss, _, _ = clipped_grpo_loss(current, old, ref, 1.0, .2, .02)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            (loss / 2).backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradients')
        optimizer.step()
        metrics.append({'epoch': epoch, 'loss': float(loss.detach()), 'grad_norm': float(norm)})
    torch.cuda.synchronize()
    report = {'synthetic': True, 'effectiveness_evidence': False,
              'four_bit': args.four_bit, 'checkpointing': args.checkpointing,
              'prompt_tokens': args.max_input_tokens, 'action_tokens': args.max_new_tokens, 'transitions_per_update': 2,
              'updates': 2, 'training_seconds': time.perf_counter() - train_start,
              'total_seconds': time.perf_counter() - started, 'metrics': metrics,
              'memory': client.gpu_memory_stats(), 'optimizer_states': len(optimizer.state)}
    report['passed'] = report['memory']['peak_reserved_gb'] is not None and report['memory']['peak_reserved_gb'] <= 21
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
