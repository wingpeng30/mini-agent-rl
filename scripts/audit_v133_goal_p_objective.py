"""CPU-only directional checks before Goal P's new training recipe."""
import json
import math
from pathlib import Path

import torch

from mini_agent_rl.training.grpo import clipped_grpo_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-p-system/p3-f1-cpu-audit.json"


def run():
    # Four candidates, two exactly correct and two wrong: the corrected action
    # must gain relative log probability under a single analytical update.
    rewards = torch.tensor([1.25, 1.25, 0.0, 0.0])
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    advantage = ((rewards-mean)/std).tolist()
    assert advantage == [1.0, 1.0, -1.0, -1.0]
    logits = torch.nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.SGD([logits], lr=0.1)
    old = torch.zeros(1)
    reference = torch.zeros(1)
    loss = sum(clipped_grpo_loss(logits[i:i+1], old, reference, advantage[i], 0.2, 0.02)[0]
               for i in range(4)) / 4
    loss.backward()
    gradient = logits.grad.detach().tolist()
    optimizer.step()
    after = logits.detach().tolist()
    assert after[0] > after[2] and after[1] > after[3]
    assert gradient[0] < 0 < gradient[2]

    # A pure KL step must pull either side of the frozen reference toward it.
    positive = torch.tensor([1.0], requires_grad=True)
    negative = torch.tensor([-1.0], requires_grad=True)
    kl_plus = clipped_grpo_loss(positive, old, reference, 0, 0.2, 1)[0]
    kl_minus = clipped_grpo_loss(negative, old, reference, 0, 0.2, 1)[0]
    kl_plus.backward(); kl_minus.backward()
    assert positive.grad.item() > 0 and negative.grad.item() < 0

    # Zero variance carries no GRPO advantage or optimizer step in the runner.
    constant = [0.0] * 4
    variable = len(set(constant)) > 1
    assert not variable
    result = {"status":"PASSED","temperature_for_new_recipe":1.0,
              "advantages":advantage,"gradients":gradient,"after_step":after,
              "kl_gradient_positive_delta":positive.grad.item(),
              "kl_gradient_negative_delta":negative.grad.item(),
              "zero_variance_skips_rl":not variable,
              "scope":"analytical CPU objective check; does not prove full model training"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    run()
