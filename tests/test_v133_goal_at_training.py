import torch
from train_v133_goal_at import temperature_logprobs
from mini_agent_rl.training.grpo import clipped_grpo_loss

def test_sampling_temperature_is_applied_before_normalization():
    logits=torch.tensor([[1.,2.,3.],[3.,1.,-1.]],requires_grad=True);target=torch.tensor([2,0])
    got=temperature_logprobs(logits,target,.7)
    expected=torch.log_softmax(logits/.7,-1)[torch.arange(2),target]
    assert torch.equal(got,expected)
    assert not torch.equal(got,torch.log_softmax(logits,-1)[torch.arange(2),target])
    (-got.mean()).backward();assert torch.isfinite(logits.grad).all()

def test_current_action_only_mask_and_normalization():
    full=torch.tensor([[9.,0.],[8.,0.],[2.,1.],[1.,2.]],requires_grad=True)
    value=temperature_logprobs(full[2:],torch.tensor([0,1]))
    (-value.mean()).backward()
    assert full.grad[:2].abs().sum()==0
    assert full.grad[2:].abs().sum()>0

def test_group_standardization_and_reference_kl():
    rewards=torch.tensor([0.,1.,1.,0.]);advantages=(rewards-rewards.mean())/rewards.std(correction=0)
    assert advantages.tolist()==[-1.,1.,1.,-1.]
    old=torch.tensor([-1.,-2.]);cur=old.clone().requires_grad_()
    loss,kl,clipped=clipped_grpo_loss(cur,old,old,1.,.2,.02)
    assert abs(float(loss)+1)<1e-6 and float(kl)==0 and float(clipped)==0
    loss.backward();assert torch.isfinite(cur.grad).all()
