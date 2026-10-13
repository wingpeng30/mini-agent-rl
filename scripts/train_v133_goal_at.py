"""Matched finite-pool AT SFT/GRPO. No collection or parameter search."""
from __future__ import annotations
import argparse,asyncio,gc,json,math,time
from pathlib import Path
import run_v133_goal_at as at
ap=at.ap;OUT=at.OUT;ROOT=at.ROOT
rows,read,write,sha=at.rows,at.read,at.write,at.sha

def groups():
    states=rows(OUT/'frozen-states.jsonl');samples=rows(OUT/'samples.jsonl')
    rewards={(r['state_id'],r['candidate']):r['reward'] for r in rows(OUT/'reward-adjudications.jsonl')}
    result=[]
    for state in states:
        candidates=sorted([dict(r,reward=rewards[(r['state_id'],r['candidate'])]) for r in samples if r['state_id']==state['state_id']],key=lambda r:r['candidate'])
        if len(candidates)!=4:raise RuntimeError('AT group size')
        vals=[r['reward'] for r in candidates];mean=sum(vals)/4;sd=(sum((v-mean)**2 for v in vals)/4)**.5
        result.append(dict(state_id=state['state_id'],mixed=sd>0,candidates=candidates,
                           advantages=[(v-mean)/sd if sd else 0. for v in vals]))
    return result

def temperature_logprobs(logits,targets,temperature=.7):
    import torch
    if temperature<=0:raise ValueError('positive sampling temperature required')
    values=torch.log_softmax(logits.float()/temperature,dim=-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(values).all():raise RuntimeError('AT nonfinite policy logprobs')
    return values

def score(client,sample,grad=False):
    import torch
    model=client._model;model.config.use_cache=False
    prompt=sample['prompt_token_ids'];actions=sample['action_token_ids'];mask=sample['prompt_attention_mask']
    if not prompt or not actions or len(mask)!=len(prompt):raise RuntimeError('AT missing exact action replay')
    device=model.device;ids=torch.tensor([prompt+actions],dtype=torch.long,device=device)
    attention=torch.tensor([mask+[1]*len(actions)],dtype=torch.long,device=device)
    with torch.enable_grad() if grad else torch.no_grad():
        logits=model(input_ids=ids,attention_mask=attention,use_cache=False).logits
        start=len(prompt)-1
        return temperature_logprobs(logits[0,start:start+len(actions)],ids[0,len(prompt):])

def client_for(adapter,corpus):
    return ap.q.make_client(adapter,deterministic=True,product_config=at.PC,model_path=ap.BASE,corpus=corpus)

def usage(kind):return sum(r['kind']==kind for r in rows(OUT/'gpu-usage.jsonl'))
def attempt(kind,stage,case):
    limits=read(OUT/'manifest.json')['limits']
    cap=limits[{'generation':'generations','search':'searches','update':'updates'}[kind]]
    if usage(kind)>=cap:raise RuntimeError('AT quota '+kind)
    ap.append(OUT/'gpu-usage.jsonl',dict(kind=kind,stage=stage,case_id=case,status='started',attempt=usage(kind)+1))

def prepare():
    at.frozen();gate=read(OUT/'attainability-gate.json')
    if not gate['passed']:raise RuntimeError('AT attainability gate failed')
    g=groups()
    paths=dict(training_runner=Path(__file__),tests=ROOT/'tests/test_v133_goal_at_training.py',
      gate=OUT/'attainability-gate.json',rewards=OUT/'reward-adjudications.jsonl',samples=OUT/'samples.jsonl',
      source_order=ap.AM/'execution-freeze.json',states=ap.AM/'candidate-training-states.jsonl',
      trajectories=ap.Z/'training-inputs.jsonl',probes=ROOT/'runs/v133-goal-ar-constrained-evidence-v1/frozen-probes.jsonl',
      cases=ap.AM/'evaluation-cases.jsonl',grpo=ROOT/'src/mini_agent_rl/training/grpo.py',
      helper=ROOT/'scripts/run_v133_goal_am.py',budget=ROOT/'scripts/run_v133_goal_o_budget.py')
    for x in paths.values():
        if not x.is_file():raise FileNotFoundError(x)
    states=rows(paths['states']);order=read(paths['source_order'])['order']['42'];probes=rows(paths['probes'])
    if len(states)!=34 or len(order)!=128 or len(set(order))!=128 or len(probes)!=32:raise RuntimeError('AT frozen quantity')
    generation=read(OUT/'actual-generation-config.json')
    for key,allowed in dict(top_p=(1.,),top_k=(0,),repetition_penalty=(None,1.),no_repeat_ngram_size=(None,0),
                            min_new_tokens=(None,0),min_length=(None,0),suppress_tokens=(None,),bad_words_ids=(None,)).items():
        if generation.get(key) not in allowed:raise RuntimeError('AT behavior scorer unsupported '+key)
    import torch
    logits=torch.tensor([[1.,2.,3.],[2.,1.,0.]],requires_grad=True);target=torch.tensor([2,0])
    values=temperature_logprobs(logits,target)
    expected=torch.log_softmax(logits/.7,-1)[torch.arange(2),target]
    if not torch.equal(values,expected):raise RuntimeError('AT temperature test')
    (-values.mean()).backward()
    if not torch.isfinite(logits.grad).all():raise RuntimeError('AT gradient test')
    write(OUT/'training-freeze.json',dict(parent='AM42A',parent_sha256=ap.AM_HASH,arms=dict(A='first supported sample SFT',B='four-sample group-standardized clipped GRPO'),
      optimizer=dict(learning_rate=5e-6,betas=[.9,.999],eps=1e-8,weight_decay=0.,max_grad_norm=1.),
      updates_per_arm=128,mixed_groups=sum(x['mixed'] for x in g),temperature=.7,clip=.2,kl_beta=.02,policy_epochs=1,
      trajectory_normalization='all assistant target tokens per trajectory',state_normalization='current action tokens only',
      mixing=dict(am_state=.25,sampled_group=.25),old_reference='both frozen AM parent, exact .7 full-softmax replay',
      order=order,paths={k:str(v) for k,v in paths.items()},hashes={k:sha(v) for k,v in paths.items()},
      gate=dict(support_success=23,complete_probes=22,partial_search=7,errors_no_increase=True,actions_no_increase=True,
                retain_rapana_and_duke=True,continuous_protection_max_loss=1,grpo_vs_sft_support_gain=2,grpo_vs_sft_complete_gain=2)))
    write(OUT/'PRE_TRAIN.md','''# AT actual frozen training prerequisites

Parent: AM seed42-A, SHA256 3bcd1f85382c1d7775fa674c1a6e1e68c80911a69b86de209d20f2ab6269280c. Product remains Goal Z.
Same 128 trajectories, 34 AM states, 16 fixed four-sample groups; 6 mixed groups carry additional loss in both arms. 50/64 samples supported.
A: first supported sampled action NLL. B: four-sample GRPO, population-standardized binary reward, clip .2, reference KL .02, one epoch. Temperature-adjusted old/current/reference probabilities are recomputed at .7; raw T=1 diagnostic logprobs are never behavior probabilities.
Both: 128 updates, one trajectory per update, AM state .25 on updates 2..68 even, sample group .25 on updates 72..87, constant-reward groups zero. LR 5e-6, AdamW (.9,.999), eps1e-8, weight_decay0, clip1, LoRA only, one GPU BF16/no quantization.
No prompt/history/tool target loss. Same seed42 order and material; independent parent reload. Parent probabilities frozen before updates; clipping uses them after shared SFT has moved the policy, not an assumed ratio of 1.
Evaluation: each arm 64 real-search cases, 24 complete and 8 partial probes, original prompt, first JSON, 16384/2048, 4 calls/3 searches, BM25 top3. Combined AT cap 640 generation attempts including prior64 samples,384 searches,256 updates. No new confirmation.
Stop on identity, numeric, GPU, archive or infrastructure failure; save scene, no automatic rerun/tuning. Semantic failure closes route normally. Passage requires >=23/29 support, >=22/24 complete, no error/action increase, protections; RL attribution additionally requires >=2 over matched SFT in both support and complete probes.
''')
    print(json.dumps(dict(prepared=True,mixed_groups=sum(x['mixed'] for x in g))),flush=True)

def frozen():
    at.frozen();cfg=read(OUT/'training-freeze.json')
    for k,p in cfg['paths'].items():
        if sha(p)!=cfg['hashes'][k]:raise RuntimeError('AT training source drift '+k)
    if sha(ap.AMADAPTER/'adapter_model.safetensors')!=cfg['parent_sha256']:raise RuntimeError('AT parent identity')
    return cfg

def train(arm):
    cfg=frozen();folder=OUT/f'seed42-{arm}';folder.mkdir(exist_ok=False)
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step,clipped_grpo_loss,ratio_statistics,require_initial_ratio
    import run_v133_goal_o_budget as budget
    if torch.cuda.device_count()!=1 or usage('update')+128>256:raise RuntimeError('AT GPU/update precondition')
    set_seed(42);client,wrapped,registry=client_for(ap.AMADAPTER,ap.Z/'corpus-train.json')
    named=client.prepare_for_training();params=[p for _,p in named]
    if not named or any('lora_' not in n for n,_ in named):raise RuntimeError('AT non-LoRA trainable')
    before=budget.parameter_hashes(client._model,lora=False);lora=budget.parameter_hashes(client._model,lora=True)
    data={r['id']:r for r in rows(ap.Z/'training-inputs.jsonl')};states=rows(ap.AM/'candidate-training-states.jsonl');g=groups()
    # Freeze exact behavior and reference probabilities before any optimizer update.
    parent={};stats=[]
    for group in g:
        if not group['mixed']:continue
        for sample in group['candidates']:
            key=(sample['state_id'],sample['candidate']);values=score(client,sample).detach().cpu().clone();parent[key]=values
            current=score(client,sample).detach().cpu();stats.append((current-values).exp())
    initial=ratio_statistics(stats);require_initial_ratio(initial,label='AT parent probability replay')
    write(folder/'parent-probabilities.jsonl',[dict(state_id=k[0],candidate=k[1],temperature=.7,old=v.tolist(),reference=v.tolist()) for k,v in parent.items()],jsonl=True)
    write(folder/'initial-engineering.json',dict(passed=True,ratios=initial,trainable_tensors=len(named),sampling_temperature=.7))
    opt=cfg['optimizer'];optimizer=torch.optim.AdamW(params,lr=opt['learning_rate'],betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'])
    for update,cid in enumerate(cfg['order'],1):
        optimizer.zero_grad(set_to_none=True);client.set_training_forward_mode()
        actions=data[cid]['actions'];denom=sum(len(a['action_token_ids']) for a in actions);tl=0.
        for a in actions:
            n=len(a['action_token_ids']);tl+=ap.am.target_loss(client,a,[1.]*n,n/denom)
        state=states[(update-2)//2] if 2<=update<=68 and update%2==0 else None;sl=0.
        if state:sl=ap.am.target_loss(client,state,[1.]*len(state['action_token_ids']),.25)
        group=g[update-72] if 72<=update<=87 else None;extra=0.;diagnostics=[]
        if group and group['mixed']:
            if arm=='A':
                sample=next(s for s in group['candidates'] if s['reward']==1)
                extra=ap.am.target_loss(client,sample,[1.]*len(sample['action_token_ids']),.25)
            else:
                for sample,advantage in zip(group['candidates'],group['advantages']):
                    current=score(client,sample,True);old=parent[(sample['state_id'],sample['candidate'])].to(current.device)
                    loss,kl,clipped=clipped_grpo_loss(current,old,old,advantage,.2,.02)
                    if not torch.isfinite(loss):raise RuntimeError('AT nonfinite GRPO loss')
                    scaled=.25*loss/4;scaled.backward();extra+=float(scaled.detach())
                    diagnostics.append(dict(candidate=sample['candidate'],reward=sample['reward'],advantage=advantage,kl=float(kl),clip_fraction=float(clipped)))
        grad=float(torch.nn.utils.clip_grad_norm_(params,1.))
        if not math.isfinite(grad) or grad<=0:raise RuntimeError('AT gradient failure')
        attempt('update',f'train-{arm}',cid);guarded_optimizer_step(optimizer,params,client.gpu_memory_stats(),21.)
        ap.append(folder/'updates.jsonl',dict(update=update,trajectory_id=cid,trajectory_loss=tl,state_id=state['state_id'] if state else None,
            state_loss=sl,group_id=group['state_id'] if group else None,group_loss=extra,gradient=grad,group_diagnostics=diagnostics))
        print(json.dumps(dict(arm=arm,update=update,trajectory_loss=tl,state_loss=sl,group_loss=extra,gradient=grad)),flush=True)
    if budget.parameter_hashes(client._model,lora=False)!=before:raise RuntimeError('AT base changed')
    changed=budget.parameter_hashes(client._model,lora=True);changed_count=sum(changed[k]!=v for k,v in lora.items())
    if changed_count!=192:raise RuntimeError('AT LoRA tensor change mismatch')
    adapter=folder/'adapter-final';client._model.save_pretrained(adapter)
    action=data[cfg['order'][0]]['actions'][-1]
    expected=client.score_action_tensor([],action['action_token_ids'],False,action['prompt_token_ids'],action['prompt_attention_mask']).detach().float().cpu()
    del client,wrapped,registry,optimizer,params,named;gc.collect();torch.cuda.empty_cache()
    reloaded,_,_=client_for(adapter,ap.Z/'corpus-train.json')
    actual=reloaded.score_action_tensor([],action['action_token_ids'],False,action['prompt_token_ids'],action['prompt_attention_mask']).detach().float().cpu()
    delta=float((expected-actual).abs().max())
    if not torch.isfinite(actual).all() or delta>.001:raise RuntimeError('AT independent reload failure')
    write(folder/'summary.json',dict(arm=arm,updates=128,base_unchanged=True,changed_lora_tensors=changed_count,reload_max_abs=delta,adapter_sha256=sha(adapter/'adapter_model.safetensors')))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train']);p.add_argument('--arm',choices=['A','B']);a=p.parse_args()
    try:
        if a.stage=='prepare':prepare()
        else:
            if not a.arm:raise ValueError('arm required')
            train(a.arm)
    except BaseException as exc:
        if not (OUT/'experiment-failure.json').exists():write(OUT/'experiment-failure.json',dict(stage=a.stage,arm=a.arm,error_type=type(exc).__name__,error=str(exc),stop_no_auto_rerun=True))
        raise
if __name__=='__main__':main()
