"""Conditional engineering/training gates for the single frozen experiment."""
import argparse
import gc
import json
import math
from pathlib import Path
import random
import time
from run_hybrid_adaptation import read,rows,write,append,create_client,evaluate,Budget
from mini_agent_rl.retrieval.local import LocalRetrieval,sha

def score(rows_,mode):
    selected=[r for r in rows_ if r['retriever']==mode]
    return dict(natural=sum(r['normalized_exact_match'] for r in selected if r['stratum']!='constructed_singlehop'),
        groups={g:sum(r['normalized_exact_match'] for r in selected if r['stratum']==g) for g in ('development','natural_supplement','constructed_singlehop')},
        invalid=sum(r['invalid_actions'] for r in selected),repeated=sum(r['repeated_actions'] for r in selected))

def qualifies(candidate,baseline,gain=3,control=None):
    return (candidate['natural']>=baseline['natural']+gain
        and all(candidate['groups'][g]>=baseline['groups'][g] for g in baseline['groups'])
        and candidate['invalid']<=baseline['invalid'] and candidate['repeated']<=baseline['repeated']
        and (control is None or candidate['natural']>=control['natural']+2))

def persistent_errors(review,current):
    wrong={r['case_id']:r for r in current if r['retriever']=='hybrid-rrf-v1' and not r['normalized_exact_match']}
    eligible=[]
    for reviewed in review:
        cid=reviewed['case_id']
        if not reviewed['complete_evidence_relation_or_repeat_candidate'] or cid not in wrong:continue
        visible={d['id']:d for t in wrong[cid]['trace'] for d in t.get('observation',{}).get('result',[])}
        if all(e['document_id'] in visible and e['quote'] in visible[e['document_id']]['text'] for e in reviewed['necessary_excerpts']):
            if wrong[cid]['answer']==reviewed['answer']:eligible.append(cid)
    return eligible

def clear():
    import torch
    gc.collect();torch.cuda.empty_cache()

def train(folder,arm,budget):
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    from run_v133_goal_o_budget import parameter_hashes
    pairs=rows(folder/'approved-training-pairs.jsonl')
    freeze=read(folder/'training-pairs-freeze.json')
    if sha(folder/'approved-training-pairs.jsonl')!=freeze['pairs_sha256']:raise RuntimeError('Training pairs drift')
    if len(pairs)!=64 or sum(p['kind']=='answer' for p in pairs)!=32 or sum(p['kind']=='search' for p in pairs)!=32:
        raise RuntimeError('Training pair quota failed')
    if len({p['case_id'] for p in pairs})<32:raise RuntimeError('Training question quota failed')
    evaluation=rows(folder/'cases.jsonl');train_rows=rows(folder/'training-inputs.jsonl')
    if len(train_rows)!=128:raise RuntimeError('Original training trajectory count changed')
    if {p['case_id'] for p in pairs}&{c['id'] for c in evaluation}:raise RuntimeError('Evaluation leakage')
    for p in pairs:
        if not p['semantic_approved']:raise RuntimeError('Unreviewed training state')
        if p['A']['target']!=p['B']['target']:raise RuntimeError('A/B supervised action mismatch')
        for variant in ('A','B'):
            action=p[variant]
            if not action['action_token_ids'] or not action['prompt_token_ids'] or len(action['prompt_token_ids'])!=len(action['prompt_attention_mask']):raise RuntimeError('Invalid target mask')
            if not all(x==1 for x in action['prompt_attention_mask']):raise RuntimeError('Unexpected padded input')
    output=folder/('train-'+arm);output.mkdir(exist_ok=False)
    budget.check();set_seed(42)
    client,pc=create_client(folder)
    named=client.prepare_for_training();params=[p for _,p in named]
    for pair in pairs:
        state=pair[arm]
        from mini_agent_rl.domain import Message
        inputs=client._inputs([Message.model_validate(m) for m in state['messages']],'training-state-preflight')
        if inputs['input_ids'][0].tolist()!=state['prompt_token_ids']:raise RuntimeError('Training input replay mismatch')
        tokenizer=client._processor.tokenizer
        target=tokenizer.encode(json.dumps(state['target'],ensure_ascii=False),add_special_tokens=False)+[tokenizer.convert_tokens_to_ids('<|im_end|>')]
        if target!=state['action_token_ids']:raise RuntimeError('Supervised action token mask mismatch')
    if any('lora_' not in name for name,_ in named):raise RuntimeError('Base weights trainable')
    base_before=parameter_hashes(client._model,lora=False);lora_before=parameter_hashes(client._model,lora=True)
    optimizer=torch.optim.AdamW(params,lr=5e-6,betas=(.9,.999),eps=1e-8,weight_decay=0)
    all_actions=[a for t in train_rows for a in t['actions']]+[p[arm] for p in pairs]
    longest=max(all_actions,key=lambda a:len(a['prompt_token_ids'])+len(a['action_token_ids']))
    client.set_training_forward_mode()
    values=client.score_action_tensor([],longest['action_token_ids'],True,longest['prompt_token_ids'],longest['prompt_attention_mask']).float()
    smoke=-values.mean();smoke.backward()
    grad=float(torch.nn.utils.clip_grad_norm_(params,1.))
    if not torch.isfinite(smoke) or not math.isfinite(grad) or grad<=0:raise RuntimeError('Longest backward smoke failed')
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise RuntimeError('Nonfinite smoke gradients')
    write(output/'smoke.json',dict(passed=True,prompt_tokens=len(longest['prompt_token_ids']),action_tokens=len(longest['action_token_ids']),
        loss=float(smoke.detach()),gradient=grad,memory=client.gpu_memory_stats(),optimizer_updates=0,target_tokens_only=True))
    optimizer.zero_grad(set_to_none=True);del smoke,values
    order=list(train_rows);random.Random(42).shuffle(order)
    auxiliary=list(pairs);random.Random(42).shuffle(auxiliary);auxiliary=auxiliary*2
    started=time.monotonic()
    for update,(trajectory,pair) in enumerate(zip(order,auxiliary),1):
        budget.check();optimizer.zero_grad(set_to_none=True);client.set_training_forward_mode()
        denom=sum(len(a['action_token_ids']) for a in trajectory['actions']);total=0.
        for action in trajectory['actions']:
            values=client.score_action_tensor([],action['action_token_ids'],True,action['prompt_token_ids'],action['prompt_attention_mask']).float()
            loss=-.75*values.sum()/denom
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite trajectory loss')
            total+=float(loss.detach());loss.backward();del values,loss
        action=pair[arm]
        values=client.score_action_tensor([],action['action_token_ids'],True,action['prompt_token_ids'],action['prompt_attention_mask']).float()
        loss=-.25*values.mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite auxiliary loss')
        extra=float(loss.detach());loss.backward();del values,loss
        grad=float(torch.nn.utils.clip_grad_norm_(params,1.))
        if not math.isfinite(grad) or grad<=0:raise RuntimeError('Invalid gradient')
        guarded_optimizer_step(optimizer,params,client.gpu_memory_stats(),21.)
        if any(not torch.isfinite(p).all() for p in params):raise RuntimeError('Nonfinite updated parameter')
        append(output/'updates.jsonl',dict(update=update,trajectory_id=trajectory['id'],pair_id=pair['pair_id'],trajectory_loss=total,
            auxiliary_loss=extra,gradient=grad,input_tokens=sum(len(a['prompt_token_ids']) for a in trajectory['actions'])+len(action['prompt_token_ids']),
            target_tokens=denom+len(action['action_token_ids']),elapsed_seconds=time.monotonic()-started,memory=client.gpu_memory_stats()))
        print(json.dumps(dict(stage='train-'+arm,update=update,loss=total+extra)),flush=True)
    if parameter_hashes(client._model,lora=False)!=base_before:raise RuntimeError('Base weight drift')
    lora_after=parameter_hashes(client._model,lora=True)
    changed=sum(lora_before[n]!=v for n,v in lora_after.items())
    if not changed:raise RuntimeError('LoRA did not update')
    adapter=output/'adapter-final';client._model.save_pretrained(adapter)
    probe=auxiliary[-1][arm]
    expected=client.score_action_tensor([],probe['action_token_ids'],False,probe['prompt_token_ids'],probe['prompt_attention_mask']).detach().float().cpu()
    del client,optimizer,params,named;clear()
    reloaded,pc=create_client(folder,adapter)
    actual=reloaded.score_action_tensor([],probe['action_token_ids'],False,probe['prompt_token_ids'],probe['prompt_attention_mask']).detach().float().cpu()
    difference=float((expected-actual).abs().max())
    if difference>.001 or not torch.isfinite(actual).all():raise RuntimeError('Adapter reload mismatch')
    write(output/'verification.json',dict(base_unchanged=True,lora_changed_parameters=changed,updates=128,reload_max_difference=difference,
        adapter_sha256=sha(adapter/'adapter_model.safetensors'),training_seconds=time.monotonic()-started,
        original_loss_weight=.75,auxiliary_loss_weight=.25,rank=8,seed=42))
    del reloaded;clear();budget.check()
    return adapter

def run(folder):
    budget=Budget(folder)
    original=read(folder/'manifest.json')
    for name,expected in original['frozen_files'].items():
        if sha(folder/name)!=expected:raise RuntimeError('Original experiment asset drift: '+name)
    if sha(folder/'base-model/model.safetensors-00001-of-00001.safetensors')!=original['base_sha256']:raise RuntimeError('Base asset drift')
    if sha(folder/'goal-z/adapter_model.safetensors')!=original['adapter_sha256']:raise RuntimeError('Goal Z asset drift')
    frozen=read(folder/'training-stage-manifest.json')
    for name,expected in frozen['files'].items():
        if sha(folder.parent/name)!=expected:raise RuntimeError('Conditional-stage asset drift: '+name)
    if sha(folder/'training-inputs.jsonl')!=frozen['training_input_sha256']:raise RuntimeError('Original training input drift')
    if (folder/'failure.json').exists():raise RuntimeError('Failure latch present; stop without retry')
    if not read(folder/'remote-stage.json')['baseline_complete']:raise RuntimeError('Baseline incomplete')
    baseline=rows(folder/'goal-z-ledger.jsonl')
    if len(baseline)!=256:raise RuntimeError('Baseline pairing incomplete')
    bscore=score(baseline,'hybrid-rrf-v1')
    engineering=None
    if (folder/'score-free-ledger.jsonl').exists():
        engineering=score(rows(folder/'score-free-ledger.jsonl'),'hybrid-rrf-v1')
        passed=qualifies(engineering,bscore)
        write(folder/'engineering-gate.json',dict(passed=passed,candidate=engineering,baseline=bscore))
        if passed:
            write(folder/'final-gate.json',dict(passed=True,route='score-free-v1',training=False));return
    review=rows(folder/'error-review.jsonl')
    current=rows(folder/'score-free-ledger.jsonl') if engineering is not None else baseline
    eligible=persistent_errors(review,current);complete=len(eligible)
    write(folder/'training-trigger.json',dict(audited_complete_evidence_persistent_errors=complete,case_ids=eligible,passed=complete>=2))
    if complete<2:raise RuntimeError('Audited complete-evidence training trigger absent')
    if not (folder/'approved-training-pairs.jsonl').exists():
        write(folder/'final-gate.json',dict(passed=False,route='training-data-gate',reason='No 64 semantically approved paired states',training=False));return
    engine=LocalRetrieval(read(folder/'retrieval-remote.json'))
    for arm in ('A','B'):
        adapter=train(folder,arm,budget)
        client,pc=create_client(folder,adapter)
        evaluate(folder,client,pc,engine,budget,'candidate-'+arm,['hybrid-rrf-v1'])
        del client;clear()
    a=score(rows(folder/'candidate-A-ledger.jsonl'),'hybrid-rrf-v1');b=score(rows(folder/'candidate-B-ledger.jsonl'),'hybrid-rrf-v1')
    passed=qualifies(b,bscore,control=a)
    write(folder/'training-gate.json',dict(passed=passed,A=a,B=b,baseline=bscore))
    if passed:
        client,pc=create_client(folder,folder/'train-B/adapter-final')
        evaluate(folder,client,pc,engine,budget,'candidate-B-bm25',['bm25-v2'])
        regression=score(rows(folder/'candidate-B-bm25-ledger.jsonl'),'bm25-v2')
        passed=qualifies(regression,score(baseline,'bm25-v2'),gain=0)
        write(folder/'bm25-regression-gate.json',dict(passed=passed,candidate=regression,baseline=score(baseline,'bm25-v2')))
    write(folder/'final-gate.json',dict(passed=passed,route='hybrid-lora',training=True,experimental_only=not passed))
    budget.check()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    try:run(args.folder)
    except Exception as exc:
        write(args.folder/'failure.json',dict(stage='conditional-training',error=repr(exc),complete=False));raise
