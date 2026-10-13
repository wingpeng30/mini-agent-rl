"""Goal AH matched continuation. GPU commands require a separate human approval file.

No resume/retry, parameter search, inherited Goal T source or stale AF sealed-set
identity guard. Product execution itself uses the unchanged Goal Q rollout.
"""
import argparse
import asyncio
import gc
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_v133_goal_ah import ROOT, OUT, Z, EXPECTED_ADAPTER, rows, sha, write_once

SOURCE=Path('/root/autodl-tmp/mini-agent-rl/runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final')
BASE=Path('/root/autodl-tmp/qwen_08b')
EXPECTED_PRODUCT=dict(max_input_tokens=2048,max_model_calls_per_case=4,max_new_tokens=128,
    max_searches_per_case=3,observation_encoding='legacy',retriever='bm25-v2',
    system_prompt='direct-v090',thinking=False,top_k=3)

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def append(path,value):
    with Path(path).open('a',encoding='utf8',newline='\n') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush()

def approved_config():
    if (OUT/'experiment-failure.json').exists():raise RuntimeError('AH failure latch set; automatic retry forbidden')
    cfg=read(OUT/'execution-freeze.json')
    if not cfg.get('pre_gpu_gates_passed'):
        raise RuntimeError('CPU preparation gate failed; current frozen plan is not GPU executable')
    approval=read(OUT/'gpu-approval.json')  # Only create after explicit human confirmation.
    if approval.get('human_confirmed') is not True or approval.get('execution_freeze_sha256')!=sha(OUT/'execution-freeze.json'):
        raise RuntimeError('No human approval for these exact experimental inputs')
    if cfg['product']!=EXPECTED_PRODUCT:raise RuntimeError('Product conditions changed')
    for name,digest in cfg['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Frozen asset changed: '+name)
    for spec in cfg['evaluation']:
        if sha(ROOT/spec['corpus'])!=spec['corpus_sha256']:raise RuntimeError('Evaluation corpus changed')
    if sha(SOURCE/'adapter_model.safetensors')!=EXPECTED_ADAPTER:raise RuntimeError('Wrong published adapter')
    if sha(BASE/'model.safetensors-00001-of-00001.safetensors')!=cfg['base_sha256']:raise RuntimeError('Wrong base model')
    if sha(BASE/'tokenizer.json')!=cfg['tokenizer_sha256']:raise RuntimeError('Tokenizer identity changed')
    release=read(ROOT/'runs/v133-goal-aa-audit-confirmation/product-decision.json')
    if not release.get('published') or release.get('selected_seed')!=42 or release.get('selected_route')!='trajectory' or release.get('adapter')!=str(SOURCE):
        raise RuntimeError('AA release identity disagrees')
    return cfg

def client(adapter):
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    return TransformersModelClient(LocalModelConfig(model_id=str(BASE),adapter_path=str(adapter),
        device='cuda',dtype='bfloat16',load_in_4bit=False,gradient_checkpointing=True,
        max_input_tokens=2048,max_new_tokens=128,enable_thinking=False,deterministic=True,
        observation_encoding='legacy',recompute_old_logprobs=False,timeout=None))

def target_loss(client_,action,denominator,weight):
    import torch
    values=client_.score_action_tensor([],action['action_token_ids'],True,
        action['prompt_token_ids'],action['prompt_attention_mask']).float()
    loss=-weight*values.sum()/denominator
    if not torch.isfinite(values).all() or not torch.isfinite(loss):raise RuntimeError('Nonfinite action loss')
    loss.backward()
    return float(loss.detach())

def broad_invalid(trace):
    seen=set();invalid=[]
    for i,a in enumerate(trace):
        reason=[]
        if a.get('parse_error'):reason.append('parse_error')
        if a.get('action')=='search':
            query=' '.join(str(a.get('query','')).casefold().split())
            if query in seen:reason.append('repeated_search')
            if i>=3:reason.append('search_over_budget')
            seen.add(query)
        if reason:invalid.append(dict(call_index=i,reasons=reason))
    return invalid

def train(seed,arm):
    cfg=approved_config()
    if seed not in [42,43] or arm not in ['A','B']:raise ValueError('Unfrozen run')
    folder=OUT/f'seed{seed}-{arm}'
    if folder.exists():raise FileExistsError('Refusing resume or overwrite')
    folder.mkdir()
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as o
    if torch.cuda.device_count()!=1:raise RuntimeError('Expected exactly one visible GPU')
    set_seed(seed);c=client(SOURCE)
    named=c.prepare_for_training();params=[p for _,p in named]
    if any('lora_' not in n for n,_ in named):raise RuntimeError('Non-LoRA trainable parameter')
    base_before=o.parameter_hashes(c._model,lora=False)
    lora_before=o.parameter_hashes(c._model,lora=True)
    data={r['id']:r for r in rows(Z/'training-inputs.jsonl')}
    judgments={r['judgment_id']:r for r in rows(OUT/'judgment-training-inputs.jsonl')}
    binding={r['update']:r['judgment_id'] for r in cfg['judgment_binding'][str(seed)]}
    opt=cfg['optimization'];optimizer=torch.optim.AdamW(params,lr=opt['learning_rate'],
        betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'])
    started=time.monotonic()
    for update,cid in enumerate(cfg['order'][str(seed)],1):
        actions=data[cid]['actions'];n=sum(len(a['action_token_ids']) for a in actions)
        optimizer.zero_grad(set_to_none=True);c.set_training_forward_mode()
        lt=sum(target_loss(c,a,n,1.) for a in actions)
        jid=binding.get(update) if arm=='B' else None;lj=0.
        if jid:
            a=judgments[jid];lj=target_loss(c,a,len(a['action_token_ids']),.25)
        grad=float(torch.nn.utils.clip_grad_norm_(params,1.))
        if not math.isfinite(grad) or grad<=0:raise RuntimeError('Invalid gradient')
        guarded_optimizer_step(optimizer,params,c.gpu_memory_stats(),21.)
        if any(not torch.isfinite(p).all() for p in params):raise RuntimeError('Nonfinite LoRA after update')
        append(folder/'updates.jsonl',dict(update=update,case_id=cid,Lt=lt,weighted_Lj=lj,
            gradient=grad,judgment_id=jid,original_tokens=n,elapsed_seconds=time.monotonic()-started))
        if update%8==0:print(json.dumps(dict(seed=seed,arm=arm,updates=update)),flush=True)
    if o.parameter_hashes(c._model,lora=False)!=base_before:raise RuntimeError('Base parameters changed')
    after=o.parameter_hashes(c._model,lora=True)
    changed=sum(after[n]!=v for n,v in lora_before.items())
    if changed!=192:raise RuntimeError('Expected all 192 original LoRA tensors to change')
    adapter=folder/'adapter-final';c._model.save_pretrained(adapter)
    sentinels=[data[cfg['order'][str(seed)][i]]['actions'][-1] for i in [0,63,127]]
    def scores(model):
        return [model.score_action_tensor([],a['action_token_ids'],False,a['prompt_token_ids'],a['prompt_attention_mask']).detach().float().cpu() for a in sentinels]
    expected=scores(c);del c,optimizer,params;gc.collect();torch.cuda.empty_cache()
    loaded=client(adapter);actual=scores(loaded)
    delta=max(float((a-b).abs().max()) for a,b in zip(expected,actual))
    if not all(torch.isfinite(a).all() for a in actual) or delta>.001:raise RuntimeError('Independent reload check failed')
    write_once(folder/'summary.json',dict(seed=seed,arm=arm,updates=128,base_unchanged=True,
        changed_lora_tensors=changed,reload_sentinels=3,reload_max_abs=delta,adapter_sha256=sha(adapter/'adapter_model.safetensors')))

def evaluate(label):
    cfg=approved_config()
    if label not in ['baseline','42-A','43-A','42-B','43-B']:raise ValueError('Unfrozen evaluation model')
    adapter=SOURCE if label=='baseline' else OUT/f'seed{label}/adapter-final'
    if label!='baseline':
        summary=read(OUT/f'seed{label}/summary.json')
        if summary['reload_max_abs']>.001 or sha(adapter/'adapter_model.safetensors')!=summary['adapter_sha256']:raise RuntimeError('Candidate engineering checks missing')
    import torch
    import run_v133_goal_q_search as q
    from mini_agent_rl.tools import SearchTool,ToolRegistry
    # Q rollout's behavior is unchanged; its obsolete J/RL identity check is
    # replaced by AH's explicit source, code, data and product identity checks.
    q.load_config=lambda:{'product':cfg['product']}
    for spec in cfg['evaluation']:
        path=OUT/f"eval-{spec['split']}-{label}.jsonl"
        if path.exists():raise FileExistsError('Refusing evaluation retry')
        c=client(adapter);wrapped=q.SearchTextAliasClient(c);registry=ToolRegistry()
        registry.register(SearchTool(read(ROOT/spec['corpus']),'bm25-v2'))
        for index,case in enumerate(rows(ROOT/spec['cases']),1):
            r=asyncio.run(q.rollout(case['question'],wrapped,registry,sample=False,seed=42))
            r.update(case_id=case['case_id'],model=label,split=spec['split'],official_answer=case['answer'],**q.score(r['answer'],case['answer']))
            r['legacy_invalid_actions']=sum(bool(a.get('parse_error')) for a in r['trace'])
            r['broad_invalid_actions']=broad_invalid(r['trace'])
            append(path,r)
            if index%8==0:print(json.dumps(dict(model=label,split=spec['split'],completed=index)),flush=True)
        del c,wrapped,registry;gc.collect();torch.cuda.empty_cache()

def mechanism(label):
    cfg=approved_config()
    if label not in ['baseline','42-A','43-A','42-B','43-B']:raise ValueError('Unfrozen model')
    adapter=SOURCE if label=='baseline' else OUT/f'seed{label}/adapter-final'
    if label!='baseline':
        summary=read(OUT/f'seed{label}/summary.json')
        if sha(adapter/'adapter_model.safetensors')!=summary['adapter_sha256']:raise RuntimeError('Candidate identity changed')
    path=OUT/f'mechanism-{label}.jsonl'
    if path.exists():raise FileExistsError('No mechanism retry')
    import torch
    from mini_agent_rl.tools import SearchTool
    c=client(adapter)
    corpus=next(s for s in cfg['evaluation'] if s['split']=='goal_z_dev')['corpus']
    tool=SearchTool(read(ROOT/corpus),'bm25-v2')
    for probe in rows(OUT/'mechanism-probes.jsonl'):
        # Diagnostic-only replay preserves exact observed tokens. No extra tool
        # evidence enters the input and no change reaches product rollout.
        def replay_inputs(messages,phase='unknown'):
            return {'input_ids':torch.tensor([probe['prompt_token_ids']],device=c._model.device),
                    'attention_mask':torch.tensor([probe['prompt_attention_mask']],device=c._model.device)}
        c._inputs=replay_inputs
        started=time.monotonic();r=c._generate_sync([],0.,42)
        record=dict(case_id=probe['case_id'],category=probe['category'],raw=r.raw_text,
            prompt_token_ids=r.prompt_token_ids,action_token_ids=r.action_token_ids,
            parse_error=r.parse_error,answer=r.content,elapsed_seconds=time.monotonic()-started)
        if r.prompt_token_ids!=probe['prompt_token_ids']:raise RuntimeError('Mechanism input changed')
        if r.tool_call:
            record['query']=r.tool_call.arguments.get('query');record['cpu_retrieved']=tool._search(record['query'])
        append(path,record)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='command',required=True)
    t=s.add_parser('train');t.add_argument('--seed',type=int,choices=[42,43],required=True);t.add_argument('--arm',choices=['A','B'],required=True)
    e=s.add_parser('evaluate');e.add_argument('--model',choices=['baseline','42-A','43-A','42-B','43-B'],required=True)
    m=s.add_parser('mechanism');m.add_argument('--model',choices=['baseline','42-A','43-A','42-B','43-B'],required=True)
    args=p.parse_args()
    try:
        if args.command=='train':train(args.seed,args.arm)
        elif args.command=='evaluate':evaluate(args.model)
        else:mechanism(args.model)
    except BaseException as exc:
        # Approval/file-exists errors also latch: operators must investigate, never retry automatically.
        if not (OUT/'experiment-failure.json').exists():
            write_once(OUT/'experiment-failure.json',dict(error=str(exc),command=vars(args),automatic_retry=False))
        raise

if __name__=='__main__':main()
