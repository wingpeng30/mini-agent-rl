"""Goal O: fixed 4-pass, 3-seed GRPO budget comparison from v118."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/v133-goal-o-budget'
CFG = OUT / 'config-frozen.json'
STATE = OUT / 'status.json'
N = None


def n_module():
    global N
    if N is None:
        import run_v133_goal_n_v118_grpo as module
        N = module
    return N


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')
        stream.flush()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_cfg():
    c = json.loads(CFG.read_text(encoding='utf-8'))
    n = n_module()
    required = {'runner': Path(__file__), 'goal_n_runner': Path(n.__file__),
                'grpo': ROOT/'src/mini_agent_rl/training/grpo.py',
                'client': ROOT/'src/mini_agent_rl/model/local.py',
                'train': n.PATHS['train'], 'new_eval': n.PATHS['new_eval'],
                'goal_G_regression': n.PATHS['goal_G_regression'],
                'v118': n.PATHS['v118'], 'goal_J_sft': n.PATHS['goal_J_sft'],
                'base_config': n.BASE/'config.json', 'tokenizer': n.BASE/'tokenizer.json'}
    for name, path in required.items():
        if sha(path) != c['hashes'][name]:
            raise RuntimeError(f'frozen asset changed: {name}')
    return c


def prepare():
    if CFG.exists() or STATE.exists():
        raise FileExistsError('Goal O output already prepared; refusing overwrite')
    n = n_module()
    if json.loads((n.OUT/'status.json').read_text(encoding='utf-8'))['status'] != 'COMPLETED_EXPERIMENT':
        raise RuntimeError('Goal N completed reference missing')
    if json.loads((n.OUT/'cpu-contract.json').read_text(encoding='utf-8'))['status'] != 'pass':
        raise RuntimeError('Goal N CPU contract missing')
    if json.loads((n.OUT/'scoring-regression.json').read_text(encoding='utf-8'))['rows'] != 96:
        raise RuntimeError('Goal N scoring regression missing')
    import torch, transformers, peft, tokenizers
    versions = {'torch': torch.__version__, 'transformers': transformers.__version__,
                'peft': peft.__version__, 'tokenizers': tokenizers.__version__}
    if versions != {'torch': '2.5.1+cu124', 'transformers': '5.17.0', 'peft': '0.20.0', 'tokenizers': '0.23.2'}:
        raise RuntimeError(f'software version changed: {versions}')
    paths = {'runner': Path(__file__), 'goal_n_runner': Path(n.__file__),
             'grpo': ROOT/'src/mini_agent_rl/training/grpo.py',
             'client': ROOT/'src/mini_agent_rl/model/local.py',
             'train': n.PATHS['train'], 'new_eval': n.PATHS['new_eval'],
             'goal_G_regression': n.PATHS['goal_G_regression'],
             'v118': n.PATHS['v118'], 'goal_J_sft': n.PATHS['goal_J_sft'],
             'base_config': n.BASE/'config.json', 'tokenizer': n.BASE/'tokenizer.json'}
    baseline = json.loads((n.OUT/'config-frozen.json').read_text(encoding='utf-8'))
    c = {**baseline, 'version': 'v133-goal-o-v1', 'hashes': {k: sha(v) for k,v in paths.items()},
         'versions': versions, 'seeds': [42,43,44], 'passes': 4, 'cases_per_pass': 16,
         'group_size': 4, 'max_groups_per_seed': 64, 'max_updates_per_seed': 64,
         'checkpoints_after_passes': [1,4], 'max_seconds_per_seed': 1800,
         'max_evaluation_seconds': 1800, 'total_gpu_cap_seconds': 7200,
         'reference_mode': 'initial_v118_adapter',
         'sampling_distribution': 'temperature_0.7_top_p_1_top_k_0',
         'policy_logprob': 'untempered_raw_policy_teacher_forced',
         'objective_note': 'GRPO ratios use raw policy logprob; temperature changes sampling distribution. This is the frozen Goal N approximation.'}
    OUT.mkdir(parents=True, exist_ok=True)
    write(CFG, c)
    write(STATE, {'status':'CPU_READY','stage':'O0','config_sha256':sha(CFG),'gpu_seconds_used':0.0})
    print(json.dumps({'status':'CPU_READY','config_sha256':sha(CFG),'seeds':c['seeds']}))


def parameter_hashes(model, *, lora):
    result = {}
    for name, value in model.named_parameters():
        if ('lora_' in name) == lora:
            result[name] = hashlib.sha256(value.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()
    return result


def reference_hashes(reference):
    return {name: hashlib.sha256(value.detach().float().cpu().numpy().tobytes()).hexdigest()
            for name, value in reference.items()}


def check_replay(client, item, old, reference_lp):
    import torch
    expected = hashlib.sha256(str(item['prompt_token_ids']).encode()).hexdigest()
    if client._context_records[-2]['prompt_sha256'] != expected or client._context_records[-1]['prompt_sha256'] != expected:
        raise RuntimeError('old/reference did not replay sampled prompt tokens')
    if len(item['action_token_ids']) != old.numel() or reference_lp.numel() != old.numel():
        raise RuntimeError('old/reference action token count mismatch')
    if item['action_mask'] != [1] * old.numel() or not torch.isfinite(old).all() or not torch.isfinite(reference_lp).all():
        raise RuntimeError('action mask or logprob invalid')
    recorded = torch.tensor(item['old_logprobs'],dtype=torch.float32)
    if (old.float().cpu()-recorded).abs().max() > 0.01:
        raise RuntimeError('old logprob changed across same-policy replay')


def save_checkpoint(client, seed_dir, pass_index, before_lora, base_before, reference, reference_before, check_item, updates):
    import torch
    current_lora = parameter_hashes(client._model,lora=True)
    if not any(before_lora[k] != current_lora[k] for k in before_lora):
        raise RuntimeError(f'no LoRA change by pass {pass_index}')
    if parameter_hashes(client._model,lora=False) != base_before:
        raise RuntimeError(f'base changed by pass {pass_index}')
    if reference_hashes(reference) != reference_before:
        raise RuntimeError(f'reference changed by pass {pass_index}')
    folder = seed_dir/f'checkpoint-pass-{pass_index}'
    if folder.exists():
        raise FileExistsError(f'checkpoint exists: {folder}')
    folder.mkdir(parents=True)
    client._model.save_pretrained(folder)
    before_lp = client.score_action_tensor([],check_item['action_token_ids'],False,
        check_item['prompt_token_ids'],check_item['prompt_attention_mask']).detach().float().cpu().tolist()
    record = {'pass':pass_index,'updates_cumulative':updates,
              'adapter_sha256':sha(folder/'adapter_model.safetensors'),
              'changed_lora_parameters':sum(before_lora[k]!=current_lora[k] for k in before_lora),
              'base_unchanged':True,'reference_unchanged':True,'check_item':{
                  'prompt_token_ids':check_item['prompt_token_ids'],
                  'prompt_attention_mask':check_item['prompt_attention_mask'],
                  'action_token_ids':check_item['action_token_ids'],
                  'logprobs':before_lp}}
    write(seed_dir/f'checkpoint-pass-{pass_index}.json',record)
    return record


def train_seed(seed):
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import clipped_grpo_loss,guarded_optimizer_step,ratio_statistics,require_initial_ratio
    c = parse_cfg(); n = n_module()
    if seed not in c['seeds']:
        raise ValueError('unfrozen seed')
    seed_dir = OUT/f'seed-{seed}'
    if seed_dir.exists():
        raise FileExistsError(f'seed run already started: {seed}')
    seed_dir.mkdir(parents=True)
    set_seed(seed)
    client = n.load_client(n.V118,c,training=True)
    named = client.prepare_for_training()
    params = [value for _,value in named]
    reference = {name:value.detach().clone() for name,value in named}
    reference_before = reference_hashes(reference)
    base_before = parameter_hashes(client._model,lora=False)
    lora_before = parameter_hashes(client._model,lora=True)
    optimizer = torch.optim.AdamW(params,lr=c['learning_rate'],betas=tuple(c['betas']),
        eps=c['optimizer_eps'],weight_decay=c['weight_decay'])
    data = rows(n.PATHS['train'])
    if len(data)!=c['cases_per_pass']:
        raise RuntimeError('frozen training length changed')
    started=time.monotonic(); total_groups=0; total_updates=0; history=[]
    for pass_index in range(1,c['passes']+1):
        pass_updates=0; pass_variable=0; pass_answer_variable=0; pass_correctness_variable=0
        for row in data:
            if time.monotonic()-started >= c['max_seconds_per_seed']:
                raise TimeoutError('per-seed cap before group')
            client._model.eval(); group=[]; started_group=time.monotonic()
            for candidate in range(c['group_size']):
                item=n.generate(client,row,c,sample=True)
                item.update(seed=seed,pass_index=pass_index,candidate=candidate)
                old=client.score_action_tensor([],item['action_token_ids'],False,
                    item['prompt_token_ids'],item['prompt_attention_mask']).detach()
                item['old_logprobs']=old.float().cpu().tolist()
                group.append(item)
                append(seed_dir/'rollouts.jsonl',item)
            summary=n.group_summary(group)
            summary.update(seed=seed,pass_index=pass_index,group_index=total_groups+1,
                action_tokens_mean=sum(item['action_tokens'] for item in group)/len(group),
                prompt_tokens=group[0]['prompt_tokens'])
            pass_variable+=int(summary['reward_variable'])
            pass_answer_variable+=int(summary['answer_reward_variable'])
            pass_correctness_variable+=int(summary['answer_correctness_variable'])
            if summary['reward_variable']:
                old_rows=[];ref_rows=[]
                for item in group:
                    old=client.score_action_tensor([],item['action_token_ids'],False,
                        item['prompt_token_ids'],item['prompt_attention_mask']).detach()
                    ref=client.reference_logprobs([],item['action_token_ids'],reference,
                        item['prompt_token_ids'],item['prompt_attention_mask']).detach()
                    check_replay(client,item,old,ref)
                    old_rows.append(old);ref_rows.append(ref)
                client.set_training_forward_mode()
                initial=[]
                for item,old in zip(group,old_rows):
                    current=client.score_action_tensor([],item['action_token_ids'],True,
                        item['prompt_token_ids'],item['prompt_attention_mask'])
                    initial.append(torch.exp(current.detach()-old).float().cpu())
                initial_stats=ratio_statistics(initial)
                require_initial_ratio(initial_stats,label='initial')
                optimizer.zero_grad(set_to_none=True)
                backwards=[];losses=[];kls=[]
                for item,old,ref in zip(group,old_rows,ref_rows):
                    current=client.score_action_tensor([],item['action_token_ids'],True,
                        item['prompt_token_ids'],item['prompt_attention_mask'])
                    expected=hashlib.sha256(str(item['prompt_token_ids']).encode()).hexdigest()
                    if client._context_records[-1]['prompt_sha256']!=expected:
                        raise RuntimeError('current did not replay sampled prompt')
                    backwards.append(torch.exp(current.detach()-old).float().cpu())
                    advantage=(item['reward']-summary['mean_reward'])/max(summary['reward_std_population'],1e-8)
                    loss,kl,_=clipped_grpo_loss(current,old,ref,advantage,c['clip_epsilon'],c['kl_beta'])
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite GRPO loss')
                    (loss/c['group_size']).backward()
                    losses.append(float(loss.detach()));kls.append(float(kl))
                backward_stats=ratio_statistics(backwards)
                require_initial_ratio(backward_stats,label='first backward')
                grad_norm=float(torch.nn.utils.clip_grad_norm_(params,c['max_grad_norm']))
                if not math.isfinite(grad_norm) or grad_norm==0:
                    raise RuntimeError('nonfinite or zero gradient')
                memory=client.gpu_memory_stats()
                guarded_optimizer_step(optimizer,params,memory,c['max_reserved_memory_gb'])
                total_updates+=1;pass_updates+=1
                summary.update(updated=True,initial_ratio=initial_stats,first_backward_ratio=backward_stats,
                    loss=sum(losses)/len(losses),kl=sum(kls)/len(kls),grad_norm=grad_norm,gpu_memory=memory)
            else:
                summary.update(updated=False,skip_reason='zero_variance')
            total_groups+=1
            if total_groups>c['max_groups_per_seed'] or total_updates>c['max_updates_per_seed']:
                raise RuntimeError('frozen group/update cap exceeded')
            summary['seconds']=round(time.monotonic()-started_group,3)
            append(seed_dir/'groups.jsonl',summary)
            history.append(summary['seconds'])
            if len(history)==2 and time.monotonic()-started+sum(history)/2*(c['max_groups_per_seed']-2)+180>c['max_seconds_per_seed']:
                raise TimeoutError('first-two group speed projection')
        pass_summary={'seed':seed,'pass':pass_index,'updates':pass_updates,'updates_cumulative':total_updates,
            'reward_variable_groups':pass_variable,'answer_reward_variable_groups':pass_answer_variable,
            'answer_correctness_variable_groups':pass_correctness_variable,
            'zero_variance_groups':c['cases_per_pass']-pass_variable,
            'elapsed_seconds':round(time.monotonic()-started,3)}
        append(seed_dir/'passes.jsonl',pass_summary)
        if pass_index in c['checkpoints_after_passes']:
            save_checkpoint(client,seed_dir,pass_index,lora_before,base_before,reference,reference_before,group[0],total_updates)
    if total_groups!=c['max_groups_per_seed']:
        raise RuntimeError('incomplete seed training')
    del client,optimizer,params,reference
    gc.collect();torch.cuda.empty_cache()
    reload_checks={}
    for pass_index in c['checkpoints_after_passes']:
        info=json.loads((seed_dir/f'checkpoint-pass-{pass_index}.json').read_text(encoding='utf-8'))
        loaded=n.load_client(seed_dir/f'checkpoint-pass-{pass_index}',c)
        check=info['check_item']
        scores=loaded.score_action_tensor([],check['action_token_ids'],False,
            check['prompt_token_ids'],check['prompt_attention_mask']).detach().float().cpu()
        delta=float((scores-torch.tensor(check['logprobs'])).abs().max())
        if delta>0.001 or sha(seed_dir/f'checkpoint-pass-{pass_index}'/'adapter_model.safetensors')!=info['adapter_sha256']:
            raise RuntimeError(f'checkpoint reload mismatch: pass{pass_index}, delta={delta}')
        reload_checks[str(pass_index)]=delta
        del loaded;gc.collect();torch.cuda.empty_cache()
    result={'status':'PASSED','seed':seed,'passes':c['passes'],'groups':total_groups,'updates':total_updates,
            'checkpoints':c['checkpoints_after_passes'],'reload_max_abs':reload_checks,
            'base_unchanged':True,'reference_unchanged':True,
            'elapsed_seconds':round(time.monotonic()-started,3)}
    write(seed_dir/'summary.json',result)
    print(json.dumps(result))


def evaluate():
    import torch
    from transformers import set_seed
    c=parse_cfg();n=n_module()
    if (OUT/'evaluation.jsonl').exists():
        raise FileExistsError('evaluation already started')
    for seed in c['seeds']:
        info=json.loads((OUT/f'seed-{seed}'/'summary.json').read_text(encoding='utf-8'))
        if info['status']!='PASSED':
            raise RuntimeError(f'seed {seed} did not pass')
    started=time.monotonic();arms=[('v118',n.V118),('goal_J_sft',n.J_ADAPTER)]
    for seed in c['seeds']:
        for pass_index in c['checkpoints_after_passes']:
            arms.append((f'seed-{seed}-pass-{pass_index}',OUT/f'seed-{seed}'/f'checkpoint-pass-{pass_index}'))
    for arm,adapter in arms:
        set_seed(c['seed'])
        client=n.load_client(adapter,c)
        for split in ('train','new_eval','goal_G_regression'):
            for row in rows(n.PATHS[split]):
                if time.monotonic()-started>=c['max_evaluation_seconds']:
                    raise TimeoutError('unified evaluation cap')
                output=n.generate(client,row,c,sample=False)
                append(OUT/'evaluation.jsonl',{**{k:output[k] for k in ('case_id','raw','action','prediction','em','f1',
                    'prompt_tokens','action_tokens','generation_seconds')},'arm':arm,'split':split})
        del client;gc.collect();torch.cuda.empty_cache()
    result=rows(OUT/'evaluation.jsonl')
    if len(result)!=len(arms)*48:
        raise RuntimeError('unified evaluation incomplete')
    summary={'status':'PASSED','elapsed_seconds':round(time.monotonic()-started,3),'sets':{}}
    for arm,_ in arms:
        summary['sets'][arm]={}
        for split in ('train','new_eval','goal_G_regression'):
            values=[item for item in result if item['arm']==arm and item['split']==split]
            summary['sets'][arm][split]={'n':len(values),'em':sum(item['em'] for item in values),
                'f1':sum(item['f1'] for item in values)/len(values),
                'actions':{kind:sum(item['action']==kind for item in values) for kind in ('answer','search','invalid')},
                'generation_seconds':round(sum(item['generation_seconds'] for item in values),3)}
    write(OUT/'evaluation-summary.json',summary)
    print(json.dumps({'status':'PASSED','rows':len(result),'seconds':summary['elapsed_seconds']}))


def run():
    c=parse_cfg()
    state=json.loads(STATE.read_text(encoding='utf-8'))
    if state['status']!='CPU_READY':
        raise RuntimeError('run already started or not prepared')
    started=time.monotonic();state.update(status='RUNNING',stage='O1',gpu_started_unix=time.time())
    write(STATE,state)
    stages=[(f'seed-{seed}', ['train-seed','--seed',str(seed)],c['max_seconds_per_seed']) for seed in c['seeds']]
    stages.append(('O2',['evaluate'],c['max_evaluation_seconds']))
    for stage,arguments,cap in stages:
        remaining=c['total_gpu_cap_seconds']-(time.monotonic()-started)
        if remaining<=0:
            state.update(status='TECHNICAL_BLOCKED',stage=stage,error='total GPU cap expired')
            break
        state.update(stage=stage,gpu_seconds_used=round(time.monotonic()-started,3));write(STATE,state)
        try:
            done=subprocess.run([sys.executable,str(Path(__file__).resolve()),*arguments],cwd=ROOT,
                timeout=min(cap,remaining),check=False)
        except subprocess.TimeoutExpired:
            state.update(status='TECHNICAL_BLOCKED',error=f'{stage} hard timeout');break
        if done.returncode:
            state.update(status='TECHNICAL_BLOCKED',error=f'{stage} exited {done.returncode}');break
        if stage.startswith('seed-'):
            summary_path=OUT/stage/'summary.json'
            if not summary_path.exists() or json.loads(summary_path.read_text(encoding='utf-8')).get('status')!='PASSED':
                state.update(status='TECHNICAL_BLOCKED',error=f'{stage} completion record missing');break
        elif not (OUT/'evaluation-summary.json').exists():
            state.update(status='TECHNICAL_BLOCKED',error='evaluation completion record missing');break
    else:
        state.update(status='EXPERIMENT_COMPLETE',stage='O3')
    state['gpu_seconds_used']=round(time.monotonic()-started,3)
    write(STATE,state);print(json.dumps(state))
    if state['status']!='EXPERIMENT_COMPLETE':
        raise RuntimeError(state['error'])


def summarize():
    c=parse_cfg();s=json.loads(STATE.read_text(encoding='utf-8'))
    summaries={str(seed):json.loads((OUT/f'seed-{seed}'/'summary.json').read_text(encoding='utf-8'))
        for seed in c['seeds'] if (OUT/f'seed-{seed}'/'summary.json').exists()}
    print(json.dumps({'state':s,'seeds':summaries,'evaluation_complete':(OUT/'evaluation-summary.json').exists()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','summarize','train-seed','evaluate'])
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.command=='train-seed':
        if args.seed is None: parser.error('--seed required for train-seed')
        train_seed(args.seed)
    else:
        if args.seed is not None: parser.error('--seed only valid for train-seed')
        globals()[args.command]()


if __name__=='__main__':
    main()
