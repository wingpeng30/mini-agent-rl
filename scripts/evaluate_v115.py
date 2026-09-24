"""Paired fixed-state and task evaluation for v115; only the frozen internal set is accepted."""
from __future__ import annotations
import argparse, asyncio, hashlib, json, re, os
from pathlib import Path
from mini_agent_rl.domain import Message, Task
from mini_agent_rl.evaluation import answer_metrics
from mini_agent_rl.tools.search import _BM25Index

ROOT=Path(__file__).resolve().parents[1]
FROZEN_ADAPTERS={'v101':'3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768',
                 'v112':'31b625a9e3934012f9ba9b2f2054006ab968caeabb5238c5934988edccf423fd'}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def tree(path):
    p=Path(path)
    return {f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*')) if f.is_file() and '.cache' not in f.relative_to(p).parts}
def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def norm(s): return ' '.join(re.sub(r'[^\w]+',' ',str(s).casefold()).split())
def normalize_ids(value):
    if not isinstance(value,list) or not value: raise ValueError('invalid_prompt_token_shape')
    if len(value)==1 and isinstance(value[0],list): value=value[0]
    if not value or any(type(x) is not int or x<0 for x in value): raise ValueError('invalid_prompt_token_ids')
    return value

def score_state(row,response,index):
    actual=None; legal=not response.parse_error and not response.usage.get('output_truncated',False)
    if response.tool_calls:
        c=response.tool_calls[0];q=c.arguments.get('query')
        legal=legal and len(response.tool_calls)==1 and c.name=='search' and isinstance(q,str) and bool(q.strip())
        if legal: actual={'action':'search','value':q}
    elif response.content.strip() and legal: actual={'action':'answer','value':response.content.strip()}
    else: legal=False
    target=row['target']; action_match=bool(actual and actual['action']==target['action'])
    target_value=target.get('answer') if target['action']=='answer' else target.get('query','')
    actual_value=actual['value'] if actual else ''
    am=answer_metrics(actual_value,target_value)
    previous=[]
    for m in row['messages']:
        if m['role']=='assistant':
            try:
                a=json.loads(m['content'])
                if a.get('action')=='search': previous.append(norm(a.get('query','')))
            except Exception: pass
    repeated=bool(actual and actual['action']=='search' and norm(actual['value']) in previous)
    hits=index.search(actual['value'],top_k=3) if actual and actual['action']=='search' else []
    old={d.get('id') for d in row.get('evidence',[]) if isinstance(d,dict)}
    new=[d for d in hits if d.get('id') not in old]
    return {'state_id':row['state_id'],'task_id':row['task_id'],'target':target,'actual_action':actual,
            'raw_output':response.raw_text,'parse_error':response.parse_error,'response':response.model_dump(),
            'protocol_valid':bool(legal),'action_match':action_match,
            'normalized_target_match':bool(action_match and am['normalized_exact_match']),
            'target_token_f1':am['token_f1'] if action_match else 0.0,'repeated_search':repeated,
            'retrieval_results':hits,'new_retrieval_evidence':new,'semantic_audit':'pending'}

def state_summary(rows):
    answers=[r for r in rows if r['target']['action']=='answer']; searches=[r for r in rows if r['target']['action']=='search']
    return {'count':len(rows),'protocol_rate':sum(r['protocol_valid'] for r in rows)/max(1,len(rows)),
            'supported_answer_count':len(answers),'supported_answer_correct_rate':sum(r['normalized_target_match'] for r in answers)/max(1,len(answers)),
            'insufficient_evidence_search_count':len(searches),'insufficient_evidence_search_rate':sum(r['action_match'] for r in searches)/max(1,len(searches)),
            'nonrepeat_search_rate':sum(not r['repeated_search'] for r in searches)/max(1,len(searches)),
            'new_top3_evidence_rate':sum(bool(r['new_retrieval_evidence']) for r in searches)/max(1,len(searches))}

async def run(args):
    from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.cli import default_reward
    from mini_agent_rl.runtime import AgentRunner
    from mini_agent_rl.storage import SQLiteStore
    from mini_agent_rl.tools import ToolRegistry,SearchTool
    out=Path(args.output)
    if out.exists(): raise ValueError('output_exists')
    corpus_path=Path(args.corpus); corpus=json.loads(corpus_path.read_text(encoding='utf8'))
    env_path=corpus_path.parent/'internal-environment-manifest.json'; env=json.loads(env_path.read_text(encoding='utf8'))
    tasks_path=corpus_path.parent/'internal-tasks-private.jsonl'; states_path=corpus_path.parent/'internal-states-private.jsonl'
    if sha(corpus_path)!=env.get('corpus_sha256') or sha(tasks_path)!=env.get('tasks_sha256') or sha(states_path)!=env.get('states_sha256'):
        raise ValueError('frozen_internal_environment_hash_mismatch')
    if args.label in FROZEN_ADAPTERS and args.expected_adapter_sha256!=FROZEN_ADAPTERS[args.label]:
        raise ValueError('expected_frozen_adapter_sha_mismatch')
    if sha(Path(args.adapter_path)/'adapter_model.safetensors')!=args.expected_adapter_sha256:
        raise ValueError('adapter_weight_sha_mismatch')
    config=LocalModelConfig(model_id=args.model_path,adapter_path=args.adapter_path,device='cuda',dtype='bfloat16',
        load_in_4bit=False,max_input_tokens=2048,max_new_tokens=128,enable_thinking=False,timeout=600,
        gradient_checkpointing=False,observation_encoding='compact-v3',deterministic=True,local_files_only=True,
        trust_remote_code=False,logprob_mode='bf16')
    client=TransformersModelClient(config)
    report={'version':'v115','mode':args.mode,'model_label':args.label,'config':config.model_dump(),
            'runner_config':{'max_steps':3,'max_concurrency':1,'temperature':0,'base_seed':42,
                             'anti_repeat_prompt':False,'enable_thinking':False,'prompt_variant':'direct-v090'},
            'corpus_sha256':sha(corpus_path),'corpus_documents':len(corpus),'retriever':'bm25-v2 top3',
            'model_files':tree(args.model_path),'adapter_files':tree(args.adapter_path),
            'determinism_env':{'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                               'PYTHONHASHSEED':os.environ.get('PYTHONHASHSEED')},
            'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src/mini_agent_rl').rglob('*.py'))}
                | {f'scripts/{p.name}':sha(p) for p in (ROOT/'scripts/evaluate_v115.py',ROOT/'scripts/build_v115_internal_environment.py')},
            'private_inputs':{},'rows':[],'complete_coverage':False,'semantic_audit':'pending'}
    registry=ToolRegistry();registry.register(SearchTool(corpus,'bm25-v2'))
    if args.mode=='states':
        path=Path(args.states); states=read_jsonl(path); index=_BM25Index(corpus)
        if sha(path)!=env.get('states_sha256') or len(states)!=100 or len({r['task_id'] for r in states})!=50:
            raise ValueError('invalid_frozen_internal_state_set')
        report['private_inputs']={'state_file_sha256':sha(path),'state_count':len(states)}
        write(out,report)
        for row in states:
            response=await client.generate([Message(**m) for m in row['messages']],registry.definitions(),temperature=0,seed=42)
            report['rows'].append(score_state(row,response,index));write(out,report)
        report['metrics']=state_summary(report['rows']);expected={r['state_id'] for r in states}
        report['complete_coverage']=len(report['rows'])==100 and len({r['state_id'] for r in report['rows']})==100 and {r['state_id'] for r in report['rows']}==expected
    else:
        path=Path(args.tasks); tasks=[Task(**r) for r in read_jsonl(path)]
        if sha(path)!=env.get('tasks_sha256') or len(tasks)!=50 or len({t.id for t in tasks})!=50:
            raise ValueError('invalid_frozen_internal_task_set')
        report['private_inputs']={'task_file_sha256':sha(path),'task_count':len(tasks)}
        write(out,report); db=SQLiteStore(str(out)+'.sqlite')
        try:
            runner=AgentRunner(client,registry,default_reward(),db,max_steps=3,max_concurrency=1,temperature=0,
                base_seed=42,anti_repeat_prompt=False,enable_thinking=False,prompt_variant='direct-v090')
            for task in tasks:
                group=await runner.run_group(task,1); rollout=db.get_rollout(group.rollout_ids[0])
                report['rows'].append({'task_id':task.id,'rollout_id':rollout.id,'final_answer':rollout.final_answer,
                    'termination':rollout.termination_reason,**answer_metrics(rollout.final_answer,task.answer),
                    'transitions':[t.model_dump() for t in db.get_transitions(rollout.id)]});write(out,report)
        finally: db.close()
        report['metrics']={'task_count':len(tasks),'normalized_em':sum(r['normalized_exact_match'] for r in report['rows'])/max(1,len(tasks)),
                           'token_f1':sum(r['token_f1'] for r in report['rows'])/max(1,len(tasks)),
                           'protocol_rate':sum(r['termination'] in {'answer','repeated_action','max_steps'} for r in report['rows'])/max(1,len(tasks))}
        report['complete_coverage']=len(report['rows'])==50 and {r['task_id'] for r in report['rows']}=={t.id for t in tasks}
    write(out,report)
    if not report['complete_coverage']: raise RuntimeError('incomplete_evaluation_coverage')
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['states','tasks'],required=True)
    for n in ('model-path','adapter-path','expected-adapter-sha256','label','corpus','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--states');p.add_argument('--tasks');a=p.parse_args()
    if (a.mode=='states') != bool(a.states) or (a.mode=='tasks') != bool(a.tasks): raise ValueError('mode_input_mismatch')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8' or os.environ.get('PYTHONHASHSEED')!='42':
        raise RuntimeError('v115 evaluation requires CUBLAS_WORKSPACE_CONFIG=:4096:8 and PYTHONHASHSEED=42 before process start')
    r=asyncio.run(run(a));print(json.dumps({'mode':r['mode'],'complete_coverage':r['complete_coverage'],'metrics':r['metrics']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
