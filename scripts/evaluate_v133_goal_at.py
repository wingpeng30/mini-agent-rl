"""Fixed AT real-search evaluation and single-state probes."""
from __future__ import annotations
import argparse,asyncio,json,time
import train_v133_goal_at as training
at=training.at;ap=training.ap;OUT=training.OUT;ROOT=training.ROOT
rows,read,write,sha=training.rows,training.read,training.write,training.sha

def prepare():
    training.frozen()
    paths=dict(runner=__file__,training_freeze=OUT/'training-freeze.json',cases=ap.AM/'evaluation-cases.jsonl',
      probes=ROOT/'runs/v133-goal-ar-constrained-evidence-v1/frozen-probes.jsonl',
      probe_inputs=ROOT/'runs/v133-goal-ar-constrained-evidence-v1/probe-inputs.jsonl',
      parent_probes=ROOT/'runs/v133-goal-ar-constrained-evidence-v1/probe-A.jsonl',
      parent_review=ROOT/'runs/v133-goal-ar-constrained-evidence-v1/probe-adjudications.jsonl',
      parent_eval=ap.AM/'eval-42-A.jsonl',parent_semantic=ap.AM/'semantic-review-seed42.jsonl')
    cases=rows(paths['cases']);probes=rows(paths['probes'])
    if len(cases)!=64 or len(probes)!=32:raise RuntimeError('AT evaluation quantities')
    inputs={r['probe_id']:r for r in rows(paths['probe_inputs']) if r['arm']=='A'}
    if set(inputs)!={r['probe_id'] for r in probes}:raise RuntimeError('AT probe mapping')
    for p in probes:
        if inputs[p['probe_id']]['prompt_token_ids']!=p['prompt_token_ids']:raise RuntimeError('AT archived probe token drift')
    write(OUT/'evaluation-freeze.json',dict(paths={k:str(v) for k,v in paths.items()},hashes={k:sha(v) for k,v in paths.items()},cases=64,complete=24,partial=8,product=at.PC))

def frozen(arm):
    training.frozen();cfg=read(OUT/'evaluation-freeze.json')
    for k,p in cfg['paths'].items():
        if sha(p)!=cfg['hashes'][k]:raise RuntimeError('AT evaluation drift '+k)
    adapter=OUT/f'seed42-{arm}/adapter-final';summary=read(adapter.parent/'summary.json')
    if sha(adapter/'adapter_model.safetensors')!=summary['adapter_sha256']:raise RuntimeError('AT candidate identity')
    return cfg,adapter

def evaluate(arm):
    cfg,adapter=frozen(arm);path=OUT/f'eval-42-{arm}.jsonl'
    if path.exists():raise FileExistsError(path)
    client,wrapped,registry=training.client_for(adapter,ap.Z/'corpus-dev.json');active={'case':'unknown'}
    original_generate=wrapped.generate;original_execute=registry.execute
    async def generate(*args,**kwargs):
        training.attempt('generation',f'eval-{arm}',active['case']);return await original_generate(*args,**kwargs)
    async def execute(*args,**kwargs):
        if args and args[0]=='search':training.attempt('search',f'eval-{arm}',active['case'])
        return await original_execute(*args,**kwargs)
    wrapped.generate=generate;registry.execute=execute
    for n,case in enumerate(rows(cfg['paths']['cases']),1):
        active['case']=case['case_id'];result=asyncio.run(ap.q.rollout(case['question'],wrapped,registry,sample=False,seed=42,product_config=at.PC))
        result.update(case_id=case['case_id'],model=f'42-{arm}',official_answer=case['answer'],**ap.q.score(result['answer'],case['answer']))
        if result['reason']=='tool_error':raise RuntimeError('AT search infrastructure failure')
        if result['model_calls']>4 or result['searches']>3:raise RuntimeError('AT rollout quota')
        for t in result['trace']:
            if len(t['prompt_token_ids'])>16384 or len(t['action_token_ids'])>2048:raise RuntimeError('AT actual token limit')
        ap.append(path,result);print(json.dumps(dict(arm=arm,case=n,case_id=case['case_id'],answer=result['answer'],em=result['em'],reason=result['reason'])),flush=True)
    write(OUT/f'eval-42-{arm}-complete.json',dict(cases=64,generation_attempts=training.usage('generation'),search_attempts=training.usage('search')))

def probe(arm):
    cfg,adapter=frozen(arm);path=OUT/f'probe-42-{arm}.jsonl'
    if path.exists():raise FileExistsError(path)
    client,wrapped,_=training.client_for(adapter,ap.Z/'corpus-dev.json');client._load()
    from mini_agent_rl.domain import Message
    inputs={r['probe_id']:r for r in rows(cfg['paths']['probe_inputs']) if r['arm']=='A'}
    for p in rows(cfg['paths']['probes']):
        row=inputs[p['probe_id']];training.attempt('generation',f'probe-{arm}',p['case_id']);start=time.monotonic()
        response=asyncio.run(wrapped.generate([Message(**m) for m in row['messages']],[],temperature=0.,seed=42))
        if response.prompt_token_ids!=row['prompt_token_ids']:raise RuntimeError('AT probe actual input mismatch')
        ap.append(path,dict(probe_id=p['probe_id'],case_id=p['case_id'],category=p['category'],model=f'42-{arm}',
            action=ap.an.parsed_action(response),answer=response.content,query=response.tool_call.arguments.get('query') if response.tool_call else None,
            raw=response.raw_text,parse_error=response.parse_error,prompt_token_ids=response.prompt_token_ids,
            action_token_ids=response.action_token_ids,usage=response.usage,elapsed_seconds=time.monotonic()-start,no_search_execution=True,training_use_forbidden=True))
        print(json.dumps(dict(arm=arm,probe=p['probe_id'],action=ap.an.parsed_action(response),answer=response.content)),flush=True)
    write(OUT/f'probe-42-{arm}-complete.json',dict(states=32,searches=0))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate','probe']);p.add_argument('--arm',choices=['A','B']);a=p.parse_args()
    try:
        if a.stage=='prepare':prepare()
        elif a.stage=='evaluate':evaluate(a.arm)
        else:probe(a.arm)
    except BaseException as exc:
        if not (OUT/'experiment-failure.json').exists():write(OUT/'experiment-failure.json',dict(stage=a.stage,arm=a.arm,error_type=type(exc).__name__,error=str(exc),stop_no_auto_rerun=True))
        raise
if __name__=='__main__':main()
