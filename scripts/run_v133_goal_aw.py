"""AW first-hop reader versus compute-matched draft review, bounded inference."""
import argparse,asyncio,json,math,time
from pathlib import Path
from types import SimpleNamespace
import run_v133_goal_ap as ap
from mini_agent_rl.domain import Message
from mini_agent_rl.model.decision_pipeline import DecisionConfig,DecisionPipeline

ROOT=ap.ROOT;OUT=ROOT/'runs/v133-goal-aw-hop-reader-v1'
PRIOR=ROOT/'runs/v133-goal-ar-constrained-evidence-v1'
PC=dict(ap.PC,action_constraint='none',observation_presentation='legacy-json')
MODES={'A':'none','C':'draft','B':'reader'}
CAPS=dict(generation=1196,search=576)
rows,read,sha=ap.rows,ap.read,ap.sha
def write(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def record(kind,stage,cid):
    used=sum(r['kind']==kind for r in rows(OUT/'gpu-usage.jsonl'))
    if used>=CAPS[kind]:raise RuntimeError('AW quota '+kind)
    ap.append(OUT/'gpu-usage.jsonl',dict(kind=kind,stage=stage,case_id=cid,attempt=used+1,time=time.time()))
def make():return ap.q.make_client(ap.AMADAPTER,deterministic=True,product_config=PC,model_path=ap.BASE,corpus=ap.Z/'corpus-dev.json')
def guard(response):
    u=response.usage
    if any(u.get(k) for k in ('context_budget_error','lossy_body_trimmed','documents_deleted','history_dropped','prompt_truncated')):raise RuntimeError('AW clipping')
    if len(response.prompt_token_ids)>PC['max_input_tokens'] or len(response.action_token_ids)>PC['max_new_tokens']:raise RuntimeError('AW length limits')
    if any(not math.isfinite(x) for x in response.old_logprobs):raise FloatingPointError('AW nonfinite')
def prepare():
    if set(p.name for p in OUT.iterdir())!={'PLAN.md'}:raise FileExistsError('AW directory not clean')
    if sha(ap.AMADAPTER/'adapter_model.safetensors')!=ap.AM_HASH or sha(ap.ZADAPTER/'adapter_model.safetensors')!=ap.Z_HASH:raise RuntimeError('AW identity')
    if not read(ap.AA/'product-decision.json')['published']:raise RuntimeError('AA identity')
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(ap.BASE),local_files_only=True)
    tok=getattr(processor,'tokenizer',processor);client,_,_=make();client._processor=processor;client._model=SimpleNamespace(device='cpu')
    inputs=[r for r in rows(PRIOR/'probe-inputs.jsonl') if r['arm']=='A']
    if len(inputs)!=32 or sum(r['category']=='complete' for r in inputs)!=24:raise RuntimeError('Probe inventory')
    for r in inputs:
        encoded=client._inputs([Message(**m) for m in r['messages']],'aw_cpu')
        if encoded['input_ids'][0].tolist()!=r['prompt_token_ids']:raise RuntimeError('A actual tokens not reconstructed')
        if encoded._context_usage['prompt_truncated']:raise RuntimeError('CPU clipping')
        ap.append(OUT/'probe-inputs.jsonl',r)
    paths=dict(runner=Path(__file__),pipeline=ROOT/'src/mini_agent_rl/model/decision_pipeline.py',q=ROOT/'scripts/run_v133_goal_q_search.py',
      local=ROOT/'src/mini_agent_rl/model/local.py',plan=OUT/'PLAN.md',inputs=OUT/'probe-inputs.jsonl',
      parent=ap.AMADAPTER/'adapter_model.safetensors',tokenizer=ap.BASE/'tokenizer.json',corpus=ap.Z/'corpus-dev.json',
      cases=ap.AM/'evaluation-cases.jsonl',rulings=PRIOR/'probe-adjudications.jsonl')
    write(OUT/'manifest.json',dict(product=PC,arms=MODES,decision_config=dict(max_auxiliary_calls=2,max_model_calls=6),caps=CAPS,
        paths={k:str(v) for k,v in paths.items()},hashes={k:sha(v) for k,v in paths.items()},A_exact_inputs=32,A_fresh_generation=True,
        parent='AM42A',parent_sha256=ap.AM_HASH,training_updates=0,new_confirmation=0))
    with (OUT/'gpu-usage.jsonl').open('x'):pass
    write(OUT/'cpu-preflight.json',dict(passed=True,exact_A=32,first_hop_labels_not_in_input=True))
    print('CPU prepare passed',flush=True)
def frozen():
    if (OUT/'experiment-failure.json').exists():raise RuntimeError('AW failure latch')
    m=read(OUT/'manifest.json')
    for k,p in m['paths'].items():
        if sha(p)!=m['hashes'][k]:raise RuntimeError('AW frozen drift '+k)
    return m
def generate_factory(wrapped,stage,cid):
    async def generate(*args,**kwargs):
        record('generation',stage,cid);start=time.monotonic();r=await wrapped.generate(*args,**kwargs);guard(r)
        return r.model_copy(update={'usage':dict(r.usage,measured_generation_seconds=time.monotonic()-start)})
    return generate
def one(wrapped,item,arm,stage):
    gen=generate_factory(wrapped,stage,item['case_id']);pipe=DecisionPipeline(DecisionConfig(mode=MODES[arm]),gen)
    r=asyncio.run(pipe([Message(**m) for m in item['messages']],[],temperature=0.,seed=42))
    return dict(arm=arm,probe_id=item['probe_id'],case_id=item['case_id'],category=item['category'],raw=r.raw_text,
      action=ap.an.parsed_action(r),answer=r.content,query=r.tool_call.arguments.get('query') if r.tool_call else None,
      parse_error=r.parse_error,prompt_token_ids=r.prompt_token_ids,action_token_ids=r.action_token_ids,usage=r.usage,
      no_search_execution=True,training_use_forbidden=True)
def engineer():
    frozen();path=OUT/'engineer.jsonl'
    if path.exists():raise FileExistsError(path)
    client,wrapped,_=make();inputs=rows(OUT/'probe-inputs.jsonl')
    for cat in ('complete','partial'):
        for item in [r for r in inputs if r['category']==cat][:3]:ap.append(path,one(wrapped,item,'B','engineer'))
    write(OUT/'engineer-complete.json',dict(passed=True,generations=12,no_search=True))
def probe(arm):
    frozen()
    if not read(OUT/'engineer-complete.json')['passed']:raise RuntimeError('Engineering missing')
    path=OUT/f'probe-{arm}.jsonl'
    if path.exists():raise FileExistsError(path)
    client,wrapped,_=make()
    for item in rows(OUT/'probe-inputs.jsonl'):
        r=one(wrapped,item,arm,'probe-'+arm);ap.append(path,r)
        print(json.dumps(dict(arm=arm,probe=r['probe_id'],action=r['action'],answer=r['answer'],query=r['query'],
          auxiliary=r['usage']['decision_pipeline']['auxiliary']),ensure_ascii=False),flush=True)
    write(OUT/f'probe-{arm}-complete.json',dict(states=32))
def evaluate(arm):
    frozen();gate=read(OUT/'probe-gate.json')
    if not gate['real_search_allowed'] or (arm!='A' and not gate['eligibility'][arm]['passed']):raise RuntimeError('AW probe gate')
    path=OUT/f'eval-{arm}.jsonl'
    if path.exists():raise FileExistsError(path)
    client,wrapped,registry=make();execute=registry.execute;active={'cid':None}
    async def search(*args,**kwargs):
        record('search','eval-'+arm,active['cid']);return await execute(*args,**kwargs)
    registry.execute=search
    for case in rows(ap.AM/'evaluation-cases.jsonl'):
        active['cid']=case['case_id'];gen=generate_factory(wrapped,'eval-'+arm,active['cid'])
        pipe=DecisionPipeline(DecisionConfig(mode=MODES[arm]),gen)
        r=asyncio.run(ap.q.rollout(case['question'],wrapped,registry,sample=False,seed=42,product_config=PC,decision_pipeline=pipe))
        if r['reason']=='tool_error':raise RuntimeError('AW search infrastructure')
        r.update(case_id=case['case_id'],arm=arm,official_answer=case['answer'],**ap.q.score(r['answer'],case['answer']))
        ap.append(path,r);print(json.dumps(dict(case_id=case['case_id'],arm=arm,answer=r['answer'],model_calls=r['model_calls'])),flush=True)
    write(OUT/f'eval-{arm}-complete.json',dict(cases=64))
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','engineer','probe','evaluate']);p.add_argument('--arm',choices=['A','C','B']);a=p.parse_args()
    try:
        if a.stage in ('probe','evaluate'):globals()[a.stage](a.arm)
        else:globals()[a.stage]()
    except BaseException as e:
        if OUT.exists() and not (OUT/'experiment-failure.json').exists():write(OUT/'experiment-failure.json',dict(stage=a.stage,error_type=type(e).__name__,error=str(e),stop_whole_sequence=True,no_auto_rerun=True))
        raise
if __name__=='__main__':main()
