"""Goal BC: bounded whole-action CAD, then gated real search. No training."""
from __future__ import annotations
import argparse, asyncio, gc, hashlib, json, math, shutil, time
from pathlib import Path
from types import SimpleNamespace
import run_v133_goal_ap as ap
from mini_agent_rl.domain import Message
from mini_agent_rl.model.context_aware_decoding import reference_messages, search_count

ROOT=ap.ROOT
OUT=ROOT/'runs/v133-goal-bc-context-aware-decoding-v1'
AW=ROOT/'runs/v133-goal-aw-hop-reader-v1'
CONFIG=OUT/'config.json'
read, rows, sha=ap.read, ap.rows, ap.sha

def write(path, value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def cfg(arm, alpha=None):
    c=read(CONFIG);pc=dict(c['product'])
    pc['context_aware_decoding']=dict(c['cad']) if arm=='B' else None
    if alpha is not None:pc['context_aware_decoding']['alpha']=alpha
    return pc

def make(arm,alpha=None):
    return ap.q.make_client(ap.AMADAPTER,deterministic=True,product_config=cfg(arm,alpha),model_path=ap.BASE,corpus=ap.Z/'corpus-dev.json')

def clipping(usage):
    return any(usage.get(k) for k in ('context_budget_error','lossy_body_trimmed','documents_deleted','history_dropped','prompt_truncated'))

def frozen():
    if (OUT/'experiment-failure.json').exists():raise RuntimeError('BC failure latch: no automatic rerun')
    m=read(OUT/'manifest.json')
    for k,p in m['paths'].items():
        if sha(p)!=m['hashes'][k]:raise RuntimeError('BC frozen drift: '+k)
    return m

def record(kind,stage,cid):
    used=rows(OUT/'gpu-usage.jsonl');c=read(CONFIG)
    if sum(r['kind']==kind for r in used)>=c['caps'][kind]:raise RuntimeError('BC total quota '+kind)
    group='engineer' if stage=='engineer' else 'probe' if stage.startswith('probe') else 'evaluate'
    if sum(r['kind']==kind and r['group']==group for r in used)>=c['stages'][group][kind]:raise RuntimeError('BC stage quota '+kind)
    ap.append(OUT/'gpu-usage.jsonl',dict(kind=kind,stage=stage,group=group,case_id=cid,time=time.time(),status='attempt_started'))

def spans(client,messages,ids):
    tok=getattr(client._processor,'tokenizer',client._processor)
    chat,_=client._present_messages(messages)
    rendered=client._processor.apply_chat_template(chat,add_generation_prompt=True,tokenize=False,enable_thinking=False)
    encoded=tok(rendered,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']!=ids:raise RuntimeError('BC offsets not actual input')
    offsets=encoded['offset_mapping'];out=[];cursor=0
    for index,m in enumerate(messages):
        if m.role!='tool':continue
        start=rendered.find(m.content,cursor)
        if start<0:raise RuntimeError('BC actual tool text missing')
        within=start
        for rank,d in enumerate(json.loads(m.content)['result'],1):
            body=d.get('text','');escaped=json.dumps(body,ensure_ascii=False)[1:-1]
            position=rendered.find(escaped,within) if escaped else within
            if position<0:raise RuntimeError('BC document not actual input')
            end=position+len(escaped)
            tokens=[i for i,(a,b) in enumerate(offsets) if b>position and a<end]
            out.append(dict(message_index=index,rank=rank,document_id=d.get('id'),title=d.get('title'),body_sha256=hashlib.sha256(body.encode()).hexdigest(),body_chars=[0,len(body)],escaped_prompt_chars=[position,end],input_token_span=[tokens[0],tokens[-1]+1] if tokens else None))
            within=end
        cursor=start+len(m.content)
    return out

def prepare():
    if (OUT/'manifest.json').exists():raise FileExistsError('BC already prepared')
    (OUT/'gpu-usage.jsonl').touch(exist_ok=False)
    if sha(ap.AMADAPTER/'adapter_model.safetensors')!=ap.AM_HASH or sha(ap.ZADAPTER/'adapter_model.safetensors')!=ap.Z_HASH:raise RuntimeError('BC adapter identity')
    if not read(ap.AA/'product-decision.json')['published']:raise RuntimeError('BC AA release')
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(ap.BASE),local_files_only=True)
    client,_,_=make('A');client._processor=processor;client._model=SimpleNamespace(device='cpu')
    items=rows(AW/'probe-inputs.jsonl')
    if len(items)!=32 or sum(r['category']=='complete' for r in items)!=24:raise RuntimeError('BC inventory')
    for item in items:
        messages=[Message(**m) for m in item['messages']]
        encoded=client._inputs(messages,'bc_cpu')
        ids=encoded['input_ids'][0].tolist()
        if ids!=item['prompt_token_ids'] or clipping(encoded._context_usage):raise RuntimeError('BC A input reconstruction')
        if search_count(messages) not in read(CONFIG)['cad']['after_searches']:raise RuntimeError('BC probe outside fixed schedule')
        refs=reference_messages(encoded._budgeted_messages)
        ref_ids=client._render_chat(refs)['input_ids'][0].tolist()
        if len(ref_ids)>cfg('A')['max_input_tokens']:raise RuntimeError('BC reference budget')
        ap.append(OUT/'probe-inputs.jsonl',dict(item,reference_messages=[m.model_dump(mode='json') for m in refs],reference_prompt_token_ids=ref_ids,document_spans=spans(client,encoded._budgeted_messages,ids)))
    cases=rows(ap.AM/'evaluation-cases.jsonl')
    if len(cases)!=64 or len(read(ap.ROOT/'runs/v133-goal-al-wide-context-v1/manifest.json')['admitted'])!=29:raise RuntimeError('BC dev inventory')
    shutil.copyfile(ap.AM/'evaluation-cases.jsonl',OUT/'evaluation-cases.jsonl')
    paths=dict(config=CONFIG,plan=OUT/'PLAN.md',inputs=OUT/'probe-inputs.jsonl',cases=OUT/'evaluation-cases.jsonl',
      prior_inputs=AW/'probe-inputs.jsonl',prior_rulings=AW/'probe-adjudications.json',prior_outputs=AW/'probe-A.jsonl',
      corpus=ap.Z/'corpus-dev.json',AM=ap.AMADAPTER/'adapter_model.safetensors',Z=ap.ZADAPTER/'adapter_model.safetensors',
      AA=ap.AA/'product-decision.json',admission=ROOT/'runs/v133-goal-al-wide-context-v1/manifest.json')
    for f in ap.BASE.iterdir():
        if f.name.endswith(('.json','.jinja')):paths['base:'+f.name]=f
    for f in (ROOT/'src/mini_agent_rl').rglob('*.py'):paths['source:'+str(f.relative_to(ROOT))]=f
    for f in (ROOT/'scripts').glob('*.py'):paths['script:'+f.name]=f
    for f in (ROOT/'tests').glob('*context_aware*'):paths['test:'+f.name]=f
    write(OUT/'manifest.json',dict(paths={k:str(p) for k,p in paths.items()},hashes={k:sha(p) for k,p in paths.items()},configuration=read(CONFIG),
      model='AM42-A',model_sha256=ap.AM_HASH,published_model='Goal Z',published_sha256=ap.Z_HASH,training_use_forbidden=True,new_confirmation=0,
      protection=read(ROOT/'runs/v133-goal-al-wide-context-v1/manifest.json')['protected'],must_retain=['musique:2hop__796221_18803','2wiki:f299dd600bdd11eba7f7acde48001122']))
    write(OUT/'cpu-preflight.json',dict(passed=True,A_exact_inputs=32,reference_only_title_text_removed=True,clipped_inputs=0,full=24,partial=8,weights_loaded=False))
    print('BC CPU prepare passed: 32 exact inputs',flush=True)

async def generate(client,wrapped,messages,stage,cid,**kwargs):
    record('primary',stage,cid)
    cad=client.config.context_aware_decoding
    expected=bool(cad and cad.alpha>0 and search_count(messages) in cad.after_searches)
    if expected:record('reference',stage,cid)
    started=time.monotonic();response=await wrapped.generate(messages,[],**kwargs)
    if clipping(response.usage):raise RuntimeError('BC actual input clipped')
    if len(response.prompt_token_ids)>client.config.max_input_tokens or len(response.action_token_ids)>client.config.max_new_tokens:raise RuntimeError('BC actual limits')
    actual=response.usage.get('context_aware_decoding',{}).get('reference_sequences',0)
    if actual!=int(expected):raise RuntimeError('BC reference accounting mismatch')
    return response.model_copy(update={'usage':dict(response.usage,elapsed_generation_seconds=time.monotonic()-started)})

def one(client,wrapped,item,arm,stage):
    response=asyncio.run(generate(client,wrapped,[Message(**m) for m in item['messages']],stage,item['case_id'],temperature=0.,seed=42))
    if response.prompt_token_ids!=item['prompt_token_ids']:raise RuntimeError('BC actual primary input drift')
    cad=response.usage.get('context_aware_decoding',{})
    if cad.get('enabled') and cad['reference_prompt_token_ids']!=item['reference_prompt_token_ids']:raise RuntimeError('BC actual reference input drift')
    return dict(arm=arm,probe_id=item['probe_id'],case_id=item['case_id'],category=item['category'],raw=response.raw_text,
      action=ap.an.parsed_action(response),answer=response.content,query=response.tool_call.arguments.get('query') if response.tool_call else None,
      parse_error=response.parse_error,prompt_token_ids=response.prompt_token_ids,action_token_ids=response.action_token_ids,usage=response.usage,
      no_search_execution=True,training_use_forbidden=True)

def release(client,wrapped):
    client._model=None;client._processor=None
    gc.collect()
    import torch
    torch.cuda.empty_cache()

def stage_info(client):
    return dict(identity=client.policy_fingerprint(),load_seconds=client.load_seconds,memory=client.gpu_memory_stats())

def engineer():
    frozen()
    if (OUT/'engineer.jsonl').exists():raise FileExistsError('BC engineering already started')
    items=rows(OUT/'probe-inputs.jsonl');selected=[r for cat in ('complete','partial') for r in [x for x in items if x['category']==cat][:2]]
    original={}
    for arm in ('A','B','alpha0'):
        client,wrapped,_=make('A' if arm=='A' else 'B',0. if arm=='alpha0' else None)
        todo=selected if arm!='alpha0' else [selected[0],selected[2]]
        for item in todo:
            row=one(client,wrapped,item,arm,'engineer');ap.append(OUT/'engineer.jsonl',row)
            if arm=='A':original[item['probe_id']]=row
            if arm=='alpha0':
                old=original[item['probe_id']]
                if row['action_token_ids']!=old['action_token_ids'] or row['usage']['generation_stop_reason']!=old['usage']['generation_stop_reason']:raise RuntimeError('BC alpha0 compatibility failed')
        write(OUT/f'engineer-{arm}-identity.json',stage_info(client));release(client,wrapped)
    write(OUT/'engineer-complete.json',dict(passed=True,primary=10,reference=4,alpha0_exact=2,search=0))

def probe(arm):
    frozen()
    if not read(OUT/'engineer-complete.json')['passed']:raise RuntimeError('BC engineering missing')
    if arm=='B' and not (OUT/'probe-A-complete.json').exists():raise RuntimeError('BC A first')
    if (OUT/f'probe-{arm}.jsonl').exists():raise FileExistsError('BC probes already started')
    client,wrapped,_=make(arm)
    for item in rows(OUT/'probe-inputs.jsonl'):
        row=one(client,wrapped,item,arm,'probe-'+arm);ap.append(OUT/f'probe-{arm}.jsonl',row)
        print(json.dumps({k:row[k] for k in ('arm','probe_id','action','answer','query')},ensure_ascii=False),flush=True)
    write(OUT/f'probe-{arm}-complete.json',dict(states=32,**stage_info(client)))

def evaluate(arm):
    frozen()
    if not read(OUT/'probe-gate.json')['passed']:raise RuntimeError('BC probe gate failed')
    if arm=='B' and not (OUT/'eval-A-complete.json').exists():raise RuntimeError('BC evaluate A first')
    if (OUT/f'eval-{arm}.jsonl').exists():raise FileExistsError('BC evaluation already started')
    client,wrapped,registry=make(arm);original=wrapped.generate;search=registry.execute;active={}
    async def counted(messages,tools,**kw):
        record('primary','evaluate-'+arm,active['cid']);active['p']+=1
        cad=client.config.context_aware_decoding
        if cad and search_count(messages) in cad.after_searches:
            record('reference','evaluate-'+arm,active['cid']);active['r']+=1
        response=await original(messages,tools,**kw)
        if clipping(response.usage):raise RuntimeError('BC real input clipping')
        if len(response.prompt_token_ids)>client.config.max_input_tokens or len(response.action_token_ids)>client.config.max_new_tokens:raise RuntimeError('BC real token limits')
        return response
    async def counted_search(*args,**kw):
        record('search','evaluate-'+arm,active['cid']);return await search(*args,**kw)
    wrapped.generate=counted;registry.execute=counted_search
    for case in rows(OUT/'evaluation-cases.jsonl'):
        active.update(cid=case['case_id'],p=0,r=0)
        row=asyncio.run(ap.q.rollout(case['question'],wrapped,registry,sample=False,seed=42,product_config=cfg(arm)))
        if row['reason']=='tool_error':raise RuntimeError('BC search infrastructure')
        if active['p']>4 or active['r']>2 or active['p']+active['r']>6 or row['searches']>3:raise RuntimeError('BC case budget')
        row.update(case_id=case['case_id'],arm=arm,reference_sequences=active['r'],model_streams=active['p']+active['r'],official_answer=case['answer'],**ap.q.score(row['answer'],case['answer']))
        ap.append(OUT/f'eval-{arm}.jsonl',row)
        print(json.dumps(dict(arm=arm,case_id=case['case_id'],answer=row['answer']),ensure_ascii=False),flush=True)
    write(OUT/f'eval-{arm}-complete.json',dict(cases=64,**stage_info(client)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','engineer','probe','evaluate']);parser.add_argument('--arm',choices=['A','B']);args=parser.parse_args()
    try:
        if args.stage in ('probe','evaluate'):globals()[args.stage](args.arm)
        else:globals()[args.stage]()
    except BaseException as exc:
        if OUT.exists() and not (OUT/'experiment-failure.json').exists():write(OUT/'experiment-failure.json',dict(stage=args.stage,arm=args.arm,error_type=type(exc).__name__,error=str(exc),no_auto_rerun=True,time=time.time()))
        raise

if __name__=='__main__':main()
