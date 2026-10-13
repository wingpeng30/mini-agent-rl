"""Finite CPU closure of Goal AI. No model load, generation or weight update."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import run_v133_goal_ai_evidence_feasibility as ai
from prepare_v133_goal_ah import documents
P=ai.OUT;O=P/'closure-v3'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(n,v,jl=False):
    with (O/n).open('x',encoding='utf8') as f:
        if jl:
            for x in v:f.write(json.dumps(x,ensure_ascii=False)+'\n')
        else:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def renderer():
    from transformers import AutoProcessor
    from mini_agent_rl.model import LocalModelConfig,TransformersModelClient
    c=object.__new__(TransformersModelClient)
    c.config=LocalModelConfig(model_id=str(ai.BASE),max_input_tokens=2048,max_new_tokens=128,enable_thinking=False,observation_encoding='legacy')
    c._processor=AutoProcessor.from_pretrained(str(ai.BASE),local_files_only=True)
    c._model=SimpleNamespace(device='cpu');c._context_records=[]
    return c,getattr(c._processor,'tokenizer',c._processor)
def initial(q):
    from mini_agent_rl.domain import Message
    from mini_agent_rl.prompts import system_prompt
    return [Message(role='system',content=system_prompt(False,'direct-v090')),Message(role='user',content=q)]
def extend(ms,query,observation):
    from mini_agent_rl.domain import Message
    obs={k:v for k,v in observation.items() if k!='latency_ms'}
    return ms+[Message(role='assistant',content=json.dumps({'action':'search','query':query},ensure_ascii=False)),Message(role='tool',content=json.dumps(obs,ensure_ascii=False))]
def decode(ids,tok):
    text=tok.decode(ids,skip_special_tokens=False);enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    assert enc['input_ids']==ids,'token roundtrip mismatch'
    ds=documents(text)
    for d in ds:
        escaped=json.dumps(d['text'],ensure_ascii=False)[1:-1]
        a=text.find(escaped,*d['input_char_span']);assert a>=0
        b=a+len(escaped);ix=[i for i,(s,e) in enumerate(enc['offset_mapping']) if s<b and e>a]
        d['body_input_char_span']=[a,b];d['body_input_token_span']=[min(ix),max(ix)+1]
    return dict(actual_prompt=text,prompt_token_ids=ids,offset_mapping=enc['offset_mapping'],documents=ds,actual_input_tokens=len(ids))
def prepare():
    cfg=ai.guard();assert not O.exists()
    assert ai.sha(ai.SOURCE/'adapter_model.safetensors')==ai.EXPECTED
    sources=[P/'gpu-train-rollouts.jsonl',P/'pilot-cases-v2.jsonl',P/'diagnostic-packets.jsonl',P/'material-source-audit-v2.jsonl',P/'old-case-source-map-v2.json',Path(__file__)]
    frozen={str(p.relative_to(ROOT)):ai.sha(p) for p in sources}
    O.mkdir();save('freeze.json',dict(inputs=frozen,limits=dict(new_cpu_queries=32,new_gpu_calls=0,updates=0),product=cfg['product']))
    c,tok=renderer();train=[];case_map={r['id']:r for r in rows(P/'pilot-cases-v2.jsonl')};matched=0
    for r in rows(P/'gpu-train-rollouts.jsonl'):
        ms=initial(r['question']);calls=[]
        for index,t in enumerate(r['trace']):
            enc=c._inputs(ms);ids=enc['input_ids'][0].tolist();assert ids==t['prompt_token_ids'],(r['case_id'],index)
            d=decode(ids,tok);d.update(call_index=index,parsed_action={k:t.get(k) for k in ['action','query','answer','parse_error']},search_observation=t.get('observation'),usage=t['usage']);calls.append(d);matched+=1
            if 'observation' in t:ms=extend(ms,t['query'],t['observation'])
        case=case_map[r['case_id']]
        train.append(dict(case_id=r['case_id'],question=r['question'],answer=r['answer'],origin=r['origin'],corpus=r['corpus'],calls=calls,
            source_case=case,prior_material=case.get('material'),product_success_counted=False))
    save('train-review-packets.jsonl',train,True)
    branches=[];diag=[]
    for p in rows(P/'diagnostic-packets.jsonl'):
        spec=next(x for x in cfg['evaluation'] if x['split']==p['split'])
        raw=next(x for x in rows(ai.AH/('eval-'+p['split']+'-baseline.jsonl')) if x['case_id']==p['case_id'])
        actual=next(x for x in rows(ai.AH/('actual-inputs-'+p['split']+'-baseline.jsonl')) if x['case_id']==p['case_id'])
        ms=initial(p['question']);prefixes=[]
        for i,t in enumerate(raw['trace']):
            ids=c._inputs(ms)['input_ids'][0].tolist();assert ids==t['prompt_token_ids'],(p['case_id'],i)
            prefixes.append(list(ms))
            if 'observation' in t:ms=extend(ms,t['query'],t['observation'])
        anchor=p.get('anchor');ids=[]
        if anchor:
            k=anchor['call_index'];pre=raw['trace'][:k];used=sum('observation' in t for t in pre)
            for j,res in enumerate(p['results']):
                bid=p['case_id']+'|'+str(j);ids.append(bid)
                seen={t['query'].casefold().strip() for t in pre if 'observation' in t};q=res['query'];repeated=q.casefold().strip() in seen
                obs=dict(ok=True,result=res['result'],searched_queries=sorted(seen|{q.casefold().strip()}))
                enc=c._inputs(extend(prefixes[k],q,obs));d=decode(enc['input_ids'][0].tolist(),tok)
                d.update(branch_id=bid,case_id=p['case_id'],split=p['split'],query=q,anchor_call=k,
                    searches_if_answer=used+1,calls_if_answer=k+2,legal_budget=used+1<=3 and k+2<=4 and not repeated,
                    repeated=repeated,context_usage=enc._context_usage,grounding='pending_semantic_review',result=res['result'])
                branches.append(d)
        diag.append(dict(p,actual_calls=actual['calls'],branch_ids=ids))
    save('diagnostic-review-packets.jsonl',diag,True);save('budget-reconstruction.jsonl',branches,True)
    save('execution-status.json',dict(remote_failure_latch=False,old_state_file_is_stale=True,gpu_complete=32,gpu_calls=94,gpu_searches=62,token_replayed_calls=matched,diagnostic_cases=len(diag),branches=len(branches),new_cpu_queries=0,new_model_calls=0,updates=0))
    print(json.dumps(read(O/'execution-status.json')),flush=True)
def guard():
    assert not (O/'failure.json').exists(),'closure failure latch'
    for p,h in read(O/'freeze.json')['inputs'].items():assert ai.sha(ROOT/p)==h,p
def search():
    guard();c,tok=renderer();packets={r['case_id']:r for r in rows(O/'train-review-packets.jsonl')};plans=rows(O/'repair-query-freeze.jsonl');assert len(plans)<=32
    out=[]
    for plan in plans:
        p=packets[plan['case_id']];k=plan['call_index'];call=p['calls'][k];q=plan['query']
        # Every entity phrase is manually approved and must appear in the actual prefix or question.
        for ent in plan['grounded_entities']:assert ai.norm(ent) in ai.norm(call['actual_prompt']),ent
        r=next(x for x in rows(P/'gpu-train-rollouts.jsonl') if x['case_id']==p['case_id']);pre=r['trace'][:k]
        assert k+2<=4 and sum('observation' in t for t in pre)+1<=3
        seen={t['query'].casefold().strip() for t in pre if 'observation' in t};assert q.casefold().strip() not in seen
        ms=initial(p['question'])
        for t in pre:ms=extend(ms,t['query'],t['observation'])
        search=ai.tool(read(ROOT/p['corpus']));result=search._search(q)
        obs=dict(ok=True,result=result,searched_queries=sorted(seen|{q.casefold().strip()}))
        enc=c._inputs(extend(ms,q,obs));d=decode(enc['input_ids'][0].tolist(),tok)
        out.append(dict(plan=plan,result=result,answer_context=d,usage=enc._context_usage,product_success_counted=False))
    save('repair-query-results.jsonl',out,True);print(json.dumps({'queries':len(out),'model_calls':0}),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','search']);args=p.parse_args()
    try:globals()[args.stage]()
    except Exception as e:
        if O.exists() and not (O/'failure.json').exists():save('failure.json',dict(stage=args.stage,type=type(e).__name__,message=str(e),automatic_retry=False))
        raise
if __name__=='__main__':main()
