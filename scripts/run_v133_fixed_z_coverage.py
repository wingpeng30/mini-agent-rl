"""Frozen-model, bounded relation coverage / rank / input / next-action diagnosis."""
from __future__ import annotations
import argparse,asyncio,hashlib,json,os,time
from pathlib import Path
import close_v133_goal_ai as old
ROOT=old.ROOT;PRIOR=old.O;OUT=ROOT/'runs/v133-goal-ai-fixed-z-coverage-v1'
read=old.read;rows=old.rows;sha=old.ai.sha;norm=old.ai.norm
def asset_key(path):
 path=Path(path)
 return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
# Explicit query binding review, made from questions and original actual inputs.
# These are NOT sufficiency decisions. No terminal answers are supplied here.
SPECS=[
 ('musique:2hop__862994_69048','Delta State','birthplace','born 30 December 1979 in Delta State','chief judge','chief justice'),
 ('musique:2hop__44522_58556','agency','question anchor',None,'OTC prescription drug marketing authority food safety system','drug agency over the counter prescription food safety'),
 ('musique:2hop__442175_56873','Nottingham Forest','listed football club','joining Derby County in 2011 and later Nottingham Forest in 2015','FA Cup winning years','FA Cup victories history'),
 ('musique:2hop__337223_56873','Nottingham Forest','notably played-for club','notably for Nottingham Forest','FA Cup winning years','FA Cup victories history'),
 ('musique:2hop__468778_88165','Hagar','question anchor',None,'spouse','husband marriage'),
 ('2wiki:f299dd600bdd11eba7f7acde48001122','Juan de la Cerda, 4th Duke of Medinaceli','father','He was the son of Don Juan de la Cerda, 4th Duke of Medinaceli','date of death','died death date'),
 ('2wiki:d94d1d720bdd11eba7f7acde48001122','Tony Parker','song performer','song recorded by NBA player Tony Parker','workplace','works employer'),
 ('musique:2hop__71034_343058','Jennifer Saunders','Shrek 2 version performer','Jennifer Saunders recorded the song for the 2004 film Shrek 2','spouse','married husband'),
 ('musique:2hop__864101_135373','St. Matthew Island','question anchor',None,'located in sea','island body of water location'),
 ('2wiki:015a72a60bde11eba7f7acde48001122','Bernard VII, Count of Armagnac','father','eldest daughter of Bernard VII, Count of Armagnac','place of birth','born birthplace'),
 ('musique:2hop__475070_532404','Sir Charles Tupper','spouse','wife of Sir Charles Tupper','place of birth','born birthplace'),
 ('2wiki:66616e2a0bde11eba7f7acde48001122','Henry VIII the Sparrow','father','fourth and youngest son of Henry VIII the Sparrow','place of death','died death place'),
 ('2wiki:b592bb5c0bd911eba7f7acde48001122','Witold Leszczyński','director','film directed by Witold Leszczyński','place of birth','born birthplace'),
 ('2wiki:f480bf400bdd11eba7f7acde48001122','Thomas Powys, 2nd Baron Lilford','father','third son of Thomas Powys, 2nd Baron Lilford','place of birth','born birthplace'),
 ('2wiki:ae714dea0bdc11eba7f7acde48001122','Margarete Sophie of Austria','mother','son of Albrecht, Duke of Württemberg, and Archduchess Margarete Sophie of Austria','date of birth','born birthday'),
 ('2wiki:085ace840bde11eba7f7acde48001122','Annapurna Devi','mother','son and the eldest child of musicians Ravi Shankar and Annapurna Devi','place of birth','born birthplace'),
 ('2wiki:d438a1020bd911eba7f7acde48001122','Stefan the First-Crowned','father','middle son of Stefan the First-Crowned','nationality','nationality citizenship'),
 ('musique:2hop__65576_80026','Charlie and the Chocolate Factory','question anchor',None,'spoilt character actor','spoiled character played by'),
 ('2wiki:66c00bf60bde11eba7f7acde48001122',"Richard O'Brien",'song writer/composer candidate',"Written by Richard O'Brien",'spouse','married wife'),
 ('2wiki:43a949ae0bdc11eba7f7acde48001122','Brian Trenchard-Smith','director','directed by Brian Trenchard-Smith','place of birth','born birthplace'),
 ('musique:2hop__495316_660579','Oktibbeha County','containing county','community located in Oktibbeha County, Mississippi','county seat','county capital Mississippi'),
 ('musique:2hop__455392_40315','California State University','institution whole system','field station of the California State University','total number of campuses','system campuses total'),
 ('musique:2hop__124932_278133','Guru Gobind Singh','father','second son of Guru Gobind Singh','father','parent father'),
 ('2wiki:13e2f86c0bde11eba7f7acde48001122','Adalbert I, Margrave of Tuscany','father','son and successor of Adalbert I, Margrave of Tuscany','place of birth','born birthplace'),
]
def save(name,value,jl=False):
 with (OUT/name).open('x',encoding='utf8') as f:
  if jl:
   for row in value:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  else:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def guard():
 assert not (OUT/'failure.json').exists(),'failure latch: no automatic retry'
 cfg=read(OUT/'freeze.json')
 for path,h in cfg['hashes'].items():assert sha(ROOT/path)==h,path
 assert sha(old.ai.SOURCE/'adapter_model.safetensors')==old.ai.EXPECTED
 return cfg
def prefixes(packet,client):
 raw=next(r for r in rows(old.ai.AH/('eval-'+packet['split']+'-baseline.jsonl')) if r['case_id']==packet['case_id'])
 ms=old.initial(packet['question']);out=[]
 for t in raw['trace']:
  ids=client._inputs(ms)['input_ids'][0].tolist();assert ids==t['prompt_token_ids']
  out.append(list(ms))
  if 'observation' in t:ms=old.extend(ms,t['query'],t['observation'])
 return raw,out
def freeze():
 old.guard();cfg=old.ai.guard();assert not OUT.exists();client,tok=old.renderer()
 packets={r['case_id']:r for r in rows(PRIOR/'diagnostic-review-packets.jsonl')};plans=[];replay_count=0
 for cid,subject,role,quote,a,b in SPECS:
  p=packets[cid];raw,pres=prefixes(p,client);replay_count+=len(pres);anchor=None;binding=None
  if quote:
   for k,call in enumerate(p['actual_calls']):
    if k+2>4 or sum('observation' in t for t in raw['trace'][:k])+1>3:continue
    decoded=old.decode(call['prompt_token_ids'],tok)
    for d in decoded['documents']:
     if quote in d['text'] and norm(subject) in norm(d['text']):
      anchor=k;binding=dict(document_id=d['document_id'],title=d['title'],quote=quote,
       body_input_token_span=d['body_input_token_span'],body_input_char_span=d['body_input_char_span'],
       subject=subject,role=role,reviewer='lead pre-retrieval role binding');break
    if anchor is not None:break
   assert anchor is not None,(cid,'expected visible bound bridge not found')
  else:
   anchor=0;assert norm(subject) in norm(p['question'])
   start=p['question'].casefold().index(subject.casefold());binding=dict(question_char_span=[start,start+len(subject)],subject=subject,role=role)
  call=p['actual_calls'][anchor]
  queries=[subject+' '+a,subject+' '+b]
  for q in queries:assert norm(subject) in norm(call['actual_prompt'])
  plans.append(dict(case_id=cid,split=p['split'],question=p['question'],anchor_call=anchor,subject=subject,role=role,
   binding=binding,queries=[dict(arm=arm,query=q,entity_sources=[binding],predicate_source='question-requested canonical relation / frozen synonym') for arm,q in zip(['A','B'],queries)],
   corpus=next(s['corpus'] for s in cfg['evaluation'] if s['split']==p['split']),
   prefix_searches=sum('observation' in t for t in raw['trace'][:anchor]),
   calls_if_answer=anchor+2,searches_if_answer=sum('observation' in t for t in raw['trace'][:anchor])+1))
 assert len(plans)==24
 paths=[Path(__file__),ROOT/'src/mini_agent_rl/model/local.py',ROOT/'src/mini_agent_rl/model/context.py',ROOT/'src/mini_agent_rl/tools/search.py',ROOT/'src/mini_agent_rl/prompts.py',ROOT/'scripts/run_v133_goal_q_search.py',
   PRIOR/'diagnostic-review-packets.jsonl',PRIOR/'diagnostic-ledger-v4.jsonl',PRIOR/'budget-reconstruction.jsonl',PRIOR/'repair-query-results.jsonl',old.ai.SOURCE/'adapter_model.safetensors',old.ai.BASE/'model.safetensors-00001-of-00001.safetensors',old.ai.BASE/'tokenizer.json']
 for s in cfg['evaluation']:
  if s['split'] in {p['split'] for p in plans}:paths += [ROOT/s['corpus'],ROOT/s['cases'],old.ai.AH/('eval-'+s['split']+'-baseline.jsonl')]
 OUT.mkdir();save('query-freeze.jsonl',plans,True)
 paths += [OUT/'query-freeze.jsonl']
 save('freeze.json',dict(hashes={asset_key(p):sha(p) for p in paths},product=cfg['product'],adapter=str(old.ai.SOURCE),
  caps=dict(new_queries=48,gpu_calls=24,coverage_documents_per_edge=12,updates=0),token_replayed_calls=replay_count,
  historical_branches=53,case_count=24,corpora={p['corpus']:len(read(ROOT/p['corpus'])) for p in plans}))
 print(json.dumps({'frozen_cases':24,'queries':48,'original_calls_replayed':replay_count}),flush=True)
def recover_freeze():
 # Continue ONLY a known manifest serialization bug, before retrieval/model work.
 # Preserve the query file; do not repeat anchor selection or token replay.
 failure=read(OUT/'failure.json')
 assert failure['stage']=='freeze' and failure['type']=='ValueError' and 'not in the subpath' in failure['message']
 assert set(p.name for p in OUT.iterdir())=={'query-freeze.jsonl','failure.json','runner-before-path-fix.py'}
 old.guard();cfg=old.ai.guard();plans=rows(OUT/'query-freeze.jsonl');assert len(plans)==24
 packet=rows(PRIOR/'diagnostic-review-packets.jsonl');replay_count=sum(len(p['actual_calls']) for p in packet)
 paths=[Path(__file__),ROOT/'src/mini_agent_rl/model/local.py',ROOT/'src/mini_agent_rl/model/context.py',ROOT/'src/mini_agent_rl/tools/search.py',ROOT/'src/mini_agent_rl/prompts.py',ROOT/'scripts/run_v133_goal_q_search.py',
   PRIOR/'diagnostic-review-packets.jsonl',PRIOR/'diagnostic-ledger-v4.jsonl',PRIOR/'budget-reconstruction.jsonl',PRIOR/'repair-query-results.jsonl',old.ai.SOURCE/'adapter_model.safetensors',old.ai.BASE/'model.safetensors-00001-of-00001.safetensors',old.ai.BASE/'tokenizer.json',OUT/'query-freeze.jsonl',OUT/'runner-before-path-fix.py']
 for s in cfg['evaluation']:
  if s['split'] in {p['split'] for p in plans}:paths += [ROOT/s['corpus'],ROOT/s['cases'],old.ai.AH/('eval-'+s['split']+'-baseline.jsonl')]
 save('implementation-recovery.json',dict(reason='external base-model path manifest serialization fix',original_failure=failure,
   authorization='User authorized implementation and necessary code repairs; this is not GPU/numerical/infrastructure/identity failure.',
   query_freeze_sha256=sha(OUT/'query-freeze.jsonl'),queries_before_fix=0,gpu_calls_before_fix=0,token_replay_not_repeated=True,automatic_experiment_retry=False))
 save('freeze.json',dict(hashes={asset_key(p):sha(p) for p in paths},product=cfg['product'],adapter=str(old.ai.SOURCE),
   caps=dict(new_queries=48,gpu_calls=24,coverage_documents_per_edge=12,updates=0),token_replayed_calls=replay_count,
   historical_branches=53,case_count=24,corpora={p['corpus']:len(read(ROOT/p['corpus'])) for p in plans}))
 (OUT/'failure.json').rename(OUT/'failure.manifest-path-v1.json')
 print(json.dumps({'manifest_recovered':True,'queries_preserved':48,'replayed_calls':replay_count,'new_queries':0,'gpu_calls':0}),flush=True)
def relation_words(relation):
 words={'born':'born birth birthplace','father':'father son daughter parent','mother':'mother son daughter parent',
  'spouse':'spouse wife husband married marriage consort','died':'died death deceased','chief':'chief justice judge',
  'campus':'campus campuses university','nationality':'nationality citizen citizenship Serbian',
  'named':'named name','seat':'seat capital','performer':'singer song recorded performer','director':'directed director',
  'OTC':'OTC prescription over counter','food':'food safety FSMA','FA Cup':'FA Cup won winner winning','spoilt':'spoilt spoiled actor character played'}
 text=relation.casefold();return set(norm(' '.join(v for k,v in words.items() if k.casefold() in text)).split()) or set(norm(relation).split())
def cpu():
 guard();client,tok=old.renderer();plans=rows(OUT/'query-freeze.jsonl');packets={p['case_id']:p for p in rows(PRIOR/'diagnostic-review-packets.jsonl')}
 ledgers={p['case_id']:p for p in rows(PRIOR/'diagnostic-ledger-v4.jsonl')};corpora={p['corpus']:read(ROOT/p['corpus']) for p in plans};indexes={path:old.ai.tool(corpus) for path,corpus in corpora.items()}
 cache={};rank_cache={};candidate=[];branches=[];new_count=0;historical=[]
 # Cached observations include original raw searches and the 53 old alternatives.
 for plan in plans:
  p=packets[plan['case_id']];raw,pres=prefixes(p,client)
  for t in raw['trace']:
   if 'observation' in t:cache[(plan['corpus'],norm(t['query']))]=t['observation']['result']
  for r in p['results']:cache[(plan['corpus'],norm(r['query']))]=r['result']
 for plan in plans:
  p=packets[plan['case_id']];raw,pres=prefixes(p,client);k=plan['anchor_call'];corpus=corpora[plan['corpus']];tool=indexes[plan['corpus']]
  def rank(query):
   key=(plan['corpus'],norm(query))
   if key not in rank_cache:
    all_docs=tool._bm25_index.search(query,top_k=len(corpus));top=tool._search(query)
    assert all_docs[:3]==top,'ranking mismatch'
    if key in cache:assert [(d['id'],d['title'],d['text']) for d in cache[key]]==[(d['id'],d['title'],d['text']) for d in top],'saved retrieval mismatch'
    rank_cache[key]=dict(corpus=plan['corpus'],query=query,ranked=[dict(id=d['id'],rank=i+1,score=d['score']) for i,d in enumerate(all_docs)])
   return rank_cache[key]
  for t in raw['trace']:
   if 'observation' in t:historical.append(dict(case_id=plan['case_id'],kind='original',query=t['query'],ranking=rank(t['query'])))
  for r in p['results']:historical.append(dict(case_id=plan['case_id'],kind='historical_alternative',query=r['query'],ranking=rank(r['query'])))
  previous={norm(t['query']) for t in raw['trace'][:k] if 'observation' in t}
  for q in plan['queries']:
   key=(plan['corpus'],norm(q['query']));cached=key in cache
   if cached:result=cache[key]
   else:
    assert new_count<48;result=tool._search(q['query']);cache[key]=result;new_count+=1
   ranking=rank(q['query']);obs=dict(ok=True,result=result,searched_queries=sorted(previous|{norm(q['query'])}))
   ms=old.extend(pres[k],q['query'],obs);enc=client._inputs(ms);ctx=old.decode(enc['input_ids'][0].tolist(),tok)
   prectx=old.decode(raw['trace'][k]['prompt_token_ids'],tok)
   branches.append(dict(branch_id=plan['case_id']+'|'+q['arm'],case_id=plan['case_id'],arm=q['arm'],query=q['query'],
    anchor_call=k,corpus=plan['corpus'],binding=plan['binding'],grounded=True,repeated=norm(q['query']) in previous,
    legal_budget=k+2<=4 and plan['searches_if_answer']<=3 and norm(q['query']) not in previous,
    calls_if_answer=k+2,searches_if_answer=plan['searches_if_answer'],cached=cached,result=result,
    original_prefix_documents=prectx['documents'],answer_context=ctx,context_usage=enc._context_usage,
    ranking=ranking,messages=[m.model_dump() for m in ms],product_success_counted=False))
  # Candidate generation is AFTER the query freeze. No candidate metadata feeds queries.
  for ei,edge in enumerate(ledgers[plan['case_id']]['required_edges']):
   subject=edge[0];entities=[subject] if not subject.startswith('?') else [plan['subject']]
   existing=p['source_guided_corpus_docs'];words=relation_words(edge[1]);ent_tokens=set(norm(' '.join(entities)).split())
   byid={d['id']:d for d in corpus};priority=[]
   for d in existing:
    if d['id'] in byid and d['id'] not in priority:priority.append(d['id'])
   scored=[]
   for d in corpus:
    text=set(norm(d['title']+' '+d['text']).split());title=set(norm(d['title']).split())
    entity_score=len(text&ent_tokens);relation_score=len(text&words)
    if not entity_score and not relation_score:continue
    scored.append((-entity_score,-relation_score,d['id']))
   for _,_,did in sorted(scored):
    if did not in priority:priority.append(did)
   reviewed=priority[:12]
   candidate.append(dict(case_id=plan['case_id'],edge_index=ei,required_edge=edge,corpus=plan['corpus'],
    candidate_documents=[byid[x] for x in reviewed],candidate_total=len(priority),review_cap=12,status='pending_semantic_review',
    corpus_absence_proven=False,query_generation_already_frozen=True))
 save('coverage-review-packets.jsonl',candidate,True);save('branches.jsonl',branches,True);save('historical-ranking.jsonl',historical,True)
 # Reproduce training-only Chelsea sentinel, no new search or model call.
 sentinel=next(r for r in rows(PRIOR/'repair-query-results.jsonl') if r['plan']['case_id']=='musique:2hop__776776_90536')
 train=next(p for p in rows(PRIOR/'train-review-packets.jsonl') if p['case_id']==sentinel['plan']['case_id'])
 raw=next(r for r in rows(old.ai.OUT/'gpu-train-rollouts.jsonl') if r['case_id']==train['case_id']);ms=old.initial(train['question']);k=sentinel['plan']['call_index']
 for t in raw['trace'][:k]:ms=old.extend(ms,t['query'],t['observation'])
 seen={norm(t['query']) for t in raw['trace'][:k] if 'observation' in t};obs=dict(ok=True,result=sentinel['result'],searched_queries=sorted(seen|{norm(sentinel['plan']['query'])}))
 enc=client._inputs(old.extend(ms,sentinel['plan']['query'],obs));ctx=old.decode(enc['input_ids'][0].tolist(),tok)
 assert ctx['prompt_token_ids']==sentinel['answer_context']['prompt_token_ids'] and ctx['actual_input_tokens']==245 and not ctx['documents']
 save('chelsea-sentinel.json',dict(case_id=train['case_id'],actual_input_tokens=245,documents=ctx['documents'],context_usage=enc._context_usage,
   raw_results=sentinel['result'],token_exact_reproduced=True,in_24_denominator=False,gpu_calls=0))
 save('cpu-summary.json',dict(cases=24,branches=len(branches),coverage_edges=len(candidate),new_queries=new_count,
   cached_queries=48-new_count,historical_ranking_rows=len(historical),ranking_cache=len(rank_cache),gpu_calls=0,updates=0))
 print(json.dumps(read(OUT/'cpu-summary.json')),flush=True)
def support_visible(edge,docs):
 # Explicit reviewed body fragments, not title/answer matching or an entailment heuristic.
 hits=[]
 for proof in edge.get('proofs',[]):
  for d in docs:
   if str(d.get('document_id',d.get('id')))==str(proof['document_id']) and proof['quote'] in d['text']:
    hits.append(dict(proof,reference=d.get('reference'),body_input_token_span=d.get('body_input_token_span'),body_input_char_span=d.get('body_input_char_span')))
 return hits
def assess():
 guard();review=rows(OUT/'coverage-review.jsonl');packets=rows(OUT/'coverage-review-packets.jsonl');bykey={(x['case_id'],x['edge_index']):x for x in packets}
 assert len(review)==len(packets) and len({(x['case_id'],x['edge_index']) for x in review})==len(review)
 for r in review:
  packet=bykey[(r['case_id'],r['edge_index'])];assert r['status'] in ['supported','conflict','unconfirmed'] and r['reason']
  if r['status']=='supported':assert r['proofs']
  for proof in r.get('proofs',[]):
   d=next(d for d in packet['candidate_documents'] if d['id']==proof['document_id']);assert proof['quote'] in d['text']
   pos=d['text'].index(proof['quote']);proof['source_char_span']=[pos,pos+len(proof['quote'])]
 groups={};out=[]
 for r in review:groups.setdefault(r['case_id'],[]).append(r)
 for b in rows(OUT/'branches.jsonl'):
  edges=groups[b['case_id']];rawdocs=b['original_prefix_documents']+b['result'];new=[support_visible(e,b['result']) for e in edges]
  before=[support_visible(e,b['original_prefix_documents']) for e in edges];received=[support_visible(e,rawdocs) for e in edges];visible=[support_visible(e,b['answer_context']['documents']) for e in edges]
  covered=all(e['status']=='supported' for e in edges);ambiguous=any(e.get('critical_ambiguity',False) for e in edges)
  complete_raw=covered and all(received);complete_actual=covered and all(visible)
  item=dict(b,reviewed_edges=edges,new_necessary_edges=sum(bool(n) and not bool(p) for n,p in zip(new,before)),
   source_chain_supported=covered,critical_ambiguity=ambiguous,complete_chain_returned=complete_raw,
   complete_chain_retained=complete_actual,visible_edge_proofs=visible,
   eligible_for_gpu=b['grounded'] and b['legal_budget'] and complete_actual and not ambiguous)
  out.append(item)
 selected=[]
 for cid in groups:
  eligible=[x for x in out if x['case_id']==cid and x['eligible_for_gpu']]
  if eligible:selected.append(sorted(eligible,key=lambda x:x['arm'])[0])
 assert len(selected)<=24 and len({x['case_id'] for x in selected})==len(selected)
 save('coverage-reviewed.jsonl',review,True);save('branch-audit.jsonl',out,True)
 save('gpu-list.jsonl',[dict(case_id=x['case_id'],branch_id=x['branch_id'],prompt_token_ids=x['answer_context']['prompt_token_ids'],messages=x['messages'],
   reviewed_edges=x['reviewed_edges'],visible_edge_proofs=x['visible_edge_proofs'],calls_if_answer=x['calls_if_answer'],searches_if_answer=x['searches_if_answer']) for x in selected],True)
 save('cpu-assessment.json',dict(cases=24,source_chain_supported=sum(all(e['status']=='supported' for e in es) for es in groups.values()),
   coverage_unconfirmed_or_ambiguous=sum(any(e['status']!='supported' or e.get('critical_ambiguity') for e in es) for es in groups.values()),
   new_edge_cases=len({x['case_id'] for x in out if x['new_necessary_edges'] and x['legal_budget']}),
   complete_raw_cases=len({x['case_id'] for x in out if x['complete_chain_returned'] and x['legal_budget'] and not x['critical_ambiguity']}),
   complete_retained_cases=len({x['case_id'] for x in out if x['eligible_for_gpu']}),gpu_list=len(selected),
   semantic_decisions_hash=sha(OUT/'coverage-review.jsonl'),gpu_list_hash=sha(OUT/'gpu-list.jsonl')))
 print(json.dumps(read(OUT/'cpu-assessment.json')),flush=True)
def gpu():
 cfg=guard();assessment=read(OUT/'cpu-assessment.json');assert sha(OUT/'gpu-list.jsonl')==assessment['gpu_list_hash']
 assert sha(OUT/'coverage-review.jsonl')==assessment['semantic_decisions_hash'];todo=rows(OUT/'gpu-list.jsonl')
 assert not (OUT/'gpu-start.json').exists(),'GPU already attempted; no retry'
 if not todo:save('gpu-summary.json',dict(calls=0,reason='no eligible states',updates=0));return
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 import torch
 from transformers import set_seed
 from mini_agent_rl.model import LocalModelConfig,TransformersModelClient
 from mini_agent_rl.domain import Message
 from run_v133_goal_p_variant import SearchTextAliasClient
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();set_seed(42)
 save('gpu-start.json',dict(calls_cap=24,selected=len(todo),device=torch.cuda.get_device_name(0),adapter_sha256=old.ai.EXPECTED,updates=0))
 client=TransformersModelClient(LocalModelConfig(model_id=str(old.ai.BASE),adapter_path=str(old.ai.SOURCE),device='cuda',dtype='bfloat16',
  load_in_4bit=False,max_input_tokens=2048,max_new_tokens=128,enable_thinking=False,deterministic=True,observation_encoding='legacy',recompute_old_logprobs=False,timeout=None))
 wrapped=SearchTextAliasClient(client);completed=[]
 with (OUT/'gpu-next-actions.jsonl').open('x',encoding='utf8') as f:
  for p in todo:
   ms=[Message.model_validate(x) for x in p['messages']];start=time.monotonic();resp=asyncio.run(wrapped.generate(ms,[],temperature=0.0,seed=42));elapsed=time.monotonic()-start
   assert resp.prompt_token_ids==p['prompt_token_ids'],'GPU actual input differs from CPU frozen input'
   r=dict(case_id=p['case_id'],branch_id=p['branch_id'],response=resp.model_dump(),latency_seconds=elapsed,
    calls_if_action=p['calls_if_answer'],searches_before_action=p['searches_if_answer'],following_search_executed=False,
    product_success_counted=False,conditional_decision_only=True)
   f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();completed.append(r)
   print(json.dumps({'gpu_completed':len(completed),'selected':len(todo)}),flush=True)
 assert sha(old.ai.SOURCE/'adapter_model.safetensors')==old.ai.EXPECTED
 save('gpu-summary.json',dict(calls=len(completed),searches=0,updates=0,adapter_sha256=old.ai.EXPECTED,device=torch.cuda.get_device_name(0)))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','recover-freeze','cpu','assess','gpu']);arg=parser.parse_args()
 try:globals()[arg.stage.replace('-','_')]()
 except Exception as e:
  if OUT.exists() and not (OUT/'failure.json').exists():save('failure.json',dict(stage=arg.stage,type=type(e).__name__,message=str(e),automatic_retry=False))
  raise
if __name__=='__main__':main()
