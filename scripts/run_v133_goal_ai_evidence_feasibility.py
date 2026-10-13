"""Goal AI: bounded remote diagnosis and GPU inference; never optimizes weights."""
from __future__ import annotations
import argparse, asyncio, hashlib, io, itertools, json, os, re, sys, zipfile
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
AH=ROOT/'runs/v133-goal-ah-entity-relation'
Z=ROOT/'runs/v133-goal-z-multihop-sft'
OUT=ROOT/'runs/v133-goal-ai-evidence-feasibility'
BASE=Path('/root/autodl-tmp/qwen_08b')
SOURCE=Z/'seed42-trajectory/adapter-final'
EXPECTED='d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        return [json.loads(l) for l in f if l.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def key(s): return hashlib.sha256(s.encode()).hexdigest()
def save(name,value,jsonl=False):
    with (OUT/name).open('x',encoding='utf-8',newline='\n') as f:
        if jsonl:
            for x in value: f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
        else: json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
def norm(s): return re.sub(r'\W+',' ',str(s).casefold()).strip()
def tool(corpus):
    from mini_agent_rl.tools.search import SearchTool
    return SearchTool(corpus,'bm25-v2')
def guard():
    if (OUT/'failure.json').exists(): raise RuntimeError('Failure latch; no automatic retry')
    cfg=read(OUT/'freeze-recovery-v2.json' if (OUT/'freeze-recovery-v2.json').exists() else OUT/'freeze.json')
    for p,h in cfg['files'].items():
        if sha(ROOT/p)!=h: raise RuntimeError('Frozen input changed: '+p)
    return cfg
def clean_docs(ds):return [(d['id'],d['title'],d['text']) for d in ds]

def freeze():
    if OUT.exists():
        allowed={'failure.freeze-preflight-v1.json','preflight-resume-authorization.json'}
        if {p.name for p in OUT.iterdir()} != allowed:
            raise FileExistsError('Goal AI already exists; only explicitly authorized preflight recovery is allowed')
        recovery=read(OUT/'preflight-resume-authorization.json')
        assert recovery['stage']=='freeze' and recovery['gpu_started'] is False
    prior=read(AH/'execution-freeze.json')
    assert sha(SOURCE/'adapter_model.safetensors')==EXPECTED
    assert sha(BASE/'tokenizer.json')==prior['tokenizer_sha256']
    assert sha(BASE/'model.safetensors-00001-of-00001.safetensors')==prior['base_sha256']
    release=read(ROOT/'runs/v133-goal-aa-audit-confirmation/product-decision.json')
    assert release['published'] and release['selected_seed']==42 and release['selected_route']=='trajectory'
    final=read(AH/'final-result-freeze.json')
    for p,h in final['files'].items():assert sha(ROOT/p)==h, p
    baseline=rows(AH/'final-source-review-baseline.jsonl')
    missing=[r for r in baseline if r['category']!='success' and not r['evidence_complete']]
    assert len(baseline)==272 and len(missing)==93
    chosen=[]
    for split,n in [('goal_z_dev',12),('aa_regression',6),('ag_regression',6)]:
        pool=[r for r in missing if r['split']==split and not r['ambiguity']]
        chosen += sorted(pool,key=lambda r:key('goal-ai-v1|'+split+'|'+r['case_id']))[:n]
    assert len(chosen)==24
    paths=[AH/'final-source-review-baseline.jsonl',AH/'final-result-freeze.json',AH/'execution-freeze.json',
           Z/'selected-train.jsonl',Z/'corpus-train.json',AH/'cpu-source-contexts.jsonl',
           ROOT/'runs/v133-goal-ag-failure-repair/repair-material-final-ag.jsonl',
           ROOT/'src/mini_agent_rl/tools/search.py',ROOT/'src/mini_agent_rl/model/local.py',
           ROOT/'src/mini_agent_rl/model/context.py',ROOT/'scripts/run_v133_goal_q_search.py',Path(__file__)]
    for spec in prior['evaluation']:
        paths += [ROOT/spec['corpus'],AH/('actual-inputs-'+spec['split']+'-baseline.jsonl'),AH/('eval-'+spec['split']+'-baseline.jsonl')]
    cfg=dict(version='goal-ai-v1',product=prior['product'],evaluation=prior['evaluation'],
        source_adapter=str(SOURCE),adapter_sha256=EXPECTED,base=str(BASE),base_sha256=prior['base_sha256'],
        tokenizer_sha256=prior['tokenizer_sha256'],files={str(p.relative_to(ROOT)):sha(p) for p in set(paths)},
        caps=dict(original_cpu_searches=279,alternative_cpu_searches=72,teacher_cpu_searches=96,
                  gpu_trajectories=32,gpu_calls=128,gpu_searches=96,source_scan_each=4096,source_pool_each=48),
        gpu_inference_authorized=True,training_authorized=False,training_updates=0,paid_api_calls=0,
        source_pool_expansion=False,automatic_retry=False,stage_time_limit=None)
    OUT.mkdir(exist_ok=True)
    save('freeze.json',cfg);save('deep-cases.jsonl',chosen,True)
    save('state.json',dict(status='frozen',gpu_started=False,training_updates=0))
    print(json.dumps({'stage':'freeze','deep_cases':24,'missing':93}),flush=True)

def index():
    cfg=guard();reviews=rows(AH/'final-source-review-baseline.jsonl');out=[];replays=[]
    for spec in cfg['evaluation']:
        split=spec['split'];cs={r['case_id']:r for r in rows(AH/('actual-inputs-'+split+'-baseline.jsonl'))}
        ts={r['case_id']:r for r in rows(AH/('eval-'+split+'-baseline.jsonl'))}
        search=tool(read(ROOT/spec['corpus']))
        for r in [x for x in reviews if x['split']==split and x['category']!='success']:
            item={k:r.get(k) for k in ['case_id','split','category','evidence_complete','ambiguity','entity_mismatch','relation_mismatch','missing_relations','observed_relations','actual_final_prompt_sha256']}
            item.update(question=cs[r['case_id']]['question'],source_quotes=r['source_quotes'])
            out.append(item)
            if r['evidence_complete']:continue
            for t in ts[r['case_id']]['trace']:
                if 'observation' not in t:continue
                got=search._search(t['query']);expected=t['observation']['result']
                assert clean_docs(got)==clean_docs(expected),(r['case_id'],t['query'])
                replays.append(dict(case_id=r['case_id'],split=split,query=t['query'],document_ids=[d['id'] for d in got],matched=True))
        print(json.dumps({'stage':'index','split':split,'completed':len(out)}),flush=True)
    assert len(out)==109 and len(replays)<=279
    save('failure-index.jsonl',out,True);save('original-query-replay.jsonl',replays,True)

def query_candidates(subject,predicate,question):
    # All candidates are independent alternatives, never a three-query continuation.
    aliases={'date of birth':'born birth date','date of death':'died death date','place of death':'died death place',
        'place of birth':'born birthplace','record-label affiliation':'record label signed affiliation',
        'located in body of water':'island sea location','chief judge':'chief judge court'}
    return list(dict.fromkeys([subject+' '+predicate,subject+' '+aliases.get(predicate,predicate.replace(' ',' ')),question]))[:3]

def diagnose():
    cfg=guard();plans=[];contexts={}
    for spec in cfg['evaluation']:
        contexts.update({(spec['split'],r['case_id']):r for r in rows(AH/('actual-inputs-'+spec['split']+'-baseline.jsonl'))})
    for r in rows(OUT/'deep-cases.jsonl'):
        c=contexts[(r['split'],r['case_id'])]; anchor=None;edge=None
        for call in c['calls']:
            for e in r['missing_relations']:
                if len(e)==3 and len(norm(e[0]))>=3 and norm(e[0]) in norm(call['actual_prompt']):
                    anchor=call;edge=e;break
            if anchor:break
        plans.append(dict(case_id=r['case_id'],split=r['split'],question=c['question'],
            missing_relations=r['missing_relations'],category=r['category'],anchor=anchor,
            proposed_edge=edge,queries=query_candidates(edge[0],edge[1],c['question']) if edge else [],
            grounding_status='literal_presence_only_pending_role_audit' if edge else 'no_grounded_subject',
            original_answer=c['answer']))
    save('diagnostic-query-freeze.jsonl',plans,True) # Before any alternative retrieval.
    packets=[];count=0
    for spec in cfg['evaluation']:
        corpus=read(ROOT/spec['corpus']);search=tool(corpus)
        cases={r['case_id']:r for r in rows(ROOT/spec['cases'])}
        for p in [v for v in plans if v['split']==spec['split']]:
            results=[dict(query=q,result=search._search(q)) for q in p['queries']];count+=len(results)
            # Separate source-guided diagnostic availability; never supplies query entities.
            supporting=set(cases[p['case_id']].get('support_titles',[]))
            packet=dict(p,results=results,source_guided_corpus_docs=[d for d in corpus if d['title'] in supporting],
                product_success_counted=False,review_status='pending_source_audit')
            packets.append(packet)
    assert count<=72
    save('diagnostic-packets.jsonl',packets,True)
    print(json.dumps({'stage':'diagnose','packets':len(packets),'alternative_searches':count}),flush=True)

def array_rows(stream):
    decoder=json.JSONDecoder();buf='';end=False
    while True:
        buf=buf.lstrip(' \t\r\n[,')
        if buf.startswith(']'):return
        try:
            value,n=decoder.raw_decode(buf)
        except json.JSONDecodeError:
            chunk=stream.read(65536)
            if not chunk:
                if not buf.strip():return
                raise ValueError('Incomplete JSON source')
            buf+=chunk;continue
        yield value;buf=buf[n:]

def source_rows(name):
    import prepare_v133_goal_z_multihop_data as s
    if name=='2wiki':
        with zipfile.ZipFile(s.WIKI) as z, z.open('train.json') as f, io.TextIOWrapper(f,encoding='utf8') as text:
            yield from array_rows(text)
    elif name=='musique':
        with zipfile.ZipFile(s.MUSIQUE) as z,z.open('data/musique_ans_v1.0_train.jsonl') as f:
            for line in f:
                if line.strip():yield json.loads(line)
    else:
        import pyarrow.parquet as pq
        for p in sorted((ROOT/'data/hotpot-full-train-v114').glob('*.parquet')):
            for batch in pq.ParquetFile(p).iter_batches(batch_size=128):yield from batch.to_pylist()

def pilot():
    cfg=guard()
    import prepare_v133_goal_z_multihop_data as s
    import prepare_v133_goal_ah_confirmation as isolation
    ids,questions,titles,entities,passages,sentences,files,excluded=isolation.registry()
    # AH was skipped by the historical helper. Only hash/index its source fields;
    # never return sealed question/answer text to the agent or inference.
    def collect(v):
        if isinstance(v,list):
            for x in v:collect(x)
        elif isinstance(v,dict):
            for k in ['id','case_id','_id']:
                if isinstance(v.get(k),(str,int)):ids.add(str(v[k]));ids.add(str(v[k]).split(':',1)[-1])
            ids.update(map(str,v.get('source_ids',[])))
            if isinstance(v.get('question'),str):questions.add(s.w.qhash(v['question']))
            if isinstance(v.get('title'),str):titles.add(s.norm(v['title']))
            for k in ['text','paragraph_text']:
                if isinstance(v.get(k),str):passages.add(isolation.digest_text(v[k]));sentences.update(isolation.fingerprints(v[k]))
            if v.get('source') in ['musique','2wiki'] and isinstance(v.get('chain'),list):entities.update(isolation.case_entities(v))
            for k,x in v.items():
                if k not in ['prompt_token_ids','action_token_ids','actual_input_token_ids','offset_mapping','old_logprobs']:collect(x)
    for p in AH.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl']:
            for v in (rows(p) if p.suffix=='.jsonl' else [read(p)]):collect(v)
    hist=s.w.history_exposure();ids.update(hist[0]);questions.update(hist[1]);titles.update(hist[2])
    pool=[];counts={};reject=Counter()
    for name in ['2wiki','musique','hotpot']:
        eligible=[];scanned=0
        for raw in itertools.islice(source_rows(name),4096):
            scanned+=1
            if name=='hotpot':
                if raw.get('type')!='bridge' or str(raw.get('answer','')).lower() in ['yes','no']:continue
                support=s.w.official_support_docs(raw)
                if len(support)!=2:continue
                st={d['title'] for d in support};ctx=raw['context']
                docs=[dict(title=t,text=' '.join(body),support=t in st) for t,body in zip(ctx['title'],ctx['sentences'])]
                c=dict(id='hotpot:'+raw['id'],source=name,source_ids=[raw['id']],question=raw['question'],answer=raw['answer'],answer_aliases=[],chain=[],docs=docs,kind='bridge')
            else:
                if not s.source_eligible(name,raw):continue
                c=s.canonical(name,raw)
            reason=s.audit_case(c,(ids,questions,titles)) if name!='hotpot' else (
                'historical_question_or_source_id' if c['id'] in ids or raw['id'] in ids or s.w.qhash(c['question']) in questions else None)
            supports=[d for d in c['docs'] if d['support']]
            if not reason and any(s.norm(d['title']) in titles for d in supports):reason='historical_support_title'
            if not reason and isolation.case_entities(c)&entities:reason='historical_entity_chain'
            if not reason and any(isolation.digest_text(d['text']) in passages or isolation.fingerprints(d['text'])&sentences for d in supports):reason='historical_support_passage'
            if reason:reject[name+':'+reason]+=1;continue
            eligible.append(c)
        selected=sorted(eligible,key=lambda c:key('goal-ai-pool|'+c['id']))[:48]
        counts[name]=dict(scanned=scanned,eligible=len(eligible),frozen=len(selected));pool+=selected
    save('candidate-pool.jsonl',pool,True)
    corpus=[];seen=set()
    for c in sorted(pool,key=lambda c:key(c['id'])):
        for d in c['docs']:
            identity=(s.norm(d['title']),d['text'])
            if identity in seen:continue
            seen.add(identity);corpus.append(dict(id=key('ai-pilot|'+d['title']+'|'+d['text'])[:24],title=d['title'],text=d['text']))
    save('pilot-corpus.json',corpus)
    by={n:sorted([c for c in pool if c['source']==n],key=lambda c:key('goal-ai-material|'+c['id'])) for n in counts}
    new=[]
    for i in range(16):
        for name in ['2wiki','musique','hotpot']:
            if i<len(by[name]) and len(new)<16:new.append(by[name][i])
    old=rows(ROOT/'runs/v133-goal-ag-failure-repair/repair-material-final-ag.jsonl');assert len(old)==16
    old_cases={c['id']:c for c in rows(Z/'selected-train.jsonl')}
    selected=[dict(old_cases[r['case_id']],origin='old_real_failure',material=r,corpus=str((Z/'corpus-train.json').relative_to(ROOT))) for r in old]
    selected += [dict(c,origin='new_official_train',corpus=str((OUT/'pilot-corpus.json').relative_to(ROOT))) for c in new]
    assert len(selected)<=32 and not ({c['id'] for c in new}&ids)
    save('pilot-cases.jsonl',selected,True)
    save('pool-summary.json',dict(counts=counts,rejections=dict(reject),corpus_documents=len(corpus),new_cases=len(new),old_cases=16,
        isolation='historical registry plus AH fingerprint-only source fields',source_files={str(s.MUSIQUE):sha(s.MUSIQUE),str(s.WIKI):sha(s.WIKI)},
        sealed_model_calls=0,no_pool_expansion=True))
    import curate_v133_goal_z_multihop as curator
    search=tool(corpus);teacher=[];counter=0
    # Old cases preserve their existing real observations, with fresh student
    # trajectories collected separately. New Hotpot has no trustworthy chain.
    for c in new:
        if c['source']=='hotpot':teacher.append(dict(case_id=c['id'],status='draft_no_official_relation_chain'));continue
        t,reason=curator.teacher(c,search);counter+=2 if t or reason not in ['hidden_answer_in_first_query','first_support_not_retrieved','second_support_already_retrieved','bridge_not_visible'] else 1
        teacher.append(dict(case_id=c['id'],status='retrieval_candidate_only' if t else 'rejected_retrieval',teacher=t,reason=reason))
    assert counter<=96
    save('teacher-retrieval-drafts.jsonl',teacher,True)
    print(json.dumps({'stage':'pilot','sources':counts,'cases':len(selected),'new':len(new)}),flush=True)

def recover_pilot():
    """Resume only the failed assembly using already-frozen candidate files."""
    cfg=guard()
    auth=read(OUT/'pilot-recovery-authorization-v2.json')
    assert auth['recover_partial_pilot'] is True and auth['rerun_source_scan'] is False
    pool=rows(OUT/'candidate-pool.jsonl');assert len(pool)<=144
    assert len(read(OUT/'pilot-corpus.json'))==1562
    old=rows(ROOT/'runs/v133-goal-ag-failure-repair/repair-material-final-ag.jsonl');assert len(old)==16
    zcases={c['id']:c for c in rows(Z/'selected-train.jsonl')}
    ac_path=ROOT/'runs/v133-goal-ac-failure-state-repair/extra-candidate-pool.json'
    ac_data=read(ac_path);ac_cases={c['id']:c for c in ac_data['cases']}
    ag_path=ROOT/'runs/v133-goal-ag-failure-repair/supplemental/candidate-pool.jsonl'
    ag_cases={c['id']:c for c in rows(ag_path)}
    maps={
      'musique:2hop__202781_126055':(ac_cases,ROOT/'runs/v133-goal-ac-failure-state-repair/corpus-extra-train.json','AC'),
      '2wiki:d7b540640bda11eba7f7acde48001122':(ac_cases,ROOT/'runs/v133-goal-ac-failure-state-repair/corpus-extra-train.json','AC'),
      'musique:2hop__104341_411087':(ag_cases,ROOT/'runs/v133-goal-ag-failure-repair/supplemental/corpus.json','AG'),
      'musique:2hop__407314_80178':(ag_cases,ROOT/'runs/v133-goal-ag-failure-repair/supplemental/corpus.json','AG'),
      '2wiki:b2addc160bdc11eba7f7acde48001122':(ag_cases,ROOT/'runs/v133-goal-ag-failure-repair/supplemental/corpus.json','AG'),
      'musique:2hop__86360_64214':(ag_cases,ROOT/'runs/v133-goal-ag-failure-repair/supplemental/corpus.json','AG')}
    selected=[];mapping=[]
    for material in old:
        cid=material['case_id']
        if cid in zcases:
            c=zcases[cid];corpus=Z/'corpus-train.json';origin='old_real_failure_z'
        else:
            assert cid in maps,cid
            source_cases,corpus,label=maps[cid];c=source_cases[cid];origin='old_real_failure_'+label.lower()
            assert c['id']==cid and c['question']==material['question']
            support=c.get('support_docs',c.get('docs',[]))
            support=[d for d in support if d.get('support',True)]
            assert len(support)==2,(cid,len(support))
        assert corpus.is_file()
        # Original reviewed search evidence must include source text for the bridge and answer.
        assert material.get('visible_documents')
        selected.append(dict(c,origin=origin,material=material,corpus=str(corpus.relative_to(ROOT))))
        mapping.append(dict(case_id=cid,origin=origin,corpus=str(corpus.relative_to(ROOT)),
                            corpus_sha256=sha(corpus),question_matches=True,visible_documents=len(material['visible_documents'])))
    new=[]
    by={n:sorted([c for c in pool if c['source']==n],key=lambda c:key('goal-ai-material|'+c['id'])) for n in ['2wiki','musique','hotpot']}
    for i in range(16):
        for name in ['2wiki','musique','hotpot']:
            if i<len(by[name]) and len(new)<16:new.append(by[name][i])
    pilocorpus=OUT/'pilot-corpus.json'
    selected += [dict(c,origin='new_official_train',corpus=str(pilocorpus.relative_to(ROOT))) for c in new]
    assert len(selected)==32 and len({c['id'] for c in selected})==32
    save('old-case-source-map-v2.json',mapping)
    save('pilot-cases-v2.jsonl',selected,True)
    save('pool-summary-v2.json',dict(counts=auth['frozen_source_counts'],recovered_from='partial pilot assembly',
        corpus_documents=len(read(pilocorpus)),new_cases=len(new),old_cases=16,training_updates=0,
        source_files={str(p.relative_to(ROOT)):sha(p) for p in [ac_path,ROOT/'runs/v133-goal-ac-failure-state-repair/corpus-extra-train.json',ag_path,ROOT/'runs/v133-goal-ag-failure-repair/supplemental/corpus.json']}))
    import curate_v133_goal_z_multihop as curator
    search=tool(read(pilocorpus));teacher=[];counter=0
    for c in new:
        if c['source']=='hotpot':teacher.append(dict(case_id=c['id'],status='draft_no_official_relation_chain'));continue
        t,reason=curator.teacher(c,search);counter+=2 if t or reason not in ['hidden_answer_in_first_query','first_support_not_retrieved','second_support_already_retrieved','bridge_not_visible'] else 1
        teacher.append(dict(case_id=c['id'],status='retrieval_candidate_only' if t else 'rejected_retrieval',teacher=t,reason=reason))
    assert counter<=96
    save('teacher-retrieval-drafts-v2.jsonl',teacher,True)
    print(json.dumps({'stage':'pilot-recovery-v2','old':16,'new':len(new),'total':len(selected),'teacher_queries':counter}),flush=True)

def gpu():
    cfg=guard()
    if (OUT/'gpu-start.json').exists():raise FileExistsError('GPU inference already attempted; no retry')
    assert sha(SOURCE/'adapter_model.safetensors')==EXPECTED
    assert sha(BASE/'model.safetensors-00001-of-00001.safetensors')==cfg['base_sha256']
    assert sha(BASE/'tokenizer.json')==cfg['tokenizer_sha256']
    cases=rows(OUT/'pilot-cases-v2.jsonl');assert len(cases)==32
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from transformers import set_seed
    from mini_agent_rl.model import LocalModelConfig,TransformersModelClient
    from mini_agent_rl.tools import ToolRegistry
    import run_v133_goal_q_search as q
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    set_seed(42);q.load_config=lambda:{'product':cfg['product']}
    case_path=OUT/'pilot-cases-v2.jsonl'
    save('gpu-start.json',dict(cases=len(cases),calls_cap=128,searches_cap=96,model=cfg['source_adapter'],
        case_sha256=sha(case_path),corpora={c['corpus']:sha(ROOT/c['corpus']) for c in cases},
        device=torch.cuda.get_device_name(0),training_updates=0))
    client=TransformersModelClient(LocalModelConfig(model_id=str(BASE),adapter_path=str(SOURCE),device='cuda',dtype='bfloat16',
        load_in_4bit=False,max_input_tokens=2048,max_new_tokens=128,enable_thinking=False,deterministic=True,
        observation_encoding='legacy',recompute_old_logprobs=False,timeout=None))
    wrapped=q.SearchTextAliasClient(client);completed=[]
    with (OUT/'gpu-train-rollouts.jsonl').open('x',encoding='utf8') as f:
        for c in cases:
            registry=ToolRegistry();registry.register(tool(read(ROOT/c['corpus'])))
            r=asyncio.run(q.rollout(c['question'],wrapped,registry,sample=False,seed=42))
            assert r['model_calls']<=4 and r['searches']<=3
            r.update(case_id=c['id'],origin=c['origin'],corpus=c['corpus'],training_use='draft_until_source_review',
                **q.score(r['answer'],c['answer']))
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();completed.append(r)
            print(json.dumps({'stage':'gpu','completed':len(completed),'cases':len(cases)}),flush=True)
    assert sha(SOURCE/'adapter_model.safetensors')==EXPECTED
    save('gpu-summary.json',dict(trajectories=len(completed),model_calls=sum(r['model_calls'] for r in completed),
        searches=sum(r['searches'] for r in completed),training_updates=0,adapter_sha256=EXPECTED,
        em_diagnostic_only=sum(r['em'] for r in completed),device=torch.cuda.get_device_name(0)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','index','diagnose','pilot','recover-pilot','gpu']);args=parser.parse_args()
    try:globals()[args.stage.replace('-','_')]()
    except Exception as e:
        if OUT.exists() and not (OUT/'failure.json').exists():save('failure.json',dict(stage=args.stage,type=type(e).__name__,message=str(e),automatic_retry=False))
        raise
if __name__=='__main__':main()
