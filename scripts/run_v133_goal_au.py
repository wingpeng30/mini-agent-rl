"""Sequential AU: frozen external reranker, no corpus or query augmentation."""
from __future__ import annotations
import argparse,asyncio,json,os,time
from pathlib import Path
import run_v133_goal_at as at
from goal_au_reranker import FrozenReranker,FeasibilityBound
ap=at.ap;ROOT=at.ROOT;OUT=ROOT/'runs/v133-goal-au-frozen-reranking-v1'
rows,read,sha,write=at.rows,at.read,at.sha,at.write
MODEL=OUT/'reranker';REPO='BAAI/bge-reranker-base'

def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    status=read(at.OUT/'final-status.json')
    if not status.get('completed'):raise RuntimeError('AT must close before AU')
    OUT.mkdir()
    parent=Path(status['research_parent_adapter']);weight=parent/'adapter_model.safetensors'
    if sha(weight)!=status['research_parent_sha256']:raise RuntimeError('AU parent identity')
    original=rows(ap.am.AL/'trajectory-B.jsonl');corpus=read(ap.Z/'corpus-dev.json')
    if len(original)!=64 or len(corpus)!=2048:raise RuntimeError('AU source bounds')
    from mini_agent_rl.tools.search import _BM25Index
    index=_BM25Index(corpus);queries={};mapping=[]
    for case in original:
        for trace in case['trace']:
            if trace.get('action')!='search' or 'observation' not in trace:continue
            q=trace['query'];candidates=index.search(q,20)
            if candidates[:3]!=trace['observation']['result']:raise RuntimeError('AU offline BM25 mismatch')
            queries.setdefault(q,dict(query=q,candidates=candidates));mapping.append(dict(case_id=case['case_id'],step=trace['step'],query=q))
    if len(queries)>194:raise RuntimeError('AU original-query cap')
    sources=rows(ap.am.AL/'review-packets.jsonl');approved=read(ap.am.AL/'manifest.json')['admitted']
    corrections={(r['case_id'],r['edge_id']):r for r in read(ap.AM/'citation-corrections.json')}
    documents={d['id']:d for d in corpus};edges=[]
    for case in sources:
        if case['case_id'] not in approved:continue
        for e in case['edges']:
            c=corrections.get((case['case_id'],e['edge_id']),e);doc=documents[c['document_id']]
            if c['quote'] not in doc['text']:raise RuntimeError('AU prior approved body quote absent')
            edges.append(dict(case_id=case['case_id'],edge_id=e['edge_id'],subject=e['subject'],relation=e['relation'],object=e['object'],
                document_id=c['document_id'],quote=c['quote'],body_sha256=__import__('hashlib').sha256(doc['text'].encode()).hexdigest()))
    if len(edges)!=58:raise RuntimeError('AU prior relationship coverage')
    write(OUT/'offline-queries.jsonl',list(queries.values()),jsonl=True);write(OUT/'query-cases.jsonl',mapping,jsonl=True);write(OUT/'approved-edges.jsonl',edges,jsonl=True)
    write(OUT/'gpu-usage.jsonl',[],jsonl=True)
    paths=dict(runner=Path(__file__),reranker=ROOT/'scripts/goal_au_reranker.py',tests=ROOT/'tests/test_v133_goal_au_reranker.py',
      parent=weight,at_status=at.OUT/'final-status.json',corpus=ap.Z/'corpus-dev.json',cases=ap.AM/'evaluation-cases.jsonl',
      original=ap.am.AL/'trajectory-B.jsonl',queries=OUT/'offline-queries.jsonl',mapping=OUT/'query-cases.jsonl',edges=OUT/'approved-edges.jsonl',
      model_client=ROOT/'src/mini_agent_rl/model/local.py',q_runner=ROOT/'scripts/run_v133_goal_q_search.py',prompts=ROOT/'src/mini_agent_rl/prompts.py')
    write(OUT/'manifest.json',dict(parent_adapter=str(parent),parent_sha256=sha(weight),product=at.PC,
      reranker_repo=REPO,bounds=dict(query_tokens=64,doc_window=384,stride=256,max_windows=8,bm25_candidates=20,return_docs=3,batch=8),
      limits=dict(generations=512,searches=384,offline_unique_queries=194,offline_pairs=194*20*8),
      paths={k:str(v) for k,v in paths.items()},hashes={k:sha(v) for k,v in paths.items()},
      offline_eligibility=dict(independent_promotions=4,complete_chain_net_decline_allowed=0)))
    write(OUT/'PLAN.md','''# AU frozen retrieval contrast

Same accepted parent, original direct-v090/first-json/legacy JSON, 16384/2048, 4 calls/3 searches, unchanged2048-document corpus. A BM25-v2 top3; B frozen BAAI/bge-reranker-base rescoring BM25 top20, return top3 FULL bodies and original BM25 metadata. No summaries, query rewrites, relation labels, hidden bridges or auxiliary scores in input.
Offline sources: only existing GoalZ AL wide-window queries, maximum194 unique,20 documents and8 windows/doc. Query <=64 reranker tokens; document windows <=384 with stride256 and exact end window. Oversized query/document closes feasibility, not silent truncation. Maximum window logit per document, ties original BM25 order thenID. Model official revision, files and hashes frozen before GPU scoring; singleGPU BF16/no quantization. Reranker cost reported separately.
Need >=4 independent confirmed necessary-document promotions from rank4..20 into3; no net decline in complete-chain returned cases using original independent query sequences. Only then A/B64 real search, maximum512 generation attempts/384 searches. Parent promotion requires >=2 supported gains on29, no errors/actions increase, retainRapana/duke and <=1 lost of7 protected. No new training or confirmation.
Identity/hash/GPU/numeric/infrastructure failure: preserve and stop whole sequence. Semantic failure/feasibility shortage: close method, no retuning or extra queries.
''')
    print(json.dumps(dict(prepared=True,queries=len(queries),edges=len(edges))),flush=True)

def frozen():
    if (OUT/'experiment-failure.json').exists():raise RuntimeError('AU failure latch')
    m=read(OUT/'manifest.json')
    for k,p in m['paths'].items():
        if sha(p)!=m['hashes'][k]:raise RuntimeError('AU frozen drift '+k)
    return m

def download():
    frozen();os.environ['HF_HUB_DISABLE_XET']='1'
    from huggingface_hub import HfApi,snapshot_download
    info=HfApi().model_info(REPO);revision=info.sha
    if not revision or len(revision)!=40:raise RuntimeError('AU official revision absent')
    write(OUT/'download-request.json',dict(repo=REPO,revision=revision,endpoint='https://huggingface.co',source='official BAAI repository'))
    snapshot_download(REPO,revision=revision,local_dir=str(MODEL),allow_patterns=['*.json','*.safetensors','sentencepiece.bpe.model','README.md'],max_workers=1)
    if not (MODEL/'model.safetensors').is_file():raise RuntimeError('AU official safetensors absent')
    files={str(p.relative_to(MODEL)):sha(p) for p in sorted(MODEL.rglob('*')) if p.is_file() and '.cache' not in p.parts}
    write(OUT/'reranker-identity.json',dict(repo=REPO,revision=revision,files=files))

def verify_model():
    identity=read(OUT/'reranker-identity.json')
    for p,h in identity['files'].items():
        if sha(MODEL/p)!=h:raise RuntimeError('AU reranker hash drift '+p)

def close_skip(reason):
    m=frozen();write(OUT/'final-status.json',dict(completed=True,accepted=False,skipped=True,reason=reason,
      research_parent_adapter=m['parent_adapter'],research_parent_sha256=m['parent_sha256'],generations=0,searches=0,updates=0))
    write(OUT/'REPORT.md','# AU closed without live evaluation\n\n'+reason+'\n\nNo generation/search/training updates; no candidate expansion or retuning. Parent unchanged.\n')

def feasibility():
    frozen();verify_model();r=FrozenReranker(str(MODEL),tokenizer_only=True);total=0
    try:
        for q in rows(OUT/'offline-queries.jsonl'):total+=len(r.pairs(q['query'],q['candidates'])[0])
    except FeasibilityBound as exc:
        write(OUT/'feasibility.json',dict(passed=False,reason=str(exc),query=q['query'],completed_pairs=total));close_skip(str(exc));return
    write(OUT/'feasibility.json',dict(passed=True,pairs=total,maximum=194*20*8))

def offline():
    m=frozen();verify_model()
    if not read(OUT/'feasibility.json')['passed']:raise RuntimeError('AU feasibility failed')
    path=OUT/'offline-ranking.jsonl'
    if path.exists():raise FileExistsError(path)
    r=FrozenReranker(str(MODEL));start=time.monotonic()
    for n,q in enumerate(rows(OUT/'offline-queries.jsonl'),1):
        top,record=r.rank(q['query'],q['candidates']);ap.append(path,record)
        print(json.dumps(dict(query=n,total=len(rows(OUT/'offline-queries.jsonl')),windows=len(record['window_scores']))),flush=True)
    write(OUT/'offline-complete.json',dict(queries=n,seconds=time.monotonic()-start,pairs=sum(len(x['window_scores']) for x in r.journal)))

def gate():
    m=frozen();ranking={r['query']:r for r in rows(OUT/'offline-ranking.jsonl')};edges=rows(OUT/'approved-edges.jsonl');links=rows(OUT/'query-cases.jsonl')
    promoted=[];cases=[]
    for cid in dict.fromkeys(e['case_id'] for e in edges):
        qs=[x['query'] for x in links if x['case_id']==cid];need={e['document_id'] for e in edges if e['case_id']==cid}
        a=set();b=set()
        for q in qs:
            record=ranking[q];a.update(record['bm25_ids'][:3]);b.update(record['reranked_ids'][:3])
            for doc in need:
                if doc in record['bm25_ids'][3:20] and doc in record['reranked_ids'][:3]:
                    promoted.append(dict(case_id=cid,query=q,document_id=doc,bm25_rank=record['bm25_ids'].index(doc)+1,rerank_rank=record['reranked_ids'].index(doc)+1))
        cases.append(dict(case_id=cid,A_chain_returned=need<=a,B_chain_returned=need<=b))
    ac=sum(c['A_chain_returned'] for c in cases);bc=sum(c['B_chain_returned'] for c in cases);count=len({p['case_id'] for p in promoted})
    passed=count>=4 and bc>=ac
    write(OUT/'offline-gate.json',dict(passed=passed,independent_promotions=count,promoted=promoted,A_complete=ac,B_complete=bc,cases=cases))
    if not passed:close_skip(f'Offline gate failed: independent promotions {count}/4, complete-chain returns A={ac}, B={bc}.')
    print(json.dumps(dict(passed=passed,independent_promotions=count,A_complete=ac,B_complete=bc)),flush=True)

def evaluate(arm):
    m=frozen();verify_model()
    if not read(OUT/'offline-gate.json')['passed']:raise RuntimeError('AU offline gate failed')
    path=OUT/f'eval-{arm}.jsonl'
    if path.exists():raise FileExistsError(path)
    if arm=='B' and not (OUT/'eval-A-complete.json').exists():raise RuntimeError('AU A beforeB')
    client,wrapped,registry=ap.q.make_client(m['parent_adapter'],deterministic=True,product_config=m['product'],model_path=ap.BASE,corpus=ap.Z/'corpus-dev.json')
    if arm=='B':
        rr=FrozenReranker(str(MODEL));bm25=registry._tools['search']._bm25_index
        def search(query):
            candidates=bm25.search(query,20);top,record=rr.rank(query,candidates)
            ap.append(OUT/'live-ranking.jsonl',record);return top
        registry._tools['search'].handler=search
    generate0=wrapped.generate;execute0=registry.execute;active={'case':'unknown'}
    def attempt(kind):
        count=sum(r['kind']==kind for r in rows(OUT/'gpu-usage.jsonl'));limit=m['limits'][{'generation':'generations','search':'searches'}[kind]]
        if count>=limit:raise RuntimeError('AU quota '+kind)
        ap.append(OUT/'gpu-usage.jsonl',dict(kind=kind,arm=arm,case_id=active['case'],attempt=count+1,status='started'))
    async def generate(*args,**kwargs):attempt('generation');return await generate0(*args,**kwargs)
    async def execute(*args,**kwargs):
        if args and args[0]=='search':attempt('search')
        return await execute0(*args,**kwargs)
    wrapped.generate=generate;registry.execute=execute
    for n,case in enumerate(rows(ap.AM/'evaluation-cases.jsonl'),1):
        active['case']=case['case_id'];result=asyncio.run(ap.q.rollout(case['question'],wrapped,registry,sample=False,seed=42,product_config=m['product']))
        if result['reason']=='tool_error':raise RuntimeError('AU search infrastructure failure '+str(result['trace'][-1].get('observation')))
        result.update(case_id=case['case_id'],model=arm,official_answer=case['answer'],**ap.q.score(result['answer'],case['answer']))
        ap.append(path,result);print(json.dumps(dict(arm=arm,case=n,case_id=case['case_id'],answer=result['answer'],em=result['em'])),flush=True)
    write(OUT/f'eval-{arm}-complete.json',dict(cases=64,reranker_seconds=sum(x['elapsed_seconds'] for x in rr.journal) if arm=='B' else 0))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','download','feasibility','offline','gate','evaluate']);p.add_argument('--arm',choices=['A','B']);a=p.parse_args()
    try:
        if a.stage=='evaluate':evaluate(a.arm)
        else:globals()[a.stage]()
    except BaseException as exc:
        if OUT.exists() and not (OUT/'experiment-failure.json').exists():write(OUT/'experiment-failure.json',dict(stage=a.stage,error_type=type(exc).__name__,error=str(exc),stop_whole_sequence=True,no_auto_rerun=True))
        raise
if __name__=='__main__':main()
