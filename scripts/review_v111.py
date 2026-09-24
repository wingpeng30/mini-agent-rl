"""Auxiliary blinded review of changed and retained-supported-but-scoring-wrong answers."""
import argparse,json
import threading
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from run_v111 import API,verify,build_evidence,code_hash
from run_v110 import read_jsonl,save,digest
from mini_agent_rl.hybrid_v110 import SQLiteAPILedger,answer_metrics
from mini_agent_rl.training.thinking_sft import _env

def review_requests(inputs,results,arms):
    by={r['task_id']:r for r in inputs}; requests={}; expected={a:[] for a in arms}
    for result in results:
        source=by[result['task_id']]
        for arm in arms:
            answer=result.get('answers',{}).get(arm,source['original_answer'])
            changed=answer!=source['original_answer']
            support=any(r.get('assessments',{}).get('original',{}).get('valid') and
                        r['assessments']['original']['status']=='supported' for r in result.get('reviews',[]))
            if arm=='D':
                d=result.get('direct_call',{}).get('parsed',{})
                support=d.get('valid') and d.get('decision')=='answer'
            mismatch=not answer_metrics(answer,source['gold'])['normalized_em']
            if not changed and not (support and mismatch):continue
            docs=result.get('evidence_by_arm',{}).get(arm,source['e0'])
            key=digest([source['task_id'],source['question'],source['original_answer'],answer,docs])
            expected[arm].append({'key':key,'task_id':source['task_id'],'changed':changed,
                                  'retained_scoring_mismatch':not changed})
            requests[key]={'source':source,'answer':answer,'documents':docs}
    return requests,expected

def main():
    p=argparse.ArgumentParser()
    for key in ('inputs','results','output','report','env-file'):p.add_argument('--'+key,required=True)
    p.add_argument('--config',default='configs/v111-hybrid.json');p.add_argument('--ledger',default='runs/v110-api.sqlite')
    p.add_argument('--arms',default='U,P,R,D');p.add_argument('--cap',type=float,default=60)
    args=p.parse_args();inputs=read_jsonl(args.inputs);results=read_jsonl(args.results);arms=args.arms.split(',')
    requests,expected=review_requests(inputs,results,arms)
    fp=digest({'inputs':inputs,'results':results,'arms':arms,'code':code_hash(),
               'review_code':Path(__file__).read_text(encoding='utf-8')})
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);manifest=Path(args.output+'.manifest.json')
    if path.exists() and not manifest.exists():raise RuntimeError('missing_review_manifest')
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))['fingerprint']!=fp:raise RuntimeError('changed_review')
    save(manifest,{'fingerprint':fp,'expected':expected})
    rows=read_jsonl(path) if path.exists() else [];done={r['key'] for r in rows}
    if len(done)!=len(rows) or done-set(requests):raise RuntimeError('review_identity_conflict')
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'));ledger=SQLiteAPILedger(args.ledger,total_cap=60)
    api=API(cfg,ledger,_env(Path(args.env_file))['DEEPSEEK_API_KEY'],'v111-review',args.cap)
    try:
        stop=threading.Event()
        def work(key,item):
            owned=SQLiteAPILedger(args.ledger,total_cap=min(60,args.cap))
            worker=API(cfg,owned,api.key,'v111-review',args.cap);worker.stop_event=stop
            try:
                review,call=verify(worker,item['source'],item['answer'],build_evidence(item['documents']),kind='blind-review',blind=True)
                return {'key':key,'task_id':item['source']['task_id'],'original':item['source']['original_answer'],
                     'answer':item['answer'],'audit':review,'call':call,'status':'ok' if review['valid'] else 'unresolved'}
            except Exception as exc:
                stop.set();return {'key':key,'task_id':item['source']['task_id'],'status':'infrastructure_error','error':str(exc)}
            finally:worker.client.close();owned.close()
        remaining=iter([(k,v) for k,v in requests.items() if k not in done])
        with ThreadPoolExecutor(max_workers=cfg['question_workers']) as pool:
            pending=set()
            def enqueue():
                while not stop.is_set() and len(pending)<cfg['question_workers']:
                    item=next(remaining,None)
                    if item is None:break
                    pending.add(pool.submit(work,*item))
            enqueue()
            while pending:
                complete,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in complete:
                    row=future.result()
                    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    rows.append(row);print(f'review {len(rows)}/{len(requests)} {row["status"]}',flush=True)
                enqueue()
        by={r['key']:r for r in rows};reports={}
        for arm,items in expected.items():
            unresolved=[];risks=[];uncertain=[]
            for e in items:
                r=by.get(e['key'],{});a=r.get('audit',{}).get('assessments',{})
                if r.get('status')!='ok':unresolved.append(e['task_id']);continue
                if e['changed'] and a['original']['status']=='supported' and a['selected']['status']!='supported':risks.append(e['task_id'])
                if e['changed'] and a['selected']['status']!='supported':uncertain.append(e['task_id'])
            reports[arm]={'expected':len(items),'changed':sum(x['changed'] for x in items),
                 'retained_supported_scoring_mismatch':sum(not x['changed'] for x in items),
                 'unresolved_protocol':unresolved,'possible_semantic_regressions':risks,
                 'changed_without_confirmed_support':uncertain,
                 'auxiliary_gate':not unresolved and not risks and not uncertain}
        save(args.report,{'version':'v111','fingerprint':fp,'unique_pairs':len(rows),'arms':reports,
             'same_model_auxiliary_only':True,'human_adjudication':False,'infrastructure_errors':sum(r['status']=='infrastructure_error' for r in rows),'expected':expected})
        print(json.dumps(reports,ensure_ascii=False,indent=2))
    finally:api.client.close();ledger.close()

if __name__=='__main__':main()
