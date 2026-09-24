"""Config-driven, resumable v110 experiment. One shared durable API budget."""
from __future__ import annotations
import argparse, hashlib, json, random, time
from pathlib import Path
import httpx
from mini_agent_rl.hybrid_v110 import (SQLiteAPILedger, BudgetExceeded, answer_metrics,
    normalize_answer, apply_selection, validate_citations_detailed)
from mini_agent_rl.hybrid_answer import CandidateResponse, anonymize_candidates, evidence_hash
from mini_agent_rl.tools.search import _BM25Index
from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.training.thinking_sft import _env

CANDIDATES = '''Generate at most four distinct, concise, complete answers using only the evidence. Respect the entity, relation, date precision and scope asked. Answers may be aliases or multi-document compositions. Do not include explanations in answer strings. Return JSON {"candidates":[{"answer":"...","citations":[{"document_id":"...","quote":"exact source text"}]}]}. Return an empty candidates list if none can be supported. Never invent a quotation.'''
SELECT = '''Choose the evidence-supported candidate that exactly answers the requested entity, relation, date precision and scope. Candidate order carries no information. Prefer a concise complete answer over explanations, but never discard required precision or entities. Return JSON {"decision":"select"|"abstain","candidate_id":"C1"|null,"citations":[{"document_id":"...","quote":"exact source text"}]}. Abstain if no option is supported.'''
VERIFY = '''Independently check EACH anonymous answer against the question and evidence, including entity, relation, date precision, answer scope and polarity. Quotation occurrence alone is not support. Return JSON {"assessments":[{"id":"A1","status":"supported"|"contradicted"|"insufficient","reason":"...","citations":[{"document_id":"...","quote":"exact text"}]}],"information_gap":"concrete missing fact or empty string","queries":["at most two distinct searches for missing facts"]}. Use only evidence. Correct aliases are acceptable. For supported answers provide verbatim evidence; never infer truth from answer order.'''
BLIND = '''Judge both anonymous answers using the question and evidence. Allow correct aliases, require requested entity, relation, polarity, scope and date precision. Return JSON {"assessments":[{"id":"A1","status":"supported"|"contradicted"|"insufficient","reason":"...","citations":[{"document_id":"...","quote":"exact text"}]}],"information_gap":"...","queries":[]}. Evaluate each independently, not by relative preference. Return exactly both IDs.'''

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

class ProtocolFailure(Exception): pass

class API:
    def __init__(self,config,ledger,key,stage,cap):
        self.config=config;self.ledger=ledger;self.key=key;self.stage=stage;self.cap=cap
        self.client=httpx.Client(timeout=300)
    def call(self,kind,system,payload,retry=0):
        cfg=self.config
        output=cfg['retry_output_tokens'] if retry else cfg['max_output_tokens']
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
        # UTF-8 byte count plus framing is a conservative token upper bound.
        input_bound=len(json.dumps(messages,ensure_ascii=False).encode())+256
        if input_bound+output>cfg['context_upper_bound']:
            raise ProtocolFailure('context_requires_document_chunking')
        request={'model':cfg['model'],'thinking':{'type':'enabled'},'temperature':cfg['temperature'],
                 'max_tokens':output,'response_format':{'type':'json_object'},'messages':messages}
        reserve=(input_bound*cfg['input_cny_per_million_upper']+output*cfg['output_cny_per_million_upper'])/1e6
        fingerprint={'request':request,'implementation':CODE_HASH,'kind':kind,'retry':retry}
        call_key=digest(payload)+'-'+digest(fingerprint)
        # Ledger enforces total. Stage cumulative ceilings also protect review reserve.
        old=self.ledger.db.execute('SELECT key FROM calls WHERE payload=? AND config=?',(
            json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':')),
            json.dumps(fingerprint,sort_keys=True,ensure_ascii=False,separators=(',',':')))).fetchone()
        if not old and self.ledger.summary()['total_charged']+reserve>self.cap:
            raise BudgetExceeded('cumulative_stage_cap')
        item=self.ledger.reserve(payload,fingerprint,self.stage,reserve)
        if item['status']=='cached': body=json.loads(item['cached_response'])
        elif item['status']!='reserved': raise RuntimeError('unresolved_previous_api_call:'+item['key'])
        else:
            started=time.perf_counter()
            try:
                response=self.client.post('https://api.deepseek.com/chat/completions',headers={'Authorization':'Bearer '+self.key},json=request)
                with self.ledger.db:
                    self.ledger.db.execute('UPDATE calls SET raw_response=? WHERE key=?',(response.text,item['key']))
                response.raise_for_status();body=response.json();usage=body.get('usage')
                if not usage or 'prompt_tokens' not in usage or 'completion_tokens' not in usage:
                    raise RuntimeError('missing_api_usage')
                charged=(usage['prompt_tokens']*cfg['input_cny_per_million_upper']+usage['completion_tokens']*cfg['output_cny_per_million_upper'])/1e6
                self.ledger.complete(item['key'],json.dumps(body,ensure_ascii=False),{**usage,'latency_seconds':time.perf_counter()-started},charged)
            except Exception as exc:
                self.ledger.mark_unknown(item['key'],str(exc));raise
        choice=body['choices'][0]
        if choice.get('finish_reason')!='stop': raise ProtocolFailure('finish_reason:'+str(choice.get('finish_reason')))
        content=choice['message'].get('content')
        if not content: raise ProtocolFailure('empty_output')
        try:return json.loads(content)
        except ValueError as exc:raise ProtocolFailure('invalid_json') from exc

def generate(api,row,docs):
    errors=[]
    for retry in range(2):
        try:
            raw=api.call('candidates',CANDIDATES,{'question':row['question'],'evidence':docs},retry)
            parsed=CandidateResponse.model_validate(raw)
            accepted=[];rejected=[];seen={normalize_answer(row['original_answer'])}
            for c in parsed.candidates:
                check=validate_citations_detailed(c.citations,docs);key=normalize_answer(c.answer)
                if not check['valid'] or not key:rejected.append(c.model_dump());continue
                if key not in seen:accepted.append(c.model_dump());seen.add(key)
            if rejected:raise ProtocolFailure('invalid_candidate_citations')
            return [{'answer':row['original_answer'],'citations':[]}]+accepted,{'valid':True,'accepted_alternatives':len(accepted),'attempt':retry+1,'errors':errors}
        except (ProtocolFailure,ValueError) as exc:errors.append(str(exc))
    return [{'answer':row['original_answer'],'citations':[]}],{'valid':False,'accepted_alternatives':0,'errors':errors}

def select(api,row,docs,candidates,variant=0):
    ordered=anonymize_candidates(row['task_id'],candidates,variant)
    payload={'question':row['question'],'evidence':docs,'candidates':[{'candidate_id':c['candidate_id'],'answer':c['answer']} for c in ordered]}
    try:
        raw=api.call('select',SELECT,payload)
        answer,status,error=apply_selection(json.dumps(raw),ordered,row['original_answer'],docs)
        return {'answer':answer,'status':status,'error':error,'raw':raw,'ordered_candidates':ordered}
    except ProtocolFailure as exc:return {'answer':row['original_answer'],'status':'fallback_protocol','error':str(exc)}

def verify(api,row,docs,selected,blind=False):
    pairs=[('original',row['original_answer']),('selected',selected)]
    random.Random(int(digest([row['task_id'],'blind' if blind else 'verify']),16)).shuffle(pairs)
    payload={'question':row['question'],'evidence':docs,'answers':[{'id':f'A{i+1}','answer':v[1]} for i,v in enumerate(pairs)]}
    raw=api.call('blind_review' if blind else 'verify',BLIND if blind else VERIFY,payload)
    if not isinstance(raw,dict):raise ProtocolFailure('review_object_required')
    assessments=raw.get('assessments',[])
    if len(assessments)!=2 or {a.get('id') for a in assessments}!={'A1','A2'}:raise ProtocolFailure('review_ids')
    mapped={}
    for a in assessments:
        if a.get('status') not in {'supported','contradicted','insufficient'}:raise ProtocolFailure('review_status')
        if a['status']=='supported' and not validate_citations_detailed(a.get('citations',[]),docs)['valid']:
            raise ProtocolFailure('review_invalid_citation')
        mapped[pairs[int(a['id'][1:])-1][0]]=a
    return {'assessments':mapped,'raw':raw,'order':[p[0] for p in pairs]}

def mechanism(api,row,index,position_check=False):
    if not row.get('provenance_valid') or not row.get('prompt_replay_exact') or evidence_hash(row['e0'])!=row['e0_sha256']:
        raise RuntimeError('unverified_input:'+row['task_id'])
    ids=row['prompt_token_ids'];mask=row['prompt_attention_mask']
    actual=hashlib.sha256(str(ids).encode()).hexdigest()
    if ids!=row['replayed_prompt_token_ids'] or mask!=row['replayed_prompt_attention_mask'] or len(ids)!=len(mask) or not ids:
        raise RuntimeError('token_replay_mismatch')
    if actual!=row['saved_prompt_sha256'] or actual!=row['replayed_prompt_sha256']:
        raise RuntimeError('prompt_hash_mismatch')
    if evidence_hash(visible_documents(row['replayed_messages']))!=row['e0_sha256']:
        raise RuntimeError('evidence_message_mismatch')
    docs=list(row['e0']);candidates,protocol=generate(api,row,docs)
    chosen=select(api,row,docs,candidates)
    b=chosen['answer']; trace=[];checks=[];c=b
    position=select(api,row,docs,candidates,1) if position_check else None
    for round_index in range(3):
        try:review=verify(api,row,docs,c)
        except ProtocolFailure as exc:
            checks.append({'error':str(exc)});c=row['original_answer'];break
        checks.append(review);assessment=review['assessments']['selected']
        if assessment['status']=='supported':break
        c=row['original_answer']
        if assessment['status']!='insufficient' or round_index>=api.config['supplemental_rounds']:break
        gap=review['raw'].get('information_gap');queries=review['raw'].get('queries',[])
        if not gap or not isinstance(queries,list):break
        queries=list(dict.fromkeys(q.strip() for q in queries if isinstance(q,str) and q.strip()))[:2]
        known={d['id'] for d in docs};new=[]
        for query in queries:
            hits=index.search(query,top_k=api.config['retrieval_top_k']);trace.append({'round':round_index+1,'gap':gap,'query':query,'results':hits})
            for d in hits:
                if d['id'] not in known:new.append({k:d[k] for k in ('id','title','text')});known.add(d['id'])
        if not new:break
        docs.extend(new)
        extra,extra_protocol=generate(api,row,docs)
        candidates=extra;selected=select(api,row,docs,candidates);c=selected['answer']
        checks.append({'regeneration_protocol':extra_protocol,'selection':selected})
    return {'task_id':row['task_id'],'original_answer':row['original_answer'],'B':b,'C':c,'protocol':protocol,
            'selection':chosen,'position_check':position,'initial_candidates':chosen.get('ordered_candidates',candidates),
            'checks':checks,'retrieval':trace,'final_evidence':docs,'status':'ok'}

def summarize(inputs,rows):
    ids={r['task_id'] for r in inputs}
    if len(ids)!=len(inputs) or len({r['task_id'] for r in rows})!=len(rows) or any(r['task_id'] not in ids for r in rows):
        raise ValueError('manifest_result_identity_conflict')
    by={r['task_id']:r for r in rows};metrics={};n=len(inputs)
    for arm in ('A','B','C'):
        scores=[answer_metrics(r['original_answer'] if arm=='A' else by.get(r['task_id'],{}).get(arm,r['original_answer']),r['gold']) for r in inputs]
        metrics[arm]={k:sum(s[k] for s in scores)/n for k in ('strict_em','normalized_em','f1')}
    base_correct=[r for r in inputs if answer_metrics(r['original_answer'],r['gold'])['normalized_em']]
    preserve={arm:sum(answer_metrics(by.get(r['task_id'],{}).get(arm,r['original_answer']),r['gold'])['normalized_em'] for r in base_correct)/max(1,len(base_correct)) for arm in ('B','C')}
    valid_b=lambda r:r.get('protocol',{}).get('valid',False) and not r.get('selection',{}).get('error') and r.get('status')=='ok'
    valid_c=lambda r:valid_b(r) and all(not x.get('error') and x.get('regeneration_protocol',{'valid':True}).get('valid') and not x.get('selection',{}).get('error') for x in r.get('checks',[]))
    protocol_rates={'B':sum(valid_b(r) for r in rows)/n,'C':sum(valid_c(r) for r in rows)/n}
    return {'tasks':n,'completed':len(rows),'metrics':metrics,'correct_original_preserve':preserve,'protocol_valid_rate':protocol_rates,
            'infrastructure_errors':sum(r['status']!='ok' for r in rows),'missing_tasks':n-len(rows)}

CODE_HASH=digest({p:Path(p).read_text(encoding='utf-8') for p in [__file__,str(Path(__file__).parent.parent/'src/mini_agent_rl/hybrid_v110.py')]})

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--corpus',required=True)
    p.add_argument('--config',default='configs/v110-hybrid.json');p.add_argument('--env-file',required=True)
    p.add_argument('--output',required=True);p.add_argument('--report',required=True);p.add_argument('--ledger',default='runs/v110-api.sqlite')
    p.add_argument('--stage',choices=['train','development'],default='train');p.add_argument('--limit',type=int,default=100);p.add_argument('--cap',type=float,default=20)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8'));inputs=read_jsonl(args.inputs)[:args.limit]
    corpus=json.loads(Path(args.corpus).read_text(encoding='utf-8'));index=_BM25Index(corpus)
    fingerprint=digest({'inputs':inputs,'corpus':corpus,'config':cfg,'code':CODE_HASH})
    root=Path(args.output);root.parent.mkdir(parents=True,exist_ok=True)
    manifest_path=Path(args.output+'.manifest.json')
    if root.exists() and not manifest_path.exists():raise RuntimeError('unmanifested_results')
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8'))['fingerprint']!=fingerprint:raise RuntimeError('run_fingerprint_changed')
    save(manifest_path,{'fingerprint':fingerprint,'task_ids':[r['task_id'] for r in inputs],'code':CODE_HASH,'config':cfg,'corpus_sha256':digest(corpus)})
    rows=read_jsonl(root) if root.exists() else []
    if len({r['task_id'] for r in rows})!=len(rows):raise RuntimeError('duplicate_results')
    if any(r.get('fingerprint')!=fingerprint or r['task_id'] not in {x['task_id'] for x in inputs} for r in rows):raise RuntimeError('result_fingerprint_mismatch')
    if any(r.get('status')!='ok' for r in rows):raise RuntimeError('prior_failed_run_requires_explicit_reconciliation')
    Path(args.ledger).parent.mkdir(parents=True,exist_ok=True)
    ledger=SQLiteAPILedger(args.ledger,total_cap=cfg['total_budget_cny'])
    api=API(cfg,ledger,_env(Path(args.env_file))['DEEPSEEK_API_KEY'],args.stage,args.cap)
    models=api.client.get('https://api.deepseek.com/models',headers={'Authorization':'Bearer '+api.key});models.raise_for_status()
    save(args.output+'.models.json',models.json())
    done={r['task_id'] for r in rows};started=time.perf_counter()
    try:
        for i,row in enumerate(inputs):
            if row['task_id'] in done:continue
            try:result=mechanism(api,row,index,position_check=args.stage=='train' and i<20)
            except Exception as exc:
                result={'task_id':row['task_id'],'original_answer':row['original_answer'],'B':row['original_answer'],'C':row['original_answer'],'status':'error','error':str(exc)}
            result['fingerprint']=fingerprint;rows.append(result)
            with root.open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
            report={**summarize(inputs,rows),'fingerprint':fingerprint,'api_upper_cny':ledger.summary()['total_charged'],'elapsed_seconds':time.perf_counter()-started}
            save(args.report,report);print(f"{len(rows)}/{len(inputs)} {result['status']} cost={report['api_upper_cny']:.4f}",flush=True)
            if result['status']!='ok':break
    finally:api.client.close();ledger.close()

if __name__=='__main__':main()
