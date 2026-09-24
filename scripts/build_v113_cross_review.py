"""Create blinded semantic review packets for the fixed-input cross replay."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports'
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def extract(messages,key,docmap):
    evidence=[];history=[]
    for m in messages:
        if m['role']=='assistant':
            try:
                d=json.loads(m['content']);history.append({k:d.get(k) for k in ('action','query','answer')})
            except Exception:pass
        if m['role']=='tool':
            try:obs=json.loads(m['content'])
            except Exception:continue
            for d in obs.get('result',[]) if isinstance(obs,dict) else []:
                if isinstance(d,dict) and isinstance(d.get('text'),str):
                    original=d.get('id') or h(d['text'])[:16];mapping=docmap.setdefault(key,{})
                    public=mapping.setdefault(original,f'doc-{len(mapping)+1}')
                    if not any(x['document_id']==public for x in evidence):evidence.append({'document_id':public,'title':d.get('title',''),'text':d['text']})
    question=next((m['content'] for m in reversed(messages) if m['role']=='user'),'')
    return question,history,evidence
def main():
    src=R/'v113-cross-results-stable.jsonl';rows=read(src)
    if len(rows)!=340 or len({(x['input_id'],x['arm']) for x in rows})!=340:raise ValueError('complete_170_by_2_required')
    grouped={}
    for row in rows:grouped.setdefault(row['input_id'],{})[row['arm']]=row
    if len(grouped)!=170 or any(set(x)!={'baseline','candidate'} for x in grouped.values()):raise ValueError('paired_coverage_required')
    packets=[];keys={};docmap={}
    for input_id,pair in sorted(grouped.items()):
        baseline=pair['baseline'];messages=next(x['messages'] for x in read(R/'v113-cross-inputs.jsonl') if x['input_hash']==input_id)
        question,history,evidence=extract(messages,'state-'+h(input_id)[:12],docmap)
        stateid='cross-'+h('v113-p1:'+input_id)[:16];swap=int(h('v113-p1-order:'+input_id)[0],16)%2==1
        A,B=(pair['candidate'],baseline) if swap else (baseline,pair['candidate'])
        def pub(record):
            retrieval=[]
            for doc in record['retrieval_results']:
                original=doc.get('id');mapping=docmap.setdefault('state-'+h(input_id)[:12],{})
                public=mapping.setdefault(original,f'doc-{len(mapping)+1}')
                retrieval.append({'document_id':public,'title':doc.get('title',''),'text':doc.get('text',''),
                    'rank':doc.get('rank'),'previously_used':doc.get('previously_used'),'new_document':doc.get('new_document')})
            return {'action':record['action_parsed'],'valid_json_action':record['valid_json_action'],
                'output_truncated':record['output_truncated'],'parse_error':record['parse_error'],
                'top3_retrieval_after_action':retrieval}
        packets.append({'blind_input_id':stateid,'question':question,'history':history,'current_evidence':evidence,
          'A':pub(A),'B':pub(B),
          'instructions':'This is a fixed, common model input from a trajectory. Evaluate both anonymous single-step actions using only the question, history and current_evidence. Do not use outside knowledge, training labels or answer keys. For each: judge whether answer is entailed by current evidence; if search, whether justified by current information and query is grounded; judge whether top3 actually addresses a specific gap, not merely contains a new document. For answer, citations must come only from current_evidence; cite retrieval output only for retrieval-gap fit. Distinguish what this fixed-input comparison can establish from full closed-loop behavior.'})
        keys[stateid]={'input_id':input_id,'baseline_origin':baseline['origin'],'candidate_origin':pair['candidate']['origin'],'A_arm':'candidate' if swap else 'baseline','B_arm':'baseline' if swap else 'candidate'}
    for name in ('v113-p1-blind-review.jsonl','v113-p1-private-key.json','v113-p1-review-document-key.json'):
        if (R/name).exists():raise FileExistsError('refuse_to_overwrite:'+name)
    out=R/'v113-p1-blind-review.jsonl';out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in packets),encoding='utf-8')
    (R/'v113-p1-private-key.json').write_text(json.dumps(keys,indent=2)+'\n',encoding='utf-8')
    (R/'v113-p1-review-document-key.json').write_text(json.dumps(docmap,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for i,(lo,hi) in enumerate(((0,57),(57,114),(114,170))):
        p=R/f'v113-p1-review-{i}.jsonl'
        if p.exists():raise FileExistsError(str(p))
        p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in packets[lo:hi]),encoding='utf-8')
    print(json.dumps({'pairs':len(packets),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
