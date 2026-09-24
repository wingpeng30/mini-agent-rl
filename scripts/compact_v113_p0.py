"""Compact blinded evidence for independent P0 review without scores/labels."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/v113-p0-blind-packet.jsonl'
OUT=ROOT/'reports/v113-p0-review-cases.jsonl'
KEY=ROOT/'reports/v113-p0-review-document-key.json'
CHUNKS=ROOT/'reports/v113-p0-review-chunks'
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def extract(messages,arm_key,docsmap):
    evidence=[];history=[]
    for m in messages:
        if m['role']=='assistant':
            try:
                obj=json.loads(m['content'])
                if isinstance(obj,dict):history.append({'action':obj.get('action'),'query':obj.get('query'),'answer':obj.get('answer')})
            except Exception:pass
        if m['role']=='tool':
            try:obs=json.loads(m['content'])
            except Exception:continue
            for d in obs.get('result',[]) if isinstance(obs,dict) else []:
                if not isinstance(d,dict) or not isinstance(d.get('text'),str):continue
                original=d.get('id') or sha(d['text'])[:16]
                mapping=docsmap.setdefault(arm_key,{})
                public=mapping.setdefault(original,f'doc-{len(mapping)+1}')
                item={'document_id':public,'title':d.get('title',''),'text':d['text']}
                if all(x['document_id']!=public for x in evidence):evidence.append(item)
    return history,evidence
def main():
    if OUT.exists() or KEY.exists() or CHUNKS.exists():raise FileExistsError('refuse_to_overwrite_v113_p0_review')
    packets=read(SOURCE);compact=[];docmap={}
    for packet in packets:
        if 'blind_case_id' in packet:
            arms={}
            for label in ('A','B'):
                events=[];doc_key=packet['blind_case_id']+':'+label
                for event in packet[label]:
                    history,evidence=extract(event['input_messages'],doc_key,docmap)
                    retrieved=[]
                    for d in event['retrieved_after_action']:
                        original=d.get('id') or sha(d.get('text',''))[:16]
                        mapping=docmap.setdefault(doc_key,{})
                        public=mapping.setdefault(original,f'doc-{len(mapping)+1}')
                        retrieved.append({'document_id':public,'title':d.get('title',''),'text':d.get('text','')})
                    action=event['action']
                    events.append({'step':event['step'],'history':history,'current_evidence':evidence,
                      'action':action,'retrieved_after_action':retrieved,
                      'budget_audit':event['budget_audit']})
                arms[label]=events
            compact.append({'blind_case_id':packet['blind_case_id'],'question':packet['question'],
              'A':arms['A'],'B':arms['B'],
              'instructions':'Use only this question and current_evidence for answer support. retrieved_after_action is visible only after that event and cannot justify its action retrospectively. Judge each action on its own: whether a search was needed, its query followed from current information, and returned material answered a specific gap. For answers, judge whether current visible documents entail the requested fact and scope. Return exact verbatim evidence quotes with event, source-before/after and public document_id. Separate scoring expression differences from evidence correctness. Do not use outside knowledge, hidden labels or guessed gold.'})
        else:
            doc_key=packet['blind_state_id']
            _,evidence=extract(packet['current_messages'],doc_key,docmap)
            compact.append({'blind_state_id':packet['blind_state_id'],'question':packet['question'],
             'current_evidence':evidence,'A':packet['A'],'B':packet['B'],
             'instructions':packet['instructions']+' Cite exact current evidence by public document_id.'})
    OUT.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in compact),encoding='utf-8')
    KEY.write_text(json.dumps(docmap,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    CHUNKS.mkdir()
    for i in range(3):
        lo=i*21;hi=min(len(compact),(i+1)*21)
        (CHUNKS/f'chunk-{i}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in compact[lo:hi]),encoding='utf-8')
    print(json.dumps({'cases':len(compact),'bytes':OUT.stat().st_size,'chunk_counts':[len(read(CHUNKS/f'chunk-{i}.jsonl')) for i in range(3)],'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
