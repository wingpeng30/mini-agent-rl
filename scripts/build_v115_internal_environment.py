"""Freeze a 50-question internal task set and corpus from official train rows only."""
from __future__ import annotations
import hashlib, json, sys, unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/"data/evidence-sft-v114-review"
OUT=ROOT/"data/evidence-sft-v115"
sys.path.insert(0,str(ROOT/"scripts"))
from build_v114_sft_candidates import load_full, documents_for
from finalize_v114_reviews import select_cohort

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def norm(s):
    s=unicodedata.normalize('NFKC',s).casefold()
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in s).split())
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def jsonl(path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def main():
    gate=read(ROOT/'reports/v114-review-gate.json')
    if gate.get('passed') is not True or gate.get('training_permitted') is not True: raise ValueError('v114_gate_not_passed')
    if gate.get('final_test_accessed') or gate.get('independent100_evaluated'): raise ValueError('forbidden_eval_access')
    manifest=read(ROOT/'reports/v114-sft-export-manifest.json')
    if manifest.get('exports',{}).get('sft-internal.jsonl',{}).get('sha256') != sha(WORK/'sft-internal.jsonl'): raise ValueError('internal_export_hash_mismatch')
    freeze=read(WORK/'internal-freeze-v3.json')
    if len(freeze.get('ids',[]))!=50 or len(set(freeze['ids']))!=50: raise ValueError('invalid_frozen_internal_ids')
    records=jsonl(WORK/'accepted.jsonl')
    internal_pool=[r for r in records if r.get('split')=='internal']
    internal=select_cohort(internal_pool,'internal',set(freeze['ids']),
                           set(gate.get('risk_category_recheck',{}).get('applied_cases',[])))
    if internal is None or len(internal)!=100 or {r['source_id'] for r in internal}!=set(freeze['ids']): raise ValueError('accepted_internal_coverage_mismatch')
    byid={}
    questions={}
    for r in internal:
        qid=r['source_id']; question=r['packet']['question']
        if qid in questions and norm(questions[qid])!=norm(question): raise ValueError('inconsistent_question_for_id')
        questions[qid]=question
    full=load_full(set(freeze['ids']))
    tasks=[]; docs={}
    for qid in freeze['ids']:
        row=full[qid]
        if norm(row['question'])!=norm(questions[qid]): raise ValueError('frozen_question_mismatch:'+qid)
        tasks.append({'id':qid,'question':row['question'],'answer':row['answer'],
                      'metadata':{'source':'hotpotqa-official-train','source_id':qid,
                                  'support_titles':list(dict.fromkeys(row['supporting_facts']['title']))}})
        for doc in documents_for(row):
            content_key=hashlib.sha256((doc['title']+'\0'+doc['text']).encode()).hexdigest()
            docs.setdefault(content_key,{'id':content_key[:20],'title':doc['title'],'text':doc['text']})
    if len(tasks)!=50 or not docs: raise ValueError('empty_internal_environment')
    OUT.mkdir(parents=True,exist_ok=True)
    task_path=OUT/'internal-tasks-private.jsonl'; corpus_path=OUT/'internal-corpus.json'
    state_path=OUT/'internal-states-private.jsonl'
    task_path.write_text(''.join(json.dumps(t,ensure_ascii=False,separators=(',',':'))+'\n' for t in tasks),encoding='utf-8')
    corpus=sorted(docs.values(),key=lambda d:d['id'])
    corpus_path.write_text(json.dumps(corpus,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    state_rows=[]
    for r in internal:
        packet=r['packet']
        state_rows.append({'state_id':r['case_id'],'task_id':r['source_id'],'question':packet['question'],
                           'messages':packet['visible_messages'],'target':packet['proposed_action'],
                           'evidence':packet.get('visible_evidence',[]),'retrieval_top3':packet.get('retrieval_top3',[])})
    state_path.write_text(''.join(json.dumps(s,ensure_ascii=False,separators=(',',':'))+'\n' for s in state_rows),encoding='utf-8')
    source_files=sorted((ROOT/'data/hotpot-full-train-v114').glob('train-*-of-*.parquet'))
    if len(source_files)!=2: raise ValueError('unexpected_official_train_shards')
    out={'version':'v115-internal-environment-v1','freeze_sha256':sha(WORK/'internal-freeze-v3.json'),
         'export_manifest_sha256':sha(ROOT/'reports/v114-sft-export-manifest.json'),
         'source_shards':{p.name:sha(p) for p in source_files},'internal_ids':list(freeze['ids']),
         'question_count':len(tasks),'document_count':len(corpus),'deduplication':'exact title+text SHA256',
         'tasks_sha256':sha(task_path),'corpus_sha256':sha(corpus_path),'states_sha256':sha(state_path),
         'scope':'50 frozen internal questions; official training source context only; no independent100/final-test'}
    (OUT/'internal-environment-manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
