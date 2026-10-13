"""Deduplicate CPU evidence reviews, never changing any model or product run."""
import argparse,json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/v133-goal-ah-entity-relation'
def rows(p):return list(map(json.loads,p.read_text(encoding='utf8').splitlines()))
def semantic_key(r):
    # Only content that was actually visible in the final call is eligible.
    evidence=sorted({(d['document_id'],d['text']) for d in r['calls'][-1]['documents']})
    return hashlib.sha256(json.dumps([r['case_id'],r['answer'],evidence],ensure_ascii=False).encode()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);args=parser.parse_args()
    known=set()
    for f in OUT.glob('actual-inputs-*-baseline.jsonl'):known.update(semantic_key(r) for r in rows(f))
    for label in ['42-A','43-A','42-B','43-B']:
        if label==args.model:break
        for f in OUT.glob(f'actual-inputs-*-{label}.jsonl'):known.update(semantic_key(r) for r in rows(f))
    result=[]
    for f in sorted(OUT.glob(f'actual-inputs-*-{args.model}.jsonl')):
        for r in rows(f):
            key=semantic_key(r)
            if key in known:continue
            result.append(dict(case_id=r['case_id'],model=r['model'],split=r['split'],question=r['question'],answer=r['answer'],semantic_review_key=key,
                calls=[{k:c[k] for k in ['call_index','actual_prompt','prompt_sha256','documents','parsed_action','offset_mapping']} for c in r['calls']]))
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result)
    path=OUT/f'novel-evidence-{args.model}.jsonl'
    if path.exists():assert path.read_text(encoding='utf8')==data
    else:path.write_text(data,encoding='utf8')
    print(json.dumps(dict(model=args.model,novel=len(result),by_split={s:sum(r['split']==s for r in result) for s in sorted({r['split'] for r in result})})))
if __name__=='__main__':main()
