"""Decode existing AH outputs on CPU. Never generates model or oracle outputs."""
import argparse,json,hashlib
from pathlib import Path
from prepare_v133_goal_ah import ROOT,OUT,rows,write_once,documents

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--split',required=True)
    args=parser.parse_args();cfg=json.loads((OUT/'execution-freeze.json').read_text(encoding='utf8'))
    spec=next(s for s in cfg['evaluation'] if s['split']==args.split)
    raw=rows(OUT/f'eval-{args.split}-{args.model}.jsonl');cases={r['case_id']:r for r in rows(ROOT/spec['cases'])}
    if len(raw)!=spec['count']:raise ValueError('Review only a completed frozen split')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen_08b',local_files_only=True)
    result=[]
    for r in raw:
        calls=[]
        for index,a in enumerate(r['trace']):
            ids=a['prompt_token_ids'];text=tokenizer.decode(ids,skip_special_tokens=False)
            encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            if encoded['input_ids']!=ids:raise ValueError('Actual token roundtrip failed')
            ds=documents(text)
            for d in ds:
                start,end=d['input_char_span'];positions=[i for i,(x,y) in enumerate(encoded['offset_mapping']) if y>start and x<end]
                d['input_token_span']=[min(positions),max(positions)+1] if positions else None
            calls.append(dict(call_index=index,actual_input_tokens=len(ids),prompt_token_ids=ids,actual_prompt=text,
                prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),documents=ds,
                parsed_action={k:a.get(k) for k in ['action','query','answer','parse_error']},raw_action=a['raw'],
                offset_mapping=encoded['offset_mapping'],search_observation=a.get('observation')))
        result.append(dict(case_id=r['case_id'],model=args.model,split=args.split,question=r['question'],answer=r['answer'],
            official_answer=cases[r['case_id']]['answer'],support_titles=cases[r['case_id']].get('support_titles',[]),
            calls=calls,em=r['em'],f1=r['f1']))
    dest=OUT/f'actual-inputs-{args.split}-{args.model}.jsonl';write_once(dest,result,jsonl=True)
    print(json.dumps({'file':str(dest),'cases':len(result),'cpu_only':True,'model_calls_added':0}))

if __name__=='__main__':main()
