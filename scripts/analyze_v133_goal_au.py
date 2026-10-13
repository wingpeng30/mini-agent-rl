"""AT mechanical evidence packets; semantic rulings supplied separately."""
import json
from pathlib import Path
import evaluate_v133_goal_at as ev
import run_v133_goal_au as au
import analyze_v133_goal_am as amreview
import finalize_v133_goal_al_report as prior
ap=ev.ap;ROOT=ev.ROOT;OUT=ROOT/'runs/v133-goal-au-frozen-reranking-v3'
au.OUT=OUT;au.MODEL=OUT/'reranker'
rows,read,write=ev.rows,ev.read,ev.write

def packets():
    manifest=au.frozen();au.verify_model()
    if manifest['parent_sha256']!=ap.AM_HASH:raise RuntimeError('AU reviewer requires current AM parent')
    tok=ap.am.tokenizer_from_base()
    parent={r['case_id']:r for r in rows(ap.AM/'eval-42-A.jsonl')}
    parent_review={r['case_id']:r for r in rows(ap.AM/'semantic-review-seed42.jsonl') if r['model']=='42-A'}
    original={r['case_id']:r for r in rows(ap.am.AL/'review-packets.jsonl')}
    corrections={(r['case_id'],r['edge_id']):r for r in read(ap.AM/'citation-corrections.json')}
    summaries={'AM':amreview.metric(list(parent.values()))};packets=[];pairs=[]
    for arm in ('A','B'):
        items=rows(OUT/f'eval-{arm}.jsonl')
        if len(items)!=64 or [x['case_id'] for x in items]!=list(parent):raise RuntimeError('AT evaluation sequence')
        summaries[arm]=amreview.metric(items)
        for row in items:
            cid=row['case_id'];base=parent[cid];final=row['trace'][-1];ids=final['prompt_token_ids']
            divergence=next((i for i,(a,b) in enumerate(zip(base['trace'],row['trace'])) if a['raw']!=b['raw'] or a['prompt_token_ids']!=b['prompt_token_ids']),None)
            pair=dict(case_id=cid,arm=arm,question=row['question'],parent_answer=base['answer'],answer=row['answer'],first_divergence=divergence,
                      parent_reason=base['reason'],reason=row['reason'],em=row['em'],f1=row['f1'],admitted=cid in parent_review)
            pairs.append(pair)
            if cid not in parent_review:continue
            edges=[]
            for edge in original[cid]['edges']:
                candidates=[(edge['document_id'],edge['quote'])]
                corr=corrections.get((cid,edge['edge_id']))
                if corr:candidates.insert(0,(corr['document_id'],corr['quote']))
                candidates.extend((e['citation']['document_id'],e['citation']['quote']) for e in parent_review[cid]['edges'] if e['edge_id']==edge['edge_id'] and e.get('citation'))
                found=next((c for d,q in candidates if (c:=prior.cite(ids,tok,d,q)) is not None),None)
                first=next((t['step'] for t in row['trace'] if any(d.get('id')==di and quote in d.get('text','') for di,quote in candidates for d in t.get('observation',{}).get('result',[]))),None)
                edges.append(dict(edge_id=edge['edge_id'],subject=edge['subject'],relation=edge['relation'],object=edge['object'],actual_visible=found is not None,citation=found,first_return_search=first))
            packets.append(dict(pair,prompt_token_ids=ids,complete_chain_visible=all(e['actual_visible'] for e in edges),edges=edges,
              identical_parent_input=ids==base['trace'][-1]['prompt_token_ids'],parent_category=parent_review[cid]['category'],
              queries=[t['query'] for t in row['trace'] if t.get('query')],trace=row['trace'],training_use_forbidden=True))
    write(OUT/'review-packets.jsonl',packets,jsonl=True);write(OUT/'paired-ledger.jsonl',pairs,jsonl=True)
    write(OUT/'mechanical-metrics.json',summaries)
    print(json.dumps(summaries),flush=True)
    for r in packets:
        print(json.dumps({k:r[k] for k in ('arm','case_id','answer','parent_answer','parent_category','reason','complete_chain_visible','identical_parent_input','queries','edges')},ensure_ascii=False),flush=True)

if __name__=='__main__':packets()

