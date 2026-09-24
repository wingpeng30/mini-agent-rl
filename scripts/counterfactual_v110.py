"""Post-hoc training diagnosis only; not eligible for v110 adoption or P2."""
import json
from pathlib import Path
from run_v110 import read_jsonl,save
from mini_agent_rl.hybrid_v110 import answer_metrics

if __name__=='__main__':
    inputs={r['task_id']:r for r in read_jsonl('backups/v110-train-inputs.jsonl')};rows=[]
    for r in read_jsonl('backups/v110-train-results.jsonl'):
        reviews=[x for x in r['checks'] if 'assessments' in x]
        preserve=bool(reviews) and reviews[0]['assessments']['original']['status']=='supported'
        answer=r['original_answer'] if preserve else r['C'];source=inputs[r['task_id']]
        rows.append({'task_id':r['task_id'],'original_answer':r['original_answer'],'C':r['C'],'hypothetical_answer':answer,
            'baseline':answer_metrics(r['original_answer'],source['gold']),'hypothetical':answer_metrics(answer,source['gold'])})
    correct=[r for r in rows if r['baseline']['normalized_em']]
    report={'version':'v110','posthoc_training_only':True,'eligible_for_P2':False,
        'rule':'If first independent review supports original, preserve it verbatim; otherwise use C.',
        'reason_not_eligible':'Chosen after observing training results; untested as a frozen prospective mechanism and C protocol reliability gate remains failed.',
        'tasks':len(rows),'metrics':{k:sum(r['hypothetical'][k] for r in rows)/len(rows) for k in ('strict_em','normalized_em','f1')},
        'changed_vs_C':sum(r['hypothetical_answer']!=r['C'] for r in rows),'correct_original_preserve':sum(r['hypothetical']['normalized_em'] for r in correct)/len(correct),'new_api_calls':0}
    save('reports/v110-posthoc-preserve-original.json',report)
    Path('reports/v110-posthoc-preserve-original.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    print(json.dumps(report,indent=2))
