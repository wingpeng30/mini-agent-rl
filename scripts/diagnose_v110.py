"""Read-only error-source and cost decomposition of completed v110 artifacts."""
import argparse,json,sqlite3
from collections import Counter
from pathlib import Path
from mini_agent_rl.hybrid_v110 import answer_metrics
from run_v110 import read_jsonl,save

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--results',required=True);p.add_argument('--ledger',default='runs/v110-api.sqlite');p.add_argument('--output',required=True);p.add_argument('--details',required=True);args=p.parse_args()
    inputs={r['task_id']:r for r in read_jsonl(args.inputs)};rows=[];counts=Counter()
    for r in read_jsonl(args.results):
        source=inputs[r['task_id']];gold=source['gold'];base=answer_metrics(source['original_answer'],gold)
        candidates=r.get('initial_candidates',[]);covered=any(answer_metrics(c['answer'],gold)['normalized_em'] for c in candidates)
        reviews=[x for x in r.get('checks',[]) if 'assessments' in x]
        original_review=reviews[0]['assessments']['original'] if reviews else {}
        category='baseline_correct' if base['normalized_em'] else 'candidate_selection_error' if covered and not answer_metrics(r['B'],gold)['normalized_em'] else 'candidate_recovery' if covered else 'candidate_missing'
        detail={'task_id':r['task_id'],'question':source['question'],'gold':gold,'original':source['original_answer'],'B':r['B'],'C':r['C'],'category':category,'original_evidence_assessment_auxiliary':original_review,'queries':r.get('retrieval',[]),'changed_after_retrieval':bool(r.get('retrieval')) and r['C']!=r['B'],
            'reverted_to_original':r['B']!=source['original_answer'] and r['C']==source['original_answer'],
            'retrieval_associated_em_gain':bool(r.get('retrieval')) and answer_metrics(r['C'],gold)['normalized_em']>answer_metrics(r['B'],gold)['normalized_em'],
            'changed_despite_original_supported':r['C']!=source['original_answer'] and original_review.get('status')=='supported'}
        rows.append(detail);counts[category]+=1
    con=sqlite3.connect(f'file:{Path(args.ledger).resolve().as_posix()}?mode=ro',uri=True);by_kind={};total=0
    for cfg,status,charge,usage in con.execute('SELECT config,status,charged,usage FROM calls'):
        kind=json.loads(cfg)['kind'];entry=by_kind.setdefault(kind,{'calls':0,'charged_upper_cny':0,'latency_seconds':0,'input_tokens':0,'output_tokens':0,'unknown':0})
        u=json.loads(usage) if usage else {};entry['calls']+=1;entry['charged_upper_cny']+=charge;total+=charge
        entry['latency_seconds']+=u.get('latency_seconds',0);entry['input_tokens']+=u.get('prompt_tokens',0);entry['output_tokens']+=u.get('completion_tokens',0);entry['unknown']+=status!='complete'
    con.close();report={'tasks':len(rows),'categories':dict(counts),'retrieval_tasks':sum(bool(r['queries']) for r in rows),'retrieval_changed_answer_tasks':sum(r['changed_after_retrieval'] for r in rows),
        'reverted_to_original':sum(r['reverted_to_original'] for r in rows),'retrieval_associated_em_gains':sum(r['retrieval_associated_em_gain'] for r in rows),'changed_despite_original_supported':sum(r['changed_despite_original_supported'] for r in rows),
        'api_by_kind':by_kind,'total_upper_cny':total,'evidence_judgments_are_auxiliary':True}
    save(args.output,report);Path(args.details).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
