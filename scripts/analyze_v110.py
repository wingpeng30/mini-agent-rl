"""Manifest-complete paired analysis and pre-registered training gates."""
import argparse,json,random
from pathlib import Path
from mini_agent_rl.hybrid_v110 import answer_metrics
from run_v110 import read_jsonl,summarize,save

def bootstrap(values):
    rng=random.Random(42);n=len(values)
    means=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    return {'mean':sum(values)/n,'ci95':[means[250],means[9750]]}

def analyze(inputs,rows):
    report=summarize(inputs,rows);by={r['task_id']:r for r in rows};paired=[]
    for r in inputs:
        result=by.get(r['task_id'],{});a=answer_metrics(r['original_answer'],r['gold'])
        item={'task_id':r['task_id'],'A':a,'original_answer':r['original_answer'],'gold':r['gold']}
        for arm in ('B','C'):
            answer=result.get(arm,r['original_answer']);item[arm]=answer_metrics(answer,r['gold']);item[arm+'_answer']=answer
        item['candidate_best_f1']=max([answer_metrics(c['answer'],r['gold'])['f1'] for c in result.get('initial_candidates',[])]+[a['f1']])
        item['candidate_contains_normalized_gold']=any(answer_metrics(c['answer'],r['gold'])['normalized_em'] for c in result.get('initial_candidates',[]))
        item['retrieval_queries']=len(result.get('retrieval',[]));paired.append(item)
    report['paired']={arm:{metric:bootstrap([x[arm][metric]-x['A'][metric] for x in paired]) for metric in ('strict_em','normalized_em','f1')} for arm in ('B','C')}
    report['paired_C_minus_B']={metric:bootstrap([x['C'][metric]-x['B'][metric] for x in paired]) for metric in ('strict_em','normalized_em','f1')}
    report['transitions']={arm:{'wrong_to_right':sum(not x['A']['normalized_em'] and x[arm]['normalized_em'] for x in paired),'right_to_wrong':sum(x['A']['normalized_em'] and not x[arm]['normalized_em'] for x in paired)} for arm in ('B','C')}
    report['candidate_oracle_f1']=sum(x['candidate_best_f1'] for x in paired)/len(paired)
    wrong=[x for x in paired if not x['A']['normalized_em']]
    report['candidate_recall_on_baseline_wrong']={'numerator':sum(x['candidate_contains_normalized_gold'] for x in wrong),'denominator':len(wrong)}
    report['retrieval_tasks']=sum(x['retrieval_queries']>0 for x in paired)
    report['position_changed']=sum(r.get('position_check') is not None and r['position_check']['answer']!=r['B'] for r in rows)
    report['numerical_gate']={arm:report['completed']==len(inputs) and not report['infrastructure_errors'] and
        report['metrics'][arm]['normalized_em']>=report['metrics']['A']['normalized_em'] and
        report['metrics'][arm]['f1']>=report['metrics']['A']['f1'] and
        report['correct_original_preserve'][arm]>=.95 and report['protocol_valid_rate'][arm]>=.98 for arm in ('B','C')}
    report['semantic_review_required_before_adoption']=True
    return report,paired

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--details',required=True);args=p.parse_args()
    report,rows=analyze(read_jsonl(args.inputs),read_jsonl(args.results));save(args.output,report)
    Path(args.details).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
