"""Local reporting for every completed arm, including failures and drift."""
import argparse
from collections import Counter
from pathlib import Path
import random
from run_hybrid_adaptation import rows,read,write

def aggregate(data):
    if not data:return {'count':0}
    return dict(count=len(data),correct=sum(r['normalized_exact_match'] for r in data),
        strict_correct=sum(r['strict_exact_match'] for r in data),
        mean_f1=sum(r['token_f1'] for r in data)/len(data),
        all_support=sum(r['both_support_seen'] for r in data),
        invalid=sum(r['invalid_actions'] for r in data),repeated=sum(r['repeated_actions'] for r in data),
        calls=sum(r['model_calls'] for r in data),searches=sum(r['searches'] for r in data),
        input_tokens=sum(r['input_tokens'] for r in data),output_tokens=sum(r['output_tokens'] for r in data),
        seconds=sum(r['elapsed_seconds'] for r in data),end_reasons=dict(Counter(r['reason'] for r in data)))

def paired(baseline,candidate):
    before={r['case_id']:r for r in baseline};after={r['case_id']:r for r in candidate}
    ids=sorted(before.keys()&after.keys());delta=[after[c]['normalized_exact_match']-before[c]['normalized_exact_match'] for c in ids]
    if not delta:return {'pairs':0}
    rng=random.Random(42);draws=sorted(sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(2000))
    return dict(pairs=len(delta),improved=[c for c,d in zip(ids,delta) if d>0],regressed=[c for c,d in zip(ids,delta) if d<0],
        unchanged=sum(d==0 for d in delta),normalized_em_delta=sum(delta)/len(delta),bootstrap95=[draws[50],draws[1949]])

def differences(baseline,candidate):
    old={r['case_id']:r for r in baseline}
    changes=[]
    for r in candidate:
        p=old[r['case_id']]
        delta=r['normalized_exact_match']-p['normalized_exact_match']
        if delta:
            changes.append(dict(case_id=r['case_id'],question=r['question'],stratum=r['stratum'],delta=delta,
                baseline_answer=p['answer'],candidate_answer=r['answer'],baseline_reason=p['reason'],candidate_reason=r['reason'],
                baseline_queries=[t.get('query') for t in p['trace'] if t.get('action')=='search'],
                candidate_queries=[t.get('query') for t in r['trace'] if t.get('action')=='search'],
                baseline_all_support=p['both_support_seen'],candidate_all_support=r['both_support_seen']))
    return changes

def report(folder):
    remote=folder/'remote-results'
    baseline=rows(remote/'goal-z-ledger.jsonl')
    modes=('bm25-v2','hybrid-rrf-v1')
    cases=rows(folder/'cases.jsonl');expected={(c['id'],m) for c in cases for m in modes}
    actual=[(r['case_id'],r['retriever']) for r in baseline]
    complete=len(actual)==256 and len(set(actual))==256 and set(actual)==expected
    budget_ok=all(r['model_calls']<=4 and r['searches']<=3 and r['auxiliary_calls']==0 and r['reasoning_mode']=='off' for r in baseline)
    groups={}
    for name in ('primary_natural_104','development','natural_supplement','constructed_singlehop','all_128'):
        data=[r for r in baseline if name=='all_128' or (name=='primary_natural_104' and r['stratum']!='constructed_singlehop') or r['stratum']==name]
        groups[name]={m:aggregate([r for r in data if r['retriever']==m]) for m in modes}
        groups[name]['paired_hybrid_minus_bm25']=paired([r for r in data if r['retriever']==modes[0]],[r for r in data if r['retriever']==modes[1]])
    old={(r['case_id'],r['retriever']):r for r in rows(folder/'local-baseline.jsonl')}
    drift=[]
    for r in baseline:
        p=old[r['case_id'],r['retriever']]
        qold=[(s.get('query'),[d['id'] for d in s.get('documents',[])]) for s in p['steps'] if s.get('action')=='search']
        qnew=[(s.get('query'),[d['id'] for d in s.get('observation',{}).get('result',[])]) for s in r['trace'] if s.get('action')=='search']
        if (p['answer'],p['reason'],qold)!=(r['answer'],r['reason'],qnew):
            drift.append(dict(case_id=r['case_id'],mode=r['retriever'],old_answer=p['answer'],new_answer=r['answer'],
                old_reason=p['reason'],new_reason=r['reason'],old_searches=qold,new_searches=qnew))
    results=dict(kind='Exposed diagnostic/regression; no independent generalization claim',baseline_complete=complete,
        budgets_passed=budget_ok,groups=groups,local_remote_changes=drift,training_started=(remote/'train-A').exists())
    for name in ('state-gate','engineering-gate','training-trigger','training-gate','bm25-regression-gate','final-gate','gpu-budget','failure','remote-stage'):
        if (remote/(name+'.json')).exists():results[name]=read(remote/(name+'.json'))
    if (remote/'state-ledger.jsonl').exists():
        states=rows(remote/'state-ledger.jsonl');results['state_diagnostics']=dict(generations=len(states),
            invalid=sum(bool(s['parse_error']) for s in states),seconds=sum(s['elapsed_seconds'] for s in states),
            input_tokens=sum(s['usage'].get('prompt_tokens',0) for s in states),output_tokens=sum(s['usage'].get('completion_tokens',0) for s in states),
            per_view={v:dict(correct_final=sum(s['normalized_exact_match'] for s in states if s['view']==v and s['final']),
                final_states=sum(s['view']==v and s['final'] for s in states)) for v in ('legacy','score-free-v1','bm25-order')})
    if (remote/'score-free-ledger.jsonl').exists():
        candidate=rows(remote/'score-free-ledger.jsonl');results['score_free']={}
        for name in ('primary_natural_104','development','natural_supplement','constructed_singlehop','all_128'):
            keep=lambda r:name=='all_128' or (name=='primary_natural_104' and r['stratum']!='constructed_singlehop') or r['stratum']==name
            a=[r for r in baseline if r['retriever']=='hybrid-rrf-v1' and keep(r)];b=[r for r in candidate if keep(r)]
            results['score_free'][name]=dict(scores=aggregate(b),paired=paired(a,b))
    arms={}
    for arm,filename,mode in (('A','candidate-A-ledger.jsonl','hybrid-rrf-v1'),('B','candidate-B-ledger.jsonl','hybrid-rrf-v1'),
                              ('B_bm25','candidate-B-bm25-ledger.jsonl','bm25-v2')):
        path=remote/filename
        if not path.exists():continue
        data=rows(path);ids=[r['case_id'] for r in data]
        complete_arm=len(ids)==128 and len(set(ids))==128 and set(ids)=={c['id'] for c in cases}
        valid=all(r['model_calls']<=4 and r['searches']<=3 and r['auxiliary_calls']==0 and r['reasoning_mode']=='off' and r['retriever']==mode for r in data)
        arm_result=dict(complete=complete_arm,budgets_passed=valid,groups={})
        for name in ('primary_natural_104','development','natural_supplement','constructed_singlehop','all_128'):
            def keep(r):return name=='all_128' or (name=='primary_natural_104' and r['stratum']!='constructed_singlehop') or r['stratum']==name
            before=[r for r in baseline if r['retriever']==mode and keep(r)];after=[r for r in data if keep(r)]
            arm_result['groups'][name]=dict(scores=aggregate(after),paired_vs_goal_z=paired(before,after))
            if arm=='B' and (remote/'candidate-A-ledger.jsonl').exists():
                control=[r for r in rows(remote/'candidate-A-ledger.jsonl') if keep(r)]
                arm_result['groups'][name]['paired_vs_A']=paired(control,after)
        arm_result['changed_cases_vs_goal_z']=differences([r for r in baseline if r['retriever']==mode],data)
        arms[arm]=arm_result
        if not complete_arm or not valid:results['incomplete_or_invalid_candidate']=True
    results['training_arms']=arms
    results['training_verification']={}
    for arm in ('A','B'):
        path=remote/('train-'+arm)
        if not path.exists():continue
        updates=rows(path/'updates.jsonl') if (path/'updates.jsonl').exists() else []
        results['training_verification'][arm]=dict(updates=len(updates),
            input_tokens=sum(r['input_tokens'] for r in updates),target_tokens=sum(r['target_tokens'] for r in updates),
            smoke=read(path/'smoke.json') if (path/'smoke.json').exists() else None,
            verification=read(path/'verification.json') if (path/'verification.json').exists() else None)
    results['experiment_complete']=bool(results.get('final-gate')) and not results.get('incomplete_or_invalid_candidate',False)
    write(folder/'diagnostic-report.json',results)
    print({k:results[k] for k in ('baseline_complete','budgets_passed','training_started')})
    print(results['groups']['primary_natural_104'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);report(parser.parse_args().folder)
