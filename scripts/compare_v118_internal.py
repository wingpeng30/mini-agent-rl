"""Pair v101/v112/v118 on the frozen v117 internal 50 questions and enforce numeric gates."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def normalized_config(r):
    c=dict(r['config']);c.pop('adapter_path',None);return c
def ensure_same(reports,mode):
    for r in reports:
        if r.get('version')!='v118' or r.get('mode')!=mode or r.get('complete_coverage') is not True:raise ValueError('incomplete_or_wrong_report:'+mode)
    ref=reports[0]
    for r in reports[1:]:
        for key in ('corpus_sha256','corpus_documents','retriever','runner_config','model_files','code_sha256','private_inputs','determinism_env'):
            if r.get(key)!=ref.get(key):raise ValueError('environment_mismatch:'+key)
        if normalized_config(r)!=normalized_config(ref):raise ValueError('model_config_mismatch')
    return True
def one(base,cand):
    b={r['task_id']:r for r in base['rows']};c={r['task_id']:r for r in cand['rows']}
    if len(b)!=50 or len(c)!=50 or set(b)!=set(c):raise ValueError('paired_50_question_coverage_required')
    original={k for k,v in b.items() if v['normalized_exact_match']==1}
    kept=sum(c[k]['normalized_exact_match']==1 for k in original)
    gains=sum(c[k]['normalized_exact_match']==1 and b[k]['normalized_exact_match']==0 for k in b)
    losses=sum(c[k]['normalized_exact_match']==0 and b[k]['normalized_exact_match']==1 for k in b)
    return {'baseline_em':sum(v['normalized_exact_match'] for v in b.values())/50,
            'candidate_em':sum(v['normalized_exact_match'] for v in c.values())/50,
            'baseline_f1':sum(v['token_f1'] for v in b.values())/50,
            'candidate_f1':sum(v['token_f1'] for v in c.values())/50,
            'baseline_correct':len(original),'preserved_correct':kept,
            'preservation':kept/len(original) if original else None,'gains':gains,'losses':losses,'net_gain':gains-losses,
            'baseline_task_protocol':sum(bool(v.get('protocol_valid')) for v in b.values())/50,
            'candidate_task_protocol':sum(bool(v.get('protocol_valid')) for v in c.values())/50}
def run(args):
    v101t,v112t,v118t=map(read,(args.v101_tasks,args.v112_tasks,args.v118_tasks))
    v101s,v112s,v118s=map(read,(args.v101_states,args.v112_states,args.v118_states))
    ensure_same([v101t,v112t,v118t],'tasks');ensure_same([v101s,v112s,v118s],'states')
    for a,b in ((v101t,v101s),(v112t,v112s),(v118t,v118s)):
        if a['corpus_sha256']!=b['corpus_sha256'] or a['adapter_files']!=b['adapter_files']:
            raise ValueError('task_state_model_or_corpus_mismatch')
    metrics=one(v101t,v118t);state=v118s['metrics']
    gates={'protocol_98':metrics['candidate_task_protocol']>=.98 and state['protocol_rate']>=.98,
           'supported_answer_90':state['supported_answer_correct_rate']>=.90,
           'insufficient_evidence_search_90':state['insufficient_evidence_search_rate']>=.90,
           'correct_preservation_95':metrics['preservation'] is not None and metrics['preservation']>=.95,
           'net_gain_at_least_3':metrics['net_gain']>=3,
           'f1_non_decrease':metrics['candidate_f1']>=metrics['baseline_f1']}
    semantic=read(args.semantic_gate) if args.semantic_gate else None
    semantic_ok=bool(semantic and semantic.get('version')=='v118-semantic-gate-v1'
        and semantic.get('semantic_gate_passed') is True and semantic.get('reviewers')==2
        and semantic.get('independent_reviews') is True and semantic.get('unresolved_disputes')==0
        and semantic.get('confirmed_semantic_regressions')==0
        and semantic.get('each_reviewer_state_coverage')=={'luna_agent':100,'deepseek_api':100})
    result={'version':'v118','internal_only':True,'independent100_opened':False,'final_test_accessed':False,
            'v101_vs_v118':metrics,'v112_reference':one(v101t,v112t),
            'state_metrics':state,'numeric_gates':gates,
            'all_numeric_gates_pass':bool(gates) and all(gates.values()),'semantic_audit':semantic,
            'semantic_gate_passed':semantic_ok,'internal_eligible':bool(gates) and all(gates.values()) and semantic_ok}
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('v101-tasks','v112-tasks','v118-tasks','v101-states','v112-states','v118-states','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--semantic-gate')
    run(p.parse_args())
