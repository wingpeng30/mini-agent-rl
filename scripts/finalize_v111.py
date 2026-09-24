"""Finalize completed v111 evidence without changing frozen results or gates."""
import hashlib,json,sqlite3,sys
from pathlib import Path
from run_v111 import code_hash
from run_v110 import save,read_jsonl

def main():
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    analysis=read('reports/v111-train-analysis.json');review=read('reports/v111-train-review.json')
    manual=read('reports/v111-train-R-evidence-review.json');retained=read('reports/v111-retained-evidence-review.json')
    manifest=read('backups/v111-train-results.jsonl.manifest.json')
    if not analysis['manifest_complete'] or analysis['tasks']!=100:raise RuntimeError('train_incomplete')
    if manifest['code']!=code_hash():raise RuntimeError('frozen_code_changed')
    with sqlite3.connect(Path('runs/v110-api.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row;calls=[dict(r) for r in db.execute('SELECT * FROM calls')]
    costs={};models={};tokens={'prompt_tokens':0,'completion_tokens':0,'reasoning_tokens':0}
    for c in calls:
        stage=c['stage'];costs.setdefault(stage,{'calls':0,'upper_cny':0,'unresolved':0})
        costs[stage]['calls']+=1;costs[stage]['upper_cny']+=c['charged']
        costs[stage]['unresolved']+=c['status']!='complete'
        if stage.startswith('v111') and c['status']=='complete':
            body=json.loads(c['raw_response']);model=body.get('model','unknown');models[model]=models.get(model,0)+1
            usage=body['usage']
            for k in ('prompt_tokens','completion_tokens'):tokens[k]+=usage.get(k,0)
            tokens['reasoning_tokens']+=usage.get('completion_tokens_details',{}).get('reasoning_tokens',0)
    total=sum(c['charged'] for c in calls);new=sum(c['charged'] for c in calls if c['stage'].startswith('v111'))
    if any(c['status']!='complete' for c in calls):raise RuntimeError('unresolved_api_calls')
    semantic={a:review['arms'][a]['auxiliary_gate'] for a in ('P','R','D')}
    semantic['R']=semantic['R'] and manual['p2_semantic_gate_passed']
    qualified=[a for a in analysis['eligible_ranking'] if semantic[a]]
    report={'version':'v111','P0':'implemented_and_cpu_verified','P1':'completed100_numerical_pass_semantic_gate_failed',
        'P2':'not_started_prespecified_semantic_gate','training_metrics':analysis['metrics'],
        'first_pass_protocol':analysis['protocol_first_pass_valid_rate'],'final_protocol':analysis['protocol_valid_rate'],
        'original_correct_preservation':analysis['correct_original_preserve'],'transitions':analysis['transitions'],
        'numerical_gate':analysis['numerical_gate'],'semantic_gate':semantic,'fully_qualified_candidates':qualified,
        'candidate_selected_on_numbers_only':analysis['selected_numerical_candidate'],
        'paired_R_vs_A':analysis['paired_vs_A']['R'],'protection_P_minus_U':analysis['paired_P_minus_U'],
        'retrieval_R_minus_P':analysis['paired_R_minus_P'],'matched_regeneration':analysis['matched_regeneration'],
        'manual_R_review':{'counts':manual['counts'],'coverage_complete':manual['coverage_complete'],
            'quotes':manual['quote_count'],'blocking_task_ids':manual['blocking_task_ids'],'human_or_independent_truth':False},
        'retained_review':{'count':retained['count'],'counts':retained['counts'],'human_or_independent_truth':False},
        'blind_review':{'unique_pairs':review['unique_pairs'],'infrastructure_errors':review['infrastructure_errors'],
            'common_P_R_support_disputes':review['arms']['R']['changed_without_confirmed_support'],'same_model_auxiliary_only':True},
        'api':{'stages':costs,'new_upper_cny':new,'cumulative_upper_cny':total,'remaining_upper_budget_cny':60-total,
            'actual_account_bill_queried':False,'response_models':models,'new_tokens':tokens,'unresolved_calls':0},
        'frozen_code_unchanged':True,'default_system_changed':False,'new_training_started':False,
        'independent100_model_evaluated':False,'final_test_accessed':False,'instance_shutdown_requested':False,
        'next_priority':'Validate question-entity/relation/measurement/time/scope entailment before replacement; evidence IDs already solve provenance, not semantic support. No posthoc retuning or holdout access this run.'}
    if qualified:raise RuntimeError('qualified_candidate_requires_P2_instead_of_failure_finalize')
    save('reports/v111-summary.json',report)
    save('reports/v111-api-summary.json',report['api'])
    print(json.dumps({'decision':report['P2'],'new_upper_cny':new,'cumulative_upper_cny':total,'qualified':qualified},ensure_ascii=False))

if __name__=='__main__':main()
