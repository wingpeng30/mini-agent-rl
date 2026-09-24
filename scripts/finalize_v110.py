"""Assemble final v110 decision from completed artifacts, without new model calls."""
import json,sqlite3
from pathlib import Path
from run_v110 import save

def load(name):return json.loads(Path(name).read_text(encoding='utf-8'))
if __name__=='__main__':
    analysis=load('reports/v110-train-analysis.json');manual=load('reports/v110-full-manual-review.json');blind=load('reports/v110-train-review-full.json');diagnosis=load('reports/v110-error-diagnosis.json')
    con=sqlite3.connect('file:runs/v110-api.sqlite?mode=ro',uri=True)
    calls=con.execute('SELECT status,charged,usage,raw_response FROM calls').fetchall();con.close()
    models={};tokens={'prompt':0,'completion':0}
    for status,cost,usage,raw in calls:
        body=json.loads(raw) if raw else {};name=body.get('model','unavailable');models[name]=models.get(name,0)+1
        data=json.loads(usage) if usage else {};tokens['prompt']+=data.get('prompt_tokens',0);tokens['completion']+=data.get('completion_tokens',0)
    cost=sum(c[1] for c in calls)
    if cost>60 or any(c[0]!='complete' for c in calls):raise RuntimeError('ledger_not_reconciled')
    if analysis['completed']!=100 or not manual['coverage_complete']:raise RuntimeError('incomplete_primary_stage')
    report={'version':'v110','decision':'P1_not_qualified_P2_not_started_keep_v101_default',
        'P0':'passed_exact_replay_200_and_official_rescore','P1':'completed_100_tasks_all_arms_failed_prespecified_gate','P2':'prepared_100_independent_tasks_986_documents_not_evaluated',
        'training_metrics':analysis['metrics'],'correct_original_preserve':analysis['correct_original_preserve'],
        'protocol_valid_rate':analysis['protocol_valid_rate'],'paired':analysis['paired'],'paired_C_minus_B':analysis['paired_C_minus_B'],
        'infrastructure_errors':analysis['infrastructure_errors'],'candidate_recall_on_baseline_wrong':analysis['candidate_recall_on_baseline_wrong'],
        'retrieval_tasks':diagnosis['retrieval_tasks'],'retrieval_associated_em_gains':diagnosis['retrieval_associated_em_gains'],
        'manual_review':{'changed_tasks':manual['unique_changed_tasks'],'changed_arm_pairs':manual['changed_arm_pairs'],'verified_citations':manual['verified_verbatim_citations'],'classifications_by_arm':manual['classifications_by_arm'],'human_external_adjudication':False},
        'blind_review_protocol_unresolved':blind['unresolved'],'api':{'calls':len(calls),'conservative_upper_cny':cost,'ceiling_cny':60,'remaining_upper_budget_cny':60-cost,'response_models':models,'tokens':tokens,'unknown_calls':0},
        'next_priority':'Prospectively test preserve-original when independently supported, plus reliable quote/citation interface; retain original numerical gates and then evaluate the frozen independent100.',
        'posthoc_diagnosis_not_adopted':load('reports/v110-posthoc-preserve-original.json'),
        'final_test_accessed':False,'default_system_changed':False,'gpu_training_started':False,'instance_shutdown_requested':False}
    save('reports/v110-summary.json',report)
    save('reports/v110-api-summary.json',report['api'])
    print(json.dumps({'decision':report['decision'],'api':report['api']},ensure_ascii=False,indent=2))
