"""Authorize access to the independent100 evaluation only after numeric and blind audit gates pass."""
import argparse,json
from pathlib import Path

def authorize(comparison_path,review_path,packet_manifest_path,output_path):
    comparison=json.loads(Path(comparison_path).read_text(encoding='utf-8'))
    review=json.loads(Path(review_path).read_text(encoding='utf-8'))
    manifest=json.loads(Path(packet_manifest_path).read_text(encoding='utf-8'))
    if comparison.get('version')!='v115' or comparison.get('internal_only') is not True:
        raise ValueError('v115_internal_comparison_required')
    if comparison.get('numeric_gates',{}).get('protocol_98') is not True or not all(comparison.get('numeric_gates',{}).values()):
        raise ValueError('internal_numeric_gate_failed')
    if comparison.get('final_test_accessed') or comparison.get('independent100_opened'):
        raise ValueError('forbidden_eval_already_accessed')
    if manifest.get('blinded') is not True or manifest.get('gold_in_packets') or manifest.get('scores_in_packets') or manifest.get('model_identity_in_packets'):
        raise ValueError('blind_packet_manifest_invalid')
    if review.get('packet_manifest_sha256')!=__import__('hashlib').sha256(Path(packet_manifest_path).read_bytes()).hexdigest():
        raise ValueError('review_packet_hash_mismatch')
    if review.get('reviewers')!=2 or review.get('independent_reviews') is not True:
        raise ValueError('two_independent_reviews_required')
    if review.get('state_case_count')!=100 or review.get('reviewed_state_cases_by_each')!={"a":100,"b":100}:
        raise ValueError('all_internal_states_must_be_reviewed_twice')
    if review.get('changed_task_case_count')!=manifest.get('changed_task_cases') or review.get('reviewed_changed_tasks_by_each')!={
        "a":manifest.get('changed_task_cases'),"b":manifest.get('changed_task_cases')}:
        raise ValueError('all_changed_task_answers_must_be_reviewed_twice')
    if review.get('unresolved_disputes')!=0 or review.get('confirmed_semantic_regressions')!=0 or review.get('support_claim')!='no_unresolved_support_issue':
        raise ValueError('semantic_gate_failed')
    output=Path(output_path)
    if output.exists(): raise ValueError('authorization_output_exists')
    result={'version':'v115-independent100-authorization-v1','authorized':True,
        'internal_comparison_sha256':__import__('hashlib').sha256(Path(comparison_path).read_bytes()).hexdigest(),
        'semantic_review_sha256':__import__('hashlib').sha256(Path(review_path).read_bytes()).hexdigest(),
        'packet_manifest_sha256':__import__('hashlib').sha256(Path(packet_manifest_path).read_bytes()).hexdigest(),
        'independent100_opened_at_authorization':False,'final_test_accessed':False,
        'conditions':'freeze code, adapters, corpus, configuration and data before opening the independent100 source'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result

def main():
    p=argparse.ArgumentParser()
    for n in ('comparison','review','packet-manifest','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    print(json.dumps(authorize(a.comparison,a.review,a.packet_manifest,a.output),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
