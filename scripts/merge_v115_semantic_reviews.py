"""Validate two complete blinded semantic reviews, bind citations, and resolve only unanimous findings."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def jsonl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def flatten_visible(value):
    if isinstance(value,str):return value
    if isinstance(value,list):return '\n'.join(flatten_visible(x) for x in value)
    if isinstance(value,dict):return '\n'.join(flatten_visible(v) for v in value.values())
    return ''
def decisions(review,field):return {r['audit_id']:r for r in review.get(field,[])}
def validate_reviewer(review,name,manifest,state_ids,task_ids,packet_hash):
    if review.get('reviewer')!=name or review.get('independent') is not True or review.get('packet_manifest_sha256')!=packet_hash:
        raise ValueError('reviewer_provenance_mismatch:'+name)
    states=decisions(review,'states');tasks=decisions(review,'changed_tasks')
    if set(states)!=state_ids or set(tasks)!=task_ids:raise ValueError('incomplete_review_coverage:'+name)
    for group,expected in ((states,state_ids),(tasks,task_ids)):
        for case_id,row in group.items():
            candidates=row.get('candidates',{})
            if set(candidates)!= {'A','B','C'}:raise ValueError('candidate_review_missing:'+name+':'+case_id)
            for alias,decision in candidates.items():
                verdict=decision.get('verdict')
                if verdict not in {'supported','unsupported','uncertain'}:raise ValueError('invalid_verdict:'+name+':'+case_id)
                quotes=decision.get('citations',[])
                if not isinstance(quotes,list) or any(not isinstance(q,str) for q in quotes):raise ValueError('invalid_citations')
                if verdict=='supported' and not quotes:raise ValueError('supported_without_verbatim_quote:'+name+':'+case_id+':'+alias)
    return states,tasks
def merge(packet_dir,a_path,b_path,mapping_path,comparison_path,out_path):
    packet_dir=Path(packet_dir);manifest_path=packet_dir/'manifest.json';manifest=load(manifest_path)
    sp=packet_dir/'states-blind.jsonl';tp=packet_dir/'changed-tasks-blind.jsonl'
    if sha(sp)!=manifest['state_packet_sha256'] or sha(tp)!=manifest['changed_task_packet_sha256']:
        raise ValueError('blind_packet_hash_mismatch')
    state_packets={r['audit_id']:r for r in jsonl(sp)};task_packets={r['audit_id']:r for r in jsonl(tp)}
    mapping=load(mapping_path);comparison=load(comparison_path)
    pa,pb=load(a_path),load(b_path);packet_hash=sha(manifest_path)
    a_states,a_tasks=validate_reviewer(pa,'a',manifest,set(state_packets),set(task_packets),packet_hash)
    b_states,b_tasks=validate_reviewer(pb,'b',manifest,set(state_packets),set(task_packets),packet_hash)
    unresolved=[];supported_regressions=[];joined=[]
    for section,pa_rows,pb_rows,source,mappings in (
        ('states',a_states,b_states,state_packets,mapping['states']),
        ('changed_tasks',a_tasks,b_tasks,task_packets,mapping['tasks'])):
        for aid,packet in source.items():
            ma=mappings[aid]['anonymous_to_model'];votes=[]
            visible=flatten_visible(packet)
            for alias in 'ABC':
                da=pa_rows[aid]['candidates'][alias];db=pb_rows[aid]['candidates'][alias]
                legal=[]
                for quote in da.get('citations',[])+db.get('citations',[]):
                    legal.append(quote in visible)
                status=da['verdict'] if da['verdict']==db['verdict'] else 'disputed'
                if status=='uncertain' or status=='disputed' or any(not x for x in legal):
                    unresolved.append({'audit_id':aid,'section':section,'alias':alias,'status':status,
                                       'citation_invalid':any(not x for x in legal)})
                votes.append({'alias':alias,'model':ma[alias],'review_a':da,'review_b':db,'unanimous':status})
            joined.append({'audit_id':aid,'section':section,'candidate_reviews':votes})
    # A regression is counted only when both independent reviewers unanimously mark v115 unsupported
    # and v101 supported for the same case. All identities are rejoined privately after blind review.
    for row in joined:
        by_model={v['model']:v for v in row['candidate_reviews']}
        if 'v101' in by_model and 'v115' in by_model:
            if by_model['v115']['unanimous']=='unsupported' and by_model['v101']['unanimous']=='supported':
                supported_regressions.append({'audit_id':row['audit_id'],'section':row['section']})
    result={'version':'v115-semantic-review-merge-v1','reviewers':2,'independent_reviews':True,
        'packet_manifest_sha256':packet_hash,'state_case_count':len(state_packets),
        'reviewed_state_cases_by_each':{'a':len(a_states),'b':len(b_states)},
        'changed_task_case_count':len(task_packets),
        'reviewed_changed_tasks_by_each':{'a':len(a_tasks),'b':len(b_tasks)},
        'unresolved_disputes':len(unresolved),'confirmed_semantic_regressions':len(supported_regressions),
        'support_claim':'no_unresolved_support_issue' if not unresolved else 'unresolved',
        'unresolved_cases':unresolved,'confirmed_regression_cases':supported_regressions,
        'numeric_gates_pass':all(comparison.get('numeric_gates',{}).values()),'joined_reviews':joined,
        'independent100_opened':False,'final_test_accessed':False}
    out=Path(out_path);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'joined_reviews','unresolved_cases','confirmed_regression_cases'}},ensure_ascii=False,indent=2))
def main():
    p=argparse.ArgumentParser()
    for n in ('packet-dir','reviewer-a','reviewer-b','private-mapping','comparison','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();merge(a.packet_dir,a.reviewer_a,a.reviewer_b,a.private_mapping,a.comparison,a.output)
if __name__=='__main__':main()
