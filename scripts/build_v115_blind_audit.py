"""Create per-case randomized, gold-free semantic-review packets and a private map."""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path

MODELS=('v101','v112','v115')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def readl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s):return ' '.join(re.sub(r'[^\w]+',' ',str(s).casefold()).split())
def anon_order(key):
    return sorted(MODELS,key=lambda m:hashlib.sha256(('v115-blind:'+key+':'+m).encode()).hexdigest())
def write_jsonl(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows),encoding='utf-8')
def clean_message(m):
    result={'role':m.get('role','')}
    if m.get('content'):result['content']=m['content']
    if m.get('tool_calls'):
        result['tool_calls']=[{'name':c.get('name'),'arguments':c.get('arguments',{})} for c in m['tool_calls']]
    return result
def clean_transition(t):
    response=t.get('response') or {};obs=t.get('observation') or {}
    calls=response.get('tool_calls') or []
    return {'step':t.get('step'), 'visible_messages':[clean_message(m) for m in t.get('messages',[])],
            'output':{'content':response.get('content',''),
                      'tool_calls':[{'name':c.get('name'),'arguments':c.get('arguments',{})} for c in calls],
                      'parse_error':response.get('parse_error')},
            'observation':{k:obs[k] for k in ('ok','result','error') if k in obs}}
def verify(reports,mode):
    for r in reports:
        if r.get('version')!='v115' or r.get('mode')!=mode or r.get('complete_coverage') is not True:raise ValueError('incomplete_report:'+mode)
    base=reports[0]
    for r in reports[1:]:
        for k in ('corpus_sha256','runner_config','code_sha256','private_inputs','determinism_env'):
            if r.get(k)!=base.get(k):raise ValueError('blind_audit_environment_mismatch:'+k)
        c1=dict(r['config']);c2=dict(base['config']);c1.pop('adapter_path',None);c2.pop('adapter_path',None)
        if c1!=c2:raise ValueError('blind_audit_model_config_mismatch')
def build(args):
    state_reports=[read(getattr(args,f'{m}_states')) for m in MODELS]
    task_reports=[read(getattr(args,f'{m}_tasks')) for m in MODELS]
    verify(state_reports,'states');verify(task_reports,'tasks')
    state_file=readl(args.states);task_file=readl(args.tasks)
    state_by={r['state_id']:r for r in state_file};task_by={r['id']:r for r in task_file}
    if len(state_by)!=100 or len(task_by)!=50:raise ValueError('frozen_internal_coverage_required')
    state_maps=[{r['state_id']:r for r in rep['rows']} for rep in state_reports]
    task_maps=[{r['task_id']:r for r in rep['rows']} for rep in task_reports]
    if any(set(m)!=set(state_by) for m in state_maps) or any(set(m)!=set(task_by) for m in task_maps):raise ValueError('paired_report_coverage_mismatch')
    state_packets=[];task_packets=[];mapping={'version':'v115-private-blind-map-v1','states':{},'tasks':{}}
    for sid,row in sorted(state_by.items()):
        aid='s-'+hashlib.sha256(('v115-audit:'+sid).encode()).hexdigest()[:12]
        order=anon_order(sid); outputs={}; mapping['states'][aid]={'state_id':sid,'task_id':row['task_id'],'anonymous_to_model':{chr(65+i):model for i,model in enumerate(order)}}
        for i,model in enumerate(order):
            observed=state_maps[MODELS.index(model)][sid]
            outputs[chr(65+i)]={'action':observed.get('actual_action'),'raw_output':observed.get('raw_output')}
        state_packets.append({'audit_id':aid,'question':row['question'],'visible_history_and_evidence':row['messages'],
            'anonymous_outputs':outputs,'review_request':'For each anonymous output, assess protocol and evidence support using only the visible history. Check entity, relation, time, quantity/unit, and scope. Quote exact supporting or contradicting text. Assess whether a search is justified and whether it obtains relevant new evidence. Mark uncertain when evidence is insufficient; do not infer a hidden answer.'})
    changed=[]
    for tid,task in sorted(task_by.items()):
        answers={m:task_maps[MODELS.index(m)][tid].get('final_answer') or '' for m in MODELS}
        if len({norm(v) for v in answers.values()})==1:continue
        aid='t-'+hashlib.sha256(('v115-audit:'+tid).encode()).hexdigest()[:12]
        order=anon_order(tid);outputs={};mapping['tasks'][aid]={'task_id':tid,'anonymous_to_model':{chr(65+i):model for i,model in enumerate(order)}}
        for i,model in enumerate(order):
            r=task_maps[MODELS.index(model)][tid]
            outputs[chr(65+i)]={'final_answer':r.get('final_answer'),
                               'trajectory':[clean_transition(t) for t in r.get('transitions',[])]}
        changed.append({'audit_id':aid,'question':task['question'],'anonymous_outputs':outputs,
            'review_request':'Compare the answer and its actual visible evidence/search trajectory for each anonymous run. Check exact entities, relation, time, number/unit, scope, source support and over-search. Quote the precise supporting or contradicting evidence and identify any clear semantic regression. Do not infer gold or scoring direction; mark uncertain if the record does not decide.'})
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    write_jsonl(out/'states-blind.jsonl',state_packets);write_jsonl(out/'changed-tasks-blind.jsonl',changed)
    (out/'private-mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest={'version':'v115-blind-audit-packets-v1','state_cases':len(state_packets),'changed_task_cases':len(changed),
        'state_packet_sha256':sha(out/'states-blind.jsonl'),'changed_task_packet_sha256':sha(out/'changed-tasks-blind.jsonl'),
        'private_mapping_sha256':sha(out/'private-mapping.json'),'blinded':True,'gold_in_packets':False,'scores_in_packets':False,
        'model_identity_in_packets':False,'randomized_anonymous_mapping_per_case':True}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
def main():
    p=argparse.ArgumentParser()
    for m in MODELS:
        p.add_argument('--'+m+'-states',required=True);p.add_argument('--'+m+'-tasks',required=True)
    p.add_argument('--states',required=True);p.add_argument('--tasks',required=True);p.add_argument('--output',required=True)
    build(p.parse_args())
if __name__=='__main__':main()
