"""Freeze blinded P0 episodes and exact P1 prompt cross-inputs from v112 raw logs."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'
OUT=REPORTS/'v113-p0-blind-packet.jsonl'
KEY=REPORTS/'v113-p0-private-key.json'
P1=REPORTS/'v113-cross-inputs.jsonl'
MANIFEST=REPORTS/'v113-diagnostics-manifest.json'
ARMS={'baseline':'v101','candidate':'v112'}

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(value):return hashlib.sha256(value.encode('utf-8')).hexdigest()
def read(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
def write_new(path,rows):
    if path.exists():raise FileExistsError('refuse_to_overwrite:'+str(path))
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
def model_action(response):
    if response.get('tool_calls'):
        calls=response['tool_calls']
        return {'kind':'tool_call','calls':[{'name':c.get('name'),'arguments':c.get('arguments')} for c in calls]}
    if response.get('parse_error'):
        return {'kind':'invalid','text':response.get('raw_text') or response.get('content','')}
    return {'kind':'text','text':response.get('content','')}

def main():
    task_reports={arm:json.loads((REPORTS/f'v112-replay2-{arm}-tasks.json').read_text(encoding='utf-8'))
                  for arm in ARMS}
    state_reports={arm:json.loads((REPORTS/f'v112-replay2-{arm}-states.json').read_text(encoding='utf-8'))
                   for arm in ARMS}
    tasks={arm:{r['task_id']:r for r in report['rows']} for arm,report in task_reports.items()}
    ids=set(tasks['baseline'])
    if len(ids)!=50 or set(tasks['candidate'])!=ids:raise ValueError('fixed_50_task_pair_required')
    if any(len(report['rows'])!=113 or report['complete_coverage'] is not True for report in state_reports.values()):
        raise ValueError('complete_113_state_pair_required')
    state_by={arm:{r['state_hash']:r for r in report['rows']} for arm,report in state_reports.items()}
    if set(state_by['baseline'])!=set(state_by['candidate']):raise ValueError('state_pair_mismatch')
    failed=sorted(h for h in state_by['baseline'] if state_by['baseline'][h]['target']['action']=='answer' and
                  (not state_by['baseline'][h]['normalized_target_match'] or not state_by['candidate'][h]['normalized_target_match']))
    validation={r['state_hash']:r for r in read(ROOT/'data/evidence-sft-v112/validation.jsonl')}
    if set(validation)!=(set(state_by['baseline']) & set(validation)):
        raise ValueError('validation_state_map_mismatch')
    # Answer targets are never copied into the blinded packet. Raw model outputs
    # and visible evidence are the only state material sent for semantic review.
    by_task={arm:{} for arm in ARMS}
    for arm in ARMS:
        for tid,row in tasks[arm].items():
            events=[]
            for transition in row['transitions']:
                response=transition['response'];usage=response.get('usage') or {}
                events.append({'step':transition['step'],'input_messages':transition['messages'],
                    'input_tokens_sha256':sha(canonical(response.get('prompt_token_ids') or [])),
                    'input_token_count':len(response.get('prompt_token_ids') or []),
                    'budget_audit':{k:usage.get(k) for k in ('prompt_truncated','truncation_stage','lossy_body_trimmed','documents_deleted','retained_tool_documents')},
                    'action':model_action(response),
                    'retrieved_after_action':(transition.get('observation') or {}).get('result',[])})
            by_task[arm][tid]=events
    packets=[];key={};scores={};swapmap={}
    for tid in sorted(ids):
        blind='case-'+sha('v113-case:'+tid)[:16]
        swap=int(sha('v113-order:'+tid)[0],16)%2==1
        left,right=('candidate','baseline') if swap else ('baseline','candidate')
        packets.append({'blind_case_id':blind,'question':tasks['baseline'][tid]['transitions'][0]['messages'][-1]['content'],
            'A':by_task[left][tid],'B':by_task[right][tid],
            'instructions':'Review only the question and the visible messages before each action. Tool results are available only after their search. Do not infer correctness from search count or answer wording alone. For each final answer judge support at the moment it was given. For each search judge whether its query follows from the question/current evidence and whether returned facts address the stated gap. Give exact quotes, categorize concerns, and mark unresolved when evidence cannot decide. Do not use outside knowledge or look for any answer key.'})
        key[blind]={'task_id':tid,'A_arm':ARMS[left],'B_arm':ARMS[right],'order_swapped':swap}
        b=tasks['baseline'][tid];c=tasks['candidate'][tid]
        scores[blind]={'baseline':{'normalized_exact_match':b['normalized_exact_match'],'strict_exact_match':b['strict_exact_match'],'token_f1':b['token_f1'],'termination':b['termination'],'final_answer':b['final_answer']},
          'candidate':{'normalized_exact_match':c['normalized_exact_match'],'strict_exact_match':c['strict_exact_match'],'token_f1':c['token_f1'],'termination':c['termination'],'final_answer':c['final_answer']},
          'answer_changed':b['final_answer']!=c['final_answer']}
        swapmap[tid]=blind
    # Add paired answer-action snapshots for every fixed supported-evidence
    # state missed by either arm. These contain no synthetic target answer.
    state_packets=[]
    for state_hash in failed:
        a,b=(state_by['baseline'][state_hash],state_by['candidate'][state_hash])
        source_state=validation[state_hash]
        tid=a['task_id'];blind='state-'+sha('v113-state:'+state_hash)[:16]
        swap=int(sha('v113-state-order:'+state_hash)[0],16)%2==1
        ordered=(('candidate',b),('baseline',a)) if swap else (('baseline',a),('candidate',b))
        state_packets.append({'blind_state_id':blind,'question':source_state['question'],
          'current_messages':source_state['messages'][:-1],
          'A':{'raw_output':ordered[0][1].get('raw_output'),'actual_action':ordered[0][1].get('actual_action')},
          'B':{'raw_output':ordered[1][1].get('raw_output'),'actual_action':ordered[1][1].get('actual_action')},
          'instructions':'Judge whether each answer is supported by facts visible before this answer. Do not use outside knowledge. Provide exact evidence quotes and label uncertainty.'})
        key[blind]={'task_id':tid,'state_hash':state_hash,'A_arm':ARMS[ordered[0][0]],'B_arm':ARMS[ordered[1][0]],'order_swapped':swap}
        scores[blind]={'baseline':{'target_match':a['normalized_target_match'],'action':a['actual_action']},
                       'candidate':{'target_match':b['normalized_target_match'],'action':b['actual_action']}}
    packets.extend(state_packets)
    p1={};prompt_occurrences={}
    for arm in ARMS:
        for tid,row in tasks[arm].items():
            for tr in row['transitions']:
                response=tr['response'];tokens=response.get('prompt_token_ids') or []
                messages=tr['messages']; token_hash=sha(canonical(tokens)); msg_hash=sha(canonical(messages))
                record=p1.setdefault(token_hash,{'input_hash':token_hash,'messages':messages,'source_message_hashes':[],
                  'prompt_token_ids':tokens,'prompt_token_sha256':token_hash,'prompt_tokens':len(tokens),
                  'step':tr['step'],'observed':{},'duplicate_output_conflicts':{}})
                if record['prompt_token_ids']!=tokens:raise ValueError('same_token_hash_different_ids')
                if msg_hash not in record['source_message_hashes']:record['source_message_hashes'].append(msg_hash)
                if arm in record['observed']:
                    if canonical(model_action(record['observed'][arm]))!=canonical(model_action(response)):
                        record['duplicate_output_conflicts'][arm]=True
                else:record['observed'][arm]=response
                prompt_occurrences.setdefault(token_hash,set()).add(arm)
    conflict_rows=[k for k,v in p1.items() if v['duplicate_output_conflicts']]
    missing=sum(len(set(ARMS)-seen) for seen in prompt_occurrences.values())
    if len(p1)!=170 or missing!=109:raise ValueError(f'unexpected_cross_design:{len(p1)}:{missing}')
    for path in (OUT,KEY,REPORTS/'v113-p0-score-key.json',P1,MANIFEST):
        if path.exists():raise FileExistsError('refuse_to_overwrite:'+str(path))
    write_new(OUT,packets)
    KEY.write_text(json.dumps(key,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (REPORTS/'v113-p0-score-key.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_new(P1,list(p1.values()))
    sources={}
    for arm in ARMS:
        for kind in ('tasks','states'):
            name=f'reports/v112-replay2-{arm}-{kind}.json';path=ROOT/name
            sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'version':'v113','source_files':sources,'p0_packet_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
      'p0_key_sha256':hashlib.sha256(KEY.read_bytes()).hexdigest(),'p0_score_key_sha256':hashlib.sha256((REPORTS/'v113-p0-score-key.json').read_bytes()).hexdigest(),
      'p1_inputs_sha256':hashlib.sha256(P1.read_bytes()).hexdigest(),'paired_tasks':len(ids),'changed_final_answers':sum(x['answer_changed'] for x in scores.values() if 'answer_changed' in x),
      'supported_answer_failure_states':len(failed),'p1_unique_inputs':len(p1),'p1_cross_outputs_missing':missing,
      'duplicate_model_output_conflict_inputs':len(conflict_rows),
      'fixed_state_overlap':len(set(state_by['baseline']) & {sha(canonical(x['messages'])) for x in p1.values()}),
      'independent100_accessed':False,'final_test_accessed':False,'gold_or_target_in_blind_packet':False}
    MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
