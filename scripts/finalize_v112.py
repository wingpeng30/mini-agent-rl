"""Join independent reviews, quarantine disagreements, and emit a fail-closed gate."""
from __future__ import annotations
import hashlib, json, math, sqlite3
from collections import Counter
from pathlib import Path
from build_v112 import ROOT, OUT, read, digest, all_candidates
from mini_agent_rl.evidence_sft_v112 import validate_dataset, validate_row, compute_candidate_hash, load_budget_policy
from review_v112 import packet

SOURCE_SHA='3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768'

def inspect_ledger(manifest):
    """Take one read-only snapshot for both budget gates and review provenance."""
    with sqlite3.connect((ROOT/'runs/v110-api.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        db.execute('BEGIN')
        ledger=db.execute('select stage,status,count(*),sum(charged) from calls group by stage,status').fetchall()
        old=db.execute("select * from calls where stage not like 'v112-%' order by key").fetchall()
        audited=db.execute("select key,payload,raw_response from calls where stage='v112-review' and status='complete' order by key").fetchall()
    total=sum(r[3] for r in ledger);new=sum(r[3] for r in ledger if r[0].startswith('v112-'))
    errors=[];policy=load_budget_policy(ROOT)
    if any(r[1]!='complete' for r in ledger):errors.append('unresolved_api_calls')
    if not math.isfinite(total) or (policy['cumulative_cap_cny'] is not None and total>policy['cumulative_cap_cny']):errors.append('cumulative_api_cap_exceeded')
    if not math.isfinite(new) or (policy['new_cap_cny'] is not None and new>policy['new_cap_cny']):errors.append('new_api_cap_exceeded')
    prior_hash=digest(old)
    if prior_hash!=manifest['opening_ledger']['rows_sha256']:errors.append('prior_ledger_rows_changed')
    gate={'passed':not errors,'errors':errors,'prior_rows_sha256':prior_hash,
          'ledger_path':'runs/v110-api.sqlite','cumulative_cny':total,'new_cny':new,'policy':policy}
    provenance={}
    for key,payload,response in audited:
        try:
            states=json.loads(payload)['states'];body=json.loads(response)
            choice=body['choices'][0]
            if choice.get('finish_reason')!='stop':continue
            reviews=json.loads(choice['message']['content'])['reviews']
            state_map=unique(states);review_map=unique(reviews)
            if set(state_map)!=set(review_map):continue
            for state_hash,state in state_map.items():
                provenance.setdefault(state_hash,[]).append((key,state,review_map[state_hash]))
        except (ValueError,TypeError,KeyError,IndexError):
            continue
    return ledger,gate,provenance

def verified_api_key(review,row,provenance):
    # These fields are appended by review_v112, not returned by the provider.
    added={'kind','reviewer','attempts','first_pass_valid'}
    original={k:v for k,v in review.items() if k not in added}
    for key,state,raw_review in provenance.get(row['state_hash'],[]):
        if state==packet(row) and original==raw_review:return key
    return None
def unique(rows):
    by={}
    for r in rows:
        key=r['state_hash']
        if key in by:raise ValueError('duplicate_review:'+key)
        by[key]=r
    return by
def audit_citations(review,row):
    if not review.get('approved'):return False
    docs=row['evidence'] if row['target']['action']=='answer' else row['evidence']+row['retrieval_results'][:3]
    texts={d['id']:d['text'] for d in docs}
    quotes=review.get('citations',[])
    if not quotes:return False
    if not all(c.get('document_id') in texts and isinstance(c.get('quote'),str) and c['quote'] and c['quote'] in texts[c['document_id']] for c in quotes):return False
    if row['target']['action']=='search':
        if not review.get('missing_fact'):return False
        prior=[json.loads(m['content']).get('query','') for m in row['messages'][:-1] if m['role']=='assistant']
        if row['target']['value'].strip().casefold() in {q.strip().casefold() for q in prior}:return False
        visible={d['text'] for d in row['evidence']}
        retrieved={d['id']:d['text'] for d in row['retrieval_results'][:3]}
        if not any(c['document_id'] in retrieved and c['quote'] in retrieved[c['document_id']]
                   and retrieved[c['document_id']] not in visible for c in quotes):return False
    return True
def counts(rows):
    return {'states':len(rows),'questions':len({r['task_id'] for r in rows}),
      'actions':dict(Counter(r['target']['action'] for r in rows)),
      'risks':dict(Counter(k for r in rows for k in r['risk_labels']))}
def run():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    ledger,budget_gate,provenance=inspect_ledger(manifest)
    candidates=all_candidates();keys={r['state_hash'] for r in candidates}
    candidate_sha=hashlib.sha256((OUT/'candidates.jsonl').read_bytes()).hexdigest()
    for path in OUT.glob('audit-workload-*.json'):
        workload=json.loads(path.read_text(encoding='utf-8'))
        if workload['candidates_sha256']!=candidate_sha:
            raise ValueError('candidates_changed_after_review_started')
        for name,expected in workload.get('supplements',{}).items():
            if Path(name).name!=name or hashlib.sha256((OUT/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('supplement_changed_after_review_started')
    api=unique(read(OUT/'api-reviews.jsonl') if (OUT/'api-reviews.jsonl').exists() else [])
    task=[]
    packets={}
    for path in sorted(OUT.glob('task-review-*.jsonl')):
        blind=OUT/path.name.replace('task-review-','blind-packet-')
        packet_rows=unique(read(blind));review_rows=read(path)
        if {r['state_hash'] for r in review_rows}!=set(packet_rows):raise ValueError('incomplete_blind_packet')
        packets.update(packet_rows);task.extend(review_rows)
    task=unique(task)
    if (set(api)|set(task))-keys:raise ValueError('foreign_review')
    accepted=[];quarantined=[];pending=[]
    for row in candidates:
        key=row['state_hash'];a=api.get(key);b=task.get(key)
        ledger_key=verified_api_key(a,row,provenance) if a else None
        if a and ledger_key is None:
            quarantined.append({'state_hash':key,'task_id':row['task_id'],'split':row['split'],
              'reason':'api_review_missing_completed_ledger_provenance','api':a,'task_agent':b});continue
        if a and not a.get('approved') or b and not b.get('approved'):
            quarantined.append({'state_hash':key,'task_id':row['task_id'],'split':row['split'],'api':a,'task_agent':b});continue
        if not a or not b:pending.append(key);continue
        if packets.get(key)!=packet(row):raise ValueError('review_packet_target_or_evidence_mismatch')
        if not audit_citations(a,row) or not audit_citations(b,row):
            quarantined.append({'state_hash':key,'task_id':row['task_id'],'split':row['split'],'reason':'invalid_or_missing_verbatim_review_citations','api':a,'task_agent':b});continue
        merged={**row,'reviews':[{**a,'candidate_hash':compute_candidate_hash(row),'ledger_key':ledger_key},
          {**b,'candidate_hash':compute_candidate_hash(row)}],'risk_labels':sorted(set(a['risk_labels'])&set(b['risk_labels']))}
        try:validate_row(merged)
        except (ValueError,TypeError,KeyError) as exc:
            quarantined.append({'state_hash':key,'task_id':row['task_id'],'reason':str(exc)});continue
        accepted.append(merged)
    train=[r for r in accepted if r['split']=='train'];val=[r for r in accepted if r['split']=='validation']
    # Initial searches are a uniform extra candidate per fixed question. Select
    # them only to represent questions otherwise missing after semantic exclusion.
    represented={r['task_id'] for r in val if r['evidence']}
    val=[r for r in val if r['evidence'] or r['task_id'] not in represented]
    # Round-robin tasks preserves question diversity before optional extra states.
    groups={}
    for row in train:groups.setdefault(row['task_id'],[]).append(row)
    selected=[]
    for i in range(max(map(len,groups.values()),default=0)):
        selected.extend(rows[i] for rows in groups.values() if i<len(rows))
    selected=selected[:400]
    if {r['task_id'] for r in val}-set(manifest['internal_ids']):raise ValueError('internal_manifest_mismatch')
    if {r['task_id'] for r in selected}-set(manifest['training_ids']):raise ValueError('train_manifest_mismatch')
    gate=validate_dataset(selected,val)
    if not budget_gate['passed']:
        gate['passed']=False;gate['errors'].extend(budget_gate['errors'])
    if {r['task_id'] for r in val}!=set(manifest['internal_ids']):
        gate['passed']=False;gate['errors'].append('not_all_50_frozen_internal_questions_covered')
    artifacts={}
    for name,rows in [('train',selected),('validation',val),('quarantine',quarantined)]:
        path=OUT/(name+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        artifacts[name]={'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(rows)}
    gate.update(version='v112',approved=gate['passed'],source_adapter_sha256=SOURCE_SHA,artifacts=artifacts,
      manifest_sha256=digest(manifest),counts={'train':counts(selected),'validation':counts(val)},
      review_coverage={'api':len(api),'task_agent':len(task),'pending':len(pending),'quarantined':len(quarantined)},
      qualified_total=len(accepted),qualified_not_selected=len(accepted)-len(selected)-len(val),
      human_ground_truth=False,budget_gate=budget_gate)
    (ROOT/'reports/v112-data-gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={'version':'v112','phase':'data_gate_passed_training_not_started' if gate['passed'] else 'data_gate_failed_no_training',
      'data_gate':gate,'ledger':ledger,'prior_921_rows_unchanged':budget_gate['prior_rows_sha256']==manifest['opening_ledger']['rows_sha256'],
      'cumulative_cny':sum(r[3] for r in ledger),'new_cny':sum(r[3] for r in ledger if r[0].startswith('v112-')),
      'remaining_cny':None if budget_gate['policy']['mode']=='uncapped' else 60-sum(r[3] for r in ledger),'budget_policy':budget_gate['policy'],'new_training_started':False,'independent100_evaluated':False,
      'final_test_accessed':False,'v101_replaced':False,'instance_shutdown_requested':False}
    training_path=ROOT/'reports/v112-training-report.json'
    if training_path.exists():
        training=json.loads(training_path.read_text(encoding='utf-8'))
        summary['new_training_started']=training.get('execute') is True
        summary['training_phases']=list(training.get('phases',{}))
        summary['source_adapter_unchanged']=training.get('source_before')==training.get('source_after')
        if 'formal' in summary['training_phases']:summary['phase']='training_complete_internal_pending'
    comparison=ROOT/'reports/v112-internal-comparison.json'
    if comparison.exists():
        internal=json.loads(comparison.read_text(encoding='utf-8'))
        summary['internal_comparison']=internal
        summary['phase']='internal_numeric_pass_semantic_pending' if all(internal['numeric_gates'].values()) else 'internal_failed_no_independent_eval'
    (ROOT/'reports/v112-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'phase':summary['phase'],'counts':gate['counts'],'review_coverage':gate['review_coverage'],'errors':gate['errors'],'new_cny':summary['new_cny']},ensure_ascii=False))
if __name__=='__main__':run()
