"""Independent serialization, evidence-position, isolation and gate verification."""
import json
from collections import Counter
import close_v133_goal_ai as c

def main():
 c.guard();cfg=c.ai.guard();_,tok=c.renderer()
 manifest=c.read(c.O/'closure-manifest-v3.json')
 for name,h in manifest['files'].items():assert c.ai.sha(c.O/name)==h,name
 assert manifest['adapter_sha256']==c.ai.EXPECTED
 evaluation=[]
 for split in cfg['evaluation']:evaluation.extend(c.rows(c.ROOT/split['cases']))
 def ident(x):return x.get('id',x.get('case_id'))
 eval_ids={ident(x) for x in evaluation};eval_q={c.ai.norm(x['question']) for x in evaluation}
 packets=c.rows(c.O/'train-review-packets.jsonl');byid={p['case_id']:p for p in packets}
 assert not eval_ids&set(byid)
 assert not eval_q&{c.ai.norm(x['question']) for x in packets}
 pairs=c.rows(c.O/'recovery-pairs-v3.jsonl');answers=c.rows(c.O/'answer-only-examples-v3.jsonl');positive=c.rows(c.O/'supported-positive-examples-v3.jsonl')
 samples=[]
 for p in pairs:samples.extend([(p['search'],None),(p['answer'],p['edge_citations'])])
 for p in answers+positive:samples.append((p['example'],p['edge_citations']))
 nproof=0
 for ex,edges in samples:
  n=ex['input_tokens'];target=ex['target_token_ids'];assert n<=2048 and 0<len(target)<=128
  assert ex['input_ids']==ex['prompt_token_ids']+target
  assert ex['labels']==[-100]*n+target and ex['loss_mask']==[0]*n+[1]*len(target)
  assert ex['attention_mask']==[1]*len(ex['input_ids'])
  assert json.loads(tok.decode(target[:-1]))==ex['target_action']
  if edges:
   text=tok.decode(ex['prompt_token_ids'],skip_special_tokens=False)
   offsets=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
   for edge in edges:
    assert edge
    for proof in edge:
     a,b=proof['input_char_span'];quote=json.dumps(proof['quote'],ensure_ascii=False)[1:-1]
     assert text[a:b]==quote
     hits=[i for i,(s,e) in enumerate(offsets) if s<b and e>a]
     assert proof['input_token_span']==[min(hits),max(hits)+1]
     nproof+=1
 query=c.rows(c.O/'repair-query-freeze.jsonl');results=c.rows(c.O/'repair-query-results.jsonl')
 assert len(query)==len(results)==16 and len({p['case_id'] for p in query})==16
 assert len(query)<=32
 coverage=Counter(k for p in pairs for k in p['error_types'])
 old=sum(p['origin'].startswith('old_real_failure_') for p in pairs)
 real=sum(p['real_failed_prefix'] for p in pairs)
 diag=c.rows(c.O/'diagnostic-ledger-v3.jsonl');branches=c.rows(c.O/'branch-audit-v3.jsonl')
 nrec=sum(x['recoverable'] for x in diag)
 gate=dict(diagnostic_min8=nrec>=8,pairs_min16=len(pairs)>=16,real_failures_min8=real>=8,
   entity_min4=coverage['entity_mismatch']>=4,relation_min4=coverage['relation_mismatch']>=4,missing_min4=coverage['missing_edge']>=4)
 counts=dict(diagnostic_recoverable=nrec,diagnostic_total=len(diag),branches=len(branches),
   recovery_pairs=len(pairs),real_goal_z_model_failure_prefixes=real,old_pool_failure_pairs=old,
   answer_only=len(answers),supported_positives=len(positive),coverage=dict(coverage),
   rejected=len([p for p in c.rows(c.O/'train-audit-ledger-v3.jsonl') if p['status']=='rejected']))
 c.save('gate-decision-v4.json',dict(counts=counts,checks=gate,passed=all(gate.values()),
   replaces='gate-decision-v3.json statistics only; semantic ledgers unchanged',
   correction='v3 compared source origin with nonexistent exact old_real_failure enum. Actual old origins have _z/_ac/_ag suffixes. The >=8 criterion concerns real published Goal Z failure prefixes, not membership in its original training set.',
   decision='NO_TRAINING; prepare necessary-relation coverage/retrieval feasibility route',
   new_cpu_queries=16,new_model_calls=0,new_gpu_calls=0,updates=0,product_success_measured=False))
 c.save('verification-v4.json',dict(passed=True,adapter_sha256=c.ai.EXPECTED,
   verified_examples=len(samples),verified_citations=nproof,evaluation_cases_isolated=len(evaluation),
   examples_max_input_tokens=max(e['input_tokens'] for e,_ in samples),
   examples_max_target_tokens=max(e['target_tokens'] for e,_ in samples),
   frozen_manifest_verified=True,no_cross_branch_document_union=True,
   semantic_review='Lead-authored per-edge judgments; these checks validate mechanics, not semantic entailment',
   no_training_launch=True))
 print(json.dumps(dict(counts=counts,checks=gate,verified_examples=len(samples),verified_citations=nproof),ensure_ascii=False))
if __name__=='__main__':main()
