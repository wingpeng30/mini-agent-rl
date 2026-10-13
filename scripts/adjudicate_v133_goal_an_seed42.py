"""One-pass primary AN adjudication; CPU only, no new evidence."""
import json
from pathlib import Path
P=Path('runs/v133-goal-an-hard-state-sft-v2'); AM=Path('runs/v133-goal-am-action-sft-v2')
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
answers={}
for p in rows(P/'review-packets-seed42.jsonl'):
 if p['reused_verdict']:continue
 cid=p['case_id']
 if cid=='musique:2hop__864101_135373':
  v=dict(category='success',reason='Both quoted edges are visible: St Matthew Island belongs to the Bering Sea unit and the sea is named for Vitus Bering. Bering is an unambiguous surname here, accepted as same referent; EM differs only by wording.',entity_relation_mismatch=False)
 elif cid=='musique:2hop__796221_18803':
  v=dict(category='incomplete_required_qualifier',reason='Royal Dutch Shell body explicitly states April 2010 trying to find a potential buyer for all operations in Finland. A potential buyer alone omits the announced activity and Finland operations scope. Complete evidence is visible; this is answer incompleteness, not a contradicted factual claim.',entity_relation_mismatch=False)
 elif cid=='musique:2hop__96194_78303':
  v=dict(category='factual_error',reason='Actual Walker cast body binds Chuck Norris to Cordell Walker. The Terror is Simon Moon in the other film, not the requested Norris role in Walker. Both necessary links remain visible.',entity_relation_mismatch=True)
 else:raise RuntimeError('Unreviewed changed combination '+cid)
 answers[cid]=v
old={p['probe_id']:p for p in rows(AM/'probe-review-seed42.jsonl') if p['model']=='42-A' and p['category']=='complete'}
raw_old={p['probe_id']:p for p in rows(AM/'probe-42-A.jsonl')}
complete={}
for p in rows(P/'probe-42-B-complete.jsonl'):
 z=old[p['probe_id']]
 assert p['prompt_token_ids']==raw_old[p['probe_id']]['prompt_token_ids']
 if p['answer']==z['answer']:
  v=dict(correct=z['correct'],reason='Identical frozen actual input and answer to matched AM A: '+z['reason'],binding_repair=False)
 else:
  assert p['case_id']=='musique:2hop__864101_135373' and p['answer']=='Bering'
  v=dict(correct=True,reason=answers[p['case_id']]['reason'],binding_repair=False)
 complete[p['probe_id']]=v
partial={m:{} for m in ['GoalZ','42-A','42-B']}
for p in rows(P/'probe-review-packets-seed42.jsonl'):
 if p['category']!='partial':continue
 cid=p['case_id'];legal=True
 if cid=='musique:2hop__337223_56873':
  legal=False;reason='2015-16 is visible only in FA Cup Europa League qualification rules, not the winning year of Nottingham Forest. Query reuses that unrelated temporal scope and does not resolve the selected team relation. This is misbinding of visible information, not hidden entity leakage.'
 elif p['repeated_query']:
  legal=False;reason='Exact prior Rapana owner April 2010 query repeated; cannot count as legal missing-edge repair.'
 else:
  reason='Query targets the missing relation of a visible bridge entity or its unambiguous question-based relational description. No hidden answer introduced. The first edge remains visible, second edge absent, and search plus final-answer budget remains.'
 partial[p['model']][p['probe_id']]=dict(legal_missing_relation_search=legal,reason=reason,entity_sources=dict(question=p['question'],bridge_entity=p['bridge_entity'],first_edge_citation=p['first_edge_citation']),query=p['query'],missing_relation=p['missing_relation'])
out=P/'primary-adjudications-seed42.json'
assert not out.exists()
out.write_text(json.dumps(dict(answers={'42-B':answers},complete={'42-B':complete},partial=partial),ensure_ascii=False,indent=2))

