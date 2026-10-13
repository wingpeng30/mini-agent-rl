"""One-pass explicit source-grounded AT reward adjudication; no sampling."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/v133-goal-at-supported-action-learning-v1'
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]
states=rows(OUT/'frozen-states.jsonl');samples=rows(OUT/'samples.jsonl')
# Keys are source state IDs, never case ordinals or target-string hits.
decisions={
'ai-answer-correction:musique:2hop__407314_80178':([0,0,0,0],'The requested collaborating singer is Jaden Smith; Justin Bieber is the tour performer and lead artist, not the requested partner.'),
'ai-answer-correction:2wiki:b2addc160bdc11eba7f7acde48001122':([0,1,1,0],'The requested institution is University of Helsinki. Professor and the subject taught name an occupation, not the institution.'),
'ai-answer-correction:2wiki:a874fa560bda11eba7f7acde48001122':([1,0,1,0],'Louis King was born in Christiansburg, Virginia. June 28, 1898 is his birth date, not birthplace.'),
'ai-answer-correction:2wiki:fa9654640bdb11eba7f7acde48001122':([1,0,1,1],'The body dates Kennedy birth to September 3, 1922. Year 1922 is a supported coarse when-answer; September 3 alone omits the identifying year.'),
'ai-recovery-post-repair-answer:musique:2hop__130364_92585':([1,0,0,1],'The first permanent European Maryland colony was founded by English settlers. John Cabot explored earlier and does not name the requested settler group.'),
'ai-recovery-post-repair-answer:musique:2hop__131631_51268':([1,0,0,1],'Colorado entered the Union August 1, 1876; March 3, 1875 is the enabling act, a different event.'),
'ai-recovery-post-repair-answer:musique:2hop__202781_126055':([1,1,1,0],'The frozen Brescia body identifies mayor Emilio Del Bono; Lene Conradi is not the mayor of the requested city.'),
}
by={s['state_id']:s for s in states};out=[]
for r in samples:
 s=by[r['state_id']]
 if r['state_id'] in decisions:
  values,reason=decisions[r['state_id']];reward=values[r['candidate']]
 elif r['state_id']=='ai-recovery-post-repair-answer:hotpot:5a84b1e75542997b5ce3ff14':
  reward=1;reason='Parliamentary privilege in the United Kingdom is a supported expanded name of the same legal doctrine.'
 elif r['state_id']=='ai-supported-answer:2wiki:4b5e77620bd911eba7f7acde48001122':
  reward=1;reason='The lifespan June 28, 1885–September 24, 1953 contains the correctly selected death endpoint. Extraneous supported birth information is not a factual or relationship error.'
 else:
  if r['preliminary_reward']!=1:raise RuntimeError('Unreviewed alternate answer '+str(r))
  reward=1;reason='The exact or equivalent answer is supported by the explicitly mapped source relationships in this state; reviewed with both necessary body citations.'
 out.append(dict(state_id=r['state_id'],candidate=r['candidate'],answer=r['answer'],reward=reward,source_supported=bool(reward),reason=reason,
  actual_citations=s['actual_citations'],review='main-agent one-pass source-grounded; no external facts',training_only=True))
with (OUT/'reward-adjudications.jsonl').open('x',encoding='utf-8') as f:
 for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps(dict(reviewed=len(out),supported=sum(r['reward'] for r in out))))
