"""Author-approved, bounded CPU queries. Run on AutoDL only, before retrieval."""
import json
from pathlib import Path
P = Path('/root/autodl-tmp/mini-agent-rl/runs/v133-goal-ai-evidence-feasibility/closure-v3')
specs = [
 (0,'Carpenter South Dakota county',['Carpenter','South Dakota']),
 (1,'major Russian city bordering the Baltic Sea',['Russian','Baltic Sea']),
 (2,'Malaysia formation year Sarawak unification',['Malaysia','Sarawak']),
 (3,'Colorado date became part of the United States',['Colorado','United States']),
 (6,'Mel Ferrer spouse',['Mel Ferrer']),
 (7,'Antarctica area size',['Antarctica']),
 (8,'mayor of Brescia',['Brescia']),
 (9,'who founded the Maryland colony and who first settled it',['Maryland']),
 (10,'Sri Lanka independence date from British Empire',['Sri Lanka','British Empire']),
 (11,'Chelsea Champions League winning manager',['Chelsea','Champions League']),
 (17,'Tacoma county location',['Tacoma']),
 (20,'John Cale mother',['John Cale']),
 (26,'McDowell County West Virginia adjacent counties',['McDowell','West Virginia']),
 (28,'Joshiy place of birth',['Joshiy']),
 (29,'Maria Nowakowska birthplace',['Maria Nowakowska']),
 (30,'Damian Green arrest headlines',['Damian Green']),
]
def main():
 packets = list(map(json.loads,(P/'train-review-packets.jsonl').read_text().splitlines()))
 output=[]
 for i,q,entities in specs:
  p=packets[i];k=len(p['calls'])-1
  output.append(dict(case_id=p['case_id'],call_index=k,query=q,grounded_entities=entities,
    predicate_source='Question requests this relation; all named entities come from question or actual prefix. No answer object supplied.',
    authority='lead manual query review before retrieval',new_queries_per_case=1))
 assert len(output)==16 and len({x['case_id'] for x in output})==16
 with (P/'repair-query-freeze.jsonl').open('x',encoding='utf8') as f:
  for x in output:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 print(json.dumps({'frozen_cpu_queries':len(output),'new_model_calls':0}))
if __name__=='__main__':main()
