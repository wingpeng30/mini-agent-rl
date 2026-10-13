"""Main-agent review of nine new seed42-B development contexts, CPU only."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/v133-goal-ah-entity-relation'
DECISIONS=[
('unsupported_by_visible_evidence',False,True,False,'Adams is among highest-paid actresses, not proved to be the globally highest-paid person. Returned features are hers but the required highest-paid scope is missing.'),
('factual_error',False,True,True,'Chuck Norris maps to Cordell Walker in Walker, Texas Ranger; Simon Moon is the other film’s character.'),
('unsupported_by_visible_evidence',False,True,False,'A Hoffman Academy Award nomination is explicit, but the claimed 78th ceremony is only in a retrieved title. A title alone cannot prove this specific recognition.'),
('unsupported_by_visible_evidence',True,True,False,'Visible Compaq developer is unrelated to the P&O merger paragraph; no Compaq merger partner proved.'),
('unsupported_by_visible_evidence',True,True,False,'Opole is maternal grandfather’s title, not father Henry VIII’s place of death.'),
('success',False,False,True,'Elizabeth’s mother Christina identified; Christina explicitly born Torgau.'),
('success',False,False,True,'Journal of a Crime directed by Keighley; his birth–death pair gives New York, New York death.'),
('unsupported_by_visible_evidence',False,True,False,'Tony Parker majority ownership and past 2011 play at ASVEL do not establish unqualified current works-at relation.'),
('unsupported_by_visible_evidence',False,True,False,'Bernard VII is Constable of France; office country is not birthplace proof.'),
]
def main():
    rows=list(map(json.loads,(OUT/'novel-dev-42-B.jsonl').read_text(encoding='utf8').splitlines()));assert len(rows)==len(DECISIONS)
    result=[]
    for row,(category,entity,relation,complete,reason) in zip(rows,DECISIONS):
        call=row['calls'][-1];quotes=[];seen=set()
        for d in call['documents']:
            if d['document_id'] in seen:continue
            seen.add(d['document_id']);quotes.append(dict(document_id=d['document_id'],quote=d['text'],input_char_span=d['input_char_span'],input_token_span=d['input_token_span'],call_index=call['call_index'],span_scope='document JSON block'))
        result.append(dict(case_id=row['case_id'],split=row['split'],model=row['model'],category=category,entity_mismatch=entity,relation_mismatch=relation,evidence_complete=complete,reason=reason,source_quotes=quotes,actual_final_prompt_sha256=call['prompt_sha256'],answer=row['answer'],reviewer='main agent',scoring_version='AH paired source rubric v5'))
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result).encode('utf8');dest=OUT/'candidate-dev-42-B-main-review.jsonl'
    if dest.exists():assert dest.read_bytes()==data
    else:dest.write_bytes(data)
    print('Reviewed',len(result))
if __name__=='__main__':main()
