"""Main-agent source review for 13 genuinely new seed43-A dev contexts."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/v133-goal-ah-entity-relation'
DECISIONS=[
('factual_error',True,False,False,'Delta State birthplace is explicit; named judge belongs to Rivers State.'),
('unsupported_by_visible_evidence',True,False,False,'Sammy Baloji is not bound to Sammy Going South; works-at location is for another entity.'),
('success',False,False,True,'Born Poplarville; Poplarville explicitly in Pearl River County.'),
('missing_required_information',False,False,True,'Both Rapana/Shell and April 2010 announcement are visible, but service stations alone omits the requested process of seeking a buyer for Finnish operations.'),
('factual_error',False,True,True,'Chuck Norris maps to Cordell Walker; Simon Moon is the other film’s character.'),
('success',False,False,True,'Paine authored Common Sense; its effect on rebellious demand explicitly stated.'),
('unsupported_by_visible_evidence',True,True,False,'Bank merger concerns Gjensidige NOR in 2003, not Compaq in 2002.'),
('success',False,False,True,'Retain the source-grounded generic political association for father Antiochus I of Seleucid Empire.'),
('unsupported_by_visible_evidence',True,True,False,'Opole is associated with maternal grandfather’s title, not father Henry VIII’s death location.'),
('factual_error',False,True,True,'Montagu’s death 25 July 1666 explicit; 1608 is his birth year.'),
('unsupported_by_visible_evidence',False,True,False,'Northern court location Kyoto does not prove Go-Daigo’s birthplace.'),
('unsupported_by_visible_evidence',False,True,False,'Father Bernard VII is Constable of France; office country does not prove birthplace.'),
('unsupported_by_visible_evidence',True,False,False,'2005 FA Cup winner Arsenal not bound to Jamie Ward’s team; visible clubs do not include Arsenal.'),
]
def main():
    rows=list(map(json.loads,(OUT/'novel-dev-43-A.jsonl').read_text(encoding='utf8').splitlines()))
    assert len(rows)==len(DECISIONS)
    results=[]
    for row,(category,entity,relation,complete,reason) in zip(rows,DECISIONS):
        call=row['calls'][-1];seen=set();quotes=[]
        for d in call['documents']:
            if d['document_id'] in seen:continue
            seen.add(d['document_id']);quotes.append(dict(document_id=d['document_id'],quote=d['text'],input_char_span=d['input_char_span'],input_token_span=d['input_token_span'],call_index=call['call_index'],span_scope='document JSON block'))
        results.append(dict(case_id=row['case_id'],split=row['split'],model=row['model'],category=category,entity_mismatch=entity,relation_mismatch=relation,evidence_complete=complete,reason=reason,source_quotes=quotes,actual_final_prompt_sha256=call['prompt_sha256'],answer=row['answer'],reviewer='main agent',scoring_version='AH paired source rubric v4'))
    dest=OUT/'candidate-dev-43-A-main-review.jsonl';data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results).encode('utf8')
    if dest.exists():assert dest.read_bytes()==data
    else:dest.write_bytes(data)
    print('Reviewed',len(results))
if __name__=='__main__':main()
