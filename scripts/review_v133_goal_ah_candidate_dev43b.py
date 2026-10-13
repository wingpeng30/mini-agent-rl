"""Main-agent source review for the three new seed43-B dev contexts."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/v133-goal-ah-entity-relation'
DECISIONS=[
('unsupported_by_visible_evidence',True,True,False,'Skateboarder and actress describe different people and occupations, not two features of one global highest-paid person. Highest-paid profession-specific scope does not prove the question’s target.'),
('missing_required_information',False,False,True,'Owner Shell and its April 2010 process of seeking a buyer for Finnish operations are visible. A potential buyer alone omits the requested process and necessary operations/location qualifier.'),
('unsupported_by_visible_evidence',True,True,False,'Oleśnica is probably the birthplace of a different Henry Wenceslaus, not father Henry VIII the Sparrow’s place of death. Father death location absent.'),
]
def main():
    rows=list(map(json.loads,(OUT/'novel-dev-43-B.jsonl').read_text(encoding='utf8').splitlines()));assert len(rows)==len(DECISIONS)
    result=[]
    for row,(category,entity,relation,complete,reason) in zip(rows,DECISIONS):
        call=row['calls'][-1];quotes=[];seen=set()
        for d in call['documents']:
            if d['document_id'] in seen:continue
            seen.add(d['document_id']);quotes.append(dict(document_id=d['document_id'],quote=d['text'],input_char_span=d['input_char_span'],input_token_span=d['input_token_span'],call_index=call['call_index'],span_scope='document JSON block'))
        result.append(dict(case_id=row['case_id'],split=row['split'],model=row['model'],category=category,entity_mismatch=entity,relation_mismatch=relation,evidence_complete=complete,reason=reason,source_quotes=quotes,actual_final_prompt_sha256=call['prompt_sha256'],answer=row['answer'],reviewer='main agent',scoring_version='AH paired source rubric v5'))
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result).encode('utf8');dest=OUT/'candidate-dev-43-B-main-review.jsonl'
    if dest.exists():assert dest.read_bytes()==data
    else:dest.write_bytes(data)
    print('Reviewed',len(result))
if __name__=='__main__':main()
