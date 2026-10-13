"""Record the main agent's once-only source review of 32 changed dev contexts.

No model calls, training, gold-based classification, or changes to frozen inputs.
"""
import hashlib,json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'runs/v133-goal-ah-entity-relation'
# index, category, entity mismatch, relation mismatch, complete, source-bound reason
DECISIONS=[
(0,'unsupported_by_visible_evidence',True,True,False,'Highest-paid skateboarder and highest-paid actress are different scopes; no paragraph proves the requested global highest-paid person or skateboarding-and-comedy features.'),
(1,'unsupported_by_visible_evidence',False,True,False,'Refuge-unit membership does not prove the island is located in the Bering Sea; sea etymology alone is insufficient.'),
(2,'factual_error',True,False,False,'Gloria is born in Delta State; the named judge is explicitly of Rivers State, a different target jurisdiction.'),
(3,'unsupported_by_visible_evidence',True,False,False,'Sammy Baloji works in Lubumbashi but is not any of the film actors identified in the film paragraph.'),
(4,'success',False,False,True,'Smith born Poplarville; Poplarville explicitly a city in Pearl River County. The answer includes the county type, repairing the previously ambiguous bare name.'),
(5,'unsupported_by_visible_evidence',False,True,False,'2015–16 concerns UEFA qualification policy, not an FA Cup win by Crossley’s team.'),
(6,'success',False,False,True,'Rapana is identified as a Royal Dutch/Shell tanker; Shell explicitly announced in April 2010 that it was seeking a buyer for Finnish operations.'),
(7,'success',False,False,True,'Brundage’s IOC presidency binds the organization; the IOC president paragraph explicitly names incumbent Thomas Bach.'),
(8,'factual_error',False,True,True,'Film star Chuck Norris maps to Cordell Walker in the visible cast table. Simon Moon is a character in Hero and the Terror, not the requested Walker role.'),
(9,'factual_error',False,True,True,'Millais painted The Blind Girl; the biographical birth–death range explicitly ends 13 August 1896. Answer 8 June 1829 is his birth date.'),
(10,'success',False,False,True,'Common Sense authored by Paine and the Paine paragraph explicitly states its role crystallizing rebellious demands.'),
(11,'unsupported_by_visible_evidence',False,True,False,'Saunders performed the Shrek 2 version; being its performer does not answer the requested performer’s spouse. Mirrorball collaboration does not establish marriage.'),
(12,'unsupported_by_visible_evidence',True,True,False,'CEMM context binds Compaq, but the merger text concerns unrelated P&O operations; no Compaq merger partner is visible.'),
(13,'success',False,False,True,'Father Antiochus I explicitly a king of Seleucid Empire. Retain the inherited generic political country-association interpretation, not birthplace or legal citizenship.'),
(14,'success',False,False,True,'Film directed by Yurek Bogayevicz; his biography explicitly says born in Poznań.'),
(15,'unsupported_by_visible_evidence',False,True,False,'Father Henry VIII identified, but duchy rule is not place of death. No death location in his visible paragraph.'),
(16,'success',False,False,True,'Film identifies director William Keighley; biography birth–death pair gives death in New York, New York.'),
(17,'success',False,False,True,'Antonio’s father Ippolito identified; Ippolito’s lifespan ends 29 June 1688. Retain the inherited accepted year precision.'),
(18,'success',False,False,True,'Robert’s father Montagu identified; Montagu’s lifespan ends 25 July 1666.'),
(19,'missing_required_information',False,False,False,'No concrete answer, and final input only gives Emperor Daigo’s reign, not the father’s birthplace chain.'),
(20,'success',False,False,True,'Armless directed by Habib Azar; biography explicitly calls him American.'),
(21,'unsupported_by_visible_evidence',True,True,False,'Edinburgh belongs to Sir James’s description, not his father’s death; final shortened paragraphs also lack the parent binding and father’s death location.'),
(22,'unsupported_by_visible_evidence',True,False,False,'Father Bernard VII identified but birthplace absent. Ornans is explicitly painter Courbet’s birthplace in another document.'),
(23,'success',False,False,True,'Antónia’s husband Lajos Batthyány identified; husband explicitly executed in Pest.'),
(24,'unsupported_by_visible_evidence',False,False,False,'Fourth duke death year 1575 is visible, but the fifth duke’s final visible excerpt does not identify his father; title numbering alone cannot establish parentage.'),
(25,'success',False,False,True,'Butse Kik performed by Yoyoy Villame; he died at Las Piñas Medical Center in Manila. Added city is explicitly supported.'),
(26,'success',False,False,True,'Film directed by Reinhold Schünzel; born in St. Pauli, explicitly part of Hamburg. Added city is supported.'),
(27,'success',False,False,True,'William II son of Francesco I; Francesco I explicitly died 1397.'),
(28,'unsupported_by_visible_evidence',False,True,False,'Laren located in Gooi region, but location is not the requested person in charge. No leadership relation visible.'),
(29,'success',False,False,True,'Sinagra’s father Diego Maradona identified; FC Barcelona explicitly signed him June 1982.'),
(30,'success',False,False,True,'Hagar’s union with Abraham is explicit; Keturah paragraph explicitly states Abraham married her after Sarah’s death.'),
(31,'unsupported_by_visible_evidence',True,True,False,'Jamie Ward’s clubs are visible, but no club FA Cup victory year; 1995 is unrelated Arthur English’s death year.'),
]

def main():
    rows=[json.loads(x) for x in (OUT/'novel-final-contexts-goal_z_dev-42-A.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(rows)==len(DECISIONS)==32
    result=[]
    for index,category,entity,relation,complete,reason in DECISIONS:
        row=rows[index];call=row['calls'][-1];quotes=[];seen=set()
        for d in call['documents']:
            if d['document_id'] in seen:continue
            seen.add(d['document_id'])
            # Whole parsed paragraph is a verifiable source excerpt; its actual
            # input location already comes from token-roundtrip CPU decoding.
            quotes.append(dict(document_id=d['document_id'],quote=d['text'],
                input_char_span=d['input_char_span'],input_token_span=d['input_token_span'],
                call_index=call['call_index'],span_scope='document JSON block'))
        result.append(dict(case_id=row['case_id'],split=row['split'],model=row['model'],
            category=category,entity_mismatch=entity,relation_mismatch=relation,
            evidence_complete=complete,reason=reason,source_quotes=quotes,
            actual_final_prompt_sha256=call['prompt_sha256'],answer=row['answer'],
            reviewer='main agent',scoring_version='AH paired source rubric v4'))
    dest=OUT/'candidate-dev-42-A-main-review.jsonl'
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result).encode('utf8')
    if dest.exists():assert dest.read_bytes()==data
    else:dest.write_bytes(data)
    print(json.dumps({'reviews':len(result),'success':sum(r['category']=='success' for r in result),'sha256':hashlib.sha256(data).hexdigest()}))

if __name__=='__main__':main()
