"""One source-bound resolution of 14 inherited AA pending rows; no inference."""
import json
from prepare_v133_goal_ah import OUT,rows,write_once

DECISIONS={
 '53433_':('unsupported_by_visible_evidence','Treaty signing is explicit, but no country-founding relation is stated; year also omits the requested full date.',None),
 '698577_':('unsupported_by_visible_evidence','Author Lawrence Sanders is visible; Mr. Lawrence is not established as a notable work by him.',None),
 '133102_':('unsupported_by_visible_evidence','Nissan is manufacturer, not full model; paragraph does not bind the car to Atta beyond page title. Title alone is insufficient.',None),
 '48784_':('unsupported_by_visible_evidence','Question typo free crops is retained and flagged. Visible parliament and 1939 annexation passages establish neither the requested 1919 government nor the group composition. No invented correction of question or fact-error claim.',None),
 '728887_':('unsupported_by_visible_evidence','Savignano paragraph concerns the Rubicon; it contains no Assomada location relation.',None),
 '596188_':('unsupported_by_visible_evidence','House passage locates it in Westcombe Park/Greenwich; Maze Hill borough statement does not bind Westcombe Park to that borough. Same Greenwich token is insufficient.',None),
 '699172_':('success','Bachelor degree at Brooklyn College and its membership in CUNY explicitly complete one valid school chain.',
     [('Lawrence Landweber','bachelor degree at','Brooklyn College','Lawrence Landweber'),('Brooklyn College','part of','City University of New York','Brooklyn College')]),
 '593688_':('success','ASC HLM membership plus its home ground in Dakar, Senegal establishes club home country, not national-team representation.',
     [('Pa Amadou Gai','plays for','ASC HLM','Pa Amadou Gai'),('ASC HLM','home ground in country','Senegal','Stade de ASC HLM')]),
 'dc15d27e':('success','Retain generic country-affiliation reading: father is explicitly Emperor Bao Dai of Vietnam. Vietnam is source-supported broader country wording despite narrower Annam gold; no birthplace or legal-citizenship inference.',
     [('Phương Mai','father','Bảo Đại','Phương Mai'),('Bảo Đại','political country affiliation','Vietnam','Phương Mai')]),
 'c400306a':('success','Director and death in New York City are explicit; New York City is the equivalent city answer.',
     [('Along Came Love','director','Bert Lytell','Along Came Love'),('Bert Lytell','died in','New York City','Bert Lytell')]),
 'fbfb55ac':('success','Original 1953 recording explicitly binds Amos Milburn; biography says born Houston and died there. Other recorded versions do not invalidate this supported original-performer chain.',
     [('One Bourbon, One Scotch, One Beer','original recording by','Amos Milburn','One Bourbon, One Scotch, One Beer'),('Amos Milburn','died in','Houston, Texas','Amos Milburn')]),
 '4acbfb1c':('success','Father relation and full birth date are explicit. Full date properly answers birthday even when gold gives only year.',
     [('Lê Hiến Tông','father','Lê Thánh Tông','Lê Hiến Tông'),('Lê Thánh Tông','birth date','25 August 1442','Lê Thánh Tông')]),
 '655505_':('missing_required_information','Unknown does not supply birth date; source identifies performer but no performer birth date. Unrelated Moffat unknown birth date is not evidence about Mauro Scocco.',None),
 '481809_':('factual_error','Horse chain explicitly yields Buckpasser as Tom Fool offspring; answer Tom Owen is explicitly a human actor, son of Bill Owen. Visible entity/type and father contradiction.',None),
}

def main():
    ledger=rows(OUT/'baseline-category-ledger-v2.jsonl')
    actual={r['case_id']:r for r in rows(OUT/'actual-inputs-aa_regression-baseline.jsonl')}
    audit=[];protect=[]
    for r in ledger:
        if r['category']!='ambiguous':continue
        key=next(k for k in DECISIONS if k in r['case_id']);cat,reason,facts=DECISIONS[key]
        case=actual[r['case_id']]
        audit.append(dict(case_id=r['case_id'],old_category='ambiguous',new_category=cat,reason=reason,
            actual_final_prompt_sha256=case['calls'][-1]['prompt_sha256'],actual_visible_documents=case['calls'][-1]['documents'],
            question_ambiguity_flag=key=='48784_',candidate_outputs_seen=False,reviewer='lead',version='AH AA pending once v1'))
        r.update(category=cat,source_bound_resolution=reason,scoring_version='AH baseline v3; explicit once-only AA pending resolution')
        if not facts:continue
        rels=[]
        for sub,pred,obj,title in facts:
            d=next(d for d in case['calls'][-1]['documents'] if d['title']==title)
            earliest=min(c['call_index'] for c in case['calls'] if any(v['document_id']==d['document_id'] and v['text']==d['text'] for v in c['documents']))
            escaped=json.dumps(d['text'],ensure_ascii=False)[1:-1];start=case['calls'][-1]['actual_prompt'].find(escaped,*d['input_char_span']);assert start>=0
            end=start+len(escaped);tt=[i for i,(a,b) in enumerate(case['calls'][-1]['offset_mapping']) if b>start and a<end]
            rels.append(dict(triple=[sub,pred,obj],document_id=d['document_id'],quote=d['text'],first_visible_call=earliest,final_visible=True,
                actual_input_char_span=[start,end],actual_input_token_span=[min(tt),max(tt)+1]))
        protect.append(dict(case_id=r['case_id'],split='aa_regression',supported_success=True,
            protected=any(x['first_visible_call']>=2 for x in rels),reason=reason,necessary_relations=rels,
            actual_final_prompt_sha256=case['calls'][-1]['prompt_sha256'],reviewer='lead'))
    assert len(audit)==14 and len(protect)==6
    write_once(OUT/'aa-pending-adjudication-v1.jsonl',audit,jsonl=True)
    write_once(OUT/'baseline-category-ledger-v3.jsonl',ledger,jsonl=True)
    write_once(OUT/'baseline-aa-additional-protection-main.jsonl',protect,jsonl=True)
    print(json.dumps({'resolved':14,'additional_supported':6,'additional_protected':sum(x['protected'] for x in protect)}))

if __name__=='__main__':main()
