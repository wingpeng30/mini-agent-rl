"""Once-only source-grounded corrections discovered by the frozen axis audit.

Preserves all prior ledgers and the pre-candidate protection membership.
"""
import json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/v133-goal-ah-entity-relation'
def rows(p):return list(map(json.loads,p.read_text(encoding='utf8').splitlines()))
def save(name,rs):
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs).encode('utf8');p=OUT/name
    if p.exists():assert p.read_bytes()==data
    else:p.write_bytes(data)
SUCCESS={
'musique:2hop__506035_325512','5abbfb7f554299114383a0d1','5ac0139a5542992a796dec79',
'5a7e6a235542997cc2c4753f','5ab74fa55542993667793fba','5a74565055429979e288290b',
'2wiki:f77e7afa0bda11eba7f7acde48001122','5a8673585542991e77181662','5a78cc3f554299029c4b5e9b',
'g004','g006','g009','musique:2hop__805610_392181','2wiki:60c4f9dc0bde11eba7f7acde48001122','2wiki:188eb6da0bde11eba7f7acde48001122'}
OTHER={'g001':('missing_required_information','Question asks common occupations plural; director is supported but omits the other explicitly common occupation screenwriter.'),
'musique:2hop__151455_120171':('factual_error','Acornsoft explicitly ceased separately in January 1986; 1985 is game availability, a different relation.')}
AXIS_OVERRIDES={
'5a8ccf10554299653c1aa129':dict(evidence_complete=False,reason='Fetteresso country not explicitly visible; Scottish architecture is not country proof. Do not assign Cowie’s quote to Fetteresso.'),
'5ade30b955429939a52fe859':dict(evidence_complete=False,reason='Indie rock and alternative rock are not proved equivalent by these paragraphs. Shared broad rock is insufficient for the narrower returned subtype.'),
'5ac168ce5542994ab5c67d15':dict(evidence_complete=False,reason='Instrumental rock and alternative rock do not establish shared alternative-rock subtype.'),
'musique:2hop__71755_463572':dict(evidence_complete=False,relation_mismatch=True,reason='Song recording released on Sharp does not establish Joni James artist label affiliation; Asian Man answer remains unsupported.'),
'5ab867475542992aa3b8c8df':dict(relation_mismatch=True,reason='Both belong to Ulster Cycle, but character is entity type, not the requested shared collection.'),
}
def main():
    inherited=rows(OUT/'baseline-category-ledger-v4.jsonl');assert len(inherited)==272
    axis={r['case_id']:r for part in ['first','second'] for r in rows(OUT/f'baseline-error-axis-review-{part}.jsonl')};assert len(axis)==124
    actual={r['case_id']:r for f in OUT.glob('actual-inputs-*-baseline.jsonl') for r in rows(f)}
    assert len(actual)==272
    corrections=[];reviews=[]
    for old in inherited:
        cid=old['case_id'];row=dict(old);context=actual[cid];call=context['calls'][-1]
        if old['category']=='success':traits=dict(entity_mismatch=False,relation_mismatch=False,evidence_complete=True,ambiguity=False,observed_relations=[],missing_relations=[],reason='Source-bound success reviewed in the pre-candidate protection audit; inherited unchanged.')
        else:
            traits=dict(axis[cid]);traits.update(AXIS_OVERRIDES.get(cid,{}))
            for q in traits['observed_relations']:
                assert any(d['document_id']==q['document_id'] and q['quote'] in d['text'] for d in call['documents']), (cid,q)
        if cid in SUCCESS:
            category='success';reason='Actual final source explicitly supports the returned semantic answer; prior unsupported classification was incorrect. '+traits['reason']
        elif cid in OTHER:category,reason=OTHER[cid]
        else:category=old['category'];reason=traits['reason']
        if category!=old['category']:
            corrections.append(dict(case_id=cid,split=old['split'],old_category=old['category'],new_category=category,reason=reason,
                actual_final_prompt_sha256=call['prompt_sha256'],source_relations=traits['observed_relations'],reviewer='main agent',
                scope='Finite 124 previously failed case axis audit, one correction version; no gold, training, GPU, or protection membership change.'))
        row.update({k:traits.get(k) for k in ['entity_mismatch','relation_mismatch','evidence_complete','ambiguity','observed_relations','missing_relations']})
        if category=='success':row.update(entity_mismatch=False,relation_mismatch=False,evidence_complete=True)
        row.update(category=category,reason=reason,answer=context['answer'],scoring_version='AH paired source rubric v5; explicit v4 inherited-category corrections',reviewer='main agent')
        # Attach exact paragraph evidence positions to every row, including
        # successes inherited from the pre-candidate source reviews.
        row['source_quotes']=[dict(document_id=d['document_id'],quote=d['text'],input_char_span=d['input_char_span'],input_token_span=d['input_token_span'],call_index=call['call_index'],span_scope='document JSON block') for d in call['documents']]
        reviews.append(row)
    assert len(corrections)==17 and sum(r['category']=='success' for r in reviews)==163
    save('baseline-axis-adjudication-corrections-v1.jsonl',corrections)
    save('baseline-category-ledger-v5.jsonl',reviews)
    print(json.dumps({'cases':len(reviews),'explicit_category_corrections':len(corrections),'success':sum(r['category']=='success' for r in reviews),'dev_success':sum(r['category']=='success' and r['split']=='goal_z_dev' for r in reviews),'protected_membership_changed':False}))
if __name__=='__main__':main()
