"""Remote-only closure export. Semantic decisions below are lead reviews, not heuristics."""
import json, hashlib
from collections import Counter
from pathlib import Path
import close_v133_goal_ai as c

# edge = subject, relation, object, source title. Full quoted bodies and exact
# input token positions are exported. A title match merely LOCATES a reviewed
# passage; it never automatically makes a missing edge supported.
def E(s,r,o,t): return [s,r,o,t]
T = [
 ('recovery',['entity_mismatch','missing_edge'],'Clark County',[E('SoDak Con','named after','South Dakota','SoDak Con'),E('Carpenter, South Dakota','located in county','Clark County','Carpenter, South Dakota')],'Different Carpenter/state was selected; bridge state is visible.'),
 ('rejected',['missing_edge','missing_qualifier'],None,[E('Vilsandi','located in','Baltic Sea','Vilsandi')],'Russian shore passage identifies Saint Petersburg area, but does not establish the requested major-city qualification/unique city. Keep as candidate, not a fully supported target.'),
 ('recovery',['missing_edge'],'1963',[E('Titik','located in country','Malaysia','Titik'),E('Malaysia','formed by unification in','1963','Malaysia')],'Formation, not Malayan independence in 1957; text explicitly unites Sarawak in 1963.'),
 ('recovery',['entity_mismatch','missing_edge'],'August 1, 1876',[E('Pawnee National Grassland','located in state','Colorado','Pawnee National Grassland'),E('Colorado','admitted to Union on','August 1, 1876','History of Colorado')],'Pawnee City, Nebraska is a different entity. Interpret US membership as state admission, as the source states.'),
 ('correct',[],'a heart attack',[E('The Path She Chose','director','Phil Rosen','The Path She Chose'),E('Phil Rosen','died of','heart attack','Phil Rosen')],'Current rollout is supported; legacy failure status does not make this a current failed prefix.'),
 ('correct',[],'Holyhead County School',[E('Claude Whatham','died in','Anglesey','Claude Whatham'),E('Holyhead County School','comprehensive established in','Anglesey in 1949','Comprehensive school')],'Resolve prior A/B disagreement: comprehensive source explicitly gives school, location and year.'),
 ('recovery',['entity_mismatch','missing_edge'],'Audrey Hepburn',[E('Cabriola','director','Mel Ferrer','Cabriola'),E('Mel Ferrer','was husband of','Audrey Hepburn','Mel Ferrer')],'Director is not film star Marisol, nor politician Marisol Touraine. Source explicitly names a spouse; no uniqueness across all life periods claimed.'),
 ('recovery',['relation_mismatch','missing_edge'],'14,000,000 square kilometres',[E('Foggy Pass','located on continent','Antarctica','Foggy Pass'),E('Antarctica','area','14,000,000 square kilometres','Antarctica')],'Returning continent name does not answer area; units must be retained.'),
 ('recovery',['entity_mismatch','missing_edge'],'Emilio Del Bono',[E('Gianluigi Scalvini','born in','Brescia','Gianluigi Scalvini'),E('Brescia','mayor in frozen source','Emilio Del Bono','Brescia')],'Active painter Pietro Scalvini is not city leader. Answer is relative to frozen source, not a present-day claim; in-charge interpreted as municipal mayor.'),
 ('recovery',['relation_mismatch','missing_edge'],'English',[E('Takoma Academy','located in','Maryland','Takoma Academy'),E('Maryland permanent European colony','created by settlers','English in 1634','History of Maryland')],'School operator is not colonial settler. English created permanent colony per passage; first refers to European colony, not first human inhabitants.'),
 ('recovery',['relation_mismatch','missing_edge'],'February 4, 1948',[E('Kavisigamuwa','located in country','Sri Lanka','Kavisigamuwa'),E('Sri Lanka / Ceylon','independent from British Empire on','February 4, 1948','Sri Lankan independence movement')],'British Empire events in 1968 do not establish Sri Lankan independence.'),
 ('rejected',['entity_mismatch','missing_edge'],None,[E('Joe McLaughlin','played for','Chelsea','Joe McLaughlin (footballer)')],'Query retrieves Chelsea manager honours, but production budgeting drops ALL document bodies in the next 245-token input. Cannot export a sufficient answer.'),
 ('rejected',['ambiguity','relation_mismatch'],None,[E('Luis Villafañe','nationality','Puerto Rican','Luis Villafañe')],'Pipeline endpoints/name resemblance do not prove named-after relation; question/entity naming ambiguous.'),
 ('answer_only',['entity_mismatch','sufficient_but_wrong'],'Jaden Smith',[E('Believe Tour','performer','Justin Bieber','Believe Tour'),E('Never Say Never','features rap by','Jaden Smith','Never Say Never (Justin Bieber song)')],'Question asks accompanying performer, not bridge performer. Rap contribution is a valid vocal contribution.'),
 ('answer_only',['relation_mismatch','sufficient_but_wrong'],'University of Helsinki',[E('Ossian Donner','father','Otto Donner','Ossian Donner'),E('Otto Donner','professor at','University of Helsinki','Otto Donner')],'Profession professor is not institution; sufficient chain already visible.'),
 ('rejected',['ambiguity'],None,[E('I Need You','songwriter','George Harrison','I Need You (Beatles song)'),E('Something','written for','Pattie Boyd','Something (Beatles song)')],'Source supports Something, but question names Something For. Do not silently repair title and use as certain target.'),
 ('answer_only',['missing_qualifier','sufficient_but_wrong'],'September 3, 1922',[E('Return of the Seven','director','Burt Kennedy','Return of the Seven'),E('Burt Kennedy','born on','September 3, 1922','Burt Kennedy')],'Month/day answer omits necessary birth year.'),
 ('rejected',['relation_mismatch','missing_edge'],None,[E('David Friesen','born in city','Tacoma','David Friesen'),E('PCCNG papers','distributed around','Tacoma','Pierce County Community Newspaper Group')],'Distribution around Tacoma does not establish Tacoma county membership, even though answer matches gold.'),
 ('correct',[],'ETH Zurich',[E('Bluebottle OS','developed at','ETH Zurich','Bluebottle OS'),E('ETH Zurich','STEM university located in','Zürich, Switzerland','ETH Zurich')],'Both roles/location explicitly stated; full chain passes.'),
 ('answer_only',['relation_mismatch','sufficient_but_wrong'],'Christiansburg, Virginia',[E('Terror Mountain','director','Louis King','Terror Mountain'),E('Louis King','born in','Christiansburg, Virginia','Louis King')],'Birth date was returned for birthplace question.'),
 ('recovery',['entity_mismatch','missing_edge'],'Margaret Davies',[E('Animal Justice','performer','John Cale','Animal Justice'),E('John Cale','mother','Margaret Davies','John Cale')],'Fictional Dashiki/actress Tracey Jones is not the real musician’s mother.'),
 ('correct',[],'Christopher Mitchum',[E('Murder in a Blue World','stars','Christopher Mitchum','Murder in a Blue World'),E('Christopher Mitchum','occupation/nationality','American businessman','Christopher Mitchum')],'Christopher and Chris are compatible aliases.'),
 ('correct',[],'24 September 1953',[E('Rhodes of Africa','director','Berthold Viertel','Rhodes of Africa'),E('Berthold Viertel','died on','24 September 1953','Berthold Viertel')],'Full chain supported.'),
 ('correct',[],'Roger Dean Stadium',[E('Sam Fishburn','played for','St. Louis Cardinals','Sam Fishburn'),E('St. Louis Cardinals','spring training venue in frozen source','Roger Dean Stadium','Roger Dean Stadium')],'Question asks team venue, not venue during Fishburn’s 1919 playing season; preserve this temporal distinction.'),
 ('correct',[],'England',[E('Delamere Way / Stockton Heath','in English county','Cheshire / England','Delamere Way')],'Direct country evidence is sufficient; requiring every official support document would incorrectly reject this shortcut.'),
 ('correct',[],'6 October 877',[E('Richilde of Provence','married to','Charles the Bald','Richilde of Provence'),E('Charles the Bald','died on','6 October 877','Charles the Bald')],'Consort plus explicit marriage supports spouse; equivalent wording allowed.'),
 ('rejected',['missing_edge','ambiguity'],None,[E('Elbert, West Virginia','located in county','McDowell County, West Virginia','Elbert, West Virginia')],'No adjacent-county edge returned. Gold Avery may involve same-name McDowell in another state; unresolved, not asserted as a proven gold error.'),
 ('correct',[],'Wallace Shawn',[E('Marvin Cohen play','staged reading featured','Wallace Shawn','Marvin Cohen (American writer)'),E('Wallace Shawn','played character','Vizzini in The Princess Bride','Wallace Shawn')],'Both role bindings visible in full bodies.'),
 ('rejected',['relation_mismatch','missing_edge'],None,[E('Kodungattu','director','Joshiy','Kodungattu'),E('Joshiy','from','Varkala','Joshiy')],'Explicit correction to v2: from is not born in; born date elsewhere in sentence does not bind birthplace.'),
 ('rejected',['relation_mismatch','missing_edge'],None,[E('Maria Nowakowska','student from','Legnica','Maria Nowakowska'),E('Battle of Legnica','took place on','9 April 1241','Battle of Legnica')],'Explicit correction to v2: student from is not birthplace. Event occurrence also does not independently prove end date; first edge already fails.'),
 ('recovery',['missing_edge'],'Parliamentary privilege',[E('Ashford MP since 1997 / First Secretary since 2017','identity','Damian Green','Damian Green'),E('Parliamentary privilege doctrine','made headlines in relation to arrest of','Damian Green','Parliamentary privilege in the United Kingdom')],'Replace repeated Damian Green query by question-grounded arrest/headlines relation.'),
 ('correct',[],'Chicago, Illinois',[E('Dopey Dicks','director','Edward Bernds','Dopey Dicks'),E('Edward Bernds','born in','Chicago, Illinois','Edward Bernds')],'Chicago, Illinois entails Chicago; EM mismatch is not a factual error.'),
]

# Original diagnoses preserve AH final scores. These are versioned explanatory
# annotations; no retrospective score changes or evaluation-to-training exports.
D = [
 ('entity_mismatch','Delta State chief judge, not Rivers State','Gloria Kemasuode','born in','Delta State','Gloria Kemasuode'),
 ('missing_edge','OTC/prescription authority → agency → food-safety system; agency alone is not system','question-defined drug agency','has food safety system','?','FDA Food Safety Modernization Act'),
 ('missing_edge','Jamie Ward team → FA Cup win year; several historical teams require qualification','Jamie Ward','joined in 2015','Nottingham Forest','Jamie Ward'),
 ('relation_mismatch','Mark Crossley team → FA Cup victory; 2015–16 qualification rule is not victory','Mark Crossley','played for notably','Nottingham Forest','Mark Crossley'),
 ('ambiguity','Hagar union/child does not establish spouse; Abraham → marriage after Sarah also missing','Hagar','union / child with','Abraham / Ishmael','Hagar'),
 ('entity_mismatch','father of fifth duke → father death; own birth year 1544 is wrong entity and relation','Juan fifth duke','birth year','1544','Juan de la Cerda, 5th Duke of Medinaceli'),
 ('relation_mismatch','Tony Parker → requested workplace/time; owner/retired former player is not current employer','Balance-toi','recorded by','Tony Parker','Balance-toi'),
 ('entity_mismatch','Shrek 2 performer → spouse; Daily’s version is for Cover Up','Holding Out for a Hero in Shrek 2','recorded by','Jennifer Saunders','Holding Out for a Hero'),
 ('missing_edge','island sea-location and sea → named for person; refuge unit alone is not direct sea location','St. Matthew Island','protected as part of','Bering Sea refuge unit','St. Matthew Island'),
 ('missing_edge','Bonne father Bernard VII → birthplace','Bonne of Armagnac','father','Bernard VII','Bonne of Armagnac'),
 ('entity_mismatch','Frances spouse Charles → birthplace; Dickey/Anselm is a different person','Frances Tupper','spouse','Charles Tupper','Frances Tupper'),
 ('missing_edge','Wenceslaus father Henry VIII Sparrow → deathplace; ruled territory is not deathplace','Wenceslaus of Krosno','father','Henry VIII the Sparrow','Wenceslaus of Krosno'),
 ('relation_mismatch','Witold birthplace; nationality/school location not birthplace','Axiliad','director','Witold Leszczyński','Axiliad'),
 ('missing_edge','Horatio father Thomas second baron → birthplace; burial/marriage place is not birthplace','Horatio Powys','father','Thomas Powys, 2nd Baron Lilford','Horatio Powys'),
 ('entity_mismatch','Philipp mother Margarete → birthday; father birthday was used','Philipp Albrecht','mother','Margarete Sophie of Austria','Philipp Albrecht, Duke of Württemberg'),
 ('relation_mismatch','Shubhendra mother Annapurna → birthplace; moved to Bombay is not born there','Shubhendra Shankar','mother','Annapurna Devi','Shubhendra Shankar'),
 ('ambiguity','father nationality; ruler of Serbia alone does not necessarily establish nationality','Stefan Vladislav','father','Stefan the First-Crowned','Stefan Vladislav'),
 ('entity_mismatch','spoilt character identification → actor; Willy Wonka actor is not supported spoilt-role binding','Johnny Depp','played','Willy Wonka','Charlie and the Chocolate Factory (film)'),
 ('missing_edge','song composer → spouse; composer himself is not spouse','The Rocky Horror Show','music/lyrics/book by',"Richard O'Brien",'The Rocky Horror Show'),
 ('relation_mismatch','Brian birthplace; English-Australian nationality is not birthplace','Day of the Assassin','director','Brian Trenchard-Smith','Day of the Assassin'),
 ('relation_mismatch','Oktibbeha county seat; school-district merger does not establish county seat','Hickory Grove Estates','located in','Oktibbeha County','Hickory Grove Estates, Mississippi'),
 ('relation_mismatch','CSU system total campuses; consortium 7 / regional 12 are different scopes','Desert Studies Center','field station of','California State University','Desert Studies Center'),
 ('relation_mismatch','Guru Gobind Singh → father; meeting Tegh Bahadur does not establish parenthood','Jujhar Singh','father','Guru Gobind Singh','Jujhar Singh'),
 ('entity_mismatch','Adalbert II → father Adalbert I → birthplace; Bologna namesake is different entity','Adalbert I','margrave of','Tuscany','Adalbert I, Margrave of Tuscany'),
]

def citations(ctx, edge):
 out=[]
 for d in ctx['documents']:
  if d['title']!=edge[3]: continue
  out.append(dict(triple=edge[:3],reference=d['reference'],document_id=d['document_id'],title=d['title'],
    quote=d['text'],input_token_span=d['body_input_token_span'],input_char_span=d['body_input_char_span']))
 return out

def example(ctx, action, tok):
 target=json.dumps(action,ensure_ascii=False)
 ids=tok.encode(target,add_special_tokens=False)+[tok.convert_tokens_to_ids('<|im_end|>')]
 assert len(ids)<=128 and len(ctx['prompt_token_ids'])<=2048
 prompt=ctx['prompt_token_ids']; labels=[-100]*len(prompt)+ids
 assert all(x==-100 for x in labels[:len(prompt)]) and labels[len(prompt):]==ids
 return dict(prompt_token_ids=prompt,target_action=action,target_token_ids=ids,
   input_ids=prompt+ids,attention_mask=[1]*(len(prompt)+len(ids)),labels=labels,
   loss_mask=[0]*len(prompt)+[1]*len(ids),target_tokens=len(ids),input_tokens=len(prompt))

def main():
 c.guard();_,tok=c.renderer()
 train=c.rows(c.O/'train-review-packets.jsonl');diag=c.rows(c.O/'diagnostic-review-packets.jsonl')
 repairs={x['plan']['case_id']:x for x in c.rows(c.O/'repair-query-results.jsonl')}
 branches={x['branch_id']:x for x in c.rows(c.O/'budget-reconstruction.jsonl')}
 ledger=[];pairs=[];answers=[];positives=[];revisions=[]
 assert len(train)==len(T)==32 and len(diag)==len(D)==24
 for i,(p,spec) in enumerate(zip(train,T)):
  status,kinds,answer,edges,reason=spec;orig=p['calls'][-1];rr=repairs.get(p['case_id']);ctx=rr['answer_context'] if status=='recovery' else orig
  proofs=[citations(ctx,e) for e in edges]
  if status in ['recovery','answer_only','correct']: assert all(proofs),(i,status,[e[3] for e,z in zip(edges,proofs) if not z])
  row=dict(case_id=p['case_id'],question=p['question'],origin=p['origin'],corpus=p['corpus'],
    target_entity_role=edges[0][0],required_relationship_description=p['question'],reviewed_visible_edges=[e[:3] for e,z in zip(edges,proofs) if z],
    missing_relations=[] if status in ['correct','answer_only','recovery'] else [reason],
    original_missing_or_misused_relation=reason,original_action=orig['parsed_action'],error_types=kinds,
    status=status,reason=reason,edge_citations=proofs,original_input_tokens=orig['actual_input_tokens'],
    original_visible_documents=orig['documents'],budget_repair=None,product_success_counted=False,
    reviewer='lead',review_version=3,training_release_approved=False)
  if rr:
   plan=rr['plan'];k=plan['call_index'];used=sum(bool(z.get('search_observation')) for z in p['calls'][:k])
   row['budget_repair']=dict(query=plan['query'],call_index=k,calls_if_answer=k+2,searches_if_answer=used+1,
      input_tokens=rr['answer_context']['actual_input_tokens'],documents=rr['answer_context']['documents'],
      chain_complete=status=='recovery',model_answer_executed=False,reason=reason)
   assert k+2<=4 and used+1<=3
  if status=='recovery':
   search_example=example(orig,dict(action='search',query=rr['plan']['query']),tok)
   answer_example=example(ctx,dict(action='answer',answer=answer),tok)
   pairs.append(dict(case_id=p['case_id'],origin=p['origin'],corpus=p['corpus'],real_failed_prefix=True,
      original_action=orig['parsed_action'],error_types=kinds,search=search_example,real_observation=rr['result'],
      answer=answer_example,edge_citations=proofs,budget=row['budget_repair'],diagnostic_only=True))
  elif status=='answer_only':answers.append(dict(case_id=p['case_id'],example=example(ctx,dict(action='answer',answer=answer),tok),edge_citations=proofs,not_a_search_recovery_pair=True))
  elif status=='correct':positives.append(dict(case_id=p['case_id'],example=example(ctx,dict(action='answer',answer=answer),tok),edge_citations=proofs,not_a_failed_prefix=True))
  ledger.append(row)
  if i in [5,9,12,13,14,15,17,18,21,24,27,28,29,30,31]:
   prior=next(x for x in c.rows(c.P/'material-source-audit-v2.jsonl') if x['case_id']==p['case_id'])
   revisions.append(dict(case_id=p['case_id'],prior_file='material-source-audit-v2.jsonl',prior_status=prior['status'],new_status=status,reason=reason,original_retained=True,score_changed=False))
 c.save('train-audit-ledger-v3.jsonl',ledger,True);c.save('recovery-pairs-v3.jsonl',pairs,True)
 c.save('answer-only-examples-v3.jsonl',answers,True);c.save('supported-positive-examples-v3.jsonl',positives,True)
 c.save('material-review-revisions-v3.jsonl',revisions,True)
 dl=[];bl=[]
 for i,(p,spec) in enumerate(zip(diag,D)):
  kind,missing,s,r,o,t=spec;last=p['actual_calls'][-1]
  original=c.decode(last['prompt_token_ids'],tok);anchor=p.get('anchor');checks=[]
  for bid in p['branch_ids']:
   b=branches[bid];grounded=not(i==5 and bid.endswith('|0'))
   # Only mother's birthday has the complete requested chain in a legal,
   # grounded independent branch. All other branches have explicit gaps.
   complete=i==14 and bid.endswith('|0') and b['legal_budget'] and grounded
   gap=missing
   if i==5 and bid.endswith('|0'):gap='Hidden father name/ordinal supplied in query before first retrieval. Excluded even though resulting documents contain both edges.'
   if i==14 and not complete:gap='Mother birthday not retrieved in this independent branch.'
   proofs=[]
   if complete:
    proofs=[citations(b,E('Philipp Albrecht','mother','Margarete Sophie of Austria','Philipp Albrecht, Duke of Württemberg')),
      citations(b,E('Margarete Sophie of Austria','born on','13 May 1870','Archduchess Margarete Sophie of Austria'))]
    assert all(proofs)
   item=dict(branch_id=bid,case_id=p['case_id'],query=b['query'],grounded=grounded,
      grounding_reason='All query entities/objects are question or actual anchor evidence' if grounded else gap,
      legal_budget=b['legal_budget'],calls_if_answer=b['calls_if_answer'],searches_if_answer=b['searches_if_answer'],
      actual_input_tokens=b['actual_input_tokens'],visible_documents=b['documents'],complete_chain=complete,
      missing_relation=None if complete else gap,edge_citations=proofs,product_success_counted=False)
   bl.append(item);checks.append({k:item[k] for k in ['branch_id','grounded','legal_budget','complete_chain','missing_relation']})
  dl.append(dict(case_id=p['case_id'],split=p['split'],question=p['question'],target_entity_role=s,
      necessary_relations=missing,reviewed_visible_relations=[[s,r,o]],edge_citations=citations(original,E(s,r,o,t)),
      original_action=last['parsed_action'],original_answer=p['original_answer'],error_types=[kind,'missing_edge'],
      actual_input_tokens=original['actual_input_tokens'],actual_visible_documents=original['documents'],
      missing_relation=missing,branch_checks=checks,recoverable=any(x['complete_chain'] for x in checks),
      source_guided_docs_diagnostic_only=p['source_guided_corpus_docs'],
      corpus_absence_proven=False,score_changed=False,training_eligible=False,review_version=3))
 c.save('diagnostic-ledger-v3.jsonl',dl,True);c.save('branch-audit-v3.jsonl',bl,True)
 coverage=Counter(k for p in pairs for k in p['error_types'])
 nrec=sum(p['recoverable'] for p in dl);nold=sum(p['origin']=='old_real_failure' for p in pairs)
 counts=dict(diagnostic_recoverable=nrec,diagnostic_total=24,branches=len(bl),
    ungrounded_branches=sum(not x['grounded'] for x in bl),recovery_pairs=len(pairs),old_real_failure_pairs=nold,
    answer_only=len(answers),supported_positives=len(positives),training_status=dict(Counter(p['status'] for p in ledger)),coverage=dict(coverage))
 checks=dict(diagnostic_min8=nrec>=8,pairs_min16=len(pairs)>=16,real_failures_min8=nold>=8,
    entity_min4=coverage['entity_mismatch']>=4,relation_min4=coverage['relation_mismatch']>=4,missing_min4=coverage['missing_edge']>=4)
 c.save('gate-decision-v3.json',dict(counts=counts,checks=checks,passed=all(checks.values()),
   decision='Do not launch action-recovery SFT. Recommend one frozen-model necessary-relation coverage/retrieval feasibility experiment; separate context-retention bug is demonstrated but not changed here.',
   product_model='Goal Z trajectory seed42',product_success_measured=False,new_gpu_calls=0,new_model_calls=0,new_cpu_queries=len(repairs),updates=0))
 files=[p for p in c.O.iterdir() if p.is_file()]
 c.save('closure-manifest-v3.json',dict(files={p.name:c.ai.sha(p) for p in files},adapter_sha256=c.ai.sha(c.ai.SOURCE/'adapter_model.safetensors'),script_sha256=c.ai.sha(Path(__file__))))
 print(json.dumps(dict(counts=counts,checks=checks,passed=all(checks.values())),ensure_ascii=False))
if __name__=='__main__':main()
