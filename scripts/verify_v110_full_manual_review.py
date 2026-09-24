"""Materialize manually adjudicated evidence notes and verify full changed-pair coverage."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from mini_agent_rl.hybrid_v110 import answer_metrics

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return [json.loads(x) for x in (ROOT / name).read_text(encoding='utf-8').splitlines() if x.strip()]


# Decisions were authored after reading questions and evidence, before gold inspection.
# citation tuples identify the task document suffix and an exact visible substring.
NOTES = {
'5ab728e45542992aa3b8c7b4': ('clear_semantic_correction', 'The question asks for a county, not a state. Museum in Deadwood plus its county seat identifies Lawrence County.', [(0,'on display at the Adams Museum & House in Deadwood, South Dakota.'),(1,'the county seat of Lawrence County')]),
'5ab72adb5542993667793f5d': ('clear_semantic_correction','Corrects district to requested territory, explicitly Hong Kong.',[(0,'The Central Harbourfront is a waterfront site in Central, Hong Kong.'),(1,"on the waterfront of Hong Kong's Central District.")]),
'5abae52a5542996cc5e49eea': ('clear_semantic_correction','Race winner James Hunt retired in 1979; 1999 is unrelated tennis evidence.',[(0,'was won by James Hunt.'),(1,'After retiring from racing in 1979')]),
'5abb1f745542996cc5e49fb5': ('equivalent_verbose_yes','Both biographies explicitly state American; expansion preserves polarity but violates exact yes/no form.',[(0,'is an American television writer, producer and director.'),(1,'is an American independent filmmaker.')]),
'5abbbd0f55429931dba144d5': ('equivalent_morphology_partial_support','Dancers and dancing express the same activity. Gregory is explicitly a dancer and Maurice choreographs; shared fame as dancers is only indirectly supported. No new semantic claim is introduced by the grammatical change.',[(1,'He is the brother of dancer Gregory Hines.'),(0,'choreographed by Maurice Hines.')]),
'5abd7ca05542993062266cab': ('supported_precision_expansion','Adds the explicitly stated country to the same headquarters location.',[(0,'US based Polaris Industries announced that it had acquired Aixam-Mega.'),(1,'Polaris is based in Roseau, Minnesota, USA.')]),
'5ac0551b5542997d64295a07': ('clear_semantic_correction','Question asks a period, not the mapmaker. Retrieved biography supplies Stuart period.',[(0,'on the 1610 map of the town by cartographer John Speed'),(1,'He is the best known English mapmaker of the Stuart period.')]),
'5ac1b7495542994ab5c67dd9': ('clear_semantic_correction','Corrects subgenre to its requested parent genre.',[(0,'Electropop solo artist Tracy + the Plastics'),(1,'Electropop is a variant of synth-pop')]),
'5ac1f7f355429964131be2ae': ('supported_precision_expansion','Adds the explicitly stated object beer to ingredients; same restriction.',[(0,'Reinheitsgebot Purity Law of 1516'),(1,'regulations limiting the ingredients in beer in Germany')]),
'5ac3e8c65542997ea680c993': ('unsupported_polarity_change','B equates length of levees and floodwalls lining three canals with canal length. This is a measurement-type mismatch. Neither equality nor inequality of canal lengths is established, so original no is also unproven; not a confirmed supported-to-incorrect regression. C retains original.',[(1,'There are 13 mi of levees and floodwalls that line the sides of the canals.'),(0,'passes through three levels (approximately 13 miles total)')]),
'5adc1309554299438c868d3b': ('clear_semantic_correction','Corrects show name to requested network through retrieved series document.',[(0,'season finale of the third season of "House"'),(1,'originally ran on the Fox network')]),
'5adc1b1a55429944faac2422': ('equivalent_scope_narrowing','Rock is explicitly shared; removes rock-and-roll elaboration attached only to Superheaven. Preserves shared rock claim.',[(0,'describing it simply as rock/rock and roll.'),(1,'Oceansize were an English rock band')]),
'5add3b665542997545bbbcd2': ('qualified_semantic_correction','Boren satisfies both proposal authorship and Oklahoma link. McCurdy is also an author and his state is absent, so original is not contradicted or conclusively excluded. Cannot claim unambiguous wrong-to-right semantic correction.',[(0,'from Senator David Boren and Representative David McCurdy in 1992'),(1,'politician from the state of Oklahoma.')]),
'5adf01425542995ec70e8f53': ('equivalent_unit_expansion','0.9 s and up to 0.9 seconds express the same bound asked by the question. GMT/UT1 caveat applies equally; no new numeric claim.',[(0,'a Victorian maritime Greenwich Mean Time signal'),(1,'these two meanings can differ by up to 0.9 s.')]),
'5adf698f5542993a75d264e2': ('clear_semantic_correction','Connects actress to The Prince and Me and its director. Ford is unsupported by film evidence.',[(0,'her appearances on film "The Prince and Me"'),(1,'a 2004 romantic comedy film directed by Martha Coolidge')]),
'5ae0fcb855429945ae959492': ('qualified_semantic_correction','Novotna exited in quarterfinals, so she was not runner-up. C identifies finalist Kournikova and verifies birthday and Russian nationality. Russian nationality does not explicitly establish born in Russia; birthplace remains unproven. B lacks birthday and nationality evidence in E0 and cannot distinguish the two finalists on those constraints.',[(0,'Novotná and Zvereva retired in their quarterfinal match'),(0,'Hingis and Kournikova reached the final where they lost'),(1,'born 7 June 1981) is a Russian former professional tennis player.')]),
'5ae3345f55429928c4239682': ('equivalent_verbose_yes','Both are explicitly American punk rock musicians; verbose affirmative preserves meaning.',[(1,'Spike Slawson is an American punk rock musician'),(0,'is an American punk rock musician.')]),
'5ae4ba9255429913cc2044e6': ('clear_semantic_correction','Question requests settlement, not arena. The evidence locates the event in Uniondale; its formal hamlet/CDP designation is not independently stated but provided by the question.',[(0,'the only Bunkhouse Stampede event to air as a pay-per-view (PPV) event.'),(0,'the Nassau Veterans Memorial Coliseum in Uniondale, New York.')]),
'5ae64cbf5542992ae0d162c1': ('equivalent_quantity_expansion','Question already says over how many; 300 fills that bound. Selected answer repeats over and units, preserving the same quantity.',[(0,'has written over 300 major-label recorded songs')]),
'5ae65d055542996d980e7b14': ('clear_semantic_correction','Connects birthplace Leeuwarden to its reported population; baseline is unrelated Columbus population. Source date is unstated, so no present-day claim is made.',[(0,'born 25 April 1962 in Leeuwarden'),(1,'a city and municipality with a population of 108,249')]),
}


def main():
    inputs = {r['task_id']: r for r in read('backups/v110-train-inputs.jsonl')}
    result_list = read('backups/v110-train-results.jsonl')
    assert len(result_list) == len({r['task_id'] for r in result_list}) == 100
    results = {r['task_id']: r for r in result_list}
    prefix = json.loads((ROOT/'reports/v110-prefix-manual-review.json').read_text(encoding='utf-8'))
    judgments = {r['task_id']: {k:v for k,v in r.items() if k not in ('offline_gold_label','arms')} for r in prefix['rows']}
    for suffix,(classification,reason,citations) in NOTES.items():
        task_id='hotpot-'+suffix
        judgments[task_id]={'task_id':task_id,'classification':classification,'reason':reason,
          'citations':[{'document_id':f'{task_id}-{idx}','quote':quote} for idx,quote in citations]}
    expected={(r['task_id'], arm) for r in result_list for arm in ('B','C') if r[arm]!=r['original_answer']}
    assert {t for t,a in expected} == set(judgments), 'Missing or extra manually reviewed tasks'
    rows=[]; verified=0
    for task_id,arm in sorted(expected):
        r=results[task_id]; source=inputs[task_id]; note=dict(judgments[task_id])
        docs=source['e0'] if arm=='B' else r['final_evidence']
        by_id={d['id']:d['text'] for d in docs}
        citations=[c for c in note['citations'] if c['document_id'] in by_id]
        assert citations
        for c in citations:
            assert c['quote'] in by_id[c['document_id']], (task_id,arm,c)
            verified+=1
        note['citations']=citations
        if arm=='B' and task_id.endswith('5ae0fcb855429945ae959492'):
            note['classification']='unknown_missing_disambiguation'
        metrics={name:answer_metrics(answer,source['gold']) for name,answer in
                 [('baseline',r['original_answer']),('selected',r[arm])]}
        note.update(arm=arm,question=source['question'],original=r['original_answer'],selected=r[arm],
                    evidence_scope='E0' if arm=='B' else 'final_evidence',
                    evidence_sha256=hashlib.sha256(json.dumps(docs,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                    offline_score={'gold':source['gold'],'metrics':metrics,
                      'normalized_correct_to_wrong':bool(metrics['baseline']['normalized_em'] and not metrics['selected']['normalized_em'])})
        rows.append(note)
    summaries={arm:dict(Counter(r['classification'] for r in rows if r['arm']==arm)) for arm in ('B','C')}
    losses=[r for r in rows if r['arm']=='C' and r['offline_score']['normalized_correct_to_wrong']]
    assert len(losses)==9
    report={'version':'v110','kind':'full_manual_evidence_bounded_review','source_tasks':100,
      'unique_changed_tasks':len(judgments),'changed_arm_pairs':len(rows),
      'coverage':{arm:sum(r['arm']==arm for r in rows) for arm in ('B','C')},'coverage_complete':True,
      'verified_verbatim_citations':verified,'classifications_by_arm':summaries,
      'C_correct_to_wrong_task_ids':[r['task_id'] for r in losses],
      'confirmed_semantic_regressions':{'B':0,'C':0},
      'automatic_review_comparison_after_manual_judgment':[
        {'task_id':'hotpot-5a7272eb5542997f827839d7','arm':'C','conflict':'Automatic supported-selected and contradicted-original overstates transfer of age from prior novel; manual time-scope unknown.'},
        {'task_id':'hotpot-5ac3e8c65542997ea680c993','arm':'B','conflict':'Automatic supports equal length using levees/floodwalls; manual rejects measurement-type equivalence and finds neither polarity proven.'},
        {'task_id':'hotpot-5ae0fcb855429945ae959492','arm':'C','conflict':'Automatic treats Russian nationality and date as complete support; manual retains explicit birthplace caveat.'},
        {'task_id':'hotpot-5ae0fcb855429945ae959492','arm':'B','agreement':'Both find selected birth criterion insufficient in E0.'},
        {'task_id':'hotpot-5add3b665542997545bbbcd2','arm':'B','agreement':'Automatic review does not call McCurdy contradicted; manual likewise treats exclusion as unresolved.'}],
      'unknown_or_qualified_are_not_confirmed_corrections':True,
      'method':'Evidence-first manual judgments; prefix judgments reused; gold used afterward only for scorer labels. Arm B uses E0, C uses final evidence. No independent external truth source.',
      'limitations':['No semantic regression confirmed does not establish zero semantic risk.','B canal polarity change is unsupported; original no is not established either.','Kournikova birthplace and B disambiguation remain unproven.','Katniss sequel-age and Boren exclusion of alternative author remain unresolved.','Automatic audit statuses and all gates remain unchanged.'],
      'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('backups/v110-train-inputs.jsonl','backups/v110-train-results.jsonl','reports/v110-prefix-manual-review.json')},
      'api_calls':0,'final_test_accessed':False,'gates_modified':False,'rows':rows}
    (ROOT/'reports/v110-full-manual-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Full train100 manual evidence review','',f"Complete coverage: {len(judgments)} changed tasks, {len(rows)} arm-pairs (B {report['coverage']['B']}, C {report['coverage']['C']}); {verified} exact citation spans verified against the appropriate arm evidence.",'',
    'All nine C normalized-correct-to-wrong changes preserve the evidence-supported referent or proposition through aliases, added precision, units or verbose yes/no wording. These remain real scoring/preservation failures; this review does not alter the failed gate or justify P2. No confirmed semantic regression was found, but unresolved scope and unsupported selection risks remain.','',
    '## All nine C scoring regressions','','| Task | Original | Selected | Classification |','|---|---|---|---|']
    def esc(x):return str(x).replace('|','\\|').replace('\n',' ')
    for r in losses: lines.append('| '+' | '.join(map(esc,[r['task_id'],r['original'],r['selected'],r['classification']]))+' |')
    lines+=['','## Counts by arm','',json.dumps(summaries,ensure_ascii=False,indent=2),'',
    '## Evidence risks and interpretation','',
    '- Katniss: a prior-novel age cannot establish age in Catching Fire. Selected age remains insufficient for the requested time scope.',
    '- Canal B: 13 miles of levees/floodwalls is not 13 miles of canal. The yes flip is unsupported; evidence does not prove original no either. C retains no.',
    '- Kournikova B: original is excluded by quarterfinal exit, but E0 lacks the birthday/nationality discriminator between two finalists. C adds birthday and Russian nationality; born in Russia is still not explicit.',
    '- Boren: selected person is supported as an Oklahoma coauthor, but the evidence does not rule out McCurdy satisfying the same condition.',
    '- Skiffle: evidence explicitly states the origin of the term; a claim about the historical musical form is broader.',
    '- Hines: dancer/dancing is the same activity, but shared fame is only indirectly supported by dancer Gregory and choreographer Maurice.',
    '', '## Comparison with automated review (read after manual judgments)', '',
    'The automated reviewer overstates support for Katniss age in the sequel, equates levee/floodwall distance with canal length, and does not retain the Russian-nationality versus birthplace distinction for C Kournikova. Its B Kournikova insufficient-evidence finding and Boren non-contradiction are consistent with this manual review. These are explicit evidence-scope disagreements, not appeals to external truth.',
    '', '## Every changed arm-pair', '']
    for r in rows:
        lines += [f"### {r['task_id']} / {r['arm']}",'',f"{r['original']} → {r['selected']}. **{r['classification']}**. {r['reason']}",'']
        lines += [f"- `{c['document_id']}`: {c['quote']}" for c in r['citations']]
        lines += ['']
    lines += ['Manual judgments were made before gold inspection for the new rows. Gold is recorded separately in JSON solely for official scorer labels. Prefix judgments were reused. This is an assistant review, not independent external truth adjudication. No API calls, no final-test access, no changes to gates.','']
    (ROOT/'reports/v110-full-manual-review.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('coverage','unique_changed_tasks','verified_verbatim_citations','classifications_by_arm','C_correct_to_wrong_task_ids')},indent=2))

if __name__=='__main__':main()
