"""Lead-adjudicated bounded confirmation repair. CPU only; never grants approval."""
import json
from collections import Counter
from prepare_v133_goal_ah import ROOT, OUT, rows, sha, write_once
import prepare_v133_goal_ah_confirmation as c
import prepare_v133_goal_z_multihop_data as s

DEST = OUT / 'confirmation-repair-v2'
SELECT = {'musique': [1,5,7,10,11,12,15,36,50,54,57,60],
          '2wiki': [2,5,10,11,12,14,15,18,19,20,21,25,26,27,29,32]}
REJECT = {
 'musique': {3:'Multiple cast members; target person unqualified.',6:'Doubles has two winners; only Mike biography supplied.',
 9:'Unqualified member question with multiple legitimate members; single-answer label.',
 14:'Same Pierre Berton chain already selected.',19:'From Eldon does not establish birthplace.',20:'Same Pierre Berton chain already selected.',
 21:'Hammill passage records sequential teams; current team underqualified.',22:'Two creators, plus already selected Affleck chain.',
 26:'Multiple band members; unqualified single-answer label.',30:'From Scranton does not establish birthplace.',
 31:'Ford between two counties does not explicitly prove shared administrative boundary.',35:'Duplicate Affleck chain.',
 40:'Multiple cast members; spouse target underqualified.',42:'Two screenwriters; target unqualified.',
 46:'Multiple group members; unqualified single-answer label.',47:'Two creators, duplicate Affleck chain.',
 49:'Passage names both Deptford Creek and Thames; near-water question unqualified.',
 51:'Paternal versus maternal grandmother unqualified.',59:'Multiple siblings; unqualified single-answer label.'},
 '2wiki': {4:'Two distinct study institutions explicitly given; single label underqualified.',
 7:'Song features Omelly as well as Meek; performer target unqualified.',8:'Supervised cartoon does not directly establish director role.',
 9:'Features music by does not distinguish score composer from included artist; role underqualified.',
 13:'Birthday asks date; year-only gold omits necessary month/day.',24:'Palestinian source descriptor contradicts Lebanon label; no gold repair allowed.'}}

def main():
    if (DEST/'final-manifest.json').exists():raise FileExistsError('No automatic finalization retry')
    kept=rows(DEST/'kept-v1-cases.jsonl'); selected=list(kept); lead=[]
    seen=set().union(*(c.case_entities(r) for r in kept))
    for name,indices in SELECT.items():
        pool=rows(DEST/f'candidate-pool-{name}.jsonl')
        helper={r['case_id']:r for r in rows(DEST/f'source-review-{name}-helper.jsonl')}
        for i,case in enumerate(pool):
            if i>max(indices) or not helper[case['id']]['accepted']:continue
            r=dict(helper[case['id']], pool_index=i, source=name, reviewer='lead', gold_modified=False)
            r['accepted']=i in indices
            r['reason']=('Lead checked full original passages: both required edges bind the intended entity and original answer, with question scope preserved.' if r['accepted'] else REJECT[name][i])
            if name=='2wiki' and i==32:
                r['relations'][1][1]='nationality description'
                bio=next(d for d in case['docs'] if d['title']=='Gyula Szőreghy')
                r['quotes']=[x for x in r['quotes'] if x['relation_index']==0]+[dict(relation_index=1,title=bio['title'],quote=bio['text'])]
                r['reason']+=' Hungarian descriptor is equivalent for country-from wording; no legal citizenship inference.'
            if name=='musique' and i==7:r['reason']+=' Full brothers share parents; Henry is explicitly Tom’s father, establishing Edward’s father.'
            if name=='musique' and i==57:r['reason']+=' Both named cast members are explicitly parents of the same child; choice does not change answer.'
            if r['accepted']:
                if c.case_entities(case)&seen:raise ValueError('Selected entity-chain overlap')
                seen.update(c.case_entities(case))
                for quote in r['quotes']:
                    title=quote.get('title',quote.get('document_title'))
                    docs=[d for d in case['docs'] if d['support'] and d['title']==title]
                    if len(docs)!=1 or quote['quote'] not in docs[0]['text']:raise ValueError('Non-verbatim support citation')
                if {q['relation_index'] for q in r['quotes']}!={0,1}:raise ValueError('Incomplete two-hop citations')
                selected.append(case)
            lead.append(r)
    assert len(selected)==64 and Counter(r['source'] for r in selected)=={'musique':32,'2wiki':32}
    ids,q,t,e,p,sp,files,excluded=c.registry()
    hist=s.w.history_exposure();ids.update(hist[0]);q.update(hist[1]);t.update(hist[2])
    for d in json.loads((OUT/'ag-regression-corpus-snapshot.json').read_text(encoding='utf8')):
        t.add(s.norm(d['title']));p.add(c.digest_text(d['text']));sp.update(c.fingerprints(d['text']))
    # v1 candidates are exposed to curation, not product inference. For new supplements
    # also exclude all v1 pools; retained 36 were approved prior to supplementation.
    v1ids=set();v1q=set();v1titles=set();v1entities=set();v1passages=set();v1sentences=set()
    for path in (OUT/'sealed-confirmation-v1').glob('candidate-pool-*.jsonl'):
        for r in rows(path):
            v1ids.update(r['source_ids']);v1q.add(s.w.qhash(r['question']));v1entities.update(c.case_entities(r))
            for d in r['docs']:
                if d['support']:v1titles.add(s.norm(d['title']))
                v1passages.add(c.digest_text(d['text']));v1sentences.update(c.fingerprints(d['text']))
    overlaps=[]
    for i,r in enumerate(selected):
        dd=[d for d in r['docs'] if d['support']]
        checks={'id':bool(set(r['source_ids'])&ids),'question':s.w.qhash(r['question']) in q,
                'title':any(s.norm(d['title']) in t for d in dd),'entity':bool(c.case_entities(r)&e),
                'passage':any(c.digest_text(d['text']) in p or c.fingerprints(d['text'])&sp for d in dd)}
        if i>=len(kept):
            checks['v1_pool']=bool(set(r['source_ids'])&v1ids or s.w.qhash(r['question']) in v1q or c.case_entities(r)&v1entities or
                any(s.norm(d['title']) in v1titles or c.digest_text(d['text']) in v1passages or c.fingerprints(d['text'])&v1sentences for d in dd))
        if any(checks.values()):overlaps.append(dict(id=r['id'],checks=checks))
    if overlaps:raise ValueError(json.dumps(overlaps))
    historical=t|v1titles
    support_titles={s.norm(d['title']) for r in selected for d in r['docs'] if d['support']}
    corpus,_=c.curator.build_corpus('confirm',selected,historical-support_titles,historical-support_titles)
    bytitle={s.norm(d['title']):d for d in corpus}
    for r in selected:
        for d in r['docs']:
            if d['support'] and bytitle[s.norm(d['title'])]['text']!=d['text']:raise ValueError('Corpus changed support paragraph')
    assert len(support_titles)==128 and len(corpus)==2048
    write_once(DEST/'source-review-main.jsonl',lead,jsonl=True)
    write_once(DEST/'final-source-cases.jsonl',selected,jsonl=True)
    cases=[{k:v for k,v in r.items() if k!='docs'}|{'support_titles':[d['title'] for d in r['docs'] if d['support']]} for r in selected]
    write_once(DEST/'final-confirm-cases.jsonl',cases,jsonl=True)
    write_once(DEST/'final-confirm-corpus.json',corpus)
    write_once(DEST/'final-isolation.json',dict(passed=True,overlaps=overlaps,history_files=files,additional_goal_g_files=hist[3],
        actual_ag_corpus_sha256=sha(OUT/'ag-regression-corpus-snapshot.json'),new_candidates_exclude_all_v1_pools=True,
        retained_v1=36,new=28,unique_support_titles=128,support_text_exact_matches=128,
        scope='Recorded IDs/questions/entity chains/support titles/passages; no claim of absolute generic entity independence. Curation-only source review is not product exposure.'))
    inputs=[DEST/f'candidate-pool-{n}.jsonl' for n in ['musique','2wiki','hotpot']]+[DEST/f'source-review-{n}-helper.jsonl' for n in ['musique','2wiki']]
    outputs=[DEST/n for n in ['source-review-main.jsonl','final-source-cases.jsonl','final-confirm-cases.jsonl','final-confirm-corpus.json','final-isolation.json']]
    write_once(DEST/'final-manifest.json',dict(passed=True,accepted=64,counts=dict(Counter(r['source'] for r in selected)),
        retained=36,supplemented=28,primary_helper_reviewed=128,helper_accepted={'musique':32,'2wiki':47},
        lead_positive_candidates_checked=len(lead),lead_new_accepted=28,hotpot_frozen_reserve=64,hotpot_used=0,
        original_gold_unchanged=True,model_filtering=False,training_changes=0,gpu_executed=False,
        cases=str((DEST/'final-confirm-cases.jsonl').relative_to(ROOT)),corpus=str((DEST/'final-confirm-corpus.json').relative_to(ROOT)),
        corpus_count=2048,files={str(path.relative_to(ROOT)):sha(path) for path in inputs+outputs}))
    print(json.dumps({'passed':True,'cases':64,'new':28,'lead_reviewed':len(lead)}))

if __name__=='__main__':main()
