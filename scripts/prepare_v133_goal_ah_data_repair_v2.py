"""User-authorized, bounded CPU confirmation supplement. No GPU or model filtering."""
import json
import sys
from collections import Counter
import prepare_v133_goal_ah_confirmation as c
import prepare_v133_goal_z_multihop_data as s
from prepare_v133_goal_ah import OUT,ROOT,rows,write_once,sha,stable

DEST=OUT/'confirmation-repair-v2'
CAP=64

def hotpot_rows():
    import prepare_v133_goal_w_data as w
    import pyarrow.parquet as pq
    files,_=w.verify_source()
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for r in batch.to_pylist():
                if r.get('type')!='bridge' or str(r.get('answer','')).casefold() in ['yes','no']:continue
                supports=w.official_support_docs(r)
                if len(supports)!=2:continue
                yield dict(id='hotpot:'+r['id'],source='hotpot',source_ids=[r['id']],question=r['question'],
                    answer=r['answer'],answer_aliases=[],chain=[],docs=[dict(d,support=True) for d in supports],kind='bridge')

def main():
    repair='--repair-cpu-encoding' in sys.argv
    if DEST.exists():
        if not repair or {p.name for p in DEST.iterdir()}!={'boundary.json','cpu-encoding-error.json'}:
            raise FileExistsError('v2 preparation already attempted; no automatic retry')
    else:DEST.mkdir()
    boundary=dict(user_authorization='2026-10-02 request to select from several datasets and resolve shortage quickly',
        candidate_cap_per_source=CAP,sources=['musique','2wiki','hotpot'],max_new_frozen_candidates=192,
        semantic_review_cap_primary=128,keep_existing_joint_qualified=36,
        target_new_counts={'musique':12,'2wiki':16},hotpot='reserve only; does not silently alter 32+32',
        training_changes=0,gpu_authorized=False,automatic_pool_expansion=False)
    if repair:
        if json.loads((DEST/'boundary.json').read_text(encoding='utf8'))!=boundary:raise ValueError('CPU repair changed frozen boundary')
    else:write_once(DEST/'boundary.json',boundary)
    ids,q,t,e,p,sp,files,excluded=c.registry()
    hist=s.w.history_exposure();ids.update(hist[0]);q.update(hist[1]);t.update(hist[2])
    original_extra=json.loads((OUT/'additional-ag-corpus-isolation-audit.json').read_text(encoding='utf8'))
    overlap={r['case_id'] for r in original_extra['overlaps']}
    v1={r['id']:r for r in rows(OUT/'sealed-confirmation-v1/isolated-candidates.jsonl')}
    kept=[v1[r['case_id']] for r in rows(OUT/'sealed-confirmation-v1/source-quality-main-v1.jsonl') if r['accepted'] and r['case_id'] not in overlap]
    for path in [OUT/'sealed-confirmation-v1/candidate-pool-musique.jsonl',OUT/'sealed-confirmation-v1/candidate-pool-2wiki.jsonl']:
        for r in rows(path):
            ids.update(r['source_ids']);ids.add(r['id']);q.add(s.w.qhash(r['question']))
            t.update(s.norm(d['title']) for d in r['docs'] if d['support']);e.update(c.case_entities(r))
            for d in r['docs']:
                p.add(c.digest_text(d['text']));sp.update(c.fingerprints(d['text']))
    for d in json.loads((OUT/'ag-regression-corpus-snapshot.json').read_text(encoding='utf8')):
        t.add(s.norm(d['title']));p.add(c.digest_text(d['text']));sp.update(c.fingerprints(d['text']))
    write_once(DEST/'kept-v1-cases.jsonl',kept,jsonl=True)
    write_once(DEST/'exposure-coverage.json',dict(files=files,additional_goal_g_files=hist[3],
        actual_ag_corpus_sha256=sha(OUT/'ag-regression-corpus-snapshot.json'),
        v1_candidate_pool_hashes={path.name:sha(path) for path in (OUT/'sealed-confirmation-v1').glob('candidate-pool-*.jsonl')},
        counts=dict(ids=len(ids),questions=len(q),titles=len(t),entities=len(e),passages=len(p)),excluded_named_sealed=excluded))
    counts={};rejections={}
    sources=[('musique',(s.canonical('musique',r) for r in s.load_musique('dev') if s.source_eligible('musique',r))),
             ('2wiki',(s.canonical('2wiki',r) for r in s.load_wiki('dev') if s.source_eligible('2wiki',r))),
             ('hotpot',hotpot_rows())]
    for name,iterator in sources:
        reject=Counter();eligible=[]
        for r in iterator:
            reason=None;supports=[d for d in r['docs'] if d['support']]
            if name!='hotpot':reason=s.audit_case(r,(ids,q,t))
            elif any(x in ids for x in r['source_ids']) or s.w.qhash(r['question']) in q:reason='historical_question_or_source_id'
            elif any(s.norm(d['title']) in t for d in supports):reason='historical_support_title'
            elif any(len(d['text'])>2200 for d in supports):reason='long_support_paragraph'
            if not reason and c.case_entities(r)&e:reason='historical_entity_chain'
            if not reason and any(c.digest_text(d['text']) in p or c.fingerprints(d['text'])&sp for d in supports):reason='historical_support_passage'
            if reason:reject[reason]+=1;continue
            eligible.append(r)
        pool=sorted(eligible,key=lambda r:stable('data-repair-v2:'+r['id']))[:CAP]
        write_once(DEST/f'candidate-pool-{name}.jsonl',pool,jsonl=True)
        counts[name]=dict(mechanical_eligible=len(eligible),frozen=len(pool));rejections[name]=dict(reject)
        print(json.dumps(dict(source=name,**counts[name])),flush=True)
    write_once(DEST/'pool-manifest.json',dict(counts=counts,rejections=rejections,quality_review='pending',
        no_model_outputs_used=True,not_training_data=True,original32plus32_unchanged=True,
        source_files={str(path.relative_to(ROOT.parent)):sha(path) for path in [s.MUSIQUE,s.WIKI]},
        pool_hashes={path.name:sha(path) for path in DEST.glob('candidate-pool-*.jsonl')}))

if __name__=='__main__':main()
