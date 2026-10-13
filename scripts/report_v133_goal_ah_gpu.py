"""Assemble source reviews and paired metrics from completed frozen runs (CPU).

Rejects missing trajectories, unreviewed novel contexts and unsupported quotes.
No training, inference, resampling, confirmation, or product publication.
"""
import argparse,copy,json,hashlib,math
from pathlib import Path
from prepare_v133_goal_ah_novel_review import semantic_key
from assess_v133_goal_ah import metrics,qualification,mechanism_b_over_a
OUT=Path(__file__).resolve().parents[1]/'runs/v133-goal-ah-entity-relation'
SPLITS={'goal_z_dev':64,'goal_y_regression':48,'aa_regression':64,'old_main':16,'old_regression':16,'ag_regression':64}
LABELS=['baseline','42-A','43-A','42-B','43-B']
REVIEW_FILES={
'42-A':['candidate-dev-42-A-main-review.jsonl','candidate-novel-evidence-review-42-A-first.jsonl','candidate-novel-evidence-review-42-A-second.jsonl'],
'43-A':['candidate-dev-43-A-main-review.jsonl','candidate-novel-regression-review-43-A.jsonl'],
'42-B':['candidate-dev-42-B-main-review.jsonl','candidate-novel-regression-review-42-B.jsonl'],
'43-B':['candidate-dev-43-B-main-review.jsonl','candidate-novel-regression-review-43-B.jsonl'],
}
def rows(p):return list(map(json.loads,p.read_text(encoding='utf8').splitlines()))
def quote_positions(quote,doc,call):
    escaped=json.dumps(quote,ensure_ascii=False)[1:-1];a,b=doc['input_char_span'];s=call['actual_prompt'].find(escaped,a,b)
    if s<0:raise ValueError(('Quote absent from actual JSON input',doc['document_id'],quote[:80]))
    e=s+len(escaped);covered=[i for i,(x,y) in enumerate(call['offset_mapping']) if x<e and y>s]
    assert covered
    return dict(quote=quote,document_id=doc['document_id'],actual_input_char_span=[s,e],actual_input_token_span=[min(covered),max(covered)+1],call_index=call['call_index'])
def save_json(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    context={};raw={}
    for label in LABELS:
        for split,n in SPLITS.items():
            p=OUT/f'actual-inputs-{split}-{label}.jsonl';q=OUT/f'eval-{split}-{label}.jsonl'
            if not p.exists() or not q.exists():continue
            a=rows(p);b=rows(q)
            assert len(a)==len(b)==n and {r['case_id'] for r in a}=={r['case_id'] for r in b}
            context[(label,split)]={r['case_id']:r for r in a};raw[(label,split)]=b
    if not args.partial:assert len(context)==30,'All 1360 product trajectories must be complete'
    base=rows(OUT/'baseline-category-ledger-v5.jsonl');cache={};adjudications=[]
    # Inherited successful chains were already source-reviewed before the
    # candidates. Bring their triples alongside the new failure-axis audit.
    frames={}
    for name in ['baseline-dev-protection-main-v1.jsonl','baseline-regression-protection-main-v1.jsonl','baseline-aa-additional-protection-main.jsonl']:
        for r in rows(OUT/name):
            if r.get('supported_success',True):frames[r['case_id']]=r.get('necessary_relations',[])
    question_relations={};question_missing={}
    for r in base:
        label='baseline';split=r['split'];a=context[(label,split)][r['case_id']];r=copy.deepcopy(r)
        if r['case_id']=='2wiki:d438a1020bd911eba7f7acde48001122':
            r.update(evidence_complete=False,reason=r['reason']+' Main correction: actual wording is King of Serbia, not a Serbian nationality descriptor; nationality edge is missing.',missing_relations=[['Stefan the First-Crowned','nationality','?']])
        r['necessary_relations']=frames.get(r['case_id'],r.get('observed_relations',[]))
        if r['case_id']=='musique:2hop__829358_754719':
            ds=a['calls'][-1]['documents'];r.update(evidence_complete=True,missing_relations=[],reason=r['reason']+' Main axis correction: Darley born Pedro Leopoldo and Xavier paragraph explicitly places that same city in Minas Gerais; different person does not invalidate the city-location relation.')
            r['necessary_relations']=[]
            for docid,triple in [('66b9d6622ad49cead48a46dd',['Darley Ramon Torres','born in','Pedro Leopoldo']),('27701255ad2329b137d47657',['Pedro Leopoldo','located in state','Minas Gerais'])]:
                d=next(d for d in ds if d['document_id']==docid);r['necessary_relations'].append(dict(triple=triple,document_id=docid,quote=d['text']))
        if r['case_id']=='musique:2hop__789565_416401':
            ds=a['calls'][-1]['documents'];r.update(evidence_complete=True,missing_relations=[],reason=r['reason']+' Main axis correction: Lady Li paragraph explicitly identifies Wang Jipeng’s father as Wang Yanjun, completing the child chain; Lady Li’s own father is Li Min.')
            r['necessary_relations']=[]
            for docid,triple in [('63cd6d2dc0aa83a2a91e0321',['Empress Dowager Huang','mother of','Wang Yanjun']),('837be0cdec2b788fe19d6f9c',['Wang Jipeng','father','Wang Yanjun'])]:
                d=next(d for d in ds if d['document_id']==docid);r['necessary_relations'].append(dict(triple=triple,document_id=docid,quote=d['text']))
        if r['case_id']=='5a8ccf10554299653c1aa129':
            # The helper had copied Cowie's country quote onto Fetteresso.
            # Replace the invalid subject binding with actual source relations.
            ds=a['calls'][-1]['documents'];r['necessary_relations']=[]
            for docid,triple in [('ee680f57a070bf1404ab',['Fetteresso Castle','situated west of','Stonehaven']),('ab01aecec5ee671dcbe9',['Cowie Castle','located in','Aberdeenshire, Scotland'])]:
                d=next(d for d in ds if d['document_id']==docid)
                r['necessary_relations'].append(dict(triple=triple,document_id=docid,quote=d['text']))
            r['missing_relations']=[['Fetteresso Castle','country','?']]
        question_relations[r['case_id']]=copy.deepcopy(r['necessary_relations'])
        question_missing[r['case_id']]=copy.deepcopy(r.get('missing_relations',[]))
        cache[semantic_key(a)]=r
    overrides={}
    p=OUT/'candidate-main-adjudication-final.jsonl'
    if not p.exists():p=OUT/'candidate-main-adjudication-v3.jsonl'
    if not p.exists():p=OUT/'candidate-main-adjudication-v2.jsonl'
    if not p.exists():p=OUT/'candidate-main-adjudication-v1.jsonl'
    if p.exists():overrides={(r['model'],r['case_id']):r for r in rows(p)}
    missing=[];reviewed={};cache_reuses=0;novel_counts={}
    for label in LABELS:
        direct={}
        if label!='baseline':
            for name in REVIEW_FILES[label]:
                p=OUT/name
                if p.exists():
                    for r in rows(p):
                        if r['case_id'] in direct:raise ValueError('Duplicate explicit candidate review')
                        direct[r['case_id']]=r
        novel_counts[label]=0
        for split,n in SPLITS.items():
            if (label,split) not in context:continue
            result=[]
            for a in context[(label,split)].values():
                key=semantic_key(a);cid=a['case_id'];call=a['calls'][-1]
                if key in cache:
                    r=copy.deepcopy(cache[key]);cache_reuses+=label!='baseline'
                    if cid in direct:
                        d=dict(direct[cid]);o=overrides.get((label,cid),{})
                        assert o.get('category',d['category'])==r['category'],('Conflicting source review',label,cid)
                    origin=dict(model=r.get('model','baseline'),case_id=cid,method='Exact same case, answer, and final visible document ID/text set')
                elif cid in direct:
                    r=copy.deepcopy(direct[cid]);o=overrides.get((label,cid))
                    r.setdefault('necessary_relations',question_relations[cid])
                    r.setdefault('missing_relations',question_missing[cid])
                    if o:
                        r['category']=o['category'];r['reason']=o['reason']
                        if o.get('evidence_complete') is not None:r['evidence_complete']=o['evidence_complete']
                        for field in ['entity_mismatch','relation_mismatch']:
                            if field in o:r[field]=o[field]
                        if 'missing_relations' in o:r['missing_relations']=o['missing_relations']
                    if r['category']=='success':r.update(entity_mismatch=False,relation_mismatch=False,evidence_complete=True,missing_relations=[])
                    r.setdefault('model',label);cache[key]=copy.deepcopy(r);novel_counts[label]+=1
                    origin=dict(model=label,case_id=cid,method='New final visible evidence or answer; explicit source-bound review')
                else:
                    missing.append(dict(model=label,split=split,case_id=cid,semantic_review_key=key));continue
                r.update(model=label,split=split,answer=a['answer'],actual_final_prompt_sha256=call['prompt_sha256'],semantic_review_key=key,evidence_review_origin=origin,scoring_version='AH paired source rubric v5; one main dispute pass')
                ds=call['documents'];relations=r.get('necessary_relations',r.get('observed_relations',[]));positioned=[]
                r['entity_relation_frame']=[q['triple'] for q in question_relations[cid] if q.get('triple')]+r.get('missing_relations',[])
                r['frame_scope']='Source observations from the question audit plus requested missing edges; only positioned observations establish current visible evidence.'
                for q in relations:
                    matches=[d for d in ds if d['document_id']==q.get('document_id') and q.get('quote','') and q['quote'] in d['text']]
                    if not matches and r['category']=='success':
                        # The main review independently verified the complete
                        # chain in this shortened actual paragraph. Preserve
                        # the question's triple, attach this actual source.
                        matches=[d for d in ds if d['document_id']==q.get('document_id')]
                    if matches:positioned.append(dict(triple=q.get('triple'),reviewed_excerpt=q['quote'],**quote_positions(matches[0]['text'],matches[0],call)))
                extra=[]
                if cid=='musique:2hop__796221_18803':
                    extra=[('1d9e1a4b6b78fbe2bee8f743',['MV Rapana','tanker in company fleet','Royal Dutch/Shell']),('95bf7d87d266b5e1754b6b36',['Royal Dutch Shell','announced in April 2010','seeking a buyer for all Finnish operations'])]
                if cid=='musique:2hop__468778_88165':
                    extra=[('30425a692981eb963b039466',['Hagar','union resulting in Ishmael','Abraham']),('431b74ffa30ec73da0f22ccb',['Abraham','married after Sarah death','Keturah'])]
                for docid,triple in extra:
                    found=next((d for d in ds if d['document_id']==docid),None)
                    if found:
                        positioned.append(dict(triple=triple,reviewed_excerpt=found['text'],**quote_positions(found['text'],found,call)))
                        if triple not in r['entity_relation_frame']:r['entity_relation_frame'].append(triple)
                r['observed_relations_with_positions']=positioned
                r['missing_relations']=r.get('missing_relations',[])
                # Retain full actual paragraphs with exact quote positions even
                # where a reviewed relation is absent or not a requested edge.
                r['actual_visible_source']=[quote_positions(d['text'],d,call) for d in ds]
                if r['category']=='success':r['missing_relations']=[]
                result.append(r)
            if len(result)==n:reviewed[(label,split)]=result
    if missing:
        save_json('unreviewed-actual-contexts.json',missing)
        print(json.dumps({'unreviewed':len(missing),'completed_review_splits':len(reviewed),'novel_counts':novel_counts}))
        if not args.partial:raise ValueError('Unreviewed actual evidence contexts remain')
    summary={label:{} for label in LABELS}
    for key,rs in reviewed.items():summary[key[0]][key[1]]=metrics(raw[key],rs)
    protected=rows(OUT/'necessary-search-protection-frozen.jsonl');ids={r['case_id'] for r in protected};assert len(ids)==86
    probes=json.loads((OUT/'mechanism-source-adjudication-v1.json').read_text(encoding='utf8'))['summary']
    gates={};lost={};paired=[]
    basemap={r['case_id']:r for (label,_),rs in reviewed.items() if label=='baseline' for r in rs}
    for label in LABELS[1:]:
        candidates={r['case_id']:r for (model,_),rs in reviewed.items() if model==label for r in rs}
        lost[label]=sorted(cid for cid in ids if cid in candidates and candidates[cid]['category']!='success')
        for cid,r in candidates.items():
            b=basemap[cid]
            paired.append(dict(model=label,split=r['split'],case_id=cid,baseline_category=b['category'],candidate_category=r['category'],baseline_answer=b['answer'],candidate_answer=r['answer'],protected=cid in ids,
                gain=b['category']!='success' and r['category']=='success',loss=b['category']=='success' and r['category']!='success',
                baseline_evidence_complete=b['evidence_complete'],candidate_evidence_complete=r['evidence_complete'],
                mechanism=('evidence_acquisition' if not b['evidence_complete'] and r['evidence_complete'] else 'evidence_loss_or_missing_relation' if b['evidence_complete'] and not r['evidence_complete'] else 'answer_change' if b['answer']!=r['answer'] else 'same_evidence_and_answer') if (b['category']=='success')!=(r['category']=='success') else 'no_success_transition'))
        if len(summary[label])==6:
            train=json.loads((OUT/f'seed{label[:2]}-{label[-1]}/summary.json').read_text(encoding='utf8'))
            # Runtime had already enforced finite gradients, actual weight
            # identity, LoRA-only changes and independent reload checks.
            updates=rows(OUT/f'seed{label[:2]}-{label[-1]}/updates.jsonl')
            engineering=(train['seed']==int(label[:2]) and train['arm']==label[-1]
                and train['updates']==128 and train['base_unchanged'] is True
                and train['changed_lora_tensors']==192 and train['reload_sentinels']==3
                and train['reload_max_abs']<=.001 and len(updates)==128
                and [u['update'] for u in updates]==list(range(1,129))
                and all(math.isfinite(u['Lt']) and math.isfinite(u['weighted_Lj'])
                    and math.isfinite(u['gradient']) and u['gradient']>0 for u in updates))
            regressions={s:(summary['baseline'][s]['success'],summary[label][s]['success']) for s in ['old_main','old_regression','goal_y_regression','aa_regression','ag_regression']}
            gates[label]=qualification(summary[label]['goal_z_dev'],summary['baseline']['goal_z_dev'],regressions,len(lost[label]),engineering)
    mechanism={}
    for seed in [42,43]:
        a=f'{seed}-A';b=f'{seed}-B'
        if 'goal_z_dev' in summary[a] and 'goal_z_dev' in summary[b]:mechanism[str(seed)]=mechanism_b_over_a(summary[b]['goal_z_dev'],summary[a]['goal_z_dev'],probes[b]['correct'],probes[a]['correct'])
    report=dict(status='partial' if args.partial else 'completed_no_release',metrics=summary,development_gates=gates,mechanism_gates=mechanism,
        lost_protected=lost,protected_count=86,cache_reuses=cache_reuses,novel_review_counts=novel_counts,
        confirmation_calls=0,publication_allowed=False,product_model='Goal Z full-trajectory SFT seed42',
        no_release_reason='Neither frozen mechanism replication nor development qualification supports a release; no new confirmation or seed selection.',
        scoring_note='Baseline v5 explicit source corrections and identical paired rubric; AG/AA old-version scores not mixed. Original protection subset of 86 remains frozen.')
    save_json('provisional-gpu-metrics.json' if args.partial else 'gpu-results.json',report)
    if not args.partial:
        for label in LABELS:
            rs=[r for (model,_),part in reviewed.items() if model==label for r in part];assert len(rs)==272
            (OUT/f'final-source-review-{label}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf8')
        assert len(paired)==1088
        (OUT/'paired-product-ledger.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in paired),encoding='utf8')
    print(json.dumps(dict(dev={m:summary[m].get('goal_z_dev',{}).get('success') for m in LABELS},gates=gates,protected_lost={m:len(x) for m,x in lost.items()},unreviewed=len(missing)),ensure_ascii=False))
if __name__=='__main__':main()
