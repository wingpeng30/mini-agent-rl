"""Verify three blinded internal audits, bind quotes to visible evidence, and unblind only afterward."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports'
def read(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def support(x):
    return {'entailed':'supported','not_applicable':'not_applicable','not_answer':'not_applicable',
            'no_answer':'no_answer'}.get(x,x)
def main():
    cases=read(R/'v113-p0-review-cases.jsonl');key=json.loads((R/'v113-p0-private-key.json').read_text(encoding='utf-8'))
    agentfiles=[R/f'v113-p0-agent-review-{i}.jsonl' for i in range(3)] + [R/f'v113-p0-agent-review-extra-{i}.jsonl' for i in range(3)]
    reviews=[read(p) for p in agentfiles]
    case_maps=[{r['case_id']:r for r in group} for group in reviews]
    expected={c.get('blind_case_id',c.get('blind_state_id')):c for c in cases}
    if len(expected)!=61 or any(not set(m)<=set(expected) for m in case_maps):raise ValueError('independent_review_coverage_mismatch')
    if any(sum(cid in m for m in case_maps)!=3 for cid in expected):raise ValueError('three_independent_reviews_per_case_required')
    quotes=0;actions=Counter();bad=[]
    for reviewer,m in enumerate(case_maps):
        for cid,review in m.items():
            packet=expected[cid]
            for arm in ('A','B'):
                part=review[arm]
                packet_events=packet.get(arm)
                trajectory=isinstance(packet_events,list)
                for event in part.get('events',[]):
                    actions[event.get('answer_support','missing')]+=1
                    step=event.get('step')
                    if not trajectory:
                        docs={d['document_id']:d['text'] for d in packet['current_evidence']}
                    else:
                        pe=next((x for x in packet_events if x['step']==step),None)
                        if pe is None:bad.append((reviewer,cid,step,'missing_step'));continue
                        docs_by_source={'current_evidence':{d['document_id']:d['text'] for d in pe['current_evidence']},
                          'retrieved_after_action':{d['document_id']:d['text'] for d in pe['retrieved_after_action']},
                          'before':{d['document_id']:d['text'] for d in pe['current_evidence']},
                          'after':{d['document_id']:d['text'] for d in pe['retrieved_after_action']}}
                    for cite in event.get('citations',[]):
                        quotes+=1;source=cite.get('source','current_evidence')
                        docs=docs if not trajectory else docs_by_source.get(source,{})
                        if cite.get('quote') not in docs.get(cite.get('document_id'),''):
                            bad.append((reviewer,cid,step,'invalid_quote'))
                        if trajectory:
                            pe=next((x for x in packet_events if x['step']==step),None)
                            action=(pe.get('action') or {}).get('kind') if pe else None
                            if action=='text' and source in ('after','retrieved_after_action'):
                                bad.append((reviewer,cid,step,'future_evidence_used_for_answer'))
    if bad:raise ValueError('citation_validation_failed:'+repr(bad[:8]))
    summary={'version':'v113','review_mode':'six blinded Codex Astra task-agent review batches provide three review passes per case; external API review blocked by outbound-data approval review',
       'cases':len(expected),'reviewer_coverage':[len(x) for x in reviews],'quote_count_verified':quotes,
       'event_label_counts':dict(actions),'citations_valid':True,'unresolved_cases_by_reviewer':[
         sum(bool(r.get('unresolved')) for r in g) for g in reviews],
       'external_api_review':'not run: automatic review rejected transfer of internal question/evidence text to external DeepSeek API; no attempt to bypass',
       'independent100_accessed':False,'final_test_accessed':False}
    mapped={}
    per_reviewer={}
    for i,m in enumerate(case_maps):
        counts=defaultdict(Counter)
        for cid,r in m.items():
            for letter in ('A','B'):
                arm=key[cid][letter+'_arm'];verdict=support(r[letter].get('final_support','uncertain'))
                counts[arm][verdict]+=1
                mapped[(cid,arm)]=mapped.get((cid,arm),[])+[verdict]
        per_reviewer[f'reviewer_{i}']={arm:dict(c) for arm,c in counts.items()}
    agreement=Counter()
    for (cid,arm),vals in mapped.items():
        agreement['three_agent_agreement' if len(set(vals))==1 else 'disagreement']+=1
        packet=expected[cid]
        if 'blind_case_id' in packet:agreement['task_final_arm_pairs_reviewed']+=1
        else:agreement['supported_state_arm_pairs_reviewed']+=1
    summary['final_support_by_arm_and_reviewer']=per_reviewer
    summary['three_agent_support_agreement']=dict(agreement)
    summary['source_packet_sha256']=hashlib.sha256((R/'v113-p0-review-cases.jsonl').read_bytes()).hexdigest()
    out=R/'v113-p0-agent-audit-summary.json'
    if out.exists():raise FileExistsError('refuse_to_overwrite_p0_audit_summary')
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
