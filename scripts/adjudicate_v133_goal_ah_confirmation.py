"""One lead source-quality adjudication; no replacement sampling or gold rewrite."""
from copy import deepcopy
from collections import Counter
from prepare_v133_goal_ah import OUT,rows,write_once,sha

def main():
    directory=OUT/'sealed-confirmation-v1'
    cases={r['id']:r for r in rows(directory/'isolated-candidates.jsonl')}
    result=[];changes=[]
    extra_reject={
        'musique:2hop__145282_21711':'Paco Godia is described as from Barcelona, not born there; birth relation absent.',
        '2wiki:db1114140bd911eba7f7acde48001122':'Västerås is where the band formed, not the performer birthplace.',
        '2wiki:2e7b57820bde11eba7f7acde48001122':'Virginia jurisdiction/migration does not establish William Randolph death location.',
        '2wiki:48816bde0bda11eba7f7acde48001122':'Two named directors; unqualified director question does not select the gold person uniquely.',
        '2wiki:1df3448c0bda11eba7f7acde48001122':'Two named directors; unqualified director question does not select the gold person uniquely.',
    }
    for old in rows(directory/'source-quality-review.jsonl'):
        r=deepcopy(old);case=cases[r['case_id']]
        # Full original paragraphs, not the helper's sometimes wrong shortened
        # citation choice, are the lead's source record for both necessary edges.
        r['lead_source_paragraphs']=[dict(title=d['title'],text=d['text']) for d in case['docs'] if d['support']]
        r['lead_review_scope']='64 source-quality decisions; preserved original gold and helper review'
        if r['case_id']=='musique:2hop__690412_526810':
            r.update(accepted=True,reason='Lead: direct principal tributary of Chao Phraya establishes the receiving river at its mouth. Accept this equivalent hydrological relation; mere shared catchment would not suffice.')
        elif r['case_id']=='2wiki:085e01bc0bde11eba7f7acde48001122':
            r.update(accepted=True,reason='Lead: birthplace is explicitly Vadul-Raşcov in Soroca County. Permit the qualified county or locality as a coarser/finer supported answer, without conflating Soroca city and county.',semantic_allowed_answers=['Vadul-Raşcov','Soroca County'])
        elif r['case_id'] in extra_reject:r.update(accepted=False,reason='Lead: '+extra_reject[r['case_id']])
        if r['accepted']!=old['accepted']:changes.append(dict(case_id=r['case_id'],helper=old,lead=r))
        result.append(r)
    count=sum(r['accepted'] for r in result)
    write_once(directory/'source-quality-main-v1.jsonl',result,jsonl=True)
    write_once(directory/'source-quality-adjudications-v1.jsonl',changes,jsonl=True)
    write_once(OUT/'confirmation-quality-gate.json',dict(passed=count==64,reviewed=64,accepted=count,
        required=64,shortfall=64-count,counts=dict(Counter(cases[r['case_id']]['source'] for r in result if r['accepted'])),
        status='stopped_source_quality_shortfall' if count!=64 else 'sealed_quality_passed',
        automatic_pool_expansion=False,confirmation_gpu_trajectories_executed=0,
        source_quality_main_sha256=sha(directory/'source-quality-main-v1.jsonl')))
    print(dict(accepted=count,required=64,shortfall=64-count))

if __name__=='__main__':main()
