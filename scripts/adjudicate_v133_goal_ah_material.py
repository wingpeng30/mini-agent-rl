"""Lead adjudication of the once-frozen AH pool; helper reviews remain immutable."""
from collections import Counter
from copy import deepcopy
from prepare_v133_goal_ah import OUT, rows, write_once

def main():
    proposals={r['judgment_id']:r for r in rows(OUT/'material-proposals.jsonl')}
    result=[]; changes=[]
    for helper in rows(OUT/'material-review-helper.jsonl'):
        r=deepcopy(helper);p=proposals[r['judgment_id']]
        r['reviewer']='lead; helper source-bound audit followed by explicit adjudication'
        r['helper_accepted']=helper['accepted']
        if p['verdict']=='supported' and any(f[1]=='country of citizenship' for f in p['target']['facts']):
            r.update(accepted=False,reason='Lead: nationality adjective alone does not establish legal country of citizenship. Avoid teaching a stronger relation than the source states.')
        elif p['verdict']=='supported' and p['target']['facts'][1][0] in ['Ray Kellogg','Pierre Falardeau','Adoor Bhasi']:
            r.update(accepted=False,reason='Lead: source states origin/from-region or mentions a rally location; it does not establish this birth/death location. Reject despite matching gold.')
        elif p['verdict']=='relation_mismatch' and r['corrected_target']['facts'][0][1]=='country of citizenship':
            fact=r['corrected_target']['facts'][0]
            fact[1]='nationality description'
            if fact[0]=='Sudhir Mishra':fact[2]='Indian'
            r['reason']='Lead: candidate wrongly binds a nationality description to birthplace. Correct fact uses the actual nationality description, without asserting legal citizenship.'
        elif p['verdict']=='missing_link' and any(s in p['case_id'] for s in ['25478_69291','02d26d9c']):
            r.update(accepted=False,reason='Lead: ambiguous question premise or unqualified film version. Do not train a definite missing edge chosen using the hidden gold chain.')
        if r['accepted']!=helper['accepted'] or r['corrected_target']!=helper['corrected_target']:
            changes.append(dict(judgment_id=r['judgment_id'],case_id=p['case_id'],helper=helper,lead=r))
        result.append(r)
    write_once(OUT/'material-review-main.jsonl',result,jsonl=True)
    write_once(OUT/'material-adjudication-v1.jsonl',changes,jsonl=True)
    print(dict(Counter(proposals[r['judgment_id']]['verdict'] for r in result if r['accepted'])))

if __name__=='__main__':main()
