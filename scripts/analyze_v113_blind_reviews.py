"""Create post-review unblinded P0/P1 aggregates while preserving raw blind reviews."""
import hashlib, json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports'
def read(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def norm_support(x):
    return {'entailed':'supported','not_answer':'no_answer','not_applicable':'not_applicable'}.get(x,x)

def main():
    # P0: per task, each of three reviewers reviewed the same 50 task pairs;
    # the additional 11 supported-state pairs are similarly triply covered.
    cases=read(R/'v113-p0-review-cases.jsonl'); key=json.loads((R/'v113-p0-private-key.json').read_text(encoding='utf-8'))
    score=json.loads((R/'v113-p0-score-key.json').read_text(encoding='utf-8'))
    files=[read(R/f'v113-p0-agent-review-{i}.jsonl') for i in range(3)] + [read(R/f'v113-p0-agent-review-extra-{i}.jsonl') for i in range(3)]
    maps=[{x['case_id']:x for x in group} for group in files]
    byid={x.get('blind_case_id',x.get('blind_state_id')):x for x in cases}
    final_cross=Counter(); per_review=Counter(); issue=Counter()
    for cid, packet in byid.items():
        verdicts={arm:[] for arm in ('v101','v112')}
        for m in maps:
            if cid not in m: continue
            r=m[cid]
            for letter in ('A','B'):
                arm=key[cid][letter+'_arm']; v=norm_support(r[letter].get('final_support','uncertain'))
                verdicts[arm].append(v); per_review[f'{arm}:{v}']+=1
                for tag in r[letter].get('trajectory_tags',[]): issue[f'{arm}:{tag}']+=1
        if any(len(verdicts[arm]) != 3 for arm in verdicts): raise ValueError('expected_three_blind_reviewers_per_arm:'+cid)
        # Preserve mixed three-reviewer outcomes; a 2/3 majority is merely a secondary annotation.
        majority={arm: Counter(vs).most_common(1)[0][0] if len(Counter(vs).most_common())==1 or Counter(vs).most_common()[0][1]>Counter(vs).most_common()[1][1] else 'no_majority' for arm,vs in verdicts.items()}
        if 'answer_changed' in score[cid]:
            official='improved' if score[cid]['candidate']['normalized_exact_match']>score[cid]['baseline']['normalized_exact_match'] else 'regressed' if score[cid]['candidate']['normalized_exact_match']<score[cid]['baseline']['normalized_exact_match'] else 'equal'
            final_cross[f'{official}|{majority["v101"]}|{majority["v112"]}']+=1
    out0={'reviewed_task_cases':50,'reviewed_supported_state_cases':11,'all_three_reviewers_cover_each_case':True,
          'per_arm_reviewer_final_support_counts':dict(per_review),'official_em_vs_three_reviewer_plurality':dict(final_cross),
          'trajectory_issue_tags':dict(issue),'scoring_and_evidence_support_are_separate':True,
          'majority_is_auxiliary_agent_judgment_not_ground_truth':True}
    # P1 has disjoint reviewer slices; unblind A/B per arm only after review artifacts are final.
    p1key=json.loads((R/'v113-p1-private-key.json').read_text(encoding='utf-8'))
    p1maps=[{x['case_id']:x for x in read(R/f'v113-p1-agent-review-{i}.jsonl')} for i in range(3)]
    behavior=Counter(); labels=Counter(); relative=Counter(); covered=Counter()
    for i,m in enumerate(p1maps):
        for cid,r in m.items():
            covered[f'reviewer_{i}']+=1
            for letter in ('A','B'):
                arm=p1key[cid][letter+'_arm']; a=r[letter]
                labels[f'{arm}:support:{norm_support(a.get("answer_support","missing"))}']+=1
                labels[f'{arm}:search_needed:{a.get("search_needed","missing")}']+=1
                labels[f'{arm}:query_grounded:{a.get("query_grounded","missing")}']+=1
                labels[f'{arm}:retrieval_gap_fit:{a.get("retrieval_gap_fit","missing")}']+=1
                for tag in a.get('issue_tags',[]): labels[f'{arm}:issue:{tag}']+=1
            raw=r.get('relative_evidence','missing')
            if 'uncertain' in raw or 'no_clear' in raw: cls='uncertain_or_no_clear'
            elif 'equivalent' in raw: cls='equivalent'
            elif 'A_better' in raw: cls='A_better'
            elif 'B_better' in raw: cls='B_better'
            else: cls='other:'+raw
            aarm=p1key[cid]['A_arm']; barm=p1key[cid]['B_arm']
            winner={'A_better':aarm,'B_better':barm}.get(cls)
            relative[cls if not winner else winner+'_better']+=1
    out1={'case_coverage_by_disjoint_reviewer':dict(covered),'actions_audited_by_unblinded_arm':dict(labels),
          'relative_evidence_comparison':dict(relative),'all_170_inputs_have_one_review':sum(covered.values())==170,
          'reviewer_agreement_claimed':False,'fixed_state_only':True}
    for name,obj in [('v113-p0-adjudication-summary.json',out0),('v113-p1-unblinded-audit-summary.json',out1)]:
        path=R/name
        if path.exists(): raise FileExistsError(str(path))
        path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'p0':out0,'p1':out1},ensure_ascii=False))
if __name__=='__main__': main()
