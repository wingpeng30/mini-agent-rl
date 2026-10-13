"""Main-agent single-pass AT semantic rulings and route closure."""
import json
from pathlib import Path
import analyze_v133_goal_at as review
import finalize_v133_goal_am_seed42 as old_review
ap=review.ap;OUT=review.OUT;ROOT=review.ROOT
rows,read,write=review.rows,review.read,review.write
CHANGES={
 'musique:2hop__337223_56873':('success','The original question asks When, not all winning years. Actual Forest body states FA Cup wins in1898 and1959. 1898 is one directly supported winning year, flagged as a partial answer set; no requirement that Crossley personally played at that win is added.',False),
 'musique:2hop__796221_18803':('unsupported_by_visible_evidence','Service stations names assets, not the asked April2010 process; merchant aircraft carrier describes Rapana, not the owner announcement. A has the explicit buyer-for-Finland clause but selects the wrong object; B answers with only ship/owner evidence and no announcement edge.',True),
 'musique:2hop__418360_67370':('factual_error','October8 2017 does not answer the question explicitly restricted to the2018 restart. The actual input lacks the necessary January19 2018 relation; no title/keyword hit substitutes for it.',True),
 '2wiki:764039340bde11eba7f7acde48001122':('unsupported_by_visible_evidence','Current query shifts to CharingCross/Edinburgh instead of father JamesHamilton. Actual input does not contain the father death-place edge. Edinburgh cannot be justified as the requested father death place by the son or burial references.',True),
}

def finalize():
    review.ev.training.frozen();packets=rows(OUT/'review-packets.jsonl');semantic=[]
    for p in packets:
        cid=p['case_id'];arm=p['arm']
        if cid in CHANGES:category,reason,mismatch=CHANGES[cid]
        elif cid=='2wiki:1dd33f080bde11eba7f7acde48001122' and arm=='A':
            category,reason,mismatch='success','Actual son-to-father clause binds JohannNicolaus to JohannChristoph; the latter body explicitly born at Arnstadt. The answer selects the father birthplace, not the son Eisenach.',False
        elif cid=='musique:2hop__96194_78303' and arm=='A':
            category,reason,mismatch='invalid_action','The generated first object does not parse into an executable action. Both film/star and Norris/CordellWalker edges were present, but no valid final answer was produced.',False
        elif cid=='musique:2hop__96194_78303' and arm=='B':
            if p['answer']!='The Terror':raise RuntimeError('Unreviewed TV character answer')
            category,reason,mismatch='factual_error','The Terror names the film killer, not ChuckNorris character in WalkerTexasRanger; actual cast table binds Norris to CordellWalker.',True
        else:
            old=next(r for r in rows(ap.AM/'semantic-review-seed42.jsonl') if r['case_id']==cid and r['model']=='42-A')
            # Same answer alone never grants sufficient evidence. Fresh citations must all exist.
            if p['answer']!=old['answer']:raise RuntimeError('Unreviewed new answer '+arm+' '+cid)
            category,reason,mismatch=old_review.RULINGS[cid]
            if category=='success' and not p['complete_chain_visible']:raise RuntimeError('Missing evidence cannot inherit success')
            reason='Existing relation ruling reused after remapping actual body citations: '+reason
        if category=='success' and (p['reason']!='answer' or not p['complete_chain_visible']):raise RuntimeError('AT unsupported success')
        semantic.append(dict(case_id=cid,arm=arm,answer=p['answer'],category=category,reason=reason,entity_relation_mismatch=mismatch,
          complete_chain_visible=p['complete_chain_visible'],edges=p['edges'],first_divergence=p['first_divergence'],queries=p['queries'],
          partial_answer_set=cid=='musique:2hop__337223_56873',training_use_forbidden=True))
    baseline={r['probe_id']:r for r in rows(ROOT/'runs/v133-goal-ar-constrained-evidence-v1/probe-adjudications.jsonl') if r['arm']=='A'}
    inputs={r['probe_id']:r for r in rows(ROOT/'runs/v133-goal-ar-constrained-evidence-v1/probe-inputs.jsonl') if r['arm']=='A'}
    probes=[]
    for arm in ('A','B'):
        for p in rows(OUT/f'probe-42-{arm}.jsonl'):
            old=baseline[p['probe_id']];correct=old['supported_correct'];legal=old['legal_bridge_search'];reason=old['adjudication_reason']
            if p['category']=='complete':
                if p['case_id']=='2wiki:764039340bde11eba7f7acde48001122':
                    if p['answer']!='Westminster':raise RuntimeError('Unreviewed father location probe')
                    correct=True;reason='Father death clause explicitly says HydePark, Westminster. Westminster is a supported coarse location answering Where, flagged as lower precision rather than exact park equivalence.'
                elif arm=='B' and p['case_id']=='musique:2hop__109242_113442':
                    if p['answer']!='8 June 1829':raise RuntimeError('Unreviewed Millais output')
                    correct=False;reason='8June1829 is the painter birth date; asked date of death is13August1896 in the same actual biography. This is birth/death relation substitution.'
                elif (p['action'],p['answer'],p['query'])!=(old['action'],old['answer'],old['query']):raise RuntimeError('Unreviewed changed complete probe '+str(p))
                legal=False
            else:
                correct=False
                if arm=='B' and p['case_id']=='musique:2hop__796221_18803':
                    legal=False;reason='Answers merchant aircraft carrier before missing owner announcement relation is returned; ship type is not the announcement process.'
                elif p['case_id']=='musique:2hop__789565_416401':
                    if p['query']!="Wang Yanjun's child":raise RuntimeError('Unreviewed child query')
                    legal=True;reason='Visible WangYanjun bridge plus missing child relation; possessive is semantically equivalent to original query.'
                elif p['case_id']=='musique:2hop__337223_56873':
                    legal=False;reason='2015-16 belongs to unrelated EuropaLeague rules, not the visible Forest FA Cup missing relation; query retains unsupported scope.'
                elif ' '.join(str(p['query']).casefold().split())!=' '.join(str(old['query']).casefold().split()):raise RuntimeError('Unreviewed partial query '+p['probe_id'])
            seen=[]
            for m in inputs[p['probe_id']]['messages']:
                if m['role']=='tool':seen.extend(json.loads(m['content']).get('searched_queries',[]))
            norm=lambda q:' '.join(str(q).casefold().split())
            repeated=p['query'] is not None and norm(p['query']) in {norm(q) for q in seen}
            probes.append(dict(p,correct=correct,legal_bridge_search=legal,reason=reason,premature_answer=p['category']=='partial' and p['action']=='answer',
                invalid_action=bool(p['parse_error']),repeated_query=repeated,precision_decline=p['case_id']=='2wiki:764039340bde11eba7f7acde48001122',
                actual_source=inputs[p['probe_id']]['documents']))
    write(OUT/'semantic-review.jsonl',semantic,jsonl=True);write(OUT/'probe-adjudications.jsonl',probes,jsonl=True)
    metrics=read(OUT/'mechanical-metrics.json');protection=read(ap.am.AL/'manifest.json')['protected'];parent=[r for r in rows(ap.AM/'semantic-review-seed42.jsonl') if r['model']=='42-A']
    old_success={r['case_id'] for r in parent if r['category']=='success'}
    result={};paired=[]
    for arm in ('A','B'):
        arm_sem=[r for r in semantic if r['arm']==arm];success={r['case_id'] for r in arm_sem if r['category']=='success'}
        err=sum(r['category'] in ('factual_error','unsupported_by_visible_evidence') for r in arm_sem)
        ps=[p for p in probes if p['model']==f'42-{arm}'];complete=sum(p['category']=='complete' and p['correct'] for p in ps)
        partial=sum(p['category']=='partial' and p['legal_bridge_search'] for p in ps)
        protection_lost=sorted((old_success-success)&set(protection));retains={'musique:2hop__796221_18803','2wiki:f299dd600bdd11eba7f7acde48001122'}<=success
        checks=dict(support_gain=len(success)>=23,complete_gain=complete>=22,errors_not_increase=err<=7,
           actions_not_increase=all(metrics[arm][k]<=metrics['AM'][k] for k in ('invalid_or_repeated','parse_invalid','repeated','overbudget_search_attempts')),
           retain_rapana_and_duke=retains,protection=len(protection_lost)<=1,partial_search=partial>=7,
           partial_actions_not_increase=not any(p['category']=='partial' and (p['premature_answer'] or p['invalid_action'] or p['repeated_query']) for p in ps))
        result[arm]=dict(supported=len(success),complete_chain=sum(r['complete_chain_visible'] for r in arm_sem),error_total=err,
          incomplete_qualifier=sum(r['category']=='incomplete_required_qualifier' for r in arm_sem),complete_probes=complete,partial_legal_search=partial,
          gained=sorted(success-old_success),lost=sorted(old_success-success),protection_lost=protection_lost,checks=checks,passed=all(checks.values()))
        for cid in sorted(old_success^success):
            r=next(x for x in arm_sem if x['case_id']==cid);packet=next(x for x in packets if x['arm']==arm and x['case_id']==cid)
            paired.append(dict(r,parent_success=cid in old_success,new_success=cid in success,classification='changed-query retrieval and/or binding' if r['complete_chain_visible'] else 'necessary relation not obtained',
                               parent_answer=packet['parent_answer']))
    result['RL_attribution']=result['B']['passed'] and result['B']['supported']>=result['A']['supported']+2 and result['B']['complete_probes']>=result['A']['complete_probes']+2
    write(OUT/'route-gate.json',result);write(OUT/'wins-losses.jsonl',paired,jsonl=True)
    # This observed outcome has no qualifying route; never choose a single metric winner.
    if result['A']['passed'] or result['B']['passed']:raise RuntimeError('Passing route requires acceptance and AV freeze, not rejected closure')
    journal=rows(OUT/'gpu-usage.jsonl');counts={k:sum(x['kind']==k for x in journal) for k in ('generation','search','update')}
    write(OUT/'final-status.json',dict(completed=True,accepted=False,status='completed_rejected',research_parent='AM42A',
      research_parent_adapter=str(ap.AMADAPTER),research_parent_sha256=ap.AM_HASH,counts=counts,new_confirmation=0,published=False))
    text=f'''# Goal AT：固定采样普通SFT/GRPO对照

两臂均未达到继续门槛，舍弃；实验父版本仍AM42-A，产品仍GoalZ。不追加采样、更新或seed43。

| 指标 | AM父版本 | A正确采样SFT | B固定组GRPO |
|---|---:|---:|---:|
| 支持成功/29 |21|{result['A']['supported']}|{result['B']['supported']}|
| 完整证据探针/24 |20|{result['A']['complete_probes']}|{result['B']['complete_probes']}|
| 部分证据合法补链/8 |7|{result['A']['partial_legal_search']}|{result['B']['partial_legal_search']}|
| 事实错误+证据不支持 |7|{result['A']['error_total']}|{result['B']['error_total']}|
| 动作问题题数/64 |1|{metrics['A']['invalid_or_repeated']}|{metrics['B']['invalid_or_repeated']}|
| EM/64 |26|{metrics['A']['EM']}|{metrics['B']['EM']}|

A赢回Bach父亲出生地：实际两边可见，回答Arnstadt。B仍答儿子Eisenach，不能归因奖励驱动绑定改善。两臂Crossley真实搜索新返回Forest正文明确赢杯1898和1959，回答1898按原When接受一个正文支持年份；单列答案集合不完整，不声称两年等价。不要求Crossley本人在夺冠时效力，该额外限定不来自题面。若要求完整年份集合，该题不计成功，A/B各再少1；路线失败不变。

两臂都丢失Rapana、BillMaher和Abercrombie父亲死亡地。A的Rapana公告链完整但答service stations，选错关系对象；B仅搜船只后答merchant aircraft carrier，缺公告边且提前回答。BillMaher未检索2018复播关系而答2017。Abercrombie查询转向本人CharingCross/Edinburgh，没有补到父亲死亡边。B同证据探针另把Millais死亡日答为出生日期。A真实Bach收益没有迁移到固定完整证据探针；没有稳定同证据绑定提升。

Westminster来自实际父亲死亡句HydePark,Westminster，允许Where的较粗地点，单列精度下降，不称公园级等价。29准入层之外35题保持未确认；EM不能替代支持成功。旧账本及评分未修改。

固定16训练状态×4采样，50/64支持、6混合组。JadenSmith组4次均答JustinBieber，零方差不能提供相对优势。两臂各128更新，原128轨迹+34状态相同，6混合组A监督首个正确样本，B四样本组标准化奖励；其他组零新增损失。B使用.7温度全softmax的old/current/reference，parent冻结后共同SFT会使策略偏移，clipped比率和KL如实保存；这是有限离线池对照，不称持续在线强化学习。loss/optimizer/历史mask测试3项通过。base未变、192LoRA更新、两次独立重载max差0。

实际额度：{counts['generation']}新增生成、{counts['search']}真实搜索、{counts['update']}更新；无新确认。两臂均无正文裁剪或长度停止。每题/调用延迟与动作分类见mechanical-metrics.json；引用、token位置、首次分歧见semantic-review.jsonl/wins-losses.jsonl。失败属于语义与门槛失败，未触发GPU/数值/身份/基础设施停止。

按顺序进入AU独立reranker可行性核验。不得把本轮B的EM比A多1解释为模型升级，也不以新增数据重跑本配方。
'''
    write(OUT/'REPORT.md',text);print(json.dumps(dict(results=result,counts=counts)),flush=True)

if __name__=='__main__':finalize()
