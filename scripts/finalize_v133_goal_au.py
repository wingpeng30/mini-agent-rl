"""One-pass AU closure using actual-input citations, without inference."""
import json
import analyze_v133_goal_au as review
import finalize_v133_goal_am_seed42 as old

ap=review.ap;OUT=review.OUT;ROOT=review.ROOT
rows,read,write=review.rows,review.read,review.write
# These are source-read judgments keyed by identity, never gold-string matches.
CHANGES={
 'musique:2hop__864101_135373':('factual_error','The question asks what the sea was named after. Both island/sea and sea/Vitus Bering bodies remain visible. Bering Sea selects the intermediate object instead of the naming object.',True),
 'musique:2hop__829358_754719':('success','Actual Darley body gives Pedro Leopoldo; actual Chico Xavier body explicitly locates this city in State of Minas Gerais. The unchanged queries now select the correctly bound state.',False),
 'musique:2hop__862994_69048':('unsupported_by_visible_evidence','Chief Justice of the United States is a role, not the chief judge of the visible birthplace Delta State. The required state-to-chief-judge edge is absent.',True),
 'musique:2hop__337223_56873':('unsupported_by_visible_evidence','The visible membership bridge is Nottingham Forest. The next query switches to Manchester United and2011. No Forest FA Cup winning-date edge is visible; another team or playing-career endpoint cannot prove it.',True),
 'musique:2hop__796221_18803':('success','The actual Shell body says April2010 seeking a potential buyer for all Finland operations. Trying to find a potential buyer is a grounded description of the asked process; its omitted asset/location detail is recorded as reduced precision, not a changed process or a newly required qualifier.',False),
 'musique:2hop__96194_78303':('factual_error','Alex Cahill belongs to Sheree J Wilson, whereas the film star is Chuck Norris and the visible cast table assigns him Cordell Walker. The first film/star edge was not returned. Related cast members do not substitute for the requested star.',True),
 'musique:2hop__703493_162341':('incomplete_required_qualifier','Best Supporting Actor names the category but omits nomination status. The actual film body proves an Academy Award nomination, not winning or receiving the award.',True),
 'musique:2hop__498251_126089':('unsupported_by_visible_evidence','The area does not name the person in charge; the required province-to-governing-person edge is absent.',True),
}

def main():
    review.au.frozen();review.au.verify_model()
    baseline=rows(ap.AM/'eval-42-A.jsonl');a=rows(OUT/'eval-A.jsonl');b=rows(OUT/'eval-B.jsonl')
    if len(a)!=64 or [r['case_id'] for r in a]!=[r['case_id'] for r in baseline]:raise RuntimeError('A sequence mismatch')
    for ref,current in zip(baseline,a):
        if ref['answer']!=current['answer'] or len(ref['trace'])!=len(current['trace']):raise RuntimeError('A baseline action mismatch')
        for rt,ct in zip(ref['trace'],current['trace']):
            for key in ('raw','prompt_token_ids','action_token_ids'):
                if rt.get(key)!=ct.get(key):raise RuntimeError('A exact archive mismatch '+key)
    write(OUT/'baseline-replay-proof.json',dict(cases=64,calls=sum(len(r['trace']) for r in a),same_raw=True,same_input_ids=True,same_output_ids=True))
    parent={r['case_id']:r for r in rows(ap.AM/'semantic-review-seed42.jsonl') if r['model']=='42-A'}
    packets=rows(OUT/'review-packets.jsonl');semantic=[]
    for p in packets:
        cid=p['case_id'];prev=parent[cid]
        if p['arm']=='B' and cid in CHANGES:category,reason,mismatch=CHANGES[cid]
        else:
            if p['answer']!=prev['answer']:raise RuntimeError('Unreviewed changed answer '+cid)
            category,reason,mismatch=old.RULINGS[cid]
            reason='Existing judgment reused after checking fresh actual-input citations: '+reason
        if category=='success' and (not p['complete_chain_visible'] or p['reason']!='answer'):raise RuntimeError('Unsupported success '+cid)
        semantic.append(dict(case_id=cid,arm=p['arm'],answer=p['answer'],category=category,reason=reason,
            entity_relation_mismatch=mismatch,complete_chain_visible=p['complete_chain_visible'],edges=p['edges'],
            first_divergence=p['first_divergence'],queries=p['queries'],reduced_answer_precision=p['arm']=='B' and cid=='musique:2hop__796221_18803',training_use_forbidden=True))
    write(OUT/'semantic-review.jsonl',semantic,jsonl=True)
    metrics=read(OUT/'mechanical-metrics.json');protected=set(read(ap.am.AL/'manifest.json')['protected'])
    good={arm:{r['case_id'] for r in semantic if r['arm']==arm and r['category']=='success'} for arm in ('A','B')}
    stats={}
    for arm in ('A','B'):
        selected=[r for r in semantic if r['arm']==arm]
        stats[arm]=dict(supported=len(good[arm]),complete_chain=sum(r['complete_chain_visible'] for r in selected),
            error_total=sum(r['category'] in ('factual_error','unsupported_by_visible_evidence') for r in selected),
            incomplete_qualifier=sum(r['category']=='incomplete_required_qualifier' for r in selected),
            entity_relation_mismatch=sum(r['entity_relation_mismatch'] for r in selected),
            source_unconfirmed=35,mechanical=metrics[arm])
    gained=sorted(good['B']-good['A']);lost=sorted(good['A']-good['B'])
    checks=dict(support_gain=stats['B']['supported']>=stats['A']['supported']+2,
        errors_not_increase=stats['B']['error_total']<=stats['A']['error_total'],
        actions_not_increase=all(metrics['B'][k]<=metrics['A'][k] for k in ('invalid_or_repeated','parse_invalid','repeated','overbudget_search_attempts')),
        retain_rapana_and_duke={'musique:2hop__796221_18803','2wiki:f299dd600bdd11eba7f7acde48001122'}<=good['B'],
        protection=len((good['A']-good['B'])&protected)<=1)
    gate=dict(stats=stats,gained=gained,lost=lost,checks=checks,passed=all(checks.values()),fixed_denominator=64,admitted_denominator=29)
    write(OUT/'route-gate.json',gate)
    changes=[]
    byarm={arm:{r['case_id']:r for r in semantic if r['arm']==arm} for arm in ('A','B')}
    for cid in gained+lost:
        pa=next(p for p in packets if p['case_id']==cid and p['arm']=='A');pb=next(p for p in packets if p['case_id']==cid and p['arm']=='B')
        divergence=next((i for i,(x,y) in enumerate(zip(pa['trace'],pb['trace'])) if x['raw']!=y['raw'] or x['prompt_token_ids']!=y['prompt_token_ids']),None)
        changes.append(dict(case_id=cid,gained=cid in gained,A=byarm['A'][cid],B=byarm['B'][cid],first_divergence=divergence,
            same_queries=pa['queries']==pb['queries'],necessary_edge_newly_visible=not pa['complete_chain_visible'] and pb['complete_chain_visible'],
            mechanism='changed retrieval ordering/context changes final relation-object selection; necessary chain visible in both arms',weights_unchanged=True))
    write(OUT/'wins-losses.jsonl',changes,jsonl=True)
    journal=rows(OUT/'gpu-usage.jsonl');counts={k:sum(r['kind']==k for r in journal) for k in ('generation','search','update')}
    if counts['generation']>512 or counts['search']>384 or counts['update']:raise RuntimeError('AU quota violation')
    if gate['passed']:raise RuntimeError('Passing AU requires AV preparation, not rejection closure')
    write(OUT/'final-status.json',dict(completed=True,accepted=False,status='completed_rejected',research_parent='AM42A',research_parent_adapter=str(ap.AMADAPTER),research_parent_sha256=ap.AM_HASH,counts=counts,new_confirmation=0,published=False))
    offline=read(OUT/'offline-gate.json');loads=rows(OUT/'component-load-times.jsonl')
    write(OUT/'REPORT.md',f'''# Goal AU：固定权重的检索排序对照

## 结论

本轮不接受reranker候选。离线五题必要文档进入top-3、完整链23→24，未转化为真实搜索净收益：实际支持成功{stats['A']['supported']}→{stats['B']['supported']}/29。当前研究父版本保持AM seed42-A；唯一发布模型仍为Goal Z。

| 指标 | A：BM25-v2 | B：BM25 top20 + BGE top3 |
|---|---:|---:|
| 支持成功/29 | {stats['A']['supported']} | {stats['B']['supported']} |
| 回答输入完整链/29 | {stats['A']['complete_chain']} | {stats['B']['complete_chain']} |
| 事实错误与证据不支持/29 | {stats['A']['error_total']} | {stats['B']['error_total']} |
| 必要限定不足/29 | {stats['A']['incomplete_qualifier']} | {stats['B']['incomplete_qualifier']} |
| EM/64 | {metrics['A']['EM']} | {metrics['B']['EM']} |
| F1 | {metrics['A']['F1']:.6f} | {metrics['B']['F1']:.6f} |
| 动作问题题数/64 | {metrics['A']['invalid_or_repeated']} | {metrics['B']['invalid_or_repeated']} |
| 模型调用/搜索 | {metrics['A']['calls']}/{metrics['A']['searches']} | {metrics['B']['calls']}/{metrics['B']['searches']} |

其余35题来源未确认，不将EM命中自动计为充分成功。58个答案组合完成一次裁定，逐边原句及实际token位置在semantic-review.jsonl。A的64题、全部192次输入、输出token及原始文本与AM档案完全一致，证明本轮匹配基线复现。

## 赢回与丢失

- Darley：Minsk Region→State of Minas Gerais。题面、查询和权重不变，检索排序改变；两臂回答输入均含Darley→Pedro Leopoldo→Minas Gerais，B正确选择地域对象。这是变化检索上下文下的绑定改善，不是新增关系取得或权重学习。
- St. Matthew：Vitus Bering→Bering Sea。两臂均保留完整岛屿→海域→命名人物链，B把中间海域当作命名对象。这是完整证据下的关系对象退步。
- Rapana保留成功：B回答“trying to find a potential buyer”，与实际April2010正文的过程一致；省略Finland资产范围另记精度下降。它不是新增成功。若更严格要求完整资产范围，B只会更低，本轮不接受结论不变。
- Hero仍失败：B查询Sheree J Wilson并答Alex Cahill，缺少电影→Chuck Norris桥接边；相关演员不能替代目标主演。

完整链进入输入仍不保证正确对象选择。离线固定查询指标与模型实际闭环指标不可混用。七题连续搜索保护保留，Rapana与公爵保留；净增至少两题的条件失败。没有更换模型、排序规则、提示或追加采样。

## 计算、网络修复与边界

实际生成{counts['generation']}次、搜索{counts['search']}次、训练更新0、新确认0。另有130个既有查询、2475个离线query-window评分；live排序记录及每个实际窗口输入在live-ranking.jsonl与reranker-actual-inputs.jsonl，加载耗时单列component-load-times.jsonl。B在线reranker合计4.386078秒，单题模型及总流程延迟保存在mechanical-metrics.json和原轨迹。未单独保存显存峰值，不能补称已验证峰值。

按用户明确授权修复下载：source /etc/network_turbo后使用hf-mirror.com，固定BAAI/bge-reranker-base revision2cfc18c9415c912f9d8155881c133215df768a70。七份文件与官方LFS SHA256/Git blob一致；权重SHA256 ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd。v1网络失败、v2 Transformers5.17 tokenizer接口失败分别保留，均未生成GPU样本；v3仅使用官方同一pair模板修复接口，7项CPU检查通过。没有重跑模型样本或调整实验参数，全部实验计算在AutoDL完成。

当前配方不值得进入累计验证；这不证明所有reranker或0.8B模型达到上限。协议约束、固定小样本GRPO、现有reranker均未形成可接受新版本，AV无新候选，不能声称208题回归保护通过。
''')
    print(json.dumps(gate,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
