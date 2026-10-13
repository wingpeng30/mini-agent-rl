"""Single-pass source-grounded AW probe closure, no new inference."""
import json
import run_v133_goal_aw as aw
OUT=aw.OUT;rows,read,sha=aw.rows,aw.read,aw.sha
def main():
    aw.frozen();baseline={r['probe_id']:r for r in rows(aw.PRIOR/'probe-adjudications.jsonl') if r['arm']=='A'}
    original={r['probe_id']:r for r in rows(aw.PRIOR/'probe-A.jsonl')}
    inputs={r['probe_id']:r for r in rows(OUT/'probe-inputs.jsonl')};ledger=[]
    for arm in ('A','C','B'):
        for r in rows(OUT/f'probe-{arm}.jsonl'):
            old=baseline[r['probe_id']];same=(r['action'],r['answer'],r['query'])==(old['action'],old['answer'],old['query'])
            if arm=='A' and (not same or r['prompt_token_ids']!=original[r['probe_id']]['prompt_token_ids'] or r['action_token_ids']!=original[r['probe_id']]['action_token_ids']):raise RuntimeError('Fresh A mismatch')
            correct=False;legal=False;reason=''
            if r['category']=='complete':
                if r['action']=='search':reason='Complete original body chain is present; final action continues search rather than answering. No search executed in probe.'
                elif same:correct=old['supported_correct'];reason='Same grounded answer, original body identities and ordered text retained: '+old['adjudication_reason']
                else:raise RuntimeError('Unreviewed new complete answer '+r['probe_id'])
            else:
                if same:legal=old['legal_bridge_search'];reason='Same query and body: '+old['adjudication_reason']
                elif r['case_id']=='musique:2hop__862994_69048':reason='Gloria birthplace chief judge leaves the actual Delta State bridge unbound; not a query for the missing state/chief-judge relation.'
                elif r['case_id']=='musique:2hop__789565_416401' and r['query']=="Wang Yanjun's child":legal=True;reason='Possessive alias of visible Wang Yanjun child relation; no hidden entity.'
                elif r['case_id']=='musique:2hop__506035_325512' and r['query']=='Poplarville, Mississippi county':legal=True;reason='Visible Poplarville bridge plus county relation; state qualifier disambiguates, no new answer entity.'
                elif r['case_id']=='musique:2hop__337223_56873':reason='Names Forest but omits the missing FA Cup winning-time relation; related membership query is not the required next edge.'
                else:raise RuntimeError('Unreviewed partial '+r['probe_id'])
            seen=[]
            for m in inputs[r['probe_id']]['messages']:
                if m['role']=='tool':seen+=json.loads(m['content']).get('searched_queries',[])
            norm=lambda s:' '.join(str(s).casefold().split())
            ledger.append(dict(arm=arm,probe_id=r['probe_id'],case_id=r['case_id'],category=r['category'],
                supported_correct=correct,legal_bridge_search=legal,binding_repair=False,reason=reason,
                premature_answer=r['category']=='partial' and r['action']=='answer',invalid_action=bool(r['parse_error']),
                repeated_query=bool(r['query']) and norm(r['query']) in {norm(s) for s in seen},
                reader_valid=bool(r['usage']['decision_pipeline']['auxiliary'] and r['usage']['decision_pipeline']['auxiliary'][0]['audit']['valid']),
                actual_input=r['prompt_token_ids'],source_documents=inputs[r['probe_id']]['documents'],training_use_forbidden=True))
    def n(arm,cat,key):return sum(r[key] for r in ledger if r['arm']==arm and r['category']==cat)
    stats={a:dict(complete=n(a,'complete','supported_correct'),partial=n(a,'partial','legal_bridge_search'),reader_valid=sum(r['reader_valid'] for r in ledger if r['arm']==a)) for a in ('A','C','B')}
    eligibility={}
    for arm in ('C','B'):
        gains=[r for r in ledger if r['arm']==arm and r['category']=='complete' and r['supported_correct'] and not baseline[r['probe_id']]['supported_correct']]
        loss=[r['probe_id'] for r in ledger if r['arm']==arm and r['category']=='complete' and not r['supported_correct'] and baseline[r['probe_id']]['supported_correct']]
        eligibility[arm]=dict(passed=stats[arm]['complete']>=22 and stats[arm]['complete']>=stats['A']['complete']+2 and any(r['binding_repair'] for r in gains) and len(loss)<=1 and stats[arm]['partial']>=7 and all(n(arm,'partial',k)<=n('A','partial',k) for k in ('premature_answer','invalid_action','repeated_query')),gained=[r['probe_id'] for r in gains],lost=loss)
    aw.write(OUT/'probe-adjudications.json',ledger)
    gate=dict(metrics=stats,eligibility=eligibility,real_search_allowed=any(r['passed'] for r in eligibility.values()))
    aw.write(OUT/'probe-gate.json',gate)
    if gate['real_search_allowed']:raise RuntimeError('Eligible real-search stage remains')
    counts={k:sum(r['kind']==k for r in rows(OUT/'gpu-usage.jsonl')) for k in ('generation','search')}
    aw.write(OUT/'final-status.json',dict(completed=True,accepted=False,research_parent='AM42A',research_parent_adapter=str(aw.ap.AMADAPTER),research_parent_sha256=aw.ap.AM_HASH,counts=counts,updates=0,new_confirmation=0,status='rejected_at_probe_gate'))
    with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:f.write(f'''# AW：逐跳读取与同预算复核

两臂均不通过，不执行64题搜索、不接受新父版本。A/C/B完整证据动作{stats['A']['complete']}/{stats['C']['complete']}/{stats['B']['complete']}/24，部分合法补链{stats['A']['partial']}/{stats['C']['partial']}/{stats['B']['partial']}/8。

A32个实际输入、生成token、动作与原档案一致。B32/32仍生成旧search/answer JSON，0/32遵守桥接字段协议；按约定invalid辅助状态传入最终调用，没有补写或重试。C有效草案也大量转为搜索：20个原成功仅保留1个；B原20成功全部转搜索。不能据此证明逐跳读取本身无效或模型到上限，因为B没有执行读取任务。C同时说明当前两步复核接口本身有严重停止决策退步。

必要正文/顺序未变，未裁剪，未执行任何草案或探针搜索。部分B的Gloria查询没绑定Delta State；Crossley查询只有人物/Forest，没有赢杯时间关系；Wang Yanjun possessive和Poplarville county是合法等价查询。逐题输入、原文和理由见probe-adjudications.json，完整原输出见probe-A/C/B.jsonl。

实际{counts['generation']}生成（工程12+A32+C64+B64）、搜索0、训练0、新确认0。单卡4090D/BF16、16384/2048、greedy、first-json，无参数修改补跑。3项CPU接口测试通过。研究父版本AM42-A，产品Goal Z。按顺序进入AX固定34状态第一跳准入；不足关闭，不补材料。
''')
    print(json.dumps(gate),flush=True)
if __name__=='__main__':main()
