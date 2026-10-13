"""Read current inputs and explicit rulings; never score from gold strings."""
import argparse, hashlib, json
import run_v133_goal_bc as bc

OUT=bc.OUT
read,rows,write=bc.read,bc.rows,bc.write

def signature(r):return (r['action'],r['answer'],r['query'])

def inspect():
    old={r['probe_id']:r for r in read(bc.AW/'probe-adjudications.json') if r['arm']=='A'}
    inputs={r['probe_id']:r for r in rows(OUT/'probe-inputs.jsonl')}
    aa={r['probe_id']:r for r in rows(OUT/'probe-A.jsonl')}
    for r in rows(OUT/'probe-B.jsonl'):
        previous=old[r['probe_id']];item=inputs[r['probe_id']]
        print(json.dumps(dict(probe_id=r['probe_id'],question=next(m['content'] for m in item['messages'] if m['role']=='user'),
          A=signature(aa[r['probe_id']]),B=signature(r),previous_supported=previous['supported_correct'],
          previous_legal=previous['legal_bridge_search'],same_as_A=signature(r)==signature(aa[r['probe_id']]),
          prior_reason=previous['reason']),ensure_ascii=False))

def finalize():
    bc.frozen()
    if not (OUT/'probe-B-complete.json').exists():raise RuntimeError('BC incomplete probes')
    old={r['probe_id']:r for r in read(bc.AW/'probe-adjudications.json') if r['arm']=='A'}
    prior={r['probe_id']:r for r in rows(bc.AW/'probe-A.jsonl')}
    inputs={r['probe_id']:r for r in rows(OUT/'probe-inputs.jsonl')}
    manual=read(OUT/'probe-new-rulings.json')
    decisions=[]
    for arm in ('A','B'):
        for output in rows(OUT/f'probe-{arm}.jsonl'):
            pid=output['probe_id'];item=inputs[pid];previous=old[pid]
            if output['prompt_token_ids']!=item['prompt_token_ids']:raise RuntimeError('Actual input mismatch')
            refs=output['usage'].get('context_aware_decoding',{})
            if refs.get('enabled'):
                if refs['reference_prompt_token_ids']!=item['reference_prompt_token_ids']:raise RuntimeError('Reference input mismatch')
                for s in refs['token_scores']:
                    if s['reference_cache_tokens']!=len(refs['reference_prompt_token_ids'])+s['step']:raise RuntimeError('Independent reference cache length mismatch')
            result=dict(arm=arm,case_id=output['case_id'],probe_id=pid,category=output['category'],
              action=output['action'],answer=output['answer'],query=output['query'],supported_correct=False,
              legal_bridge_search=False,binding_repair=False,evidence=[],actual_input_sha256=hashlib.sha256(str(output['prompt_token_ids']).encode()).hexdigest(),
              input_spans=item['document_spans'],training_use_forbidden=True)
            if signature(output)==signature(prior[pid]):
                result.update(supported_correct=previous['supported_correct'],legal_bridge_search=previous['legal_bridge_search'],
                  reason='Exact actual input and action combination; reuse source-grounded ruling: '+previous['reason'],ruling='exact_combination_reuse')
            elif output['parse_error']:
                result.update(reason='First object fails original action parser; no later object selected.',ruling='invalid_action')
            elif output['category']=='complete' and output['action']=='search':
                result.update(reason='Complete original body chain remains visible, but action continues searching. Probe query not executed.',ruling='sufficient_but_search')
            else:
                key=arm+'|'+pid
                if key not in manual:raise RuntimeError('Missing explicit semantic ruling '+key)
                value=manual[key]
                if value['output_signature']!=list(signature(output)):raise RuntimeError('Ruling action mismatch '+key)
                if value['case_id']!=output['case_id']:raise RuntimeError('Ruling case mismatch')
                result.update({k:value[k] for k in ('supported_correct','legal_bridge_search','binding_repair','reason')})
                for evidence in value['evidence']:
                    m=item['messages'][evidence['message_index']]
                    if m['role']!='tool':raise RuntimeError('Citation outside tool body')
                    candidates=[d for d in json.loads(m['content'])['result'] if d['id']==evidence['document_id']]
                    if len(candidates)!=1:raise RuntimeError('Citation document ambiguity')
                    body=candidates[0]['text'];start=body.find(evidence['quote'])
                    if start<0:raise RuntimeError('Quote not in actual body')
                    span=next(s for s in item['document_spans'] if s['message_index']==evidence['message_index'] and s['document_id']==evidence['document_id'])
                    from transformers import AutoTokenizer
                    if not hasattr(finalize,'tokenizer'):finalize.tokenizer=AutoTokenizer.from_pretrained(str(bc.ap.BASE),local_files_only=True)
                    tokenizer=finalize.tokenizer
                    text=tokenizer.decode(output['prompt_token_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
                    if encoded['input_ids']!=output['prompt_token_ids']:raise RuntimeError('Quote offsets not actual tokens')
                    quote_start=span['escaped_prompt_chars'][0]+len(json.dumps(body[:start],ensure_ascii=False)[1:-1])
                    quote_end=quote_start+len(json.dumps(evidence['quote'],ensure_ascii=False)[1:-1])
                    indexes=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>quote_start and a<quote_end]
                    if not indexes:raise RuntimeError('No quote input tokens')
                    result['evidence'].append(dict(evidence,body_char_span=[start,start+len(evidence['quote'])],input_token_span=[indexes[0],indexes[-1]+1],body_sha256=span['body_sha256']))
                if not result['evidence']:raise RuntimeError('New semantic ruling without visible body evidence')
                result['ruling']='single_pass_primary_agent_body_adjudication'
            seen={q for m in item['messages'] if m['role']=='tool' for q in json.loads(m['content']).get('searched_queries',[])}
            normalize=lambda s:' '.join(str(s).casefold().split())
            result.update(premature_answer=output['category']=='partial' and output['action']=='answer',invalid_action=bool(output['parse_error']),
              repeated_query=bool(output['query']) and normalize(output['query']) in {normalize(q) for q in seen})
            decisions.append(result)
    def count(arm,cat,key):return sum(r[key] for r in decisions if r['arm']==arm and r['category']==cat)
    metrics={a:dict(complete_correct=count(a,'complete','supported_correct'),partial_legal_search=count(a,'partial','legal_bridge_search'),
      partial_premature=count(a,'partial','premature_answer'),partial_invalid=count(a,'partial','invalid_action'),partial_repeated=count(a,'partial','repeated_query'),
      total_invalid=sum(r['invalid_action'] for r in decisions if r['arm']==a)) for a in ('A','B')}
    aa={r['probe_id']:r for r in decisions if r['arm']=='A'};bb={r['probe_id']:r for r in decisions if r['arm']=='B'}
    gained=[pid for pid,r in bb.items() if r['category']=='complete' and r['supported_correct'] and not aa[pid]['supported_correct']]
    lost=[pid for pid,r in bb.items() if r['category']=='complete' and not r['supported_correct'] and aa[pid]['supported_correct']]
    binding=[pid for pid in gained if bb[pid]['binding_repair']]
    gates=dict(complete_at_least_22=metrics['B']['complete_correct']>=22,complete_gain_at_least_2=metrics['B']['complete_correct']>=metrics['A']['complete_correct']+2,
      binding_repair=bool(binding),lost_at_most_one=len(lost)<=1,partial_at_least_7=metrics['B']['partial_legal_search']>=7,
      partial_not_lower=metrics['B']['partial_legal_search']>=metrics['A']['partial_legal_search'],
      partial_errors_not_increased=all(metrics['B'][k]<=metrics['A'][k] for k in ('partial_premature','partial_invalid','partial_repeated')))
    write(OUT/'probe-adjudications.json',decisions)
    changes=[]
    rawA={r['probe_id']:r for r in rows(OUT/'probe-A.jsonl')};rawB={r['probe_id']:r for r in rows(OUT/'probe-B.jsonl')}
    for pid in gained+lost:
        a,b=rawA[pid],rawB[pid];i=next((i for i,(x,y) in enumerate(zip(a['action_token_ids'],b['action_token_ids'])) if x!=y),min(len(a['action_token_ids']),len(b['action_token_ids'])))
        scores=b['usage'].get('context_aware_decoding',{}).get('token_scores',[])
        changes.append(dict(probe_id=pid,case_id=b['case_id'],change='gained' if pid in gained else 'lost',A=signature(a),B=signature(b),
          first_different_token_index=i,token_scores=scores[i] if i<len(scores) else None,reason=bb[pid]['reason'],evidence=bb[pid]['evidence'],input_spans=inputs[pid]['document_spans']))
    write(OUT/'changes.json',changes)
    gate=dict(passed=all(gates.values()),conditions=gates,metrics=metrics,gained=gained,lost=lost,binding_repaired=binding)
    write(OUT/'probe-gate.json',gate)
    write(OUT/'adjudication-freeze.json',dict(code_sha256=bc.sha(__file__),manual_sha256=bc.sha(OUT/'probe-new-rulings.json'),decisions_sha256=bc.sha(OUT/'probe-adjudications.json'),original_manifest_sha256=bc.sha(OUT/'manifest.json'),evaluation_used_for_training=False))
    if gate['passed']:
        print(json.dumps(gate,ensure_ascii=False));return
    usage=rows(OUT/'gpu-usage.jsonl');counts={k:sum(r['kind']==k for r in usage) for k in ('primary','reference','search')}
    actual=rows(OUT/'engineer.jsonl')+rows(OUT/'probe-A.jsonl')+rows(OUT/'probe-B.jsonl')
    costs={arm:dict(primary_tokens=sum(len(r['action_token_ids']) for r in actual if r['arm']==arm),
      reference_forward_passes=sum(r['usage'].get('context_aware_decoding',{}).get('reference_forward_passes',0) for r in actual if r['arm']==arm),
      measured_seconds=sum(r['usage']['elapsed_generation_seconds'] for r in actual if r['arm']==arm)) for arm in ('A','B','alpha0')}
    write(OUT/'costs.json',costs)
    write(OUT/'final-status.json',dict(completed=True,accepted=False,status='rejected_at_probe_gate',counts=counts,training_updates=0,new_confirmation=0,
      research_parent='AM42-A',research_parent_sha256=bc.ap.AM_HASH,published_model='Goal Z',reason='Fixed CAD gate failed; no real search or training triggered'))
    report=f'''# Goal BC：固定权重的证据感知解码

**探针未通过，结束本冻结CAD配方；未运行64题真实搜索或训练。**

AM固定权重普通解码A完整证据正确{metrics['A']['complete_correct']}/24，CAD B为{metrics['B']['complete_correct']}/24；部分证据合法补链A/B为{metrics['A']['partial_legal_search']}/{metrics['B']['partial_legal_search']}/8。新增{len(gained)}题，丢失{len(lost)}题，修复实体/角色/关系对象{len(binding)}题。门槛各项见probe-gate.json。

## 新增与丢失
'''
    for change in changes:report+=f"\n- {change['change']} `{change['case_id']}`：A={change['A']}；B={change['B']}。{change['reason']}\n"
    report+=f'''
## 工程、依据与成本
34项CPU测试通过，32个A输入逐token匹配档案；10次GPU工程通过，alpha0两条输入/生成token/停止与普通A一致。参考只清空正文/标题，不修改主输入；两路独立缓存逐token长度核验通过，无正文裁剪。实际引用字符/token跨度见probe-adjudications.json，首处分歧三路分数见changes.json。

固定alpha1整条动作CAD，只在第一/第二次搜索后启用，FP32计算2full-reference；原direct-v090、16384/2048、greedy、first-json、legacy、BM25-v2/top3。没有强制回答、重试、模板修改或参数扫描。

实际{counts['primary']}次主生成+{counts['reference']}条参考序列，合计{counts['primary']+counts['reference']}条计算序列；搜索{counts['search']}，训练0，新确认0。逐token前向与耗时见costs.json，加载/显存见engineer及probe identity文件。参考是保留历史搜索但移除文档内容的输入，不能解释为纯模型先验。

本轮是同证据条件决策诊断，不能将探针变化计为自主搜索成功或权重学习。开发29题支持成功没有重新评价，历史AM21/29不重写；其他35题仍未确认。研究参考保持AM42-A，唯一发布模型Goal Z。

## 决定
不进入真实搜索，不准备本配方蒸馏，不调整alpha/参考范围补跑。负结果限定此冻结解码配置，不代表所有证据感知方法或0.8B能力上限。旧AW/AX/AY结果和材料准入仍保留，不将未训练路线算作训练失败。
'''
    with (OUT/'REPORT.md').open('x',encoding='utf-8') as f:f.write(report)
    print(json.dumps(gate,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['inspect','finalize']);args=parser.parse_args()
    globals()[args.stage]()
