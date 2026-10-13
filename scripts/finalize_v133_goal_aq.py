"""One-pass source-grounded AQ probe closure. CPU only; no new model/search calls."""
import hashlib
import json
import statistics
from pathlib import Path
import run_v133_goal_aq as aq
from mini_agent_rl.model.grounded_action_audit import audit_grounded_action, map_documents

OUT=aq.OUT
# These are explicit primary-agent adjudications of changed outputs, keyed by state ID.
# They are not model supervision, inference inputs, or decisions indexed by list order.
CHANGED={
 "complete:musique:2hop__864101_135373":dict(supported_correct=False,error_type="relation_object_category",
   reason="The visible island body places St. Matthew in the Bering Sea unit, and Bering Sea is named for Vitus Bering. Arctic Ocean occurs only as a destination in Bering's voyage, not the named-after object."),
 "complete:musique:2hop__703493_162341":dict(supported_correct=False,error_type="recognition_qualification",
   reason="Director Hoffman is identified, but the body explicitly supports an Academy Award nomination for Best Supporting Actor, not receipt of an unqualified Academy Award. The awards-list body does not establish a win."),
 "complete:musique:2hop__158275_35739":dict(supported_correct=False,error_type="search_after_complete",
   reason="Windows 8 body explicitly states Firefox's Aurora version was first made available in September 2013. A had answered that date; B searches again in a complete-evidence state. No search is executed."),
 "complete:musique:2hop__69765_43945":dict(supported_correct=True,error_type="supported_paraphrase",
   reason="The Common Sense body explicitly says Paine marshaled moral and political arguments to encourage common people in the Colonies to fight for egalitarian government. This is a supported explanation of the work's importance, alongside the existing crystallized-independence explanation; no new binding gain."),
 "complete:2wiki:5dac1e8e0bdd11eba7f7acde48001122":dict(supported_correct=True,error_type="equivalent_location",
   reason="The same director birthplace supports New York. The abbreviated New York answer retains the city referent of A's New York, New York; no binding gain."),
 "complete:2wiki:49d965aa0bde11eba7f7acde48001122":dict(supported_correct=True,error_type="coarser_supported_location",
   reason="Butse Kik is by Yoyoy Villame; his body says he died at Las Pinas Medical Center in Manila. Manila is a directly supported city answer to Where, though it is coarser than the hospital answer and is not an equivalent hospital name. Report granularity separately; no binding gain."),
 "complete:musique:2hop__530579_13529":dict(supported_correct=True,error_type="coarser_supported_date",
   reason="Sinagra is Diego Maradona's son; Barcelona body says In June 1982, Diego Maradona was signed. 1982 preserves the supported signing year for When, though it omits the month. Report precision loss separately; no binding gain."),
 "partial:musique:2hop__862994_69048:step1":dict(legal_bridge_search=False,error_type="bridge_not_used",
   reason="Gloria's body explicitly provides Delta State. B repeats Gloria birthplace chief judge without naming that available bridge; it is a valid search action but not the frozen criterion's bridge-targeted missing-relation search."),
 "partial:musique:2hop__506035_325512:step1":dict(legal_bridge_search=False,error_type="birthplace_employment_role_mismatch",
   reason="Smith was born in Poplarville. Harrison County occurs as a later police/sheriff jurisdiction, not as the county containing Poplarville. B queries the employment county instead of the visible birthplace bridge."),
 "partial:musique:2hop__337223_56873:step1":dict(legal_bridge_search=True,error_type="query_scope_repaired",
   reason="Crossley's player body names Nottingham Forest among his clubs, so it is a legitimate visible club candidate for the FA Cup missing relation. B removes A's unrelated 2015-16 Europa League scope. This is query legality only: no result or success claim is made."),
 "partial:musique:2hop__796221_18803:step1":dict(legal_bridge_search=False,error_type="unrelated_entity_binding",
   reason="The Rapana passage names the Anglo Saxon Royal Dutch/Shell tanker group. Transavia Denmark belongs to an unrelated airline cessation passage and is not Rapana's owner; mere presence of the name does not license B's query binding."),
}


def finalize():
    aq.frozen()
    if (OUT/"probe-adjudications.jsonl").exists():raise FileExistsError("AQ one-pass review already exists")
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(aq.ap.BASE),local_files_only=True)
    tokenizer=getattr(processor,"tokenizer",processor)
    prior={(r["model"],r["arm"],r["probe_id"]):r for r in aq.rows(aq.PRIOR/"probe-adjudications-revision.jsonl")}
    inputs={(r["model"],r["arm"],r["probe_id"]):r for r in aq.rows(OUT/"probe-inputs.jsonl")}
    states={r["probe_id"]:r for r in aq.rows(OUT/"frozen-probes.jsonl")}
    edge_source={r["probe_id"]:r["edges"] for r in aq.rows(aq.ROOT/"runs/v133-goal-ao-relation-contrast-sft-v2/probe-evidence-seed42.jsonl") if r["model"]=="42-A"}
    reviews=[];evidence=[];comparisons=[];mechanical={}
    for model in ("AM","Z"):
        baseline={r["probe_id"]:r for r in aq.rows(OUT/f"probe-{model}-A.jsonl")}
        for arm in ("A","B"):
            outputs=aq.rows(OUT/f"probe-{model}-{arm}.jsonl")
            if len(outputs)!=32:raise RuntimeError("AQ probe inventory incomplete")
            mechanical[f"{model}-{arm}"]=dict(actions_valid=sum(not r["parse_error"] for r in outputs),
                answers=sum(r["action"]=="answer" for r in outputs),searches=sum(r["action"]=="search" for r in outputs),
                schema_valid=sum(r.get("grounded_audit",{}).get("schema_valid") is True for r in outputs),
                audit_fields_generated=sum("links" in json.loads(r["raw"]) for r in outputs),
                mean_input_tokens=statistics.mean(len(r["prompt_token_ids"]) for r in outputs),
                mean_output_tokens=statistics.mean(len(r["action_token_ids"]) for r in outputs),
                max_output_tokens=max(len(r["action_token_ids"]) for r in outputs),
                mean_generation_seconds=statistics.mean(r["elapsed_seconds"] for r in outputs),
                warm_generation_mean_seconds=statistics.mean(r["elapsed_seconds"] for r in outputs[1:]),
                first_call_seconds_includes_lazy_load=outputs[0]["elapsed_seconds"])
            for r in outputs:
                key=(model,arm,r["probe_id"]);item=inputs[key]
                if r["prompt_token_ids"]!=item["prompt_token_ids"]:raise RuntimeError("actual input mismatch in closure")
                docs,offsets=map_documents(tokenizer,r["prompt_token_ids"],item["messages"])
                bitem=inputs[(model,"A",r["probe_id"])]
                if [(d["document_id"],d["body_sha256"]) for d in docs]!=[(d["document_id"],d["body_sha256"]) for d in bitem["documents"]]:
                    raise RuntimeError("semantic reuse evidence body differs")
                reference=[]
                if r["category"]=="complete":
                    for edge in edge_source[r["probe_id"]]:
                        citation=edge["citation"]
                        value=dict(links=[dict(subject=str(edge["subject"]),relation=str(edge["relation"]),object=str(edge["object"]),
                            document_id=citation["document_id"],quote=citation["quote"])],missing_relation="",action="answer",answer="offline-reference-only")
                        audit=audit_grounded_action(json.dumps(value),docs,offsets)
                        if not audit["exact_quotes_valid"]:raise RuntimeError(f"existing edge quote not actually visible: {r['probe_id']}")
                        reference.append(dict(edge_id=edge["edge_id"],subject=edge["subject"],relation=edge["relation"],object=edge["object"],
                            citation=audit["links"][0],origin="existing AO corrected evidence; offline only, not generated by model"))
                else:
                    citation=states[r["probe_id"]]["first_edge_citation"]
                    value=dict(links=[dict(subject=states[r["probe_id"]]["question"],relation="existing visible first edge",
                        object=states[r["probe_id"]]["bridge_entity"],document_id=citation["document_id"],quote=citation["quote"])],
                        missing_relation="offline reference only",action="search",query="offline-reference-only")
                    audit=audit_grounded_action(json.dumps(value),docs,offsets)
                    if not audit["exact_quotes_valid"]:raise RuntimeError("partial first-edge quote is not visible")
                    reference.append(dict(origin="AN first-edge citation only; current spans revalidated; missing-edge target excluded",
                        citation=audit["links"][0]))
                evidence.append(dict(model=model,arm=arm,probe_id=r["probe_id"],case_id=r["case_id"],
                    actual_input_sha256=hashlib.sha256(json.dumps(r["prompt_token_ids"]).encode()).hexdigest(),reference_edges=reference,
                    documents=[{k:v for k,v in d.items() if k not in ("text","serialized_body")} for d in docs],training_use_forbidden=True))
                review=dict(model=model,arm=arm,probe_id=r["probe_id"],case_id=r["case_id"],category=r["category"],
                    action=r["action"],answer=r["answer"],query=r["query"],supported_correct=None,grounded_links_correct=None,
                    binding_repair=False,legal_bridge_search=None,premature_answer=False,invalid_action=bool(r["parse_error"]),
                    repeated_query=False,adjudication_reason="Goal Z descriptive only; semantic success not scored.",training_use_forbidden=True)
                if model=="AM":
                    old=prior[(model,"A",r["probe_id"])]
                    same=(r["action"],r["answer"],r["query"])==(baseline[r["probe_id"]]["action"],baseline[r["probe_id"]]["answer"],baseline[r["probe_id"]]["query"])
                    if arm=="A" or same:
                        for field in ("supported_correct","legal_bridge_search","premature_answer","invalid_action","repeated_query"):
                            review[field]=old.get(field)
                        review["adjudication_reason"]="Exact action/output reuse after actual body/hash/order and citation-location revalidation. "+old["adjudication_reason"]
                        review["error_type"]="unchanged_decision"
                    else:
                        decision=CHANGED[r["probe_id"]]
                        review.update({k:v for k,v in decision.items() if k!="reason"})
                        review["adjudication_reason"]=decision["reason"]
                    if arm=="B":
                        audit=r["grounded_audit"]
                        if audit["schema_valid"] or audit["links"]:
                            raise RuntimeError("closure's no-generated-audit finding must be re-adjudicated if any audit exists")
                        review["grounded_links_correct"]=False
                        review["audit_failure"]="links and missing_relation not generated"
                    else:
                        review["grounded_links_correct"]=False
                        review["audit_failure"]="not_requested_in_A"
                    if r["category"]=="partial":
                        previous={" ".join(q.casefold().split()) for q in states[r["probe_id"]]["previous_queries"]}
                        review["repeated_query"]=" ".join((r["query"] or "").casefold().split()) in previous
                elif arm=="B":
                    review["grounded_links_correct"]=False
                    review["audit_failure"]="links and missing_relation not generated"
                reviews.append(review)
                if arm=="B":
                    comparisons.append(dict(model=model,probe_id=r["probe_id"],question=states[r["probe_id"]]["question"],
                        A_raw=baseline[r["probe_id"]]["raw"],B_raw=r["raw"],review=review,evidence_reference=[model,arm,r["probe_id"]]))
    aq.write(OUT/"probe-adjudications.jsonl",reviews,jsonl=True)
    aq.write(OUT/"probe-evidence.jsonl",evidence,jsonl=True)
    aq.write(OUT/"paired-decisions.jsonl",comparisons,jsonl=True)
    aq.write(OUT/"mechanical-summary.json",mechanical)
    aq.gate()
    gate=aq.read(OUT/"probe-gate.json")
    if gate["passed"]:raise RuntimeError("unexpected pass: report requires explicit real-search follow-up")
    if aq.count("generations")!=64 or aq.count("searches")!=0:raise RuntimeError("AQ quota accounting mismatch")
    bad={arm:sum(r["model"]=="AM" and r["arm"]==arm and r["category"]=="partial" and r["legal_bridge_search"] is True for r in reviews) for arm in ("A","B")}
    report=f"""# Goal AQ：单次结构化关系核对结果

## 结论

冻结方案未通过探针门槛，按计划停止，不进行64题真实搜索、不训练、不运行新确认、不发布。AM完整证据动作A {gate['A_complete']}/24、B {gate['B_complete']}/24；部分证据合法桥接搜索A {bad['A']}/8、B {bad['B']}/8。没有新增支持成功，完整证据丢失{len(gate['losses'])}个原成功状态。

更关键的结果是**64次B输出全部仍是普通search/answer JSON，0/64生成links或missing_relation，0/64通过结构化核对字段检查**。AM与Goal Z均为0/32。原动作解析仍全部成功，因此这不是多JSON、截断、工具派发或引用校验器拦截造成的失败；模型没有执行新增输出协议，不能据此判断显式关系链核对本身没有价值，也不能宣称0.8B能力达到上限。

## 赢回、丢失与查询变化

- 没有赢回的完整证据成功。St. Matthew Island案例由正确Vitus Bering变成Arctic Ocean：命名关系对象错绑成航行目的地。Windows 8/Firefox案例由September 2013改为继续搜索，日期正文完整可见。
- 原有Darley地域、Huang亲属、Bach父亲出生地和Hoffman奖项限定失败没有修复。Academy Award未保留正文明确的nomination关系，仍不计充分回答。
- Common Sense改为正文支持的鼓励殖民地民众争取平等政府；不是新绑定收益。New York是同指称地点缩写。Manila与1982分别是正文明确支持的更粗地点/年份：本轮问题只问Where/When，计支持但单列精度下降，不称等价医院名称或完整日期。若要求原医院/月份粒度，这两题也会丢失；宽松计分仍无法过门槛。
- 部分证据Crossley查询去掉无关2015–16范围，针对可见Nottingham Forest候选补问FA Cup关系，合法性改善；零搜索，不能记取得证据或真实成功。Gloria没有使用已可见Delta State桥接；Smith把任职Harrison County错当出生地桥接；Rapana错绑无关航空公司Transavia Denmark。部分合法补链由7降至5。

## 工程、身份与计数

远程RTX4090D、单卡BF16、不量化。Goal Z/AM实际adapter哈希分别匹配AA发布身份及冻结AM候选身份。16384/2048、greedy、thinking=false、first-json-v1、legacy预算与legacy-json不变；只改系统提示。默认入口未改变。

生产_inputs CPU预检128条输入；A64条逐token重现，B仅系统消息变化。正文ID/哈希/排序/重复项全部保持；652个正文出现保存位置映射；最大输入2371 token，无裁剪。8项工程测试通过。实际生成均在预算内，原始生成token、解析动作、停止和延迟保留。

复用AP有效A64条（AM/Z各32），新增B64次GPU生成，0搜索，0训练，0新确认。A复用须区分历史执行与本轮执行。运行继续出现已有CuBLAS/算子非位级确定性及参考kernel警告，未调环境或重跑；greedy不等于逐位可复现。首个状态延迟包含惰性模型加载，暖调用延迟在mechanical-summary单列。

Goal Z仅描述输出和协议遵循，不把未裁定状态当0正确，也不根据它临时选模型。主要成绩为条件探针，不是64题自主搜索EM/F1或产品升级。

## 后续决定与复现

结束本次单调用结构化提示路线，不修改格式补跑，不把评价状态导入训练。今后若验证显式核对，应先解决新协议能否被模型执行，再比较正确核对是否改善动作；任何协议示范/教师或训练须使用隔离训练来源，单列成本与匹配对照。本次没有可进入蒸馏的系统收益，AM历史真实搜索21/29仍为实验候选结果，Goal Z仍为唯一发布模型。

冻结计划/manifest/cpu-preflight、probe输入和四份输出、引用证据、128条裁定、64对动作以及probe-gate均在本目录。prepare/probe/gate入口见PLAN；gate失败可独立完成本报告，不依赖不存在的评价文件。关闭脚本仅CPU、禁止追加生成；明确裁定以case_id/probe_id关联，引用查验不能替代语义判断。
"""
    with (OUT/"REPORT.md").open("x",encoding="utf-8",newline="\n") as handle:
        handle.write(report)
    aq.write(OUT/"final-status.json",dict(status="completed_stopped_at_probe_gate",new_generations=64,reused_A=64,
        searches=0,training_updates=0,new_confirmation_calls=0,published=False,manifest_sha256=aq.sha(OUT/"manifest.json"),
        finalizer_sha256=aq.sha(Path(__file__)),gate_sha256=aq.sha(OUT/"probe-gate.json")))
    print(json.dumps(dict(completed=True,gate=gate)))


if __name__=="__main__":finalize()
