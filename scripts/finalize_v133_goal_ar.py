"""One-pass main-agent AR rulings, grounded in preserved actual spans.

These explicit case-keyed decisions were written after reviewing the frozen
outputs and source quotations. No case ordinal or gold-match inference.
"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_v133_goal_ar as ar

C_CHANGES={
 "complete:musique:2hop__864101_135373":(True,"Bering is the same named explorer as Vitus Bering in the actual sea body; surname is an allowed referent abbreviation."),
 "complete:musique:2hop__151750_141308":(True,"Apple Corps Ltd. differs only by terminal punctuation; original company and release-label links remain present."),
 "complete:musique:2hop__703493_162341":(False,"Best Supporting Actor drops Academy Award nomination status. Body says nominated, not received this award."),
 "complete:2wiki:f299dd600bdd11eba7f7acde48001122":(False,"Complete father and 1575 death links remain visible; additional search fails to answer this sufficient state."),
}
B_CHANGES={
 "complete:musique:2hop__864101_135373":(False,"Selected sea naming sentence explicitly gives Vitus Bering but final Bering Sea substitutes the relation subject for its object."),
 "complete:musique:2hop__506035_325512":(False,"Smith birthplace and Poplarville/Pearl River County are visible and selected; B unnecessarily searches after sufficient evidence."),
 "complete:musique:2hop__418360_67370":(True,"2018 is a directly supported coarser year for the table's January 19, 2018 start. The question asks When; preserve precision-decline flag, not exact-date equivalence."),
 "complete:musique:2hop__703493_162341":(False,"Best Supporting Actor omits Academy Award nomination status; selected sentence says nominated, not award receipt."),
 "complete:musique:2hop__158275_35739":(False,"Firefox September 2013 is visible but B searches for Chrome's Windows 8 version, a different browser/role."),
 "complete:musique:2hop__69765_43945":(True,"Common Sense body explicitly says Paine encouraged colonial common people to fight for egalitarian government; this supported effect answers the original question."),
 "complete:2wiki:764039340bde11eba7f7acde48001122":(False,"Father James Hamilton died in Hyde Park/Westminster. Charing Cross is Abercrombie's own death place; B searches the wrong person/place."),
 "complete:2wiki:f299dd600bdd11eba7f7acde48001122":(False,"29 May 1594 is the 5th duke's own death date; requested father is 4th duke, whose body gives death in1575."),
 "complete:2wiki:49d965aa0bde11eba7f7acde48001122":(True,"Las Piñas Medical Center identifies the literal hospital in the death sentence; omitting Manila does not substitute a different location."),
 "complete:musique:2hop__530579_13529":(False,"Sinagra's father is Maradona and Barcelona body already states June1982 signing; B continues searching rather than answering."),
}
# Selected evidence is scored separately from the action. True means the
# selected subset covers the required binding links, allowing document-local
# pronoun resolution; irrelevant extra spans are preserved in the ledger.
B_SELECTED={
 "musique:2hop__864101_135373":(True,"Selected island Bering Sea-unit and explicit named-for-Vitus sentence cover the required chain; extra Sea of Sardinia is irrelevant."),
 "musique:2hop__478978_124498":(True,"Selected album singer and Matthieu son-of-Louis sentence cover both edges; sister/album-style extras are unnecessary."),
 "musique:2hop__829358_754719":(False,"Minsk administrative entity does not bind Pedro Leopoldo to Minas Gerais; selected Darley birthplace plus Minsk is a wrong geographic chain."),
 "musique:2hop__789565_416401":(True,"Selected Huang mother-of-Wang-Yanjun and Wang-Jipeng father-Wang-Yanjun clauses support the child chain; final Lady Li is still the cousin/wife."),
 "musique:2hop__506035_325512":(True,"Smith born-Poplarville and Poplarville-in-Pearl-River clauses cover the chain; employment and stadium extras are irrelevant."),
 "musique:2hop__151750_141308":(False,"Selected label/company clause omits the album release-label edge; BNA Nashville cannot supply it."),
 "musique:2hop__418360_67370":(True,"Selected book author and visible season-start table cover the asked chain; the final answer is coarser than the complete date."),
 "musique:2hop__703493_162341":(False,"Selected nomination clause does not include a director-Hoffman binding clause; event Murrow fragment and film producers do not replace that edge."),
 "musique:2hop__158275_35739":(False,"Chrome October2013, canceled Firefox, and Windows7 SP1 clauses omit the required Firefox Aurora September2013 availability relation."),
 "musique:2hop__435087_55984":(True,"Selected Carlile album and song-written-by-Phil clauses cover the chain; EP/TV extras do not add proof."),
 "musique:2hop__109242_113442":(True,"Selected painting-by-Millais and Millais death-date cover the two necessary relations; rower extra is irrelevant."),
 "musique:2hop__69765_43945":(True,"Selected Paine/Common Sense effect clauses support the relationship and final paraphrase in visible document context."),
 "2wiki:982f8e440bdb11eba7f7acde48001122":(True,"Magazine publisher Sigma Xi and society founded1886 are both selected."),
 "2wiki:67ecf3100bdc11eba7f7acde48001122":(True,"Selected Yurek birthplace and directed-Anna clauses cover both relations; unrelated director and duplicate retained."),
 "2wiki:3a3d60020bdd11eba7f7acde48001122":(False,"Selected Anais origin omits song performer binding; Jewish genetic origins are unrelated."),
 "2wiki:ef9d40a40bdb11eba7f7acde48001122":(True,"Selected Armless director and Habib American identity cover both edges; other directors remain irrelevant."),
 "2wiki:5dac1e8e0bdd11eba7f7acde48001122":(False,"Selected director bio/birthplace does not contain the requested film-to-director binding; film link is only elsewhere in the input."),
 "2wiki:1dd33f080bde11eba7f7acde48001122":(False,"Selected father Arnstadt and son Eisenach birthplace fragments omit the named son-to-father link; final chooses son's location."),
 "2wiki:764039340bde11eba7f7acde48001122":(True,"Selected Abercrombie son-of-James and James died-Hyde-Park clauses cover the chain; own Charing-Cross death is a distractor."),
 "2wiki:7bda387a0bdb11eba7f7acde48001122":(False,"Gerald birthday is selected but Xavier biography fragment omits father link; Gerald Ford family is unrelated."),
 "2wiki:f299dd600bdd11eba7f7acde48001122":(False,"4th duke lifespan is selected but 5th duke son-of-4th clause is not; 6th duke son-of-5th is the wrong direction for the question."),
 "2wiki:49d965aa0bde11eba7f7acde48001122":(False,"Singer identity and hospital death are selected; song-to-performer binding is not among selected spans."),
 "2wiki:ee3233920bda11eba7f7acde48001122":(False,"Selected FrancescoII/WilliamII/illegitimate-son fragments and dowry omit requested father FrancescoI/death1397 chain."),
 "musique:2hop__530579_13529":(False,"Selected Sinagra son-of-Maradona omits Barcelona's explicit June1982 signing sentence."),
}
B_PARTIAL={
 "musique:2hop__864101_135373":(False,False,"Search omits the visible Bering Sea entity; generic body-of-water named-after is not the bridge-targeted missing relation."),
 "musique:2hop__478978_124498":(True,True,"Query uses visible Matthieu Chedid and missing father relation; selected album clause supplies bridge, unrelated extra songs noted."),
 "musique:2hop__862994_69048":(False,False,"Virginia Dare is a retrieved distractor, not Gloria's birthplace Delta State; chief-judge query is entity-mismatched."),
 "musique:2hop__829358_754719":(True,True,"Visible Darley birthplace Pedro Leopoldo is used for the missing administrative entity."),
 "musique:2hop__789565_416401":(True,True,"Visible Huang child Wang Yanjun is used for missing child relation; unrelated Wang Jian selections noted."),
 "musique:2hop__506035_325512":(True,False,"Poplarville Mississippi county query is legal from actual birthplace sentence; selected employment sentence alone does not support birthplace."),
 "musique:2hop__337223_56873":(False,False,"Generic Mark Crossley team FA cup date does not target a visible named club; selected FA Cup qualification rules prove neither team nor victory date."),
 "musique:2hop__796221_18803":(False,False,"Transavia Denmark is an unrelated airline; query abandons MV Rapana/Shell owner relation."),
}


def finalize():
    ar.frozen()
    inputs={(r["arm"],r["probe_id"]):r for r in ar.rows(ar.OUT/"probe-inputs.jsonl")}
    baseline={r["probe_id"]:r for r in ar.rows(ar.PRIOR/"probe-adjudications-revision.jsonl") if r["model"]=="AM" and r["arm"]=="A"}
    reviews=[]
    for arm in ("A","C","B"):
        for r in ar.rows(ar.OUT/f"probe-{arm}.jsonl"):
            old=baseline[r["probe_id"]];item=inputs[(arm,r["probe_id"])]
            out=dict(old,arm=arm,action=r["action"],answer=r["answer"],query=r["query"],raw=r["raw"],
                current_input_sha256=hashlib.sha256(json.dumps(r["prompt_token_ids"]).encode()).hexdigest(),
                selected_evidence_correct=False,binding_repair=False,source_review_hash=ar.sha(ar.PRIOR/"probe-adjudications-revision.jsonl"))
            if r["category"]=="complete":
                changes=C_CHANGES if arm=="C" else B_CHANGES if arm=="B" else {}
                if r["probe_id"] in changes:
                    out["supported_correct"],out["adjudication_reason"]=changes[r["probe_id"]]
                else:
                    if (r["action"],r["answer"],r["query"])!=(old["action"],old["answer"],old["query"]):
                        raise RuntimeError("unreviewed changed output: "+r["probe_id"])
                    out["adjudication_reason"]="Same action/answer and complete original-body hashes/order preserved; reuse existing grounded ruling. "+old["adjudication_reason"]
                out.update(legal_bridge_search=False,premature_answer=False,invalid_action=bool(r["parse_error"]),repeated_query=False)
                if arm=="B":
                    out["selected_evidence_correct"],out["selected_evidence_reason"]=B_SELECTED[r["case_id"]]
                    out["precision_decline"]=r["case_id"]=="musique:2hop__418360_67370"
            else:
                out["supported_correct"]=False
                if arm=="B":
                    out["legal_bridge_search"],out["selected_evidence_correct"],out["adjudication_reason"]=B_PARTIAL[r["case_id"]]
                elif arm=="C" and r["case_id"]=="musique:2hop__337223_56873":
                    out["legal_bridge_search"]=False;out["adjudication_reason"]="Reordered Mark Crossley/team/2015-16 query retains unsupported time scope and omits named bridge club; no improvement over A."
                elif arm=="C" and r["case_id"]=="musique:2hop__862994_69048":
                    out["legal_bridge_search"]=False;out["adjudication_reason"]="Gloria Kemasuode birthplace chief judge does not name the already visible Delta State bridge; it leaves the chief-judge subject underspecified instead of targeting the missing state's relation."
                else:
                    if r["query"]!=old["query"]:raise RuntimeError("unreviewed partial query")
                    out["adjudication_reason"]="Same bridge-targeted query and actual original bodies; reuse prior query ruling. "+old["adjudication_reason"]
                out.update(premature_answer=r["action"]=="answer",invalid_action=bool(r["parse_error"]),repeated_query=False)
                if r["query"]:
                    seen=[]
                    for m in item["messages"]:
                        if m["role"]=="tool":seen.extend(json.loads(m["content"]).get("searched_queries",[]))
                    out["repeated_query"]=' '.join(r["query"].casefold().split()) in {' '.join(x.casefold().split()) for x in seen}
            if arm=="B":out["selection_audit"]=r["selection_audit"]
            reviews.append(out)
    ar.write(ar.OUT/"probe-adjudications.jsonl",reviews,jsonl=True)
    ar.gate()
    gate=ar.read(ar.OUT/"probe-gate.json")
    if gate["real_search_allowed"]:raise RuntimeError("AR real-search phase remains required")
    text='''# Goal AR：受约束证据选择结果

AR未通过探针门槛，结束该机制，不运行64题真实搜索。A原动作20/24；C索引正文+约束普通动作19/24；B索引正文+约束证据选择14/24。部分证据合法补链搜索分别7/8、6/8、4/8。没有新增正确完整证据案例。

## 得到的机制结论

B的32次输出全部遵守证据地址协议，地址全部来自实际输入；约束解决了AQ的0/64协议遵循问题。但格式执行没有转化为正确关系绑定。St. Matthew Island选中了命名对象Vitus Bering的正文，最终仍答Bering Sea；Huang题选择了母子及父子正文，最终仍答Lady Li；Abercrombie选择了父亲Hyde Park死亡原句，却查询本人死亡地Charing Cross。这些是同证据下的绑定/决策失败，不是搜索没有执行或正文丢失。

C仅失去公爵父亲死亡年题：完整链仍可见却继续搜索。B失去命名对象、Smith出生县、Firefox日期、Abercrombie父亲死亡地、公爵父亲死亡年、Maradona签约日期六个原成功状态。部分B查询另错绑Virginia Dare和Transavia Denmark；Bering查询没有带上可见桥接实体。Bill Maher题仅答2018，按原When允许更粗但正文支持的年份，单列精度下降，不称完整日期等价。

选择正文地址只保证原句可见，不能保证选择覆盖完整链；B还出现选择正确关系却最终动作错误、漏掉第一跳、选择干扰人物、重复地址。逐题理由与原句/token位置见probe-adjudications.jsonl及probe-inputs.jsonl。既有冻结引用中的错误定位不直接用作关系判定，沿用AP/AQ明确修订依据。

## 工程与额度

实际远程AM/GoalZ权重哈希匹配，RTX4090D/BF16/greedy/16384输入/2048输出，4调用/3搜索上限保持。96份CPU生产输入中A32份逐token匹配档案，C/B原始工具正文/顺序/重复项一致，64份语法编译成功，最大输入3658token，无裁剪。40项远程CPU测试通过。首轮测试因scripts导入路径缺失7项失败，修正测试运行路径后通过；两个日志都保留。

实际新增72次GPU生成：工程8、C32、B32；A复用32次旧输出。搜索0，训练更新0，新确认0。生成停止、全token与实际输入引用位置保留。现有CuBLAS非位级确定性/参考kernel警告保持原样，未改参重跑。依赖固定xgrammar0.2.8、apache-tvm-ffi0.1.14.post1，wheel及哈希保存，Torch/Transformers未升级。

## 后续

舍弃C/B研究配置，研究父版本仍AM42-A；产品仍GoalZ。按照顺序检查AS材料资格，不补数据，然后进行AT动作可达性与可靠奖励检查；只有其资格通过才匹配训练。当前结果不能声称0.8B达到能力上限，也不提供可蒸馏的系统正收益。
'''
    with (ar.OUT/"REPORT.md").open("x",encoding="utf-8") as f:f.write(text)
    ar.write(ar.OUT/"final-status.json",dict(status="completed_rejected_at_probe_gate",new_generations=72,reused_A=32,
        searches=0,training_updates=0,new_confirmation_calls=0,research_parent="AM42A",published=False,
        finalizer_sha256=ar.sha(Path(__file__)),gate_sha256=ar.sha(ar.OUT/"probe-gate.json")))

if __name__=="__main__":finalize()
