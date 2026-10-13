"""One bounded, source-grounded AY material decision; no inference or retrieval."""
import json,hashlib
from pathlib import Path
import run_v133_goal_ay as ay
from mini_agent_rl.model.grounded_action_audit import visible_documents,map_documents
ap=ay.ap
RULINGS={
 'musique:2hop__131012_90774':('granularity_unconfirmed','1907 is supported. The question does not explicitly require month/day; do not turn a shorter date into a proven binding error.'),
 'musique:2hop__737453_68172':('candidate','Kevin Ellis competed in skeleton; 2006 is his participation year, whereas the visible sport body explicitly gives its first Olympic contest as 1928.'),
 'musique:2hop__143907_30581':('missing_relation','The song-to-Cat Power edge is visible, but the Miami co-residence relation and Iron & Wine are not in the actual returned bodies. Bill Monroe cannot be repaired as a sufficient-chain decision.'),
 'musique:2hop__80857_159536':('scope_and_label_unconfirmed','The visible comparison is explicitly La Liga, while the question leaves the competition scope unspecified and the official target is a biographical description rather than a player name. Do not silently add a scope or train the incompatible official target.'),
 'musique:2hop__197667_26537':('scope_and_answer_type_unconfirmed','Main consumption in central Tibet does not explicitly prove origin. Highest elevation also permits a numerical height; 8848m is visible and must not automatically be treated as a factual error instead of Mount Everest.'),
 'musique:2hop__97293_297038':('scope_unconfirmed','The weapon body names campaigns in both Angola and Mozambique. The question does not uniquely choose one country, and Angola versus Portuguese Angola is not automatically an entity error. Do not manufacture a unique sufficient target.'),
 'musique:2hop__130364_92585':('missing_relation','The Maryland school edge is visible; the returned French Sudan body does not prove who first formed the Maryland colony. Required second edge is absent.'),
 'musique:2hop__128892_54659':('missing_entity_binding','WPGG is bound to Atlantic City; Agra Taj Mahal is a different entity. A 1983 casino construction snippet is present but does not explicitly bind that casino to Atlantic City in its body. The original demonstration also retrieves the Agra Taj Mahal body. Do not use title or gold to fill the location edge.'),
 'musique:2hop__157305_72062':('relation_unconfirmed','Hendra is bound to UN service and WHO public-health coordination is described, but its UN specialized-agency membership is not stated by the body; the list title alone cannot prove that edge. The US Public Health Service is the wrong jurisdiction, but the required sufficient corrective chain remains unconfirmed.'),
 'musique:2hop__487591_7666':('candidate','Reed is explicitly a four-season University of Kansas player; the KU body separately gives one womens outdoor track-and-field title, not thirteen total titles. The college affiliation is sufficient for this sports question without converting an unrelated workplace into education.'),
 'musique:2hop__53127_32505':('missing_relation','The British Empire emergence edge is visible, but no actual body contains the required sea-communications strategy. A wrong answer without the necessary second relation is not this stages sufficient-evidence target.'),
 'musique:2hop__776776_90536':('missing_relation','Joe McLaughlin played for Chelsea. No returned body identifies the Champions League manager; Peter Grant is a music manager in a band-management paragraph, not the required Chelsea role.'),
}
CITES={
 'musique:2hop__737453_68172':[
 ('f1f62ffc010d37f7c301eddf','Kevin Ellis (born June 29, 1973), is an American skeleton racer who competed from 1999 to 2006.',['Kevin Ellis','sport','skeleton']),
 ('1be3e54ebad6d98784af7aaa','It was first contested at the 1928 Winter Olympics in St. Moritz and again in 1948 Winter Olympics',['skeleton','first Olympic contest','1928'])],
 'musique:2hop__487591_7666':[
 ('6d1499ccd8d98da4a26485c4','Tyrel Reed (born April 8, 1989) is an American basketball player who played for University of Kansas.',['Tyrel Reed','college affiliation','University of Kansas']),
 ('a65f490d952a9ce30a513879',"one in women's outdoor track and field",['University of Kansas','national womens outdoor track-and-field championships','one'])]
}
def citations(tok,ids,specs):
    messages=ap.restore_tool_messages(ap.parse_archived_messages(tok,ids));mapped,offsets=map_documents(tok,ids,messages);out=[]
    for did,quote,triple in specs:
        doc=next((d for d in mapped if d['document_id']==did and quote in d['text']),None)
        if doc is None:return None
        escaped=json.dumps(quote,ensure_ascii=False)[1:-1];start=doc['serialized_char_span'][0]+doc['serialized_body'].index(escaped);end=start+len(escaped)
        positions=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
        out.append(dict(document_id=did,quote=quote,triple=triple,body_sha256=doc['body_sha256'],body_char_span=[doc['text'].index(quote),doc['text'].index(quote)+len(quote)],input_token_span=[min(positions),max(positions)+1],message_index=doc['message_index']))
    return out
def main():
    ay.aw.frozen();completed=ap.read(ay.OUT/'collection-complete.json');data=ap.rows(ay.OUT/'collected-trajectories.jsonl')
    if len(data)!=32 or completed['cases']!=32:raise RuntimeError('AY incomplete')
    packets={r['case_id']:r for r in ap.rows(ay.OUT/'review-packets.jsonl')}
    if set(packets)!=set(RULINGS):raise RuntimeError('Missing keyed review')
    original={r['id']:r for r in ap.rows(ap.Z/'training-inputs.jsonl')};tok=ap.am.tokenizer_from_base();candidates=[]
    for r in data:
        cid=r['case_id'];last=r['trace'][-1]
        if any(t.get('repeated') for t in r['trace']):raise RuntimeError('Unexpected unreviewed repeated state')
        verdict,reason=RULINGS.get(cid,('not_selected','Normalized official answer agrees. This is not proof of evidence sufficiency or a new semantic score; no new corrective target selected.'))
        messages=ap.restore_tool_messages(ap.parse_archived_messages(tok,last['prompt_token_ids']))
        mapped,_=map_documents(tok,last['prompt_token_ids'],messages)
        record=dict(case_id=cid,state_id=f'AY:{cid}:call-{last["step"]}',question=r['question'],original_action={k:last.get(k) for k in ('action','query','answer')},status=verdict,reason=reason,actual_input_token_count=len(last['prompt_token_ids']),actual_documents=mapped,teacher_generated=False,training_admitted=False)
        if verdict=='candidate':
            current=citations(tok,last['prompt_token_ids'],CITES[cid]);demo=citations(tok,original[cid]['actions'][-1]['prompt_token_ids'],CITES[cid])
            if current is None or demo is None:raise RuntimeError('Candidate citations not in both actual and original demonstration prefixes')
            record.update(current_citations=current,demonstration_citations=demo,eligible_before_teacher=True,proposed_action=dict(action='answer',answer='1928' if cid=='musique:2hop__737453_68172' else 'one'))
            candidates.append(record)
        ap.append(ay.OUT/'state-adjudications.jsonl',record)
    if len(candidates)>=8:raise RuntimeError('Teacher stage needed: cannot close sufficient material')
    ay.write(ay.OUT/'material-gate.json',dict(passed=False,distinct_candidates_before_teacher=len(candidates),teacher_verified_corrections=0,required=8,teacher_calls=0,teacher_download=False,reason='One bounded review of fixed32 found fewer than8 eligible sufficient-chain error states. No corpus expansion or replacement.',candidate_case_ids=[r['case_id'] for r in candidates]))
    ay.write(ay.OUT/'final-status.json',dict(completed=True,accepted=False,status='closed_at_material_gate',generation=completed['generation'],search=completed['search'],updates=0,teacher_generation=0,corrections_before_teacher=len(candidates),research_parent='AM42-A',published_model='Goal Z'))
    report=f'''# Goal AY：真实状态采集完成，未达到训练准入

修复了计数账本初始化，保留v1零生成失败现场。v2的CPU预检验证初始化、独立计数和额度拒绝后，完成固定前32个训练题采集：{completed['generation']}次GPU生成、{completed['search']}次真实搜索、0次训练更新。模型仍为AM seed42-A，使用训练冻结语料和16384/2048、原提示及动作协议。没有评价题进入训练。

一次实际正文审核得到{len(candidates)}个教师前合格候选：Kevin Ellis（入奥年份1928，而非参赛年份2006）和Tyrel Reed（女子室外田径1项，而非全校13项）。两题必要引用均已在当前错误前缀及原示范前缀定位，保存正文哈希、字符和实际token位置。候选不是教师已核验纠正，也不是训练收益。

其他不匹配分别是缺少必要关系、角色绑定未被正文证明、题义范围或答案粒度未确认。年份1907不因省略月日自动判错；8848m不因不同于山名自动判事实错误；联合开发、主要消费地、文档标题均不替代来源或关系证明。32题均无重复查询；没有需要追加教师采样的充分后重复状态。

至少8个不同训练题的门槛未满足，因此未下载4B教师、未运行教师生成、未训练A/B。不扩32题、不修改规则补满。20个规范化答案一致的案例只记未选取纠正，不冒充完整语义审核或支持成功率。12个不匹配由case_id关联一次裁定，原训练标签和旧分数不改。

AZ依赖至少8个可靠纠正状态，未触发。AW和AX也未接受版本，因此BA与BB均不触发；不宣布回归通过或产品升级。该结果是固定材料范围未获得足够真实充分证据纠错状态，不是已训练真实状态蒸馏后得到负收益。
'''
    with (ay.OUT/'REPORT.md').open('x',encoding='utf8') as f:f.write(report)
    print(json.dumps(dict(eligible_candidates=len(candidates),teacher=0,updates=0,closed=True)),flush=True)
if __name__=='__main__':main()
