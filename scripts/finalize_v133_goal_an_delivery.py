"""Goal AN CPU final delivery checks and human report; no model calls."""
import json,hashlib,math
from collections import Counter
from pathlib import Path
import run_v133_goal_an as an
P=an.OUT
r=an.read; rr=an.rows
cfg=an.frozen()
g=r(P/'seed42-gate.json');assert not g['route_passed']
u=Counter(x['kind'] for x in rr(P/'gpu-usage.jsonl'))
assert u=={'generation':250,'search':130,'update':128},u
assert not list(P.glob('*43*'))
updates=rr(P/'seed42-B/updates.jsonl');assert len(updates)==128
assert sum(x['state_id'] is not None for x in updates)==34
assert abs(sum(x['state_weight'] for x in updates if x['state_id'])-34)<1e-10
for x in updates:
 for k in ['trajectory_loss','state_loss']:
  if x[k] is not None:assert math.isfinite(x[k])
s=r(P/'seed42-B/summary.json');assert s['base_unchanged'] and s['reload_max_abs']==0 and s['updates']==128
weight=P/'seed42-B/adapter-final/adapter_model.safetensors'
assert hashlib.sha256(weight.read_bytes()).hexdigest()==s['adapter_sha256']
eval=rr(P/'eval-42-B.jsonl');assert len(eval)==64
for x in eval:
 assert len(x['trace'])<=4 and x['searches']<=3
 for t in x['trace']:
  assert len(t['prompt_token_ids'])<=16384 and len(t['action_token_ids'])<=2048
  assert len(t['prompt_attention_mask'])==len(t['prompt_token_ids'])
probes=[x for f in [P/f'probe-{m}-{k}.jsonl' for m,k in [('GoalZ','partial'),('42-A','partial'),('42-A','training-fit'),('42-B','complete'),('42-B','partial'),('42-B','training-fit')]] for x in rr(f)]
assert len(probes)==56
sources={x['probe_id']:x for f in ['partial-probes.jsonl','complete-probes.jsonl','training-fit-probes.jsonl'] for x in rr(P/f)}
for x in probes:
 assert x['prompt_token_ids']==sources[x['probe_id']]['prompt_token_ids']
 assert x['no_search_execution']
fit={}
for m in ['42-A','42-B']:
 fit[m]=[dict(probe_id=x['probe_id'],answer=x['answer']) for x in rr(P/f'probe-{m}-training-fit.jsonl')]
validation=dict(passed=True,counts=dict(u),new_trajectories=64,new_probes=56,finite_losses=True,adapter_hash=s['adapter_sha256'],reload_max_abs=s['reload_max_abs'],base_unchanged=True,seed43_executed=False,original_frozen_identity_verified=True,probe_inputs_exact=True,actual_budget_checked=True,training_fit=fit)
if (P/'delivery-validation.json').exists():assert r(P/'delivery-validation.json')==validation
else:an.write(P/'delivery-validation.json',validation)
an.write(P/'final-status.json',dict(status='complete_seed42_gate_failed',route_passed=False,seed43_executed=False,released=False,counts=dict(u),authorization='User authorized GPU and waived repeated confirmation; frozen continuation gates remain binding.'))
A=g['scores']['42-A'];B=g['scores']['42-B'];ma=g['metrics']['42-A'];mb=g['metrics']['42-B']
old={x['case_id']:x for x in rr(an.AM/'semantic-review-seed42.jsonl') if x['model']=='42-A'}
changes=[]
for x in rr(P/'semantic-review-seed42.jsonl'):
 if x['category']!=old[x['case_id']]['category']:
  changes.append(dict(case_id=x['case_id'],old_category=old[x['case_id']]['category'],new_category=x['category'],reason=x['reason'],edges=x['edges']))
an.write(P/'support-changes.json',changes)
report=f'''# Goal AN：集中难例状态监督，seed42收尾

**结论：未产生额外收益，按冻结门槛停止，不执行seed43。** 普通状态SFT（AM A）仍是当前开发诊断中更好的候选；唯一已发布产品仍为Goal Z。没有发布AN adapter。

| 指标 | 普通状态SFT A（复用AM seed42） | 集中难例B seed42 |
|---|---:|---:|
| 64题EM | {ma['EM']}/64 | {mb['EM']}/64 |
| F1 | {ma['F1']:.6f} | {mb['F1']:.6f} |
| 支持成功 | {A['supported']}/29 | {B['supported']}/29 |
| 实际完整链可见 | 26/29 | 26/29 |
| 事实错误+可见证据不支持 | 7 | 7 |
| 必要限定缺失 | 1 | 2 |
| 完整证据探针正确 | 20/24 | 20/24 |
| 部分证据合法补链查询 | 7/8 | 7/8 |
| 部分证据提前回答/无效/重复 | 0/0/0 | 0/0/0 |
| 动作问题题数 | 1 | 2 |
| 解析失败/重复/超预算尝试 | 0/1/0 | 0/1/1 |
| 调用/实际搜索 | 192/128 | 194/130 |
| 最大实际输入/输出token | 2248/30 | 2248/30 |
| 正文裁剪/长度停止 | 0/0 | 0/0 |
| 平均每题秒数 | {ma['mean_question_seconds']:.3f} | {mb['mean_question_seconds']:.3f} |

29题准入层之外35题保留来源未确认；64分母EM并不等于证据充分成功。A为历史匹配结果，延迟仅描述各次运行，不作严格速度结论。

## 赢回、丢失及真实机制

新增支持成功0题，丢失1题：`musique:2hop__796221_18803`（Rapana）。两臂第一次分歧在调用索引2，即第三次调用；之前查询、返回证据和回答输入token完全相同。Shell原文明确说明2010年4月正在寻找买家接手芬兰全部业务。A回答该活动与范围；B仅回答“a potential buyer”，遗漏活动及芬兰范围。两条引用与回答时token位置在`support-changes.json`，不是检索失败或丢文。

`musique:2hop__864101_135373`由“Vitus Bering”变为“Bering”。当前输入只有一个相关同指称人物，按既定等价表达规则仍判支持成功；它是EM损失，不是关系错绑。

`musique:2hop__96194_78303`从“Simon Moon”变为“The Terror”，仍为另一作品角色。实际Walker人物表明确Chuck Norris饰Cordell Walker，相关正文全部可见。没有绑定改善。

64题另外两项变化：Hagar题“Keturah”变“Ishmael”，EM损失但来源未确认，不新增事实错误裁定；Jamie Ward题由错误回答变成第四次search尝试，以max_steps停止。原重复搜索题`musique:2hop__560534_91104`仍然重复。全部同题首次分歧见`paired-ledger-seed42.jsonl`。

## 训练难例及探针

4个训练难例两臂均3/4：合作歌手仍误答Justin Bieber，应为Jaden Smith；机构、出生日期（含1922）、出生地点均正确。这意味着加权既未修复剩余训练错误，也未增加完整证据探针正确数，不能宣布局部拟合改善。

24个完整证据探针输入与AM A原档案token精确一致，B未新增任何实体/角色/关系对象修复。Darley地域、Huang亲属、Bach父亲出生地及奖项限定仍失败。

8个部分证据探针两臂动作均为search，其中7个合法。Crossley查询中的2015–16来自可见FA Cup资格规则，却被套到所问球队夺冠年份；它是可见信息范围错绑，非隐藏实体泄漏。Goal Z原部分探针为6/8合法，额外失败为Rapana重复原查询；A/B均修复该重复。本轮探针没有执行检索，不计自主搜索成功。

7个旧连续搜索保护题全部保留，公爵AM新增成功保留，Rapana丢失。支持成功不足23、完整探针不足22、绑定修复为0、动作问题增加、未保留两项AM收益等继续条件均失败。详细逐项见`seed42-gate.json`。

## 实施、身份与额度

独立从AA发布Goal Z seed42加载，源adapter SHA为`d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52`。B实际adapter SHA为`{s['adapter_sha256']}`；LoRA变更192个tensor，base未改变，独立重载最大差0。

128条完整轨迹、34状态和顺序均与AM一致。4个难例状态系数2.956521739，其余0.739130435，34系数总和34；仅状态级加权，当前JSON token全部等权。每条状态在更新2、4至68各出现一次。损失为轨迹归一NLL加0.25乘状态系数乘当前动作归一NLL。128更新，LR5e-6，AdamW(0.9,0.999)、eps1e-8、wd0、clip1，仅LoRA。单卡BF16，不量化。

评估16384/2048、direct-v090、first-json-v1、thinking=false、legacy、BM25-v2/top3、4调用/3搜索保持冻结。实际新增1次训练128更新、64真实轨迹、56单状态检查，共250次GPU生成、130次实际搜索；未达到额度上限，未追加采样、新确认或训练。

训练GPU峰值allocated约11.842GB、reserved约14.219GB。有限损失及预算、输入一致性、adapter哈希和停止门槛检查通过，见`delivery-validation.json`。准备阶段v1的CPU配置字段错误现场保留，v2只修复该读取；收尾CPU审核曾误从评分摘要读取token字段，改为原probe档案核对，修复前无裁定落盘，无新增GPU或改评规则。

## 后续方向

保留AM普通状态SFT局部收益19→21/29，停止此集中权重配方，不增加权重或遍数。结果支持下一次研究更明确的“同上下文不同角色/对象”监督，而非继续强化同4条状态；具体方案应仍用训练题、匹配对照和冻结额度。该建议尚未执行，本轮没有证明所有实体关系监督无效。

本轮一次主审核仅3个改变且未能完全复用的准入组合，其余26个精确复用问题—输入—答案裁定；完整探针23个相同组合复用，1个Bering简称核验。所有判断读取实际正文与逐边引用，不按案例序号指定通过。原档案和评分保留。
'''
(P/'REPORT.md').write_text(report,encoding='utf-8')
print(json.dumps(validation,ensure_ascii=False))


