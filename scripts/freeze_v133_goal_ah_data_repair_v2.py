"""Version CPU premises after explicit bounded data-supplement authorization."""
import json
from prepare_v133_goal_ah import ROOT,OUT,sha,write_once

def main():
    path=OUT/'execution-freeze.json'; old=json.loads(path.read_text(encoding='utf8'))
    if (OUT/'execution-freeze.v1.json').exists():raise FileExistsError('Already versioned; no automatic re-freeze')
    for name,digest in old['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Original frozen input changed: '+name)
    repair=OUT/'confirmation-repair-v2'
    m=json.loads((repair/'final-manifest.json').read_text(encoding='utf8'))
    if not m['passed'] or m['accepted']!=64:raise ValueError('Incomplete confirmation repair')
    for name,digest in m['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Repaired data changed')
    # Portable POSIX-relative paths are also accepted by Windows pathlib.
    def portable(value):
        if isinstance(value,str):return value.replace('\\','/')
        if isinstance(value,list):return [portable(v) for v in value]
        if isinstance(value,dict):return {k.replace('\\','/'):portable(v) for k,v in value.items()}
        return value
    cfg=portable(old)
    cfg.update(pre_gpu_gates_passed=True,confirmation_quality=m,
        confirmation=dict(count=64,cases=m['cases'].replace('\\','/'),corpus=m['corpus'].replace('\\','/'),
            corpus_sha256=sha(ROOT/m['corpus']),scores='frozen-before-model-access',route='B only, baseline plus both seeds',
            seed_for_release=42),
        freeze_version='execution v2; bounded user-authorized confirmation data repair only',
        human_gpu_confirmation_received=False,gpu_authorized=False,
        data_supplement=dict(frozen_candidates=192,primary_helper_reviews=128,lead_positive_reviews=53,
            new_confirm_cases=28,training_changes=0,hotpot_used=0,automatic_pool_expansion=False))
    extras=list(repair.glob('*.json'))+list(repair.glob('*.jsonl'))+[
        ROOT/'scripts/prepare_v133_goal_ah_data_repair_v2.py',ROOT/'scripts/finalize_v133_goal_ah_data_repair_v2.py',
        ROOT/'scripts/freeze_v133_goal_ah_data_repair_v2.py',ROOT/'tests/test_goal_ah_confirmation_v2.py']
    cfg['files'].update({str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in extras})
    (OUT/'execution-freeze.v1.json').write_bytes(path.read_bytes())
    path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8')
    statepath=OUT/'state.json';(OUT/'state.confirmation-shortfall-v1.json').write_bytes(statepath.read_bytes())
    state=dict(status='awaiting_specific_gpu_confirmation',gpu_started=False,gpu_authorized=False,
        cpu_training_material_ready=True,confirmation_accepted=64,confirmation_required=64,
        experiment_count_changed=False,automatic_retry=False,execution_freeze_sha256=sha(path))
    statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for name in ['PRECONDITIONS','REPORT']:
        p=OUT/(name+'.md');(OUT/(name+'.v1.md')).write_bytes(p.read_bytes())
    pre=(OUT/'PRECONDITIONS.md').read_text(encoding='utf8')
    start=pre.index('状态：');end=pre.index('## 实际模型与环境')
    pre=pre[:start]+'状态：**训练材料与新确认集CPU门槛通过；GPU尚未启动，等待用户对本版具体计划确认。**\n\n'+pre[end:]
    pre=pre.replace('CPU 收尾版，尚不允许启动','数据补齐v2，等待具体GPU确认')
    start=pre.index('新确认准备：');end=pre.index('## 唯一训练设计与对照')
    pre=pre[:start]+'''新确认集v2：原准备包最终合格36题（MuSiQue20、2Wiki16），原包及失败账本保留。用户随后授权跨数据集补齐：完整官方来源先做机械隔离，固定哈希各冻结64候选，MuSiQue、2Wiki、HotpotQA合计192；只初审前两者128题，Hotpot64只做备用、不进入本轮。初审MuSiQue32/64、2Wiki47/64通过；主agent复核固定顺序中53条初审阳性，排除25条歧义、缺证、金标冲突或同链重复，最终选12+16题。新增训练题仍为0。

最终64题构成32+32，原问题、金标、别名、官方链均未修改。历史隔离覆盖现有runs数据、历史Goal G、AA及AG已曝光确认集和实际AG全2048篇语料；新增28题还排除旧确认准备的所有512候选。最终题间实体链/支持标题不重复，128段支持原文与实际2048篇新语料逐字一致。候选审核可读原文，不做产品输出、得分或oracle检索过滤；这不保证产品能检索到证据，必须由真实搜索验证。

原训练材料冻结不变。`confirmation-repair-v2/final-manifest.json`、`final-isolation.json`为新确认集最终判据；旧v1失败记录仅作历史。此次是一次有上限、用户授权的确认数据补充，不能解释为允许自动扩池或重跑实验。

'''+pre[end:]
    pre=pre.replace('当前36/64缺口使这个条件阶段不能执行。','新确认64题已冻结；仍须先通过开发资格、B对A机制门槛及工程验收，才可进入这192条条件确认。')
    pre=pre.replace('12项CPU测试通过；','原12项CPU测试和新增5项确认完整性测试通过；')
    pre=pre.replace('已实现远端命令，但当前CPU门槛不允许执行：','远端训练、开发/回归和固定输入机制命令已实现；当前缺少本版具体GPU批准，不允许执行：')
    pre=pre.replace('材料、确认质量或隔离不足也不扩池改参凑数。','本版之后材料、确认质量或隔离不足也不自动扩池改参凑数。')
    pre+='\n本版另外修复冻结清单的Windows反斜杠相对路径，统一为POSIX兼容路径；只改变路径表示，全部原冻结文件哈希仍一致。条件确认需在开发审核完成后按冻结资格安排，不能提前探查确认得分。\n'
    (OUT/'PRECONDITIONS.md').write_text(pre,encoding='utf8')
    report='''# Goal AH CPU准备结果 v2

**数据缺口已补齐，训练材料和新确认集CPU门槛通过；GPU尚未启动，等待本版具体计划确认。产品仍为Goal Z完整轨迹seed42。**

唯一训练设计、参数和实验数量保持：原128条轨迹加64条训练专用判断，A/B各seed42、43，每次128更新，共4次训练512更新。训练数据没有扩大，辅助材料及原轨迹哈希未变。

## 有界补齐结果

用户授权后，从本地已验来源的MuSiQue、2Wiki、HotpotQA官方数据筛查。机械隔离合格余量分别110、1151、52665条；各只冻结64候选，合计192。前两者128条原文初审通过32、47；主agent检查53条阳性并选28条，原36条合格题保留，最终MuSiQue32+2Wiki32。Hotpot备用64未使用。

不修改原问题或金标：出生地/from错配、多导演/多角色歧义、仅有厂牌发行或间接县界关系、生日缺日期、Palestinian与Lebanon标签冲突等候选排除。初审阳性不是最终可用数；主审25条排除包含同链去重。

隔离包括全部记录的历史训练、评价与候选池、Goal G外部历史、AA/AG回归和实际AG完整语料。28条新增还排除旧512候选；最终64题实体链互不重复。新2048篇冻结搜索语料逐字保存128段必要支持原文，不按模型或oracle检索表现挑题。仅检查语料含证据，真实取证是否成功留给产品评价。

## 验证与边界

原12项CPU测试与新增5项确认检查通过；原冻结文件哈希全部一致；新确认问题/金标/链未改、原文引用、来源配比、实际语料和冻结文件哈希通过。修复Windows冻结路径表示，保证远端POSIX使用。

GPU身份、有限数值、LoRA变化、基础权重不变和独立重载尚未执行。没有训练、模型推理或新确认得分。辅助agent初审及主agent裁定不冒充独立人类金标准。

诊断结论和13题复核/115题继承的边界保持，见REPORT.v1.md；未重新全量改评。AA与AG仍仅作回归。候选审核原文的接触不进入训练、提示示范或能力保留损失。

## 可审阅文件

- PRECONDITIONS.md：完整人读GPU前提。
- execution-freeze.json：v2实际冻结；execution-freeze.v1.json保留旧冻结。
- confirmation-repair-v2/final-manifest.json：64题、2048篇语料及全部输入哈希。
- confirmation-repair-v2/source-review-main.jsonl：53条主审原文关系/引用/理由。
- confirmation-repair-v2/final-isolation.json：完整曝光检查及128段语料逐字匹配。
- diagnostic-ledger-v1.jsonl、training-material-examples.jsonl：原诊断与四类训练示例。

已执行CPU命令：`python -m unittest discover -s tests -p test_goal_ah_confirmation_v2.py -v`。未生成gpu-approval.json；运行器必须获得绑定本版冻结哈希的具体用户确认才可启动。
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf8')
    write_once(OUT/'cpu-verification-v2.json',dict(original_tests=12,new_confirmation_tests=5,
        original_frozen_assets_unchanged=True,new_confirmation_ready=True,gpu_executed=False,
        execution_freeze_sha256=sha(path),original_freeze_sha256=sha(OUT/'execution-freeze.v1.json')))
    print(json.dumps(state))

if __name__=='__main__':main()
