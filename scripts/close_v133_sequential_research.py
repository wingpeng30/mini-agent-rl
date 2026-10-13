"""Close the approved finite sequence; no GPU, retrieval, or training."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def put(p,obj):
    with p.open('x',encoding='utf-8') as f:f.write(obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
    dirs=dict(AR='v133-goal-ar-constrained-evidence-v1',AS='v133-goal-as-frozen-recovery-inventory-v1',AT='v133-goal-at-supported-action-learning-v1',AU='v133-goal-au-frozen-reranking-v3')
    paths={g:ROOT/'runs'/d/('status.json' if g=='AS' else 'final-status.json') for g,d in dirs.items()}
    states={g:read(p) for g,p in paths.items()}
    if any(s.get('accepted') for s in states.values()):raise RuntimeError('A new accepted component requires AV execution')
    if any(s['research_parent']!='AM42A' for s in states.values()):raise RuntimeError('Parent drift')
    av=ROOT/'runs/v133-goal-av-cumulative-validation-v1';av.mkdir(exist_ok=False)
    put(av/'status.json',dict(status='not_triggered_no_new_accepted_candidate',generations=0,searches=0,updates=0,regression_pass_claim=False,new_confirmation=0,published=False))
    put(av/'REPORT.md','# Goal AV：无累计新版本，不触发最终验证\n\nAR未通过探针、AS库存不足跳过、AT两臂未通过、AU真实搜索没有净增。本序列没有被接受的新系统或权重组件；按“有进步后基于新版本最后实验”的条件，AV不触发。\n\n不是208题回归验证通过，也不代表AM具有发布资格。AM仍是21/29的研究参考，唯一产品模型仍为Goal Z。本阶段新增生成、搜索、训练、新确认均0，不重复测同一父模型来补满计划。\n')
    out=ROOT/'runs/v133-sequential-method-checks-v1';out.mkdir(exist_ok=False)
    hashes={g:hashlib.sha256(p.read_bytes()).hexdigest() for g,p in paths.items()}
    put(out/'manifest.json',dict(stage_status_files={g:str(p) for g,p in paths.items()},sha256=hashes,
        accepted_components=[],research_parent='AM42A',published_model='Goal Z',counts=dict(generation=968,search=512,update=256),
        reranker_offline_queries=130,reranker_offline_window_pairs=2475,new_confirmation=0,AV='not triggered'))
    put(out/'REPORT.md','''# AR→AV有限顺序研究：最终结果

没有一条新方法达到预设继续条件，保留AM普通状态SFT为实验参考（21/29支持成功）；唯一已发布产品模型仍为Goal Z。不宣称AM已通过完整回归或新确认。

| 阶段 | 观察 | 决定 |
|---|---|---|
| AR约束证据地址/动作协议 | 完整证据A/C/B为20/19/14；B32/32协议可执行，语义退步 | 舍弃，不启动真实搜索 |
| AS既有失败恢复 | 1/24恢复、10对材料、关系错配3例，低于8/24、16对、4例门槛 | 跳过，不扩池 |
| AT固定样本普通SFT/GRPO | AM/A/B支持21/20/19；完整证据探针20/20/19 | 两臂舍弃，不执行seed43 |
| AU固定BGE检索排序 | 离线五题必要文档升入top3、完整链23→24；真实支持21→21，EM26→24 | 舍弃，不累积 |
| AV最终验证 | 没有新的合格累计版本 | 不触发，不声称208题保护通过 |

## 实际进展与机制结论

协议约束解决了显式字段不生成的问题，但没有解决关系对象选择；语法可执行不能替代语义正确。AT在固定训练状态里存在6个正确/错误混合组，能够执行匹配训练，但该小样本一次更新配方没有泛化收益；不能推广为所有RL无效。AU确实让部分必要文档进入前三，却不能确保正确使用它们：Darley地域绑定赢回，St.Matthew完整链下命名对象丢失，两题抵消。

因此当前证据支持的瓶颈包含完整证据下对象选择不稳定，以及自主查询缺链；不支持“再增加上下文或保证JSON即可解决”。这些负结果限定了本轮具体配方，不证明模型已达到能力上限。后续若改变方向，需另行冻结能直接验证对象选择的匹配机制实验，不能继续调本轮提示、权重或排序来追分。

## 额度与执行

新增968次生成、512次真实搜索、256次训练更新；另130个既有查询的2475个reranker窗口评分。没有新增数据、候选采集、新确认或发布。各阶段CPU核验、原始token、工具正文、引用位置、模型身份与损失记录分目录保存。

远程SSH使用用户提供的端口25320与connect.cqa1.seetacloud.com。本地受限网络导致的DNS失败不代表AutoDL关机。HF官方远程访问失败后，按用户指示启用/etc/network_turbo及hf-mirror.com；固定官方revision和七文件哈希一致。网络与tokenizer接口修复前失败档案保留，均没有GPU样本；修复后未重新采样或改变实验条件。

结果入口：AR、AS、AT、AU-v3、AV各自REPORT.md；AU保留v1网络失败与v2CPU接口失败。所有计算在远程AutoDL，结果同步回本地。
''')
    marker='## AR→AV最终收尾（2026-10-05）'
    addition='''

## AR→AV最终收尾（2026-10-05）

此条替代此前“AT执行中/结果待评价”的进度文字，不修改历史分数。AR约束协议未改善语义，AS库存不足跳过，AT普通SFT/GRPO支持21→20/19，AU排序支持21→21且EM26→24，均未接受新版本。AV因无新累计候选不触发，不能声称完整回归保护通过。

- [总报告](../runs/v133-sequential-method-checks-v1/REPORT.md)
- [AT结果](../runs/v133-goal-at-supported-action-learning-v1/REPORT.md)
- [AU结果与网络修复](../runs/v133-goal-au-frozen-reranking-v3/REPORT.md)
- [AV条件收尾](../runs/v133-goal-av-cumulative-validation-v1/REPORT.md)

当前研究参考保持AM seed42-A（21/29支持），产品保持Goal Z。已完成方法不再作为待办，不增加数据、采样或微调参数来补门槛。本序列实际968次新增生成、512次搜索、256次更新、新确认0。负结果只限定已冻结配方，不代表0.8B能力上限。
'''
    for name in ('v133-sequential-research-plan.md','v133-post-an-project-roadmap.md','v133-experiment-index.md'):
        p=ROOT/'docs'/name;s=p.read_text(encoding='utf-8')
        if marker in s:raise RuntimeError('Closure already appended')
        p.write_text(s+addition,encoding='utf-8')
    print(json.dumps(dict(completed=True,accepted_components=[],AV='not_triggered',report=str(out/'REPORT.md'))))
if __name__=='__main__':main()
