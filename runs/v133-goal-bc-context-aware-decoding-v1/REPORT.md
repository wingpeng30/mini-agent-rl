# Goal BC：固定权重的证据感知解码

**探针未通过，结束本冻结CAD配方；未运行64题真实搜索或训练。**

AM固定权重普通解码A完整证据正确20/24，CAD B为16/24；部分证据合法补链A/B为7/5/8。新增1题，丢失5题，修复实体/角色/关系对象1题。门槛各项见probe-gate.json。

## 新增与丢失

- gained `2wiki:1dd33f080bde11eba7f7acde48001122`：A=('answer', 'Eisenach', None)；B=('answer', 'Arnstadt', None)。The actual body binds Johann Nicolaus to father Johann Christoph, whose body explicitly says born at Arnstadt. A's Eisenach belongs to the son or father's employment/death. CAD corrects the father/birthplace object while the complete evidence is unchanged.

- lost `musique:2hop__864101_135373`：A=('answer', 'Vitus Bering', None)；B=('invalid', '', None)。First object fails original action parser; no later object selected.

- lost `musique:2hop__506035_325512`：A=('answer', 'Pearl River County', None)；B=('invalid', '', None)。First object fails original action parser; no later object selected.

- lost `musique:2hop__109242_113442`：A=('answer', '13 August 1896', None)；B=('answer', '8 June 1829', None)。The question asks Millais's death date. CAD selected 8 June 1829, the birth endpoint of the visible lifespan, instead of 13 August 1896. This is birth/death relation binding regression, not missing evidence.

- lost `2wiki:764039340bde11eba7f7acde48001122`：A=('answer', 'Hyde Park, Westminster', None)；B=('search', '', 'Charing Cross Edinburgh place of death')。Complete original body chain remains visible, but action continues searching. Probe query not executed.

- lost `2wiki:f299dd600bdd11eba7f7acde48001122`：A=('answer', '1575', None)；B=('search', '', 'Don Juan de la Cerda, 2nd Duke of Medinaceli date of death')。Complete original body chain remains visible, but action continues searching. Probe query not executed.

## 工程、依据与成本
34项CPU测试通过，32个A输入逐token匹配档案；10次GPU工程通过，alpha0两条输入/生成token/停止与普通A一致。参考只清空正文/标题，不修改主输入；两路独立缓存逐token长度核验通过，无正文裁剪。实际引用字符/token跨度见probe-adjudications.json，首处分歧三路分数见changes.json。

固定alpha1整条动作CAD，只在第一/第二次搜索后启用，FP32计算2full-reference；原direct-v090、16384/2048、greedy、first-json、legacy、BM25-v2/top3。没有强制回答、重试、模板修改或参数扫描。

实际74次主生成+36条参考序列，合计110条计算序列；搜索0，训练0，新确认0。逐token前向与耗时见costs.json，加载/显存见engineer及probe identity文件。参考是保留历史搜索但移除文档内容的输入，不能解释为纯模型先验。

本轮是同证据条件决策诊断，不能将探针变化计为自主搜索成功或权重学习。开发29题支持成功没有重新评价，历史AM21/29不重写；其他35题仍未确认。研究参考保持AM42-A，唯一发布模型Goal Z。

## 决定
不进入真实搜索，不准备本配方蒸馏，不调整alpha/参考范围补跑。负结果限定此冻结解码配置，不代表所有证据感知方法或0.8B能力上限。旧AW/AX/AY结果和材料准入仍保留，不将未训练路线算作训练失败。
