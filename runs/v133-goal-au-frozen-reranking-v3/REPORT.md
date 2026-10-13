# Goal AU：固定权重的检索排序对照

## 结论

本轮不接受reranker候选。离线五题必要文档进入top-3、完整链23→24，未转化为真实搜索净收益：实际支持成功21→21/29。当前研究父版本保持AM seed42-A；唯一发布模型仍为Goal Z。

| 指标 | A：BM25-v2 | B：BM25 top20 + BGE top3 |
|---|---:|---:|
| 支持成功/29 | 21 | 21 |
| 回答输入完整链/29 | 26 | 25 |
| 事实错误与证据不支持/29 | 7 | 7 |
| 必要限定不足/29 | 1 | 1 |
| EM/64 | 26 | 24 |
| F1 | 0.533906 | 0.500328 |
| 动作问题题数/64 | 1 | 1 |
| 模型调用/搜索 | 192/128 | 193/129 |

其余35题来源未确认，不将EM命中自动计为充分成功。58个答案组合完成一次裁定，逐边原句及实际token位置在semantic-review.jsonl。A的64题、全部192次输入、输出token及原始文本与AM档案完全一致，证明本轮匹配基线复现。

## 赢回与丢失

- Darley：Minsk Region→State of Minas Gerais。题面、查询和权重不变，检索排序改变；两臂回答输入均含Darley→Pedro Leopoldo→Minas Gerais，B正确选择地域对象。这是变化检索上下文下的绑定改善，不是新增关系取得或权重学习。
- St. Matthew：Vitus Bering→Bering Sea。两臂均保留完整岛屿→海域→命名人物链，B把中间海域当作命名对象。这是完整证据下的关系对象退步。
- Rapana保留成功：B回答“trying to find a potential buyer”，与实际April2010正文的过程一致；省略Finland资产范围另记精度下降。它不是新增成功。若更严格要求完整资产范围，B只会更低，本轮不接受结论不变。
- Hero仍失败：B查询Sheree J Wilson并答Alex Cahill，缺少电影→Chuck Norris桥接边；相关演员不能替代目标主演。

完整链进入输入仍不保证正确对象选择。离线固定查询指标与模型实际闭环指标不可混用。七题连续搜索保护保留，Rapana与公爵保留；净增至少两题的条件失败。没有更换模型、排序规则、提示或追加采样。

## 计算、网络修复与边界

实际生成385次、搜索257次、训练更新0、新确认0。另有130个既有查询、2475个离线query-window评分；live排序记录及每个实际窗口输入在live-ranking.jsonl与reranker-actual-inputs.jsonl，加载耗时单列component-load-times.jsonl。B在线reranker合计4.386078秒，单题模型及总流程延迟保存在mechanical-metrics.json和原轨迹。未单独保存显存峰值，不能补称已验证峰值。

按用户明确授权修复下载：source /etc/network_turbo后使用hf-mirror.com，固定BAAI/bge-reranker-base revision2cfc18c9415c912f9d8155881c133215df768a70。七份文件与官方LFS SHA256/Git blob一致；权重SHA256 ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd。v1网络失败、v2 Transformers5.17 tokenizer接口失败分别保留，均未生成GPU样本；v3仅使用官方同一pair模板修复接口，7项CPU检查通过。没有重跑模型样本或调整实验参数，全部实验计算在AutoDL完成。

当前配方不值得进入累计验证；这不证明所有reranker或0.8B模型达到上限。协议约束、固定小样本GRPO、现有reranker均未形成可接受新版本，AV无新候选，不能声称208题回归保护通过。
