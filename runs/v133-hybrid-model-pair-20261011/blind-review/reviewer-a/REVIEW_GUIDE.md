# 盲审操作说明

每个 JSONL 行是一个去重后的输出包。只依据包中的问题、实际最终输入、模型答案、结束原因、冻结参考关系和正文判断；不要推测或补入模型身份、EM/F1或汇总分数。相同问题/实际输入token/答案/结束原因只评一次，映射回两臂时复用同一票。

语义标签：`correct` 表示准确、充分回答，允许等价表达；`partial` 表示有正确部分但缺少必要答案或关系；`factual_error` 表示主要结论错误；`no_answer` 表示没有给出答案；`indeterminate` 表示题目来源或题义无法可靠裁定。来源状态是 `unresolved` 时必须用 `indeterminate`，不得当作错题。

证据支持必须比语义正确更严格：每个被判为支持的必要关系都要出现在 `covered_edge_ids`，并在 `citations` 中逐关系给出同一 edge ID、冻结来源文档ID和实际最终输入中可逐字找到的短引文。只看见文档标题、曾被搜索返回但未进入最终输入、或引用了无关句子都不算支持。答案语义不依赖检索证据；没有足够证据时仍可判 `correct`，但不要覆盖关系或添加引用。

每行输出一个 JSON 对象，不要改 review_id，不要省略任何包：

```json
{"review_id":"原样复制","semantic_category":"correct|partial|factual_error|no_answer|indeterminate","reason":"简短说明实体/关系/时间/限定条件","covered_edge_ids":["relation:1"],"citations":[{"edge_id":"relation:1","document_id":"文档ID","quote":"最终输入中逐字可见的短引文"}]}
```

审阅文件必须为 `review-a.jsonl` 或 `review-b.jsonl`，一行一个结果、UTF-8编码。完成后检查行数与review_id唯一性；不要修改原始review-packets.jsonl。
