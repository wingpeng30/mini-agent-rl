# v121 控制器协议失败诊断与下次实验格式门槛

本报告只复核冻结的开发实验原始输出，不重新运行模型。B 使用最终的 `output-v3-audit/arm-B.json`，C/D 使用完整固定状态运行的 `output-v2/arm-C.json`、`arm-D.json`。逐条分类保存在 `backups/v121/v121-controller-diagnosis-final.json`。

| 组别 | 控制器调用 | 主要失败 |
|---|---:|---|
| B 查询复核 | 46 | 41 次外层 `answer` 内的 JSON 没有形成合法字符串；3 次输出旧 `search` 动作；1 次内层 JSON 无效；1 次 `answer` 类型错误 |
| C 充分性检查 | 50 | 49 次输出旧 `search` 动作；1 次把多个动作拼在一起，外层 JSON 无效 |
| D 充分性检查 | 50 | 同 C；因检查阶段失败，没有进入 Q 的有效比较 |

原 SFT 目标使用 `{"action":"search","query":"..."}` 或 `{"action":"answer","answer":"..."}`。v121 的 Q/S 却要求先输出旧 `answer` 外壳，再在 `answer` 字符串里编码包含证据、引用、状态等多个字段的 JSON。这改变了模型生成任务。对于 C/D，绝大多数输出仍是旧 `search` 动作；对于最终 B，模型尝试了新结构，但经常不能完成双层 JSON 转义。现有证据将主要故障定位在**训练目标、提示和运行器的协议不一致**。解析器按既定协议拒绝这些文本，不能称为误拒绝。也没有证据表明示例缺失是主因；本轮没有固定示例对照。

记录保存了原始输出和输入 token，但没有逐次保存生成 token 数、`parse_error` 和 `output_truncated`。因此不能从这批记录独立算出截断率。部分长输出可能被 256 token 上限截断；即使如此，C/D 的 98 次旧 `search` 输出已经足以解释其主要失败。B/C/D 的 0/50 是完整流程得分，不能解释为 Q/S 在正常运行后的语义正确率。

下次实验统一采用当前 SFT 学到的唯一运行时动作外壳。Q/S 若需要记录证据状态，应将其作为单独诊断字段由运行器记录，或先构造经双审的对应训练目标并重新训练；不能临时要求现有 adapter 输出嵌套 JSON。每次改动提示、模板、tokenizer、解码参数、工具定义或输出字段时，都生成新的设置文件及 SHA-256。先用训练目标和最终设置下的少量模型原始输出运行 `scripts/check_action_protocol_contract.py`：

```text
python scripts/check_action_protocol_contract.py --sft data/evidence-sft-v117/frozen/train.jsonl --settings frozen-settings.json --probe probe.jsonl
```

`probe.jsonl` 每行包含 `raw_output`、`output_truncated`、`settings_sha256`；后者必须等于设置文件的 SHA-256。只有原始动作格式通过率达到 98% 才能开展完整开发评测。协议通过之后，还需单独审核搜索必要性、证据支持和答案正确率，不能用协议通过率代替语义验收。后续若决定做 SFT，训练目标导出也必须先通过同一动作解析器，且训练时与评估时的系统提示和目标协议一起冻结。v121 旧报告与结果保留，不追溯改写。
