# Goal AM seed42结果

固定64题，准入29题；Goal Z复用AL大窗口。探索门槛通过：False。未确认35题仅报告EM/F1。

| 指标 | Goal Z | A普通状态SFT | B字段加权SFT |
|---|---:|---:|---:|
| EM /64 | 23 | 26 | 27 |
| 支持成功 /29 | 19 | 21 | 21 |
| 事实错误+证据不支持 | 7 | 7 | 7 |
| 无效/重复/预算外动作题 | 8 | 1 | 1 |
| 完整链探针正确 /24 | 19 | 20 | 20 |
| 缺链探针提前回答 /8 | 0 | 0 | 0 |

A/B都看过相同128轨迹+34状态并各更新128次；B相对A才衡量字段加权收益。A相对Goal Z测量继续训练加状态监督的整个配方，不能单独归因于新增34状态。

A新增成功：['2wiki:f299dd600bdd11eba7f7acde48001122', 'musique:2hop__796221_18803']。丢失成功：[]。
A门槛：{'supported_net_plus_two': True, 'errors_not_higher': True, 'invalid_or_repeated_not_higher': True, 'parse_invalid_not_higher': True, 'repeated_not_higher': True, 'overbudget_not_higher': True, 'protected_losses_at_most_one': True, 'complete_probe_gain_two': False, 'missing_probe_premature_not_higher': True}。

B新增成功：['2wiki:f299dd600bdd11eba7f7acde48001122', 'musique:2hop__796221_18803']。丢失成功：[]。
B门槛：{'supported_net_plus_two': True, 'errors_not_higher': True, 'invalid_or_repeated_not_higher': True, 'parse_invalid_not_higher': True, 'repeated_not_higher': True, 'overbudget_not_higher': True, 'protected_losses_at_most_one': True, 'complete_probe_gain_two': False, 'missing_probe_premature_not_higher': True}。

通过仅继续seed43冻结复现；未通过则停止，不扩样本或改参。产品发布仍需完整回归与新确认。
