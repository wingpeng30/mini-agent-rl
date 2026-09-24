"""Build the v113 diagnostics result narrative and machine summary."""
import hashlib, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
DOCS = ROOT / "docs"


def read_json(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p0 = read_json("v113-p0-agent-audit-summary.json")
    p0_unblinded = read_json("v113-p0-adjudication-summary.json")
    p1 = read_json("v113-p1-agent-audit-summary.json")
    p1_unblinded = read_json("v113-p1-unblinded-audit-summary.json")
    manifest = read_json("v113-diagnostics-manifest.json")
    block = read_json("v113-p0-external-api-block.json")
    cross = [json.loads(x) for x in (R / "v113-cross-results-stable.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ledger = read_json("v113-ledger-reconciliation.json")
    gate = read_json("v113-p2-source-gate.json")
    by_input = {}
    for row in cross:
        by_input.setdefault(row["input_id"], {})[row["arm"]] = row
    action_matrix = Counter()
    search_new = Counter()
    for pair in by_input.values():
        ba, ca = pair["baseline"]["action_parsed"]["action"], pair["candidate"]["action_parsed"]["action"]
        action_matrix[f"{ba}->{ca}"] += 1
        for arm in ("baseline", "candidate"):
            row = pair[arm]
            if row["action_parsed"]["action"] == "search":
                search_new[arm + ("_with_new_top3_doc" if any(d.get("new_document") for d in row["retrieval_results"][:3]) else "_without_new_top3_doc")] += 1
    # P0 official metrics and blinded support audit remain separate measures.
    scores = read_json("v113-p0-score-key.json")
    changed = [v for v in scores.values() if v.get("answer_changed")]
    score_delta = Counter()
    for row in changed:
        diff = row["candidate"]["normalized_exact_match"] - row["baseline"]["normalized_exact_match"]
        score_delta["improved" if diff > 0 else "regressed" if diff < 0 else "equal"] += 1
    summary = {
        "version": "v113",
        "phase": "P0_P1_complete_P2_blocked_no_training",
        "p0": {"paired_tasks": 50, "answer_changed": 15, "normalized_em_changes": dict(score_delta),
               "reviewed_cases": p0["cases"], "reviewed_supported_failure_states": 11,
               "quotes_verified": p0["quote_count_verified"], "blinded_review_coverage": p0["reviewer_coverage"],
               "external_api_review": "blocked by automatic data-transfer approval review", "agent_support_adjudication": p0_unblinded,
               "unresolved_semantic_cases": p0["three_agent_support_agreement"].get("disagreement")},
        "p1": {"exact_token_inputs": 170, "same_arm_source_outputs_reused": 231, "cross_inferences": 109,
               "verified_quotes": p1["verified_quotes"], "action_matrix": dict(action_matrix),
               "top3_new_document_counts_not_usefulness": dict(search_new),
               "semantic_review": p1_unblinded,
               "relative_evidence_reviews_are_disjoint_agent_slices": True,
               "fixed_state_only_not_closed_loop_causal_effect": True,
               "execution_provenance": "The stable result records cross_inference slots and exact model/adapter/input hashes, but has no host, PID, execution timestamp, or separate remote receipt; execution location is unverified.",
               "stable_result_sha256": sha(R / "v113-cross-results-stable.jsonl")},
        "decision": {"priority": "answer_generation_and_evidence_scope_review, while retaining retrieval as a likely useful behavior change; no causal attribution from fixed-state actions alone",
                     "p2_eligible_new_questions": gate["eligible_new_questions_after_historical_exclusion"],
                     "p2_required_questions": 50, "p2_passed": False, "p3_started": False, "p4_started": False,
                     "independent100_accessed": False, "final_test_accessed": False, "grpo_started": False},
        "autodl": {"shutdown_requested": False,
                   "last_available_receipt": "reports/v112-remote-receipt.json (2026-09-22)",
                   "live_status_verified_at_v113_completion": False},
        "ledger": ledger,
        "external_api_block": block,
        "hashes": {"diagnostics_manifest": sha(R / "v113-diagnostics-manifest.json"),
                   "p0_packet": sha(R / "v113-p0-review-cases.jsonl"),
                   "p1_packet": sha(R / "v113-p1-blind-review.jsonl"),
                   "stable_cross_results": sha(R / "v113-cross-results-stable.jsonl")},
    }
    summary_path = R / "v113-summary.json"
    if summary_path.exists():
        raise FileExistsError(str(summary_path))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    doc = f'''# v113：证据约束退化诊断与固定状态交叉评测

日期：2026-09-24。v113 完成 P0 盲化轨迹诊断和 P1 固定输入交叉评测。v101 保留为原模型对照，v112 仅作为候选 adapter。v111 的 78% 是训练侧 API 混合流程成绩，不是本地模型成绩。

## 冻结与范围

对照来自冻结的 v112 replay2 原始轨迹。诊断清单及源轨迹哈希见 `reports/v113-diagnostics-manifest.json`。P0 覆盖完整 50 题配对端到端轨迹、15 个最终答案发生变化的问题，以及 11 个充分证据但任一模型未匹配状态目标的配对状态。盲审材料只含问题、当时实际可见证据与匿名动作；gold、评分和模型身份保留在私有映射中。独立100题和 final-test 均未访问。

最近可用的 AutoDL 回执来自 v112（2026-09-22），当时实例运行且 GPU 占用 1 MiB。本任务没有发出关机请求；但本地未保存 v113 的主机、PID、运行时间或远程执行收据，故不能据此确认 v113 完成时的实时实例状态。v101 adapter SHA-256 为 `3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768`，v112 adapter SHA-256 为 `31b625a9e3934012f9ba9b2f2054006ab968caeabb5238c5934988edccf423fd`。

## P0：配对轨迹和证据支持

15 个规范化 EM 改变中，官方评分改善 {score_delta['improved']} 题、退化 {score_delta['regressed']} 题、持平 {score_delta['equal']} 题。该评分变化与“答案是否被当时可见证据支持”分开保存。六个盲审 agent 分块合计为每个配对案例提供三次审阅，1552 条逐字引文均通过校验。审阅者来自同一 Astra 模型系列，结论不是人工真值。在 50 个终答配对的辅助多数判断中，改善 5 题表现为支持由否转是、1 题双方皆支持；退化 3 题双方仍被判断支持、1 题候选不确定、1 题候选被反证。14 个模型-终答判断存在三审分歧，均保留，不转成训练标签。API 独立复核因外发内部问题和证据文本的自动审批被拒，未尝试绕过。

逐题盲审及去标识映射、评分私钥、审阅原始文件保存在报告目录。归因应按证据事件读取，答错本身不等于遗忘。观察到的风险包含关系/范围和时间事实不足、过早作答或继续搜索、检索虽有新增文档但未填补目标缺口，以及评分表达差异；争议留待人工或获准的独立复核。

## P1：相同输入上的动作诊断

从双模型端到端轨迹抽取 170 个 token 完全一致的输入，复用 231 个已有同臂输出；稳定结果文件将其余 109 个槽标记为 cross_inference。交叉结果共 340 条。固定状态动作矩阵为：v101/v112 同为搜索 61，同为回答 73，只有 v101 回答而 v112 搜索 36，反向 0。v112 的 97 次搜索中，94 次 top3 含新文档；v101 的 61 次搜索均含新文档。新文档比例不代表相关性或补缺成功。

三名 agent 分别审阅互不重叠的 57/57/56 个匿名输入，746 条引用全部通过当前证据或搜索后 top3 引文校验；回答禁止引用未来检索。因为没有交叉复核，不能声称 reviewer agreement。按单步动作审阅，v101 有 109 个回答，其中 70 个被判支持、24 个不支持、11 个不确定或其他争议；v112 有 73 个回答，其中 57 个被判支持、9 个不支持、7 个不确定/反证。v101 的 61 次搜索全部被判 query 有依据，缺口匹配为 42 是、16 部分、3 否；v112 的 97 次搜索有 94 个 query 有依据、3 个否，缺口匹配为 44 是、25 部分、28 否。这说明 v112 更常触发搜索，但相当部分 top3 未能填补明确缺口。审阅将答案支持、搜索必要性、query 依据和检索缺口匹配分开记录。结果是固定状态单步诊断，不构成替代整条轨迹的因果效果。结果记录 temperature=0、seed=42+step；没有相应执行日志，无法独立确认运行主机及 CUDA 确定性设置。该结果仅作为输入、模型和 adapter 哈希匹配的固定状态输出记录，不能宣称执行来源或位级确定性已经独立验证。

## 决策：停止 P2–P4

优先调查答案生成的关系、时间、数量和范围表达，以及证据充分时的答案保留；继续保留 v112 显示出的搜索倾向作为候选行为变化，但目前数据无法把单步动作变化归因成完整闭环准确率提升。仅凭 50 题评分变化和盲审不足以宣称遗忘或确定单一根因。

训练来源隔离复核发现，600 条训练来源在排除既往 v090/v111/v112 使用后只剩 29 个候选问题；这 29 个均出现在历史 v105 answer-repair 导出中。排除该历史开发暴露后，新可用问题为 0，未达到内部验收新建至少 50 题门槛。因此只交付缺口报告，不构建新训练标签，不启动 SFT，也不访问独立100题；P4 和 GRPO 均不执行。`scripts/check_v113_stage_gate.py` 对 P2/P3/P4 的 prerequisite fail-closed。来源/隔离细目见 `reports/v113-p2-source-gate.json`。

账本 `runs/v110-api.sqlite` 在本次启动时为 1225 笔已结算调用、63.861056 元；另有一次外部 P0 审阅调用处于 unknown，保守暂计 0.376952 元。由于该未知结果触发停止规则，没有继续 API 调用。自动审批拒绝和账本状态见 `reports/v113-p0-external-api-block.json`、`reports/v113-ledger-reconciliation.json`。

机器可读摘要：`reports/v113-summary.json`。交叉结果：`reports/v113-cross-results-stable.jsonl`。归档和逐文件哈希回执：`reports/v113-backup-verification.json`。
'''
    doc_path = DOCS / "v113-diagnostics-results.md"
    if doc_path.exists():
        raise FileExistsError(str(doc_path))
    doc_path.write_text(doc, encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "docs": str(doc_path), "p0": summary["p0"], "p1": summary["p1"], "decision": summary["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
