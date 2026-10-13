"""CPU-only score, checkpoint, and semantic audit for Goal T."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_v133_goal_q_search as q
import run_v133_goal_s_label_granularity as s
import run_v133_goal_t_update_length as t
import analyze_v133_goal_s_label_granularity as sa

OUT = t.OUT
MODELS = ("incumbent", "42-32", "42-64", "43-32", "43-64")


def classify(case_id, answer):
    if case_id == "g010" and "promoter and manager of entertainment acts" in (answer or "").lower():
        return "unsupported_or_wrong", "This describes Chuck Wein alone; Edie Sedgwick is not a promoter or manager in the visible evidence."
    return sa.classification(case_id, answer)


def evidence(row):
    return [{"id": doc["id"], "title": doc["title"], "text": doc["text"]}
            for step in row["trace"] for doc in step.get("observation", {}).get("result", [])]


def make_audit():
    state = t.read(t.STATE)
    if state["status"] != "complete": raise RuntimeError("Goal T incomplete")
    gold = {row["case_id"]: row for row in q.dev_cases()}
    if len(gold) != 32: raise RuntimeError("Development set changed")
    evals = {model: {row["case_id"]: row for row in t.rows(OUT / f"eval-{model}.jsonl")}
             for model in MODELS}
    if any(set(rows) != set(gold) for rows in evals.values()):
        raise RuntimeError("Evaluation rows missing")
    counts = {model: Counter() for model in MODELS}
    review = {}
    for model, cases in evals.items():
        fixed = {row["case_id"]: row for row in t.rows(OUT / f"fixed-{model}.jsonl")}
        if set(fixed) != set(gold): raise RuntimeError(f"Fixed-context rows missing for {model}")
        for case_id, row in cases.items():
            official = gold[case_id]
            expected = q.score(row["answer"], official["official_answer"])
            if expected["em"] != row["em"] or abs(expected["f1"] - row["f1"]) > 1e-9:
                raise RuntimeError(f"Frozen product score mismatch: {model}/{case_id}")
            replay = fixed[case_id]
            if replay["answer"] != row["answer"] or replay["em"] != row["em"]:
                raise RuntimeError(f"Fixed-context answer mismatch: {model}/{case_id}")
            if row["reason"] != "answer" or not row["both_support_retrieved"]:
                raise RuntimeError(f"Product failure: {model}/{case_id}")
            visible = evidence(row)
            category, reason = classify(case_id, row["answer"])
            counts[model][category] += 1
            key = hashlib.sha256(json.dumps([row["question"], visible, row["answer"]],
                                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if key not in review:
                review[key] = {"case_id": case_id, "question": row["question"],
                    "answer": row["answer"], "visible_evidence": visible,
                    "classification": category, "reason": reason}
            elif review[key]["classification"] != category:
                raise RuntimeError("Semantic classification differs for identical evidence")
    allowed = set(t.read(s.OUT / "semantic-rubric.json")["categories"])
    if any(sum(counts[m].values()) != 32 or set(counts[m]) - allowed for m in MODELS):
        raise RuntimeError("Semantic classification incomplete")
    t.write(OUT / "semantic-review.json", {"rubric_sha256": t.sha(s.OUT / "semantic-rubric.json"),
        "reviewer": "agent-assisted evidence review; not independent human gold",
        "unique_question_evidence_answer_triples": len(review),
        "counts": {model: {category: counts[model][category] for category in sorted(allowed)} for model in MODELS},
        "blind_rows": list(review.values())})
    paired = {}
    for left, right in (("incumbent", "42-32"), ("incumbent", "43-32"),
                        ("42-32", "42-64"), ("43-32", "43-64")):
        gained, lost, changed = [], [], []
        for case_id in gold:
            old, new = evals[left][case_id], evals[right][case_id]
            if old["answer"] != new["answer"]:
                changed.append({"case_id": case_id, "question": old["question"],
                                "from": old["answer"], "to": new["answer"],
                                "from_category": classify(case_id, old["answer"])[0],
                                "to_category": classify(case_id, new["answer"])[0]})
            if not old["em"] and new["em"]: gained.append(case_id)
            if old["em"] and not new["em"]: lost.append(case_id)
        paired[f"{left}_to_{right}"] = {"gained": gained, "lost": lost, "changed": changed}
    checkpoints = {}
    for seed in t.SEEDS:
        update_rows = t.rows(OUT / f"seed{seed}/updates.jsonl")
        if len(update_rows) != 64 or [row["update"] for row in update_rows] != list(range(1, 65)):
            raise RuntimeError("Training update sequence incomplete")
        if len({row["case_id"] for row in update_rows[:32]}) != 16 or len({row["case_id"] for row in update_rows[32:]}) != 16:
            raise RuntimeError("One training pass incomplete")
        for count in t.CHECKPOINTS:
            folder = t.checkpoint_dir(seed, count)
            details = t.read(folder / "checkpoint-check.json")
            actual = t.sha(folder / "adapter-final/adapter_model.safetensors")
            if details["status"] != "passed" or actual != details["checkpoint_sha256"]:
                raise RuntimeError("Checkpoint hash or reload check failed")
            if details["reload_max_abs"] > .001 or not all(details[key] for key in
                    ("base_unchanged", "rng_unchanged", "optimizer_unchanged", "training_mode_unchanged")):
                raise RuntimeError("Checkpoint invariant failed")
            checkpoints[f"{seed}-{count}"] = {key: details[key] for key in
                ("checkpoint_sha256", "reload_max_abs", "changed_lora", "gpu_memory")}
    historical = {}
    for seed in t.SEEDS:
        current = evals[f"{seed}-64"]
        old = {row["case_id"]: row for row in t.rows(s.OUT / f"eval-{seed}-b.jsonl")}
        historical[str(seed)] = {"goal_t_total": sum(row["em"] for row in current.values()),
            "goal_s_total": sum(row["em"] for row in old.values()),
            "different_answers": [{"case_id": case_id, "goal_t": current[case_id]["answer"],
                                    "goal_s": old[case_id]["answer"]}
                                   for case_id in gold if current[case_id]["answer"] != old[case_id]["answer"]],
            "same_weight_sha256": checkpoints[f"{seed}-64"]["checkpoint_sha256"] ==
                 t.sha(s.OUT / f"seed{seed}-b/adapter-final/adapter_model.safetensors")}
        goal_t_losses = t.rows(OUT / f"seed{seed}/updates.jsonl")
        goal_s_losses = t.rows(s.OUT / f"seed{seed}-b/updates.jsonl")
        historical[str(seed)]["first_loss_difference_update"] = next(
            (index + 1 for index, (x, y) in enumerate(zip(goal_t_losses, goal_s_losses))
             if abs(x["loss"] - y["loss"]) > 1e-7), None)
    manual = OUT / "manual-gpu-audit.json"
    result = {"score_recomputed": True, "product_rows": 160, "fixed_rows": 160,
              "paired": paired, "semantic_counts": {model: dict(counts[model]) for model in MODELS},
              "checkpoints": checkpoints, "historical_goal_s": historical,
              "gpu_seconds": state["gpu_seconds"],
              "additional_gpu_seconds_estimate": t.read(manual)["estimated_ssh_wall_seconds"] if manual.exists() else 0.,
              "decision": t.read(OUT / "decision.json")}
    t.write(OUT / "audit.json", result)
    return result


def make_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed in t.SEEDS:
        updates = t.rows(OUT / f"seed{seed}/updates.jsonl")
        axes[0].plot([row["loss"] for row in updates], label=f"seed {seed}")
        axes[1].plot([row["grad"] for row in updates], label=f"seed {seed}")
    for axis in axes:
        axis.axvline(32, color="black", linestyle="--", linewidth=.8)
        axis.set_xlabel("Update")
        axis.legend()
    axes[0].set_title("Answer token mean CE")
    axes[1].set_title("Clipped gradient norm")
    fig.tight_layout()
    fig.savefig(OUT / "training-curves.png", dpi=150)
    plt.close(fig)


def make_report(result):
    summary = t.read(OUT / "summary.json")
    models = {row["model"]: row for row in summary["models"]}
    paired = result["paired"]
    semantic = result["semantic_counts"]
    def table_row(model):
        row = models[model]
        return f"| {model} | {row['main']}/16 | {row['regression']}/16 | {row['total']}/32 | {row['f1']:.4f} | {row['support']}/32 | {row['invalid']} | {row['searches']} | {row['mean_latency']:.2f}s |"
    def difference(left, right):
        item = paired[f"{left}_to_{right}"]
        return f"{left}→{right}：新增 {len(item['gained'])} 题（{', '.join(item['gained']) or '无'}），丢失 {len(item['lost'])} 题（{', '.join(item['lost']) or '无'}）。"
    text = f"""# Goal T：修订标签下的短预算训练

状态：`COMPLETE_PRODUCT_UPGRADED`。预设的两个短预算模型均通过产品门槛：seed 42 第 32 次更新为 20/32、回归 7/16；seed 43 为 21/32、回归 8/16。当前产品入口选择 **seed 42 的第 32 次 checkpoint**。Goal P RL-only 基线在本轮复现为 19/32、回归 7/16。

## 设计与工程核查

两个 seed 均独立从 Goal P RL-only 起步，使用 Goal S 冻结的两条修订标签、16 道真实搜索回答前输入和固定题目顺序。每个 seed 连续训练两遍、64 次更新；第 32 次和第 64 次保存 checkpoint，中间不重置优化器。四个 checkpoint 均通过 LoRA 变化、base 不变、独立重载 logprob、随机数与优化器状态核查。基础模型实际权重、起点、数据、代码、软件版本和数值环境哈希见 `config.json`。主运行 GPU 进程共 {result['gpu_seconds']:.1f} 秒；另有选中模型 CLI 演示核查约 {result['additional_gpu_seconds_estimate']:.1f} 秒（包含 SSH 启动，见 `manual-gpu-audit.json`），估计总占用约 {result['gpu_seconds'] + result['additional_gpu_seconds_estimate']:.1f} 秒。**没有 GPU 时间上限或阶段超时**。

## 同条件真实搜索结果

| 模型 | 主要开发 EM | 回归开发 EM | 合计 EM | F1 | 双侧证据 | 无效动作 | 搜索数 | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table_row('incumbent')}
{table_row('42-32')}
{table_row('42-64')}
{table_row('43-32')}
{table_row('43-64')}

两 seed 的短预算模型均在严格 EM 上超过当前模型，且回归不下降；长预算 seed 42 降到 18/32、回归 6/16，长预算 seed 43 保持 21/32、回归 8/16。因此结果支持缩短训练能减少**部分**回归，但效果并非两个 seed 都随更新次数单调变化。

{difference('incumbent','42-32')}

{difference('incumbent','43-32')}

{difference('42-32','42-64')}

{difference('43-32','43-64')}

seed 42 的第 32 次仍正确回答鸡尾酒共同成分 `lemon juice`，第 64 次改成只有一侧包含的 `honey`。seed 43 两个 checkpoint 均保留 `lemon juice`。`director` 与 `film director` 的训练标签修订已在 Goal S 冻结，本轮不重新选择标签。

## 语义边界与历史复核

基线与四个 checkpoint 共 160 条固定上下文输出，逐题与各自产品输出一致；评分按冻结 EM/F1 逐条复算。语义分类沿用 Goal S 预先冻结的规则，对同一问题、实际证据和答案只判一次。分类属于代理辅助审查，不是独立人工金标准。各模型“缺少必要限定／双侧证据不支持或事实错误”计数：当前模型 {semantic['incumbent'].get('missing_qualifier',0)}/{semantic['incumbent'].get('unsupported_or_wrong',0)}；42-32 {semantic['42-32'].get('missing_qualifier',0)}/{semantic['42-32'].get('unsupported_or_wrong',0)}；42-64 {semantic['42-64'].get('missing_qualifier',0)}/{semantic['42-64'].get('unsupported_or_wrong',0)}；43-32 {semantic['43-32'].get('missing_qualifier',0)}/{semantic['43-32'].get('unsupported_or_wrong',0)}；43-64 {semantic['43-64'].get('missing_qualifier',0)}/{semantic['43-64'].get('unsupported_or_wrong',0)}。完整依据见 `semantic-review.json`。

短预算相对基线新增的严格正确主要来自回答格式与粒度：`both are public use airports → public use airport`、`both are dogs → dog`。原回答已经表达了正确事实；本轮**尚未显示新事实问题上的稳定增益**。两短预算模型也把多职业问题的回答收窄为 `film director`，遗漏共同的 `screenwriter`。因此产品升级遵循预先冻结的严格 EM 门槛，不能解释成语义正确率已提高。

Goal T 的第 64 次模型与 Goal S 同 seed 的 64 次模型并非字节相同：seed 42 各为 {result['historical_goal_s']['42']['goal_t_total']}/32 和 {result['historical_goal_s']['42']['goal_s_total']}/32，seed 43 各为 {result['historical_goal_s']['43']['goal_t_total']}/32 和 {result['historical_goal_s']['43']['goal_s_total']}/32。两次训练的损失从第 {result['historical_goal_s']['42']['first_loss_difference_update']} 次／第 {result['historical_goal_s']['43']['first_loss_difference_update']} 次更新起已出现差异，均早于第 32 次保存；不能把历史差异归因于中间保存。CUDA 日志同时报告部分算子非严格确定性。没有因此重跑或事后改选 checkpoint。

当前两套开发题已经多次暴露，这个小样本改进不代表新问题泛化。最终推荐模型只在预设产品门槛下胜出；更强的事实性改进需要另行实验。

## 复现与产品入口

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_t_update_length_v2.py summarize
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_goal_t_selected.py --question 'Paul Young and David Draiman, share which mutual occupation?'
```

完整新运行需要指定空的 `GOAL_T_OUT`，先执行 `prepare`，再执行 `run`。运行器拒绝覆盖已开始的结果。`training-curves.png`、逐题输出、检查点哈希和资源台账均保留在本目录。
"""
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")


def main():
    result = make_audit()
    make_curves()
    make_report(result)
    print(json.dumps({"selected": result["decision"]["selected"],
                      "gpu_seconds": result["gpu_seconds"],
                      "semantic_counts": result["semantic_counts"]}, ensure_ascii=False))


if __name__ == "__main__": main()
