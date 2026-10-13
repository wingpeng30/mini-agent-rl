"""CPU-only audit and report for the completed Goal S run."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_v133_goal_q_search as q
import run_v133_goal_s_label_granularity as s

OUT = s.OUT
MODELS = ("incumbent", "42-a", "42-b", "43-a", "43-b")

# This review is keyed only by case and answer, so the same answer gets the
# same decision regardless of model identity. All remaining pairs were
# inspected against the visible evidence and judged sufficient.
REVIEWS = {
    "5ae30de055429928c42395cf": {
        "public use": ("missing_qualifier", "The answer omits airport; public use alone does not name the shared entity type."),
    },
    "5a7789215542997042120a99": {
        "honey": ("unsupported_or_wrong", "Only the Bee's Knees passage lists honey; both list lemon juice."),
    },
    "5a73583e55429901807dafd0": {
        "drummer": ("unsupported_or_wrong", "The Mike Sparks passage calls him a guitarist and vocalist and names someone else as drummer."),
    },
    "5a78cc3f554299029c4b5e9b": {
        "designer": ("unsupported_or_wrong", "The Lukas Rossi passage supports musician, not designer."),
    },
    "5ac259bc5542992f1f2b386a": {
        "director": ("missing_qualifier", "Both visible passages specify film director."),
    },
    "5ac169c25542994ab5c67d1a": {
        "director": ("missing_qualifier", "The question asks for a specific occupation and both passages support film director."),
    },
    "g000": {
        "voice actress": ("unsupported_or_wrong", "Takako Honda is called a voice actress; Milla Jovovich is only called an actress in the visible passage."),
    },
    "g001": {
        "director": ("missing_qualifier", "The plural-occupation question and both passages also support screenwriter."),
        "film director": ("missing_qualifier", "The plural-occupation question and both passages also support screenwriter."),
    },
    "g004": {
        "magazine": ("missing_qualifier", "Both passages specify women's magazine, the shared publication category asked for."),
    },
    "g005": {
        "singer-songwriter": ("unsupported_or_wrong", "The Albert Raisner passage supports producer, not singer-songwriter."),
    },
    "g006": {
        "band": ("missing_qualifier", "Both passages explicitly support the more informative common type rock band."),
    },
    "g010": {
        "an American promoter and manager of entertainment acts": ("unsupported_or_wrong", "This describes Chuck Wein, not Edie Sedgwick."),
        "actress": ("unsupported_or_wrong", "This describes Edie Sedgwick, not Chuck Wein."),
    },
    "g018": {
        "first Chief Technology Officer of the United States": ("unsupported_or_wrong", "This describes Aneesh Chopra, not Tim Kaine."),
        "senator": ("unsupported_or_wrong", "This describes Tim Kaine, not Aneesh Chopra."),
        "attorney": ("unsupported_or_wrong", "This describes Tim Kaine, not Aneesh Chopra."),
    },
}


def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def docs(row):
    return [{"id": document["id"], "title": document["title"], "text": document["text"]}
            for step in row["trace"] for document in step.get("observation", {}).get("result", [])]


def classification(case_id, answer):
    return REVIEWS.get(case_id, {}).get(answer, ("sufficient", "The answer states a common attribute supported by both visible passages; phrasing may differ from strict EM."))


def audit():
    state = s.read(s.STATE)
    if state["status"] != "complete": raise RuntimeError("Goal S run incomplete")
    frozen = {row["case_id"]: row for row in q.dev_cases()}
    if len(frozen) != 32: raise RuntimeError("Development set changed")
    evals = {model: {row["case_id"]: row for row in s.rows(OUT / f"eval-{model}.jsonl")} for model in MODELS}
    if any(set(items) != set(frozen) for items in evals.values()): raise RuntimeError("Missing product rows")
    for model, items in evals.items():
        for case_id, row in items.items():
            expected = q.score(row["answer"], frozen[case_id]["official_answer"])
            if expected["em"] != row["em"] or abs(expected["f1"] - row["f1"]) > 1e-9:
                raise RuntimeError(f"Frozen score mismatch: {model} {case_id}")
            if row["reason"] != "answer" or not row["both_support_retrieved"]:
                raise RuntimeError(f"Product evidence or stop failure: {model} {case_id}")
    review_rows = {}
    counts = {model: Counter() for model in MODELS}
    for model, items in evals.items():
        for case_id, row in items.items():
            evidence = docs(row)
            key = hashlib.sha256(json.dumps([row["question"], evidence, row["answer"]],
                                   ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            category, reason = classification(case_id, row["answer"])
            counts[model][category] += 1
            if key not in review_rows:
                review_rows[key] = {"case_id": case_id, "question": row["question"],
                    "answer": row["answer"], "visible_evidence": evidence,
                    "classification": category, "reason": reason}
            elif review_rows[key]["classification"] != category:
                raise RuntimeError("Inconsistent blind classification")
    allowed = set(s.read(OUT / "semantic-rubric.json")["categories"])
    for model in MODELS:
        if sum(counts[model].values()) != 32 or set(counts[model]) - allowed:
            raise RuntimeError("Semantic review incomplete")
    s.write(OUT / "semantic-review.json", {"rubric_sha256": file_sha(OUT / "semantic-rubric.json"),
        "reviewer": "agent-assisted evidence review, not independent human gold",
        "unique_question_evidence_answer_triples": len(review_rows),
        "fixed_counts": {key: {category: counts[key][category] for category in allowed} for key in ("42-a", "42-b")},
        "product_counts": {key: {category: counts[key][category] for category in allowed} for key in MODELS},
        "blind_rows": list(review_rows.values())})
    paired = {}
    for left, right in (("42-a", "42-b"), ("43-a", "43-b"), ("incumbent", "42-b"), ("incumbent", "43-b")):
        gained, lost, changed = [], [], []
        for case_id in frozen:
            old, new = evals[left][case_id], evals[right][case_id]
            if old["answer"] != new["answer"]:
                changed.append({"case_id": case_id, "question": old["question"], "from": old["answer"],
                                "to": new["answer"], "from_category": classification(case_id, old["answer"])[0],
                                "to_category": classification(case_id, new["answer"])[0]})
            if not old["em"] and new["em"]: gained.append(case_id)
            if old["em"] and not new["em"]: lost.append(case_id)
        paired[f"{left}_to_{right}"] = {"gained": gained, "lost": lost, "changed": changed}
    fit = {}
    for model in MODELS[1:]:
        product = evals[model]
        fixed = s.rows(OUT / f"fixed-{model}.jsonl")
        if len(fixed) != 32 or any(row["answer"] != product[row["case_id"]]["answer"] for row in fixed):
            raise RuntimeError(f"Fixed-context replay mismatch: {model}")
        training = s.rows(OUT / f"fit-{model}.jsonl")
        if len(training) != 16: raise RuntimeError("Training-fit rows incomplete")
        fit[model] = {"original_em": sum(x["original_em"] for x in training),
                      "revised_em": sum(x["revised_em"] for x in training)}
    baseline_fixed = s.rows(OUT / "fixed-42-incumbent.jsonl")
    if len(baseline_fixed) != 32 or any(row["answer"] != evals["incumbent"][row["case_id"]]["answer"] for row in baseline_fixed):
        raise RuntimeError("Baseline fixed-context replay mismatch")
    result = {"strict_scores_recomputed": True, "product_rows": 160,
              "fixed_rows": 160, "training_fit_rows": 64,
              "semantic_counts": {key: dict(counts[key]) for key in MODELS},
              "paired": paired, "training_fit": fit,
              "gpu_seconds": state["gpu_seconds"], "selected": s.read(OUT / "decision.json")["selected"]}
    s.write(OUT / "audit.json", result)
    return result


def curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed in (42, 43):
        for arm in ("a", "b"):
            updates = s.rows(OUT / f"seed{seed}-{arm}/updates.jsonl")
            if len(updates) != 64: raise RuntimeError("Training curve incomplete")
            axes[0].plot([x["loss"] for x in updates], label=f"{seed}-{arm}", alpha=.85)
            axes[1].plot([x["grad"] for x in updates], label=f"{seed}-{arm}", alpha=.85)
    axes[0].set(title="Answer token mean CE", xlabel="Update", ylabel="Loss")
    axes[1].set(title="Clipped gradient norm", xlabel="Update", ylabel="Norm")
    for axis in axes: axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "training-curves.png", dpi=150)
    plt.close(fig)


def report(audit_data):
    summary = s.read(OUT / "summary.json")
    paired = audit_data["paired"]
    counts = audit_data["semantic_counts"]
    def line(model):
        x = next(row for row in summary["models"] if row["model"] == model)
        return f"| {model} | {x['main']}/16 | {x['regression']}/16 | {x['total']}/32 | {x['mean_f1']:.4f} | {x['support']}/32 | {x['invalid']} | {x['searches']} |"
    body = f"""# Goal S：训练答案粒度对真实搜索的影响

状态：`COMPLETE_NO_PRODUCT_UPGRADE`。仅修订两条训练标签，原始数据和产品搜索设置保持冻结。修订标签在两个 seed 中都优于各自原标签对照，但只有 seed 43 超过当前最佳 19/32；seed 42 为 18/32、回归 6/16，因此按预设双 seed 门槛保留 Goal P RL-only。

## 冻结干预与工程核查

16 道训练题的真实可见证据逐题审查后，仅把 SWMRS/We the Kings 的 `band` 改为 `rock band`，把 Jacques Tourneur/Djibril Diop Mambéty 的 `director` 改为 `film director`。两侧文档均直接支持新增限定。`documentary` 和 Josh Trank/Jonathan Frakes 等存在表述或证据边界的题保留原标签。逐题修订的证据来自训练题，详见 `labels.json`；检验答案粒度这一假设本身受到此前已暴露开发题失分的启发，结果不是独立验证。

起点、tokenizer、96 篇语料、16/32 题、Goal R 实际输入 token、Goal Q 搜索入口和基础模型**实际权重文件**哈希均写入 `config.json`。每臂独立从 Goal P RL-only 加载，两遍、每题每遍两次，64 次更新；除两条答案目标外，输入 token、顺序、seed、优化设置一致。四个 checkpoint 均通过 base 不变、LoRA 变化和独立重载 logprob 核查。GPU 进程合计 {audit_data['gpu_seconds']:.1f} 秒，低于 7200 秒。

## 同条件真实搜索

| 模型 | 主要开发 EM | 回归开发 EM | 合计 EM | F1 | 双侧证据 | 无效动作 | 搜索数 |
|---|---:|---:|---:|---:|---:|---:|---:|
{line('incumbent')}
{line('42-a')}
{line('42-b')}
{line('43-a')}
{line('43-b')}

seed 42：修订标签相对原标签新增 {len(paired['42-a_to_42-b']['gained'])} 题、丢失 {len(paired['42-a_to_42-b']['lost'])} 题；新增为 `{', '.join(paired['42-a_to_42-b']['gained'])}`。seed 43 新增 {len(paired['43-a_to_43-b']['gained'])} 题、丢失 {len(paired['43-a_to_43-b']['lost'])} 题；新增为 `{', '.join(paired['43-a_to_43-b']['gained'])}`。两次均恢复了 `director → film director` 相关题；机场题也从 `public use` 恢复为 `public use airport`。后者不是训练题的直接标签替换，说明干预存在跨题迁移，但数据量不足以证明稳定因果机制。

相对当前最佳，seed 42 修订臂新增 `{', '.join(paired['incumbent_to_42-b']['gained'])}`，丢失 `{', '.join(paired['incumbent_to_42-b']['lost'])}`。seed 43 新增 `{', '.join(paired['incumbent_to_43-b']['gained'])}`，丢失 `{', '.join(paired['incumbent_to_43-b']['lost'])}`。两臂均把鸡尾酒共同成分 `lemon juice` 错答为仅一侧具有的 `honey`；seed 43 仍有这一事实错误。`Democratic` 与 `Democratic Party` 等严格 EM 差异须单独看待。

## 机制诊断与训练拟合

基线保存的 32 道回答前上下文在各模型上重放，160 条固定上下文输出与各自产品输出逐题相同。训练拟合按原标签／修订标签分别为：42-A {audit_data['training_fit']['42-a']['original_em']}/16、{audit_data['training_fit']['42-a']['revised_em']}/16；42-B {audit_data['training_fit']['42-b']['original_em']}/16、{audit_data['training_fit']['42-b']['revised_em']}/16；43-A {audit_data['training_fit']['43-a']['original_em']}/16、{audit_data['training_fit']['43-a']['revised_em']}/16；43-B {audit_data['training_fit']['43-b']['original_em']}/16、{audit_data['training_fit']['43-b']['revised_em']}/16。训练拟合口径受两条不同目标影响，不能代替开发集产品成绩。

按训练前冻结的语义规则，模型名隐藏后审查独特的“问题—实际证据—答案”组合，再映射回模型。分类是代理辅助解释，**不是独立人工金标准**；每条理由和证据见 `semantic-review.json`。严格 EM 保持原口径。各模型“缺少必要限定／无双侧支持或事实错误”计数：当前最佳 {counts['incumbent'].get('missing_qualifier',0)}/{counts['incumbent'].get('unsupported_or_wrong',0)}；42-A {counts['42-a'].get('missing_qualifier',0)}/{counts['42-a'].get('unsupported_or_wrong',0)}；42-B {counts['42-b'].get('missing_qualifier',0)}/{counts['42-b'].get('unsupported_or_wrong',0)}；43-A {counts['43-a'].get('missing_qualifier',0)}/{counts['43-a'].get('unsupported_or_wrong',0)}；43-B {counts['43-b'].get('missing_qualifier',0)}/{counts['43-b'].get('unsupported_or_wrong',0)}。这一审查不会追溯触发复现：seed 43 已由产品总 EM 上升且回归不低于 A 的预设门槛触发。

## 结论与使用

这组配对结果支持：**原训练标签的粒度至少解释了部分 SFT 退步**。两处小修订在两个 seed 都减少了部分过粗回答，使修订臂比同 seed 原标签臂多答对 3 和 4 题。它没有稳定改善产品整体：seed 42 低于现有 19/32，回归也下降；seed 43 达 20/32、回归 8/16，但按预先冻结的双 seed 规则不能单独发布。当前唯一推荐和演示模型继续是 Goal P RL-only。

这些都是反复暴露的开发题，且 CUDA 日志提示部分算子不能严格确定性复现。两 seed 的差异及有限样本使结果不能代表新问题泛化。没有根据 seed 43 的较好成绩追加训练或改动标签。

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_s_label_granularity.py summarize
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_goal_s_selected.py --question 'Paul Young and David Draiman, share which mutual occupation?'
```

完整新运行须指定空的 `GOAL_S_OUT`，先复制冻结的 `labels.json`，再运行 `prepare` 与 `run`；历史运行目录拒绝覆盖。`run` 在产品分数无法直接判断复现时会等待语义审查文件，再以相同命令继续。`training-curves.png`、全部输出和预算台账保留在本目录。
"""
    (OUT / "REPORT.md").write_text(body, encoding="utf-8")


def main():
    result = audit()
    curves()
    report(result)
    print(json.dumps({"selected": result["selected"], "gpu_seconds": result["gpu_seconds"],
                      "semantic_counts": result["semantic_counts"]}, ensure_ascii=False))


if __name__ == "__main__": main()
