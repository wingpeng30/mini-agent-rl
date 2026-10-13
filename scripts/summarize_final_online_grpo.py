"""Summarize the final fixed-data online-GRPO run without model calls or reviewers.

The main comparison is am-serial/sft42/grpo42/grpo43. am-batch is an
efficiency comparison only; base-untrained is an additional bare-model baseline.
Missing or partial branches stay visibly incomplete.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any


BRANCHES = ("am-serial", "sft42", "grpo42", "grpo43")
BARE_BRANCH = "base-untrained"
GROUPS = (("natural104", 104), ("all128", 128), ("single24", 24))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_rows(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if number == len(lines):
                warnings.append(f"Ignored an unfinished last JSONL record: {path.name}:{number}")
            else:
                raise ValueError(f"Malformed JSONL: {path}:{number}") from None
    return result


def case_id(row: dict[str, Any]) -> str:
    value = row.get("case_id", row.get("id"))
    if value is None:
        raise ValueError("Evaluation record has no case_id/id")
    return str(value)


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        key = case_id(row)
        if key in result:
            raise ValueError(f"Duplicate evaluation case ID: {key}")
        result[key] = row
    return result


def number(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if row.get(key) is not None:
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite metric {key}")
            return value
    return default


def select(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "all128":
        return rows
    def single(row: dict[str, Any]) -> bool:
        return row.get("stratum") in {"constructed_singlehop", "singlehop", "single24"}
    return [row for row in rows if single(row) == (group == "single24")]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def aggregate(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    index_rows(rows)
    n = len(rows)
    strict = sum(number(row, "exact_match", "strict_exact_match") for row in rows)
    normalized = sum(number(row, "normalized_exact_match", "exact_match") for row in rows)
    seconds = sum(number(row, "elapsed_seconds") for row in rows)
    answered = sum(row.get("reason") == "answer" or bool(row.get("answer")) for row in rows)
    return dict(count=n, expected=expected, complete=n == expected,
                strict_em_count=strict, normalized_em_count=normalized,
                strict_em_rate=strict / n if n else None,
                normalized_em_rate=normalized / n if n else None,
                mean_f1=mean(number(row, "f1", "token_f1") for row in rows) if n else None,
                answered=answered, answer_completion_rate=answered / n if n else None,
                invalid_actions=sum(number(row, "invalid_actions") for row in rows),
                repeated_actions=sum(number(row, "repeated_actions") for row in rows),
                model_calls=sum(number(row, "model_calls") for row in rows),
                searches=sum(number(row, "searches") for row in rows),
                input_tokens=sum(number(row, "input_tokens") for row in rows),
                output_tokens=sum(number(row, "output_tokens") for row in rows),
                sum_question_seconds=seconds,
                mean_question_seconds=seconds / n if n else None,
                p95_question_seconds=percentile([number(row, "elapsed_seconds") for row in rows], .95),
                peak_allocated_gib=max((number(row, "gpu_peak_allocated_bytes") / 2**30 for row in rows), default=0),
                peak_reserved_gib=max((number(row, "gpu_peak_reserved_bytes") / 2**30 for row in rows), default=0),
                end_reasons=dict(Counter(row.get("reason", "unknown") for row in rows)))


def paired(before: list[dict[str, Any]], after: list[dict[str, Any]], draws: int = 2000) -> dict[str, Any]:
    a, b = index_rows(before), index_rows(after)
    ids = sorted(a.keys() & b.keys())
    n = len(ids)
    deltas = [number(b[key], "normalized_exact_match", "exact_match") -
              number(a[key], "normalized_exact_match", "exact_match") for key in ids]
    f1_deltas = [number(b[key], "f1", "token_f1") - number(a[key], "f1", "token_f1") for key in ids]
    result: dict[str, Any] = dict(pairs=n, identical_case_sets=a.keys() == b.keys(),
                                before_only=sorted(a.keys() - b.keys()), after_only=sorted(b.keys() - a.keys()))
    if not n:
        return result
    rng = random.Random(42)
    bootstrap_em, bootstrap_f1 = [], []
    for _ in range(draws):
        sample = [rng.randrange(n) for _ in range(n)]
        bootstrap_em.append(sum(deltas[i] for i in sample) / n)
        bootstrap_f1.append(sum(f1_deltas[i] for i in sample) / n)
    changed = []
    for key, delta in zip(ids, deltas):
        if delta:
            changed.append(dict(case_id=key, change="gain" if delta > 0 else "loss",
                                question=b[key].get("question"),
                                baseline_answer=a[key].get("answer"), candidate_answer=b[key].get("answer"),
                                baseline_reason=a[key].get("reason"), candidate_reason=b[key].get("reason")))
    result.update(normalized_em_delta=mean(deltas), mean_f1_delta=mean(f1_deltas),
                  normalized_em_bootstrap95=[percentile(bootstrap_em, .025), percentile(bootstrap_em, .975)],
                  mean_f1_bootstrap95=[percentile(bootstrap_f1, .025), percentile(bootstrap_f1, .975)],
                  gains=sum(d > 0 for d in deltas), losses=sum(d < 0 for d in deltas),
                  unchanged=sum(d == 0 for d in deltas), changed_cases=changed)
    return result


def training(folder: Path, warnings: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = read_json(folder / "summary.json")
    updates = read_rows(folder / "updates.jsonl", warnings)
    groups = read_rows(folder / "group-metrics.jsonl", warnings)
    seconds = number(summary, "training_seconds")
    samples = summary.get("samples")
    if samples is None:
        path = folder / "rollouts.jsonl"
        samples = 0
        if path.exists():
            with path.open("r", encoding="utf-8") as stream:
                samples = sum(bool(line.strip()) for line in stream)
    result = dict(summary)
    result.update(update_records=len(updates), group_records=len(groups),
                  planned_groups=128 if folder.name.startswith("grpo") else 0,
                  planned_updates=256 if folder.name.startswith("grpo") else 128,
                  sampled_trajectories=samples,
                  updates_per_second=len(updates) / seconds if seconds > 0 else None,
                  groups_per_second=len(groups) / seconds if seconds > 0 else None)
    return result, updates + groups


def efficiency_metrics(folder: Path, data: list[dict[str, Any]]) -> dict[str, Any]:
    summary = read_json(folder / "evaluation-summary.json")
    count = int(summary.get("current_process_cases", len(data)))
    wall = number(summary, "evaluation_wall_seconds")
    measured = data[-count:] if count else []
    input_tokens = sum(number(row, "input_tokens") for row in measured)
    output_tokens = sum(number(row, "output_tokens") for row in measured)
    memory = summary.get("gpu_memory") or {}
    return dict(summary, measured_cases=count, wall_seconds=wall or None,
                scope="current process segment" if summary.get("resumed") else "full evaluation",
                questions_per_minute=60 * count / wall if wall > 0 else None,
                input_tokens_per_second=input_tokens / wall if wall > 0 else None,
                output_tokens_per_second=output_tokens / wall if wall > 0 else None,
                peak_allocated_gib=memory.get("peak_allocated_gb"),
                peak_reserved_gib=memory.get("peak_reserved_gb"))


def write_curves(run_dir: Path, curves: dict[str, list[dict[str, Any]]], warnings: list[str]) -> str | None:
    records = []
    for branch, points in curves.items():
        for i, point in enumerate(points, 1):
            records.append(dict(branch=branch, step=point.get("update", point.get("group", i)),
                                loss=point.get("loss"), reward=point.get("mean_reward", point.get("reward_mean")),
                                kl=point.get("kl", point.get("mean_kl"))))
    with (run_dir / "training-curves.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("branch", "step", "loss", "reward", "kl"))
        writer.writeheader()
        writer.writerows(records)
    if not records:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        warnings.append("Matplotlib unavailable; raw training curves are saved as CSV.")
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, metric in zip(axes, ("loss", "reward", "kl")):
        for branch in curves:
            points = [r for r in records if r["branch"] == branch and r[metric] is not None]
            if points:
                axis.plot([r["step"] for r in points], [r[metric] for r in points], label=branch)
        axis.set(xlabel="Logged update/group", ylabel=metric)
        axis.grid(alpha=.25)
        if axis.lines:
            axis.legend(fontsize=8)
    fig.suptitle("Fixed-data continued SFT / online GRPO (objectives are different)")
    fig.tight_layout()
    path = run_dir / "training-curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def summarize(run_dir: Path, plot: bool = True) -> dict[str, Any]:
    warnings: list[str] = []
    evaluations = {name: read_rows(run_dir / "branches" / name / "evaluation.jsonl", warnings)
                   for name in (*BRANCHES, "am-batch", BARE_BRANCH)}
    for data in evaluations.values():
        index_rows(data)
    report: dict[str, Any] = dict(kind="exposed fixed-data regression; no new semantic review", status=read_json(run_dir / "status.json"),
                                  groups={}, training={}, warnings=warnings, new_semantic_review=False,
                                  bootstrap=dict(draws=2000, seed=42, method="paired percentile"))
    for group, expected in GROUPS:
        data = {name: select(evaluations[name], group) for name in BRANCHES}
        report["groups"][group] = dict(branches={name: aggregate(rows, expected) for name, rows in data.items()},
                                       paired_vs_am={name: paired(data["am-serial"], data[name]) for name in BRANCHES[1:]},
                                       paired_vs_sft={name: paired(data["sft42"], data[name]) for name in ("grpo42", "grpo43")})
    report["complete"] = all(report["groups"][group]["branches"][name]["complete"]
                             for group, _ in GROUPS for name in BRANCHES)
    report["complete"] = report["complete"] and all(report["groups"]["all128"]["paired_vs_am"][name]["identical_case_sets"] for name in BRANCHES[1:])
    bare = evaluations[BARE_BRANCH]
    report["bare_model"] = dict(branch=BARE_BRANCH, groups={})
    for group, expected in GROUPS:
        before = select(bare, group)
        report["bare_model"]["groups"][group] = dict(
            metrics=aggregate(before, expected),
            paired_to_branches={name: paired(before, select(evaluations[name], group)) for name in BRANCHES})
    report["complete"] = report["complete"] and all(
        report["bare_model"]["groups"][group]["metrics"]["complete"] for group, _ in GROUPS)
    report["complete"] = report["complete"] and all(
        report["bare_model"]["groups"]["all128"]["paired_to_branches"][name]["identical_case_sets"] for name in BRANCHES)
    curves = {}
    for name in BRANCHES[1:]:
        report["training"][name], curves[name] = training(run_dir / "branches" / name, warnings)
    serial, batch = evaluations["am-serial"], evaluations["am-batch"]
    efficiency = dict(serial=efficiency_metrics(run_dir / "branches" / "am-serial", serial),
                      batch=efficiency_metrics(run_dir / "branches" / "am-batch", batch),
                      answer_comparison=paired(serial, batch))
    ai, bi = index_rows(serial), index_rows(batch)
    common = sorted(ai.keys() & bi.keys())
    efficiency["exact_answer_changes"] = [key for key in common if ai[key].get("answer") != bi[key].get("answer")]
    serial_rate = efficiency["serial"].get("questions_per_minute")
    batch_rate = efficiency["batch"].get("questions_per_minute")
    efficiency["batch_throughput_ratio"] = batch_rate / serial_rate if serial_rate and batch_rate else None
    report["efficiency"] = efficiency
    comprehensive_json = run_dir / "comprehensive-metrics.json"
    comprehensive_markdown = run_dir / "comprehensive-metrics.md"
    if comprehensive_json.exists() or comprehensive_markdown.exists():
        report["comprehensive_metrics"] = dict(
            json=comprehensive_json.name if comprehensive_json.exists() else None,
            markdown=comprehensive_markdown.name if comprehensive_markdown.exists() else None)
    if plot:
        report["training_curve"] = write_curves(run_dir, curves, warnings)
    lines = ["# 固定0.8B：继续SFT与在线GRPO最终结果", "",
             "本轮仅使用现有训练材料及已暴露128题；下列是EM/F1与机械指标，没有新的语义审核或独立泛化结论。",
             "SFT与GRPO从同一个AM42-A父模型起步；两个RL seed验证追加训练，不代表整条历史训练链双seed复现。裸0.8B额外评价，不参与训练。",
             "", f"四臂及裸模型128题结果：{'完成' if report['complete'] else '未完成（不完整分母逐臂列出）'}。", ""]
    for group, expected in GROUPS:
        lines += [f"## {group}（预定{expected}题）", "",
                  "| 分支 | 已完成 | 严格EM | 归一化EM | F1 | 回答完成 | 调用/搜索 | 平均/P95秒 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in BRANCHES:
            m = report["groups"][group]["branches"][name]
            completion = f"{m['answered']} ({100*m['answer_completion_rate']:.1f}%)" if m['count'] else "0 (—)"
            lines.append(f"| {name} | {m['count']}/{expected} | {m['strict_em_count']:g} | {m['normalized_em_count']:g} | {fmt(m['mean_f1'])} | {completion} | {m['model_calls']:g}/{m['searches']:g} | {fmt(m['mean_question_seconds'])}/{fmt(m['p95_question_seconds'])} |")
        lines += ["", "| 比较 | 完整配对 | 新增/丢失 | EM差值百分点 | 配对95%区间 |", "|---|---:|---:|---:|---:|"]
        for base, key in (("AM", "paired_vs_am"), ("SFT", "paired_vs_sft")):
            for name, m in report["groups"][group][key].items():
                ci = m.get("normalized_em_bootstrap95")
                interval = f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]" if ci else "—"
                delta = 100*m["normalized_em_delta"] if "normalized_em_delta" in m else None
                lines.append(f"| {name} vs {base} | {m['pairs']}/{expected} | {m.get('gains', 0)}/{m.get('losses', 0)} | {fmt(delta, 2)} | {interval} |")
        lines += [""]
    lines += ["## 裸0.8B与现有训练链对比", "",
              "裸模型使用同一题集、语料、greedy及调用预算，直接加载基础权重而不加载LoRA；不是新增训练分支。", "",
              "| 集合 | 裸模型已完成 | 严格/归一化EM | F1 | 完成回答 |", "|---|---:|---:|---:|---:|"]
    for group, expected in GROUPS:
        m = report["bare_model"]["groups"][group]["metrics"]
        lines.append(f"| {group} | {m['count']}/{expected} | {m['strict_em_count']:g}/{m['normalized_em_count']:g} | {fmt(m['mean_f1'])} | {m['answered']} |")
    lines += ["", "| 裸模型→分支 | 集合 | 完整配对 | 新增/丢失 | EM差值百分点 | 配对95%区间 |", "|---|---|---:|---:|---:|---:|"]
    for group, expected in GROUPS:
        for name, m in report["bare_model"]["groups"][group]["paired_to_branches"].items():
            ci = m.get("normalized_em_bootstrap95")
            interval = f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]" if ci else "—"
            delta = 100*m["normalized_em_delta"] if "normalized_em_delta" in m else None
            lines.append(f"| {name} | {group} | {m['pairs']}/{expected} | {m.get('gains', 0)}/{m.get('losses', 0)} | {fmt(delta, 2)} | {interval} |")
    lines += [""]
    lines += ["## 训练与资源", "", "| 分支 | 采样轨迹 | 实际组/更新 | 零方差组 | 训练秒 | 峰值reserved GiB |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in report["training"].items():
        lines.append(f"| {name} | {m['sampled_trajectories']} | {m.get('groups', m['group_records'])}/{m.get('updates', m['update_records'])} | {m.get('zero_variance', '—')} | {fmt(m.get('training_seconds'))} | {fmt(m.get('peak_reserved_gb'))} |")
    lines += ["", "GRPO每seed最多128组、每组最多2次策略更新；SFT一遍128次更新。名义题数相同不表示计算量相同，零方差组可能没有更新。", "",
              "## batch4效率对照", "",
              "| 路径 | 计时覆盖 | wall秒 | 题/分钟 | 输入/输出token每秒 | allocated/reserved GiB |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in ("serial", "batch"):
        m = efficiency[name]
        lines.append(f"| {name} | {m['measured_cases']} ({m['scope']}) | {fmt(m['wall_seconds'])} | {fmt(m['questions_per_minute'])} | {fmt(m['input_tokens_per_second'])}/{fmt(m['output_tokens_per_second'])} | {fmt(m['peak_allocated_gib'])}/{fmt(m['peak_reserved_gib'])} |")
    lines += ["", f"batch/serial吞吐比：{fmt(efficiency['batch_throughput_ratio'])}；同题答案文本变化{len(efficiency['exact_answer_changes'])}/{len(common)}。",
              "吞吐使用实际进程wall时间；batch各题重叠耗时不相加作wall。若续跑，速度只覆盖该进程完成的题段。效率对照不与主质量四臂混算。"]
    if report.get("training_curve"):
        lines += ["", "![训练曲线](training-curves.png)"]
    lines += ["", "## 简历可用实际值", ""]
    natural = report["groups"]["natural104"]["branches"]
    baseline = natural["am-serial"]
    bare_natural = report["bare_model"]["groups"]["natural104"]["metrics"]
    if report["complete"]:
        rl = "; ".join(f"seed{name[-2:]} {natural[name]['normalized_em_count']:g}/104" for name in ("grpo42", "grpo43"))
        lines.append(f"- 在固定0.8B模型与现有128条训练材料上完成继续SFT及在线GRPO双seed比较；已暴露104道自然多跳题上，裸模型归一化EM {bare_natural['normalized_em_count']:g}/104，AM {baseline['normalized_em_count']:g}/104，SFT {natural['sft42']['normalized_em_count']:g}/104，GRPO {rl}。")
    else:
        lines.append("- 实验尚未完成；不生成最终模型提升表述，实际已完成分支见上表。")
    identity = report["efficiency"]["answer_comparison"]
    es, eb = efficiency["serial"], efficiency["batch"]
    lines.append(f"- 实现在线真实搜索采样及训练/评价账本；固定AM的串行/batch4实测吞吐{fmt(es['questions_per_minute'])}→{fmt(eb['questions_per_minute'])}题/分钟，吞吐比{fmt(efficiency['batch_throughput_ratio'])}，峰值reserved显存{fmt(es['peak_reserved_gib'])}→{fmt(eb['peak_reserved_gib'])}GiB；{identity['pairs']}个配对题答案文本变化{len(efficiency['exact_answer_changes'])}题。")
    lines += ["", "逐题新增/丢失与配对F1区间见summary.json。保留全部结果，不按评价分选择seed或回改训练配方。"]
    if report.get("comprehensive_metrics"):
        links = []
        if comprehensive_markdown.exists():
            links.append("[完整机械质量、效率、检索覆盖与训练指标](comprehensive-metrics.md)")
        if comprehensive_json.exists():
            links.append("[机器可读指标](comprehensive-metrics.json)")
        lines += ["", "；".join(links) + "。这些指标由现有账本派生，不增加模型调用或语义审核。"]
    if warnings:
        lines += ["", "## 记录限制", ""] + [f"- {warning}" for warning in warnings]
    (run_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error("run_dir must be an existing directory")
    report = summarize(args.run_dir, plot=not args.no_plot)
    print(json.dumps(dict(complete=report["complete"], report=str(args.run_dir / "REPORT.md"),
                          warnings=report["warnings"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
