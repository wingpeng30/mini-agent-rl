"""Persist a dependency graph and select evidence-accepted CPU data work.

This is a local selector, not a worker daemon or a semantic oracle. The native
Codex Goal executes next-prompt.md, audits the result, and invokes this selector.
No API client, model loader, trainer, or remote command is imported or executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v124-forensic/goal-v128"
FORBIDDEN = ("final-test", "final_test", "finaltest", "independent100",
             "independent-100", "independent_100", "holdout100", "holdout-100",
             "independent 100", "holdout 100")
ALLOWED_EXACT = frozenset({
    "reports/v116-sft-preflight.json",
    "reports/v117-local-preflight.json",
    "reports/v122-p3-decision.json",
    "reports/v122-internal-preflight.json",
    "reports/v122-internal-scope-impossibility.json",
    "reports/v122-train-preflight.json",
})
ARCHIVE_SUFFIXES = frozenset({
    ".zip", ".7z", ".rar", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".zst",
})
POLICY = """目标：修复数据和审核流程，交付达到既定实验准备门槛的数据。最终入选集没有已知未解决的语义矛盾、格式错误、争议票或单审标签；不得声称数学意义上保证所有语义无误。
原始数据和旧票不可覆盖。隔离库存逐项处理：有依据修复为新版本；无法修复者保留理由并排除，不能要求所有坏数据被强制救回。修复/排除账目完整不等于训练集达标。
禁止扫描或读取 final-test 与独立100题，禁止通过别名或压缩包绕过。仅访问各任务明确输入白名单；白名单需记录规范化路径与来源。
禁止本地/远端模型生成、SFT、GRPO、smoke、GPU实例、电源操作及自动训练入口。training_allowed 永远 false；data_ready_for_experiment 只有最终门禁真通过才可 true。不恢复费用记账；不打印、归档或提交凭据。
沿用单次候选池/语料冻结及最多100新增候选、5000来源初筛的已授权上限，先核对已经使用的预算；不得自动新建第二池、换种子、降风险配额或重复审核有效否决。若需新池，先做具体设计和不足证书，再等待用户对新范围的明确授权。
仅允许已授权的 DeepSeek 和隔离 Luna 独立审查；最多同时两个 GPT-6 Luna 子agent且仅必要时使用，审查者不可参与开发。旧输入完全等价且票有效才复用；重写问题/证据/截断均重建绑定。主agent判断不冒充第二票。未知调用结果先查日志及远端实际状态，不自动重发；技术更正最多一次，语义否决不重试求通过。
无新证据时不重新打包或重跑相同审核。每轮写入具体变化、证据与尚缺项；待审、失败、阻断与通过必须区分。不得以填写验收布尔值代替阅读原始材料。
本文件的调度只校验验收记录完整性、哈希及依赖；语义判断必须由执行者逐项审计真实证据，双审票也须逐状态检验。"""


def node(id, title, deps, work, criteria):
    return {"id": id, "title": title, "depends_on": deps, "work": work,
            "criteria": {f"{id}.{i}": text for i, text in enumerate(criteria, 1)}}


PLAN = {"version": "v128-goal-dag-v1", "policy": POLICY, "nodes": [
    node("G0", "可信基线与历史表述纠正", [],
         "从明确原始行、有效修订及引用重算当前合并量、严格配对、票据覆盖、隔离数量和内部集缺口。核查此前162内嵌动作的统计、重复和生成来源，特别纠正临时脚本错误；不将记录中的动作称为已证明由模型产生。",
         ["保存输入清单/哈希、原始行复算结果与旧报告差异。",
          "明确正式候选、待审候选、单审、双审与条件上界；记录内嵌动作来源和去重仍待查项。",
          "给出实验准备未通过的逐条原因，train/data-ready标志不混用。"]),
    node("G1", "修正审核与门禁实现", ["G0"],
         "检查引用逐文档及时点绑定、多文档关系推理、无文档搜索、风险交集、严格配对、格式与token、审核恢复去重。修正主审修订应用、重复ID/同题异ID、旧票等价证明和硬编码通过标志；冻结审核契约后再送审。",
         ["相关缺陷有原始失败例及修复后的有意义回归测试。",
          "格式/引用/token/隔离检查实际读取最终输入；封存访问在读取前拒绝。",
          "风险配额只计有效双审交集，临时标签不冒充审核标签；审核契约版本可追溯。"]),
    node("G2", "全量隔离修订与来源审计", ["G1"],
         "对120隔离状态、107个v117修订状态及35未成对问题按ID建立一张去重处置表。复核v126/v127已有修订，查162条内嵌搜索的构造来源及原始消息，过滤已审核或重叠状态。修订不得把后验答案/标题加入先前查询，来源归因改题需明确任务变化。保留训练/内部用途边界，不扩池。",
         ["三个库存逐ID覆盖并去重，修复候选/无法修复排除/待证据原因完整，不丢孤立状态。",
          "所有修订候选有新输入哈希、前后差异、真实来源、动作时点、检索/截断/token重放及重复检查。",
          "未成对题无现有合法轨迹时保留缺口；gold条件合成不冒充模型轨迹或独立批准。"]),
    node("G3", "恢复独立审核通道", ["G1"],
         "核对已授权AutoDL连接与审核环境（仅无卡/CPU任务），检查已有请求日志及身份。先校验受限凭据配置是否可用，不在文件中保存秘密。现有结果未知请求只对账；无认证条件时给出阻断与恢复条件，继续其他独立任务。",
         ["确认已授权审核者身份、服务可达性与输入协议版本；没有擅自切换模型。",
          "同输入不重复请求、技术更正上限、逐条保存和网络结果未知停止机制有验证证据。",
          "Luna隔离规则和来源白名单已就绪，审核者互不看到对方票及配额。"]),
    node("G4", "达到新内部验收集门槛", ["G2", "G3"],
         "先重新验证v123不足证书适用范围：是当前冻结票据/规则上界，不是所有语义可能性定理。审计契约错误仅影响依赖结论，禁止重复有效否决求通过。核对一次性建池预算与授权；既有池不可达且预算已耗尽时阻断，提交具体新方案等授权，不自行开始第二池。若有合法可行方案，冻结一次语料/输入后双审，按固定顺序停止及选择。",
         ["独立复算50题/100状态/50严格配对，四类风险各至少5状态且来自双审交集。",
          "新集未用于开发调参，整体候选与训练/历史暴露ID和规范化题面隔离，引用和检索token均复核。",
          "范围对象集合或空间比较边界真实，预算/单次冻结/复用票据等价性可核验。"]),
    node("G5", "训练候选有效独立双审", ["G2", "G3", "G4"],
         "仅在新内部集通过后完成训练补充双审。优先复用真实等价票据，对改变的v117状态、合法抽取状态和现有v122候选按完整题组每批最多20状态审核；范围沿用既有池，不新增训练来源。分歧/无效/单审全部排除训练候选。达到门槛或证明剩余不可达后停止请求。",
         ["每个入选动作都有两名获授权独立审核者的有效批准及输入哈希绑定；重复请求为零或有明确技术更正记录。",
          "最终问题/实体/时间/范围/单位及答案证据链无已知未解决矛盾；不将格式通过当语义通过。",
          "所有票据有真实身份、来源及可复算引用，保留否决/分歧和未审完整账目。"]),
    node("G6", "统一数据集独立复算", ["G4", "G5"],
         "用独立验证器从最终行和原始票据重算既定395状态、211题、100严格配对门槛、45%-55%答案比例、至少40%非空搜索与至多10%空搜索、四类训练风险至少20状态。冻结前核对原始规范的精确计数口径，不在此改门槛。证明25个新完整配对及20个新关系风险。预检不启动模型或训练。",
         ["最终数量/问题/严格配对/动作比例/风险全部达到原门槛，数字由实际行与票据重算。",
          "JSON/action schema、token边界、输入唯一性、引用、实际检索上下文及配对连续性全部通过。",
          "训练和完整新内部候选双向隔离，最终输入零争议/零无效/零单审，内部验收门槛有效。"]),
    node("G7", "最终验收与可恢复交付", ["G6"],
         "逐项复核用户三个基本要求和全部门禁。从文件重建而非复述旧摘要；核对公开代码与私有资产、最终归档成员和哈希，记录来源缺陷及仍不确定之处。data_ready_for_experiment仅在全部门槛真通过时设true；training_allowed仍false。此时才可标记主Goal完成，不运行实验。",
         ["对无已知矛盾、全量隔离处理、实验数据门槛和零格式错误逐项给出权威证据。",
          "数据导出、审核票据、验收代码、使用说明和本地备份均可重建并哈希一致，无封存/秘密文件。",
          "data_ready_for_experiment为true且training_allowed为false，所有前置验收当前仍有效。"]),
]}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def checked_path(relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or ":" in relative:
        raise ValueError("invalid_relative_path")
    if any(part.casefold().endswith(tuple(ARCHIVE_SUFFIXES)) for part in p.parts):
        raise ValueError("archive_paths_rejected_before_read")
    resolved = (ROOT / p).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("outside_workspace")
    resolved_text = resolved.as_posix().casefold()
    if any(t in resolved_text for t in FORBIDDEN):
        raise ValueError("sealed_path_rejected_before_read")
    normalized = p.as_posix().casefold()
    resolved_relative = resolved.relative_to(ROOT.resolve()).as_posix().casefold()
    allowed_prefixes = ("docs/", "scripts/", "tests/", "data/evidence-sft-v124-forensic/",
                        "data/evidence-sft-v123/internal/insufficiency-certificate.json")
    requested_allowed = normalized.startswith(allowed_prefixes) or normalized in ALLOWED_EXACT
    resolved_allowed = (resolved_relative.startswith(allowed_prefixes)
                        or resolved_relative in ALLOWED_EXACT)
    if not requested_allowed or not resolved_allowed:
        raise ValueError("evidence_path_not_allowlisted")
    return resolved


def read_bytes(relative):
    return checked_path(relative).read_bytes()


def verify_receipt(n, r, plan_hash, deps, reader=read_bytes):
    if r.get("node") != n["id"] or r.get("plan_sha256") != plan_hash:
        raise ValueError("receipt_plan_or_node_mismatch")
    if r.get("dependencies") != deps:
        raise ValueError("dependency_receipt_changed")
    if r.get("status") not in {"passed", "failed", "blocked"} or not r.get("auditor"):
        raise ValueError("missing_status_or_auditor")
    evidence = r.get("evidence", [])
    if not evidence:
        raise ValueError("no_evidence")
    for e in evidence:
        if digest(reader(e["path"])) != e["sha256"]:
            raise ValueError("evidence_changed")
    if r["status"] == "passed":
        checks = r.get("criteria", {})
        if set(checks) != set(n["criteria"]):
            raise ValueError("incomplete_acceptance_criteria")
        for check in checks.values():
            refs = check.get("evidence", [])
            if check.get("result") != "passed" or not check.get("explanation") or not refs:
                raise ValueError("criterion_not_proven")
            if any(type(i) is not int or not 0 <= i < len(evidence) for i in refs):
                raise ValueError("invalid_evidence_reference")
    elif not r.get("reason") or not r.get("resume_condition"):
        raise ValueError("failure_requires_reason_and_resume_condition")


def evaluate(plan, plan_hash, receipts, reader=read_bytes):
    seen = set()
    for n in plan["nodes"]:
        if n["id"] in seen or not set(n["depends_on"]) <= seen or not n["criteria"]:
            raise ValueError("invalid_or_non_topological_dag")
        seen.add(n["id"])
    states, accepted_hashes = {}, {}
    for n in plan["nodes"]:
        id = n["id"]
        if any(states.get(d) != "passed" for d in n["depends_on"]):
            states[id] = "waiting_dependencies"
            continue
        deps = {d: accepted_hashes[d] for d in n["depends_on"]}
        raw = receipts.get(id)
        if raw is None:
            states[id] = "ready"
            continue
        try:
            r = json.loads(raw)
            verify_receipt(n, r, plan_hash, deps, reader)
            states[id] = r["status"]
            if r["status"] == "passed":
                accepted_hashes[id] = digest(raw)
        except (KeyError, TypeError, ValueError, OSError):
            states[id] = "stale_or_invalid"
    eligible = [n["id"] for n in plan["nodes"]
                if states[n["id"]] in {"ready", "stale_or_invalid"}]
    return states, accepted_hashes, (eligible[0] if eligible else None)


def write(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "next", "accept"])
    parser.add_argument("--receipt")
    args = parser.parse_args()
    path = BASE / "plan.json"
    if args.command == "init":
        if path.exists() and path.read_bytes() != encode(PLAN):
            raise ValueError("existing_plan_differs_version_plan_explicitly")
        write(path, encode(PLAN))
    plan_blob = path.read_bytes()
    plan = json.loads(plan_blob)
    plan_hash = digest(plan_blob)
    receipts = {n["id"]: (BASE / "receipts" / (n["id"] + ".json")).read_bytes()
                for n in plan["nodes"] if (BASE / "receipts" / (n["id"] + ".json")).exists()}
    states, hashes, selected = evaluate(plan, plan_hash, receipts)
    if args.command == "accept":
        candidate = json.loads(read_bytes(args.receipt))
        n = next(n for n in plan["nodes"] if n["id"] == candidate["node"])
        if any(states[d] != "passed" for d in n["depends_on"]):
            raise ValueError("dependencies_not_passed")
        verify_receipt(n, candidate, plan_hash, {d: hashes[d] for d in n["depends_on"]})
        blob = encode(candidate)
        # Every version is retained before replacing the current receipt.
        write(BASE / "history" / (n["id"] + "-" + digest(blob) + ".json"), blob)
        write(BASE / "receipts" / (n["id"] + ".json"), blob)
        receipts[n["id"]] = blob
        states, hashes, selected = evaluate(plan, plan_hash, receipts)
    summary = {"plan_sha256": plan_hash, "states": states, "next": selected,
               "all_receipts_accepted": all(s == "passed" for s in states.values()),
               "training_allowed": False}
    write(BASE / "status.json", encode(summary))
    if selected:
        n = next(n for n in plan["nodes"] if n["id"] == selected)
        prompt = (f"当前子目标 {selected}：{n['title']}\n\n{plan['policy']}\n\n"
                  f"工作：{n['work']}\n\n验收条件：\n" +
                  "\n".join(f"- {k}: {v}" for k, v in n["criteria"].items()) +
                  "\n\n执行后逐项核验，保存验收票（node, plan_sha256, dependencies, auditor, status, "
                  "evidence:[{path,sha256}], criteria:{条件ID:{result,explanation,evidence:[索引]}}）。"
                  "失败或阻断则填reason和resume_condition。验收票不得替代语义双审。\n"
                  "运行 python -m scripts.v128_goal_workflow accept --receipt <仓库内相对路径>；"
                  "随后重读next-prompt.md，继续新解锁任务。真正外部阻断时处理其他ready节点；"
                  "无可做节点则报告具体缺口并保持主目标未完成，不空转。\n\n"
                  f"plan_sha256={plan_hash}\ndependencies={json.dumps({d: hashes[d] for d in n['depends_on']})}\n")
    else:
        prompt = ("所有子目标验收票当前有效。执行独立最终完成审计，真实达标后才可完成主Goal。"
                  if summary["all_receipts_accepted"] else
                  "没有可执行子目标。核对阻断票和真实外部状态，不重发未知API请求，不降低门槛。")
    write(BASE / "next-prompt.md", prompt.encode("utf-8"))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
