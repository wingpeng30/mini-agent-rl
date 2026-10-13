"""Freeze the one-pass semantic adjudication and final Goal AK report."""
from __future__ import annotations

import hashlib
import json
import statistics
import time
import shutil
from collections import Counter
from pathlib import Path

import run_v133_goal_aj_relation_prompt as aj

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ak-single-action-v1"
AJ_OUT = ROOT / "runs/v133-goal-aj-relation-prompt-v1"

# Case-ID keyed rulings are the result of the frozen, one-pass review. No case is
# inferred from row position. All descriptions below refer to saved AK outputs.
B_CATEGORY = {
    "musique:2hop__864101_135373": ("success", "The returned Bering Sea passage states it is named for Vitus Bering; the first result links St. Matthew Island to the Bering Sea."),
    "musique:2hop__478978_124498": ("success", "The Je dis aime result identifies Matthieu Chedid; the returned Matthieu Chedid body says he is the son of Louis Chedid."),
    "musique:2hop__862994_69048": ("factual_error", "Gloria Kemasuode's body says Delta State; the model applies the chief judge of Rivers State, Adama Lamikanra, to the wrong jurisdiction."),
    "musique:2hop__829358_754719": ("factual_error", "The Darley body says he was born in Pedro Leopoldo, Brazil; the answer Minsk Region comes from an unrelated retrieved page."),
    "musique:2hop__789565_416401": ("factual_error", "The visible biography makes Lady Li Wang Jipeng's cousin, not the child of Wang Yanjun; this confuses the queried relation and target person."),
    "musique:2hop__506035_325512": ("factual_error", "The body says Larkin I. Smith was born in Poplarville, in Pearl River County; Harrison County refers to his later police and sheriff work."),
    "musique:2hop__337223_56873": ("no_answer_budget_exhausted", "Three searches did not retrieve the required Nottingham Forest cup-win edge; the fourth action used the final call to search again."),
    "musique:2hop__796221_18803": ("no_answer_repeated_action", "The action repeated a query and ended without an answer; retain the prior missing-information status."),
    "musique:2hop__151750_141308": ("success", "The same returned source and answer identify Apple Corps Ltd; the suffix is an accepted organization-name variant."),
    "musique:2hop__418360_67370": ("no_answer_budget_exhausted", "The necessary 2018 schedule appeared in returned evidence, but the agent continued searching through its remaining calls and stopped without answering."),
    "musique:2hop__96194_78303": ("factual_error", "Chuck Norris is identified as Cordell Walker in the series cast; Simon Moon is the villain in the film Hero and the Terror."),
    "musique:2hop__703493_162341": ("incomplete_required_qualifier", "The answer says Academy Award but omits that the recognition was a nomination for Best Supporting Actor."),
    "musique:2hop__158275_35739": ("success", "The returned browser passage directly binds Firefox, its Windows 8 version, and September 2013."),
    "musique:2hop__435087_55984": ("success", "The returned album/song sources support Brandi Carlile performing The Story and Phil Hanseroth writing it."),
    "musique:2hop__109242_113442": ("success", "The returned painting and Millais biography support the creator chain and death date."),
    "musique:2hop__69765_43945": ("success", "The returned sources identify Thomas Paine as author and directly state the reason the work mattered."),
    "2wiki:982f8e440bdb11eba7f7acde48001122": ("success", "The returned publisher and Sigma Xi sources support the founding year 1886."),
    "2wiki:67ecf3100bdc11eba7f7acde48001122": ("success", "The returned film and director biography support Yurek Bogayevicz and Poznań."),
    "2wiki:3a3d60020bdd11eba7f7acde48001122": ("no_answer_invalid_action", "The first-action protocol still received an unparseable completion; no answer was produced."),
    "2wiki:ef9d40a40bdb11eba7f7acde48001122": ("success", "American is an equivalent nationality description for United States."),
    "2wiki:5dac1e8e0bdd11eba7f7acde48001122": ("success", "New York, New York is a supported equivalent of New York."),
    "2wiki:1dd33f080bde11eba7f7acde48001122": ("no_answer_repeated_action", "The answer-time input contained both required edges, but the model repeated the same search rather than answering."),
    "2wiki:764039340bde11eba7f7acde48001122": ("unsupported_by_visible_evidence", "The final answer Edinburgh is not supported by the visible evidence for the father's place of death; the required edge was not retrieved."),
    "2wiki:7bda387a0bdb11eba7f7acde48001122": ("success", "The returned father and birth-date sources support 24 August 1921."),
    "2wiki:f299dd600bdd11eba7f7acde48001122": ("factual_error", "The question targets the 4th Duke, who died in 1575; the answer 20 January 1544 belongs to the 2nd Duke, a different person."),
    "2wiki:49d965aa0bde11eba7f7acde48001122": ("success", "The returned Yoyoy Villame body states he died at Las Piñas Medical Center in Manila; the answer gives the supported place more specifically than the short gold label."),
    "2wiki:ee3233920bda11eba7f7acde48001122": ("success", "The returned father relation and Francesco I Crispo biography support 1397."),
    "musique:2hop__498251_126089": ("unsupported_by_visible_evidence", "The visible source says Laren is in the Gooi region; that geography does not identify who was in charge. The authority relation was not retrieved."),
    "musique:2hop__530579_13529": ("success", "The returned Diego Sinagra and Barcelona passages support the father relation and June 1982 signing date."),
}

A_CORRECTIONS = {
    "musique:2hop__506035_325512": ("success", "incomplete_required_qualifier", "The answer 'Pearl River' omits the required county entity type; the question asks which county, and the source names Pearl River County."),
    "musique:2hop__703493_162341": ("success", "incomplete_required_qualifier", "The answer 'Academy Award' omits nomination status and Best Supporting Actor, which are needed to identify the reported recognition."),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_lines(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    status = aj.read(OUT / "status.json")
    if status.get("status") != "stage_complete" or status.get("stage") != "report-pre-adjudication":
        raise RuntimeError("AK trajectory/report stages are not complete")
    for name in ("REPORT.md", "final-ledger.jsonl", "source-review-ak.jsonl", "decision.json"):
        if (OUT / name).exists():
            raise FileExistsError(OUT / name)

    a, b = load_lines(OUT / "trajectory-A.jsonl"), load_lines(OUT / "trajectory-B.jsonl")
    if len(a) != 64 or len(b) != 64:
        raise RuntimeError("expected complete A/B trajectories")
    admission = {r["case_id"]: r for r in load_lines(aj.ADMISSION) if r.get("admission") == "admitted"}
    old_reviews = {r["case_id"]: r for r in load_lines(aj.PRIOR_REVIEWS)
        if r.get("split") == "goal_z_dev" and r.get("input_and_answer_identical_to_prior") is True}
    if len(admission) != 29 or set(B_CATEGORY) != set(admission) or len(old_reviews.keys() & admission.keys()) != 29:
        raise RuntimeError("admitted cases or exact-replay historical reviews changed")
    shutil.copyfile(Path(__file__), OUT / "source-finalizer-ak.py")
    aa = {r["case_id"]: r for r in a}; bb = {r["case_id"]: r for r in b}
    edges = {}
    for e in load_lines(aj.EDGES):
        if e.get("status") == "pass":
            edges.setdefault(e["case_id"], []).append(e)
    review_rows, ledger = [], []
    for case_id in [r["case_id"] for r in a]:
        ar, br = aa[case_id], bb[case_id]
        old = old_reviews.get(case_id)
        admission_status = case_id in admission
        a_cat = old.get("category") if old else "unknown_unconfirmed"
        a_reason = "Reused exact Goal Z/A input-and-answer source review; A input/action replay matched Goal AJ." if old else "Unconfirmed source status; excluded from semantic error totals."
        if case_id in A_CORRECTIONS:
            prior, revised, correction_reason = A_CORRECTIONS[case_id]
            if a_cat != prior:
                raise RuntimeError(f"unexpected prior A category for {case_id}: {a_cat}")
            a_cat, a_reason = revised, correction_reason
        b_cat, b_reason = B_CATEGORY.get(case_id, ("unknown_unconfirmed", "Not in the source-admitted layer."))
        if admission_status:
            review_rows.append({"case_id": case_id, "question": ar["question"], "official_answer": ar["official_answer"],
                "A_answer": ar.get("answer"), "A_category": a_cat, "A_basis": a_reason,
                "B_answer": br.get("answer"), "B_category": b_cat, "B_basis": b_reason,
                "B_action_reason": br["reason"], "B_actual_queries": [x.get("query") for x in br["trace"] if x.get("query")],
                "necessary_edges": [[e.get("subject"), e.get("relation"), e.get("object")] for e in edges.get(case_id, [])],
                "training_use_forbidden": True, "review_version": "AK primary one-pass"})
        ledger.append({"case_id": case_id, "question": ar["question"], "admission": "admitted" if admission_status else "unconfirmed",
            "A_answer": ar.get("answer"), "A_em": ar["em"], "A_category": a_cat,
            "A_calls": ar["model_calls"], "A_searches": ar["searches"],
            "B_answer": br.get("answer"), "B_em": br["em"], "B_category": b_cat,
            "B_reason": br["reason"], "B_calls": br["model_calls"], "B_searches": br["searches"],
            "B_queries": [x.get("query") for x in br["trace"] if x.get("query")],
            "necessary_edges": [[e.get("subject"), e.get("relation"), e.get("object")] for e in edges.get(case_id, [])],
            "B_review_basis": b_reason if admission_status else "unconfirmed; excluded from semantic error totals"})

    a_cat_counts = Counter(r["A_category"] for r in review_rows)
    b_cat_counts = Counter(r["B_category"] for r in review_rows)
    supported = {"success"}
    a_success = {r["case_id"] for r in review_rows if r["A_category"] in supported}
    b_success = {r["case_id"] for r in review_rows if r["B_category"] in supported}
    gains, losses = sorted(b_success-a_success), sorted(a_success-b_success)
    if gains != ["musique:2hop__864101_135373"]:
        raise RuntimeError(f"unexpected one-pass supported-success gains: {gains}")
    protected = {r["case_id"] for r in load_lines(OUT / "protection-frozen.jsonl")}
    a_map, b_map = aa, bb
    protected_lost = sorted(c for c in protected if a_map[c]["em"] and not b_map[c]["em"])
    metrics = {}
    context = {}
    for arm, data in (("A", a), ("B", b)):
        calls = [c for row in data for c in row["trace"]]
        usage = [c.get("context_usage", {}) for c in calls]
        metrics[arm] = {"cases": 64, "EM": sum(x["em"] for x in data), "mean_F1": statistics.mean(x["f1"] for x in data),
            "parse_invalid_cases": sum(x["reason"] == "invalid_action" for x in data),
            "repeated_action_cases": sum(x["reason"] == "repeated_action" for x in data),
            "invalid_or_repeated_cases": sum(x["reason"] in ("invalid_action", "repeated_action", "tool_error") for x in data),
            "model_calls": sum(x["model_calls"] for x in data), "searches": sum(x["searches"] for x in data),
            "mean_searches_per_case": statistics.mean(x["searches"] for x in data),
            "mean_generation_seconds": statistics.mean(c["model_seconds"] for c in calls),
            "mean_question_seconds": statistics.mean(x["elapsed_seconds"] for x in data)}
        context[arm] = {"calls": len(calls), "max_input_tokens": max(len(c.get("prompt_token_ids", [])) for c in calls),
            "input_prompt_truncated_calls": sum(bool(u.get("prompt_truncated")) for u in usage),
            "lossy_body_trimmed_calls": sum(bool(u.get("lossy_body_trimmed")) for u in usage),
            "documents_deleted": sum(len(u.get("documents_deleted", [])) if isinstance(u.get("documents_deleted", 0), list)
                                      else int(u.get("documents_deleted", 0) or 0) for u in usage)}

    prior_cats = Counter(old_reviews[c].get("category") for c in admission)
    errors_a = a_cat_counts["factual_error"] + a_cat_counts["unsupported_by_visible_evidence"]
    errors_b = b_cat_counts["factual_error"] + b_cat_counts["unsupported_by_visible_evidence"]
    counts = {"engineering_generations": 16, "A_trajectory_generations": metrics["A"]["model_calls"],
        "B_trajectory_generations": metrics["B"]["model_calls"], "total_generations": 16+metrics["A"]["model_calls"]+metrics["B"]["model_calls"],
        "searches": metrics["A"]["searches"]+metrics["B"]["searches"], "training_updates": 0, "new_confirmation_calls": 0}
    gates = {"two_new_supported_cases": len(gains) >= 2, "admitted_supported_net_plus_two": len(b_success)-len(a_success) >= 2,
        "factual_plus_unsupported_not_higher": errors_b <= errors_a,
        "invalid_or_repeated_not_higher": metrics["B"]["invalid_or_repeated_cases"] <= metrics["A"]["invalid_or_repeated_cases"],
        "protected_losses_at_most_one": len(protected_lost) <= 1,
        "passed": len(gains) >= 2 and len(b_success)-len(a_success) >= 2 and errors_b <= errors_a and
                  metrics["B"]["invalid_or_repeated_cases"] <= metrics["A"]["invalid_or_repeated_cases"] and len(protected_lost) <= 1}
    revisions = [{"case_id": k, "prior_A_category": v[0], "revised_A_category": v[1], "reason": v[2]}
        for k,v in A_CORRECTIONS.items()]
    result = {"goal": "AK", "status": "complete", "scope": "Goal Z seed42 fixed 64-question development behavior experiment; no model release",
        "model_sha256": aj.EXPECTED_ADAPTER, "metrics": metrics, "admitted_cases": 29, "unconfirmed_cases": 35,
        "admitted_semantic_categories_A": dict(a_cat_counts), "admitted_semantic_categories_B": dict(b_cat_counts),
        "admitted_error_plus_unsupported": {"A": errors_a, "B": errors_b},
        "admitted_supported_success": {"A": len(a_success), "B": len(b_success), "net": len(b_success)-len(a_success),
            "gained": gains, "lost": losses}, "all64_em_gained": sorted(c for c in aa if bb[c]["em"] and not aa[c]["em"]),
        "all64_em_lost": sorted(c for c in aa if aa[c]["em"] and not bb[c]["em"]),
        "entity_relation_mismatch_cases": ["musique:2hop__506035_325512", "musique:2hop__789565_416401",
            "musique:2hop__862994_69048", "musique:2hop__829358_754719", "musique:2hop__96194_78303",
            "2wiki:f299dd600bdd11eba7f7acde48001122", "musique:2hop__498251_126089"],
        "answer_with_complete_chain_visible": {"A": a_cat_counts["success"]+a_cat_counts["factual_error"],
            "B": b_cat_counts["success"]+b_cat_counts["factual_error"], "denominator": 29,
            "definition": "one-pass source review of the actual Goal AK evidence chain; unanswered cases are not counted as answered"},
        "context_budget": context, "protected_cases": sorted(protected), "protected_EM_losses": protected_lost,
        "first_action_protocol": {"offline_generations_reviewed": 305, "engineering_generations": 16,
            "engineering_reconciliation": aj.read(OUT / "engineering-reconciliation.json"),
            "B_parse_invalid_cases": metrics["B"]["parse_invalid_cases"]},
        "counts": counts, "exploration_gate": gates, "prior_A_review_categories_before_explicit_corrections": dict(prior_cats),
        "explicit_review_revisions": revisions,
        "semantic_review_limits": "All 29 admitted questions reviewed once by case ID; all 35 unconfirmed cases remain excluded. The source ledger has one known narrow-quote defect in the Bering Sea edge; the actual retrieved Bering Sea body directly supports the corrected relation.",
        "next_route": "Stop relation-prompt route. Prepare a separate Goal Z evidence-retention comparison; this run does not begin it.",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    result["finalizer_sha256"] = digest(Path(__file__))
    for row in review_rows:
        aj.append(OUT / "source-review-ak.jsonl", row)
    for row in ledger:
        aj.append(OUT / "final-ledger.jsonl", row)
    aj.write(OUT / "decision.json", result)
    m = metrics
    lines = ["# Goal AK：首个JSON动作边界与关系提示A/B", "",
        "## 结论", "",
        f"单动作输出修复了AJ的大部分多JSON解析故障，但关系提示没有带来搜索产品收益：A为{m['A']['EM']}/64、B为{m['B']['EM']}/64，B少4题。29题来源准入层中，A有{len(a_success)}题证据支持成功，B有{len(b_success)}题；B新增1题、丢失{len(losses)}题。",
        "", "保留关系提示路线的门槛未通过：新支持案例不到2题，准入层支持成功未净增2题，B无效或重复动作6题高于A的4题。B还丢失1/7条连续搜索保护题。停止提示蒸馏路线。", "",
        "## 固定64题结果", "", "| 指标 | A：direct-v090 | B：relation-v1 |", "|---|---:|---:|"]
    for key,label in (("EM","EM / 64"),("mean_F1","平均F1"),("parse_invalid_cases","解析无效题"),
        ("repeated_action_cases","重复动作题"),("invalid_or_repeated_cases","无效或重复动作题"),
        ("model_calls","模型调用"),("searches","搜索数"),("mean_searches_per_case","每题搜索"),
        ("mean_generation_seconds","每次生成秒"),("mean_question_seconds","每题秒")):
        x,y=m["A"][key],m["B"][key]
        lines.append(f"| {label} | {x:.3f} | {y:.3f} |" if isinstance(x,float) else f"| {label} | {x} | {y} |")
    lines += ["", "## 实际变化", "",
        f"来源准入层裁定：A支持成功{a_cat_counts['success']}/29，B支持成功{b_cat_counts['success']}/29；A事实错误与证据不支持合计{errors_a}，B为{errors_b}。可见完整链上的回答中，A为{a_cat_counts['success']+a_cat_counts['factual_error']}/29，B为{b_cat_counts['success']+b_cat_counts['factual_error']}/29。另有B实体／关系错配7题、必要限定不完整1题、无答案5题。逐题理由见`source-review-ak.jsonl`和`final-ledger.jsonl`。", "",
        f"唯一新增支持成功是`musique:2hop__864101_135373`：B检索到Bering Sea正文中‘named for Vitus Bering’，并从St. Matthew Island检索到该海域，回答由‘Bering’修正为‘Vitus Bering’。B失去3条A侧支持成功：`musique:2hop__418360_67370`、`2wiki:3a3d60020bdd11eba7f7acde48001122`、`2wiki:1dd33f080bde11eba7f7acde48001122`。Larkin案例的A答案‘Pearl River’另修订为限定不完整，B则错误回答Harrison County。", "",
        f"按账本分解，A有{len(result['all64_em_gained'])}题新增EM、{len(result['all64_em_lost'])}题丢失EM；35题来源未确认，不能加入语义错误分母。", "",
        "## 机制与输入", "",
        f"首个JSON边界把AJ的多对象输出转成单动作，解析无效从AJ的58题降到本轮B的{m['B']['parse_invalid_cases']}题；工程检查16/16从保存输出核销通过，原比较器类型失败仍保留。动作边界本身没有改善充分性判断。", "",
        f"B搜索{m['B']['searches']}次、A搜索{m['A']['searches']}次，多37次；B仍有{m['B']['invalid_or_repeated_cases']}题出现无效或重复动作。准入层有6题在完整必要链可见时答错，主要错在实体/关系绑定：Harrison County被套到出生地、把Vitus的全名压成错误实体、把电影反派当成演员角色，以及把第2代与第4代Medinaceli公爵混淆。", "",
        f"B的输入预算压力明显增加：系统提示从21增至74 token；A正文裁剪4次、删除文档0份，B正文裁剪{context['B']['lossy_body_trimmed_calls']}次、删除文档{context['B']['documents_deleted']}份，13次输入触发预算压缩；B最大2045/2048 token。上下文变化可能参与了退化，不能把差异全归为关系判断。", "",
        "## 保护、修订与下一步", "",
        f"连续搜索保护集冻结7题，B丢失{len(protected_lost)}题：{', '.join(protected_lost)}。A来源审核有两条明确修订：Pearl River缺少County限定，Academy Award缺少提名和Best Supporting Actor限定；理由和旧值均另存。", "",
        "下一步按失败分支只准备固定Goal Z的证据保留对照，保持direct-v090、检索器和搜索预算。利用本轮裁剪与删除账本及Chelsea哨兵，先验证保留完整正文能否恢复；本报告不启动后续GPU实验。", "",
        f"本轮共{counts['total_generations']}次生成、{counts['searches']}次真实搜索，训练更新0次；不运行新确认集、不发布模型或提示。详见`decision.json`。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    decision_path = OUT / "decision.json"
    checks = {name: digest(OUT / name) for name in ("manifest.json", "REPORT.md", "decision.json", "source-finalizer-ak.py",
        "final-ledger.jsonl", "source-review-ak.jsonl", "trajectory-A.jsonl", "trajectory-B.jsonl",
        "engineering-reconciliation.json", "failure.json")}
    aj.write(OUT / "checksums-final.json", checks)
    aj.write(OUT / "status.json", {"status": "complete", "stage": "finalized", "gpu_generations": counts["total_generations"],
        "searches": counts["searches"], "training_updates": 0, "failure_record_preserved": True})
    print(json.dumps({"A_EM": m["A"]["EM"], "B_EM": m["B"]["EM"], "A_supported": len(a_success),
        "B_supported": len(b_success), "gained": gains, "lost": losses, "errors_A": errors_a, "errors_B": errors_b,
        "gate": gates, "generations": counts["total_generations"], "searches": counts["searches"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
