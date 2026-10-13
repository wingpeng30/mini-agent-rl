"""One-pass, CPU-only Goal AO v2 scoring and delivery finalizer."""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ao-relation-contrast-sft-v2"
AM = ROOT / "runs/v133-goal-am-action-sft-v2"
AN = ROOT / "runs/v133-goal-an-hard-state-sft-v2"
AL = ROOT / "runs/v133-goal-al-wide-context-v1"

MANUAL = {
    ("musique:2hop__796221_18803", "42-A"): {
        "category": "factual_error", "entity_relation_mismatch": True,
        "reason": "The complete visible chain identifies Shell and its April 2010 Finland divestment process. The answer 'service stations' selects an unrelated Shell paragraph and does not answer the requested activity.",
        "review_basis": "The actual final input contains both required edges and the quoted Shell sentence; the answer instead matches a separate 2007 service-station sentence.",
    },
    ("musique:2hop__418360_67370", "*"): {
        "category": "unsupported_by_visible_evidence", "entity_relation_mismatch": False,
        "reason": "The actual final input establishes that Bill Maher authored the book, but contains no 2018 return-start date. The answer 'October 8, 2017' is not supported by the visible evidence; no external correction is used to call it a factual error.",
        "review_basis": "Only the first edge is cited in the actual input; the necessary Bill Maher schedule edge is absent.",
    },
    ("musique:2hop__96194_78303", "42-B"): {
        "category": "factual_error", "entity_relation_mismatch": True,
        "reason": "The actual input states that Hero and the Terror stars Chuck Norris and separately maps Chuck Norris to Cordell Walker in Walker, Texas Ranger. 'Simon Moon' is the film character, not his Walker character.",
        "review_basis": "Both necessary edges are visible in the actual answer input, including the role-specific television cast table and separate film-role sentence.",
    },
    ("musique:2hop__498251_126089", "*"): {
        "category": "unsupported_by_visible_evidence", "entity_relation_mismatch": False,
        "reason": "The actual input places Laren in North Holland and the Gooi region but contains no person-in-charge relationship. B's answer matches the official label, but that label is not evidence; neither answer is a supported success.",
        "review_basis": "The required North Holland office-holder edge is absent from the final input. A also answers with a region rather than a person.",
    },
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, items):
    Path(path).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")


def summary(model):
    flat = OUT / f"summary-{model}.json"
    return read(flat if flat.exists() else OUT / f"seed{model}/summary.json")


def metric(items):
    calls = [t for r in items for t in r["trace"]]
    return {
        "cases": len(items), "EM": sum(r["em"] for r in items),
        "F1": statistics.mean(r["f1"] for r in items),
        "action_problem_cases": sum(
            r["reason"] in ("invalid_action", "repeated_action") or any(
                t.get("action") == "search" and "observation" not in t for t in r["trace"]
            ) for r in items
        ),
        "parse_failures": sum(r["reason"] == "invalid_action" for r in items),
        "repeated_action_cases": sum(r["reason"] == "repeated_action" for r in items),
        "overbudget_search_attempts": sum(t.get("action") == "search" and t["step"] >= 3 for t in calls),
        "model_calls": len(calls), "searches": sum(r["searches"] for r in items),
        "max_input_tokens": max(len(t["prompt_token_ids"]) for t in calls),
        "max_action_tokens": max(len(t["action_token_ids"]) for t in calls),
        "length_stops": sum(t.get("usage", {}).get("generation_stop_reason") == "length" for t in calls),
        "body_trims": sum(t.get("usage", {}).get("lossy_body_trimmed", 0) for t in calls),
        "mean_question_seconds": statistics.mean(r["elapsed_seconds"] for r in items),
        "mean_generation_seconds": statistics.mean(t["model_seconds"] for t in calls),
    }


def main():
    packets = rows(OUT / "review-packets-seed42-corrected.jsonl")
    prior = {(r["case_id"], r["combination_sha256"]): r for r in rows(AM / "semantic-review-seed42.jsonl")}
    reviews = []
    primary = {}
    seen = {}
    for p in packets:
        key = (p["case_id"], p["combination_sha256"])
        old = prior.get(key)
        if old is not None:
            verdict = {k: old[k] for k in ("category", "reason", "entity_relation_mismatch")}
            source = "exact_AM_question_input_answer"
        else:
            verdict = MANUAL.get((p["case_id"], p["model"])) or MANUAL.get((p["case_id"], "*"))
            if verdict is None:
                raise RuntimeError(f"new answer combination lacks frozen one-pass ruling: {p['model']} {p['case_id']}")
            verdict = {k: verdict[k] for k in ("category", "reason", "entity_relation_mismatch")}
            source = "AO_one_pass_actual_input_review"
            if key not in primary:
                note = MANUAL.get((p["case_id"], p["model"])) or MANUAL.get((p["case_id"], "*"))
                primary[key] = dict(case_id=p["case_id"], combination_sha256=p["combination_sha256"],
                                    models=[], answer=p["answer"], question=p["question"],
                                    category=note["category"], reason=note["reason"], review_basis=note["review_basis"],
                                    actual_edges=p["edges"], actual_input_sha256=hashlib.sha256(
                                        json.dumps(p["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest(),
                                    training_use_forbidden=True)
        if verdict["category"] == "success" and not p["complete_chain_visible"]:
            raise RuntimeError(f"success with an incomplete actual input: {p['model']} {p['case_id']}")
        if key in seen and seen[key] != verdict["category"]:
            raise RuntimeError("same case/input/answer received inconsistent category")
        seen[key] = verdict["category"]
        if source == "AO_one_pass_actual_input_review":
            primary[key]["models"].append(p["model"])
        reviews.append(dict(case_id=p["case_id"], model=p["model"], answer=p["answer"],
                            category=verdict["category"], reason=verdict["reason"],
                            entity_relation_mismatch=verdict["entity_relation_mismatch"],
                            complete_chain_visible=p["complete_chain_visible"], edges=p["edges"],
                            combination_sha256=p["combination_sha256"], adjudication_source=source,
                            training_use_forbidden=True))
    if len(reviews) != 58 or len(primary) != 4:
        raise RuntimeError(f"expected 58 answer combinations and 4 new unique combinations, got {len(reviews)} / {len(primary)}")
    write_jsonl(OUT / "semantic-review-seed42.jsonl", reviews)
    write_json(OUT / "primary-adjudications-seed42.json", {
        "scope": "4 new unique answer combinations only; exact AM combinations reused",
        "new_unique_combination_count": 4,
        "new_case_count": len({r["case_id"] for r in primary.values()}),
        "answers": list(primary.values()), "training_use_forbidden": True,
    })

    old_probe_reviews = {(r["model"], r["probe_id"]): r for r in rows(AM / "probe-review-seed42.jsonl")}
    old_complete = {(r["probe_id"], r["action"], r.get("answer"), r.get("query")): r
                    for r in old_probe_reviews.values() if r["category"] == "complete"}
    partial_sources = {r["probe_id"]: r for r in rows(AN / "partial-probes.jsonl")}
    an_rulings = read(AN / "primary-adjudications-seed42.json")["partial"]
    probe_packets = rows(OUT / "probe-review-packets-seed42.jsonl")
    probe_reviews = []
    for p in probe_packets:
        if p["category"] == "complete":
            old = old_complete.get((p["probe_id"], p["action"], p.get("answer"), p.get("query")))
            if old is None:
                raise RuntimeError(f"unreviewed complete probe output: {p['model']} {p['probe_id']}")
            if p["exact_am_verdict"] is None:
                raise RuntimeError("complete probe declared non-exact but did not include prior verdict")
            probe_reviews.append(dict(model=p["model"], probe_id=p["probe_id"], category="complete",
                                      action=p["action"], answer=p.get("answer"), query=p.get("query"),
                                      correct=old["correct"], binding_repair=old.get("binding_repair", False),
                                      reason=old["reason"], adjudication_source="exact_AM_probe_input_action",
                                      input_sha256=hashlib.sha256(json.dumps(p["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest(),
                                      edges=next(r["edges"] for r in rows(OUT / "probe-evidence-seed42.jsonl")
                                                 if r["model"] == p["model"] and r["probe_id"] == p["probe_id"]),
                                      training_use_forbidden=True))
        else:
            src = partial_sources[p["probe_id"]]
            if p["prompt_token_ids"] != src["prompt_token_ids"]:
                raise RuntimeError(f"partial probe input mismatch: {p['probe_id']}")
            old = an_rulings[p["model"]][p["probe_id"]]
            if p["action"] == "search":
                folded = " ".join(p["query"].lower().split())
                repeated = any(folded == " ".join(q.lower().split()) for q in src["previous_queries"])
                if p.get("query") == old.get("query"):
                    legal = bool(old["legal_missing_relation_search"]) and not repeated
                    ruling_source = "exact_AN_query_ruling"
                elif (p["probe_id"] == "partial:musique:2hop__506035_325512:step1"
                      and p["model"] == "42-B" and "mississippi" in src["first_edge_citation"]["quote"].lower()
                      and "poplarville" in folded and "county" in folded):
                    legal = not repeated
                    ruling_source = "AO_bounded_equivalence_review_visible_state_name_plus_missing_county"
                else:
                    raise RuntimeError(f"partial query changed without a frozen bounded ruling: {p['probe_id']} {p['query']}")
            else:
                repeated = False
                legal = False
                ruling_source = "no_search_action"
            probe_reviews.append(dict(model=p["model"], probe_id=p["probe_id"], category="partial",
                                      action=p["action"], answer=p.get("answer"), query=p.get("query"),
                                      legal_missing_relation_search=legal, premature_answer=p["action"] == "answer",
                                      invalid_action=p["action"] == "invalid", repeated_query=repeated,
                                      previous_queries=src["previous_queries"], bridge_entity=src["bridge_entity"],
                                      missing_relation=src["missing_relation"], first_edge_citation=src["first_edge_citation"],
                                      adjudication_source=ruling_source + "_and_current_action_check",
                                      training_use_forbidden=True))
    write_jsonl(OUT / "probe-review-seed42.jsonl", probe_reviews)

    evals = {m: rows(OUT / f"eval-{m}.jsonl") for m in ("42-A", "42-B")}
    sem = {m: [r for r in reviews if r["model"] == m] for m in ("42-A", "42-B")}
    z_review = {r["case_id"]: r["B"] for r in rows(AL / "semantic-review.jsonl")}
    z_success = {c for c, r in z_review.items() if r["category"] == "success"}
    z_errors = sum(r["category"] in ("factual_error", "unsupported_by_visible_evidence") for r in z_review.values())
    protections = read(AN / "manifest.json")["protection"]
    must_retain = read(AN / "manifest.json")["must_retain"]
    complete_scores = {}
    partial_scores = {}
    scores = {}
    metrics = {"GoalZ": metric(rows(AL / "trajectory-B.jsonl"))}
    metrics.update({m: metric(evals[m]) for m in evals})
    for m in evals:
        rset = sem[m]
        cats = Counter(r["category"] for r in rset)
        success_ids = {r["case_id"] for r in rset if r["category"] == "success"}
        pset = [r for r in probe_reviews if r["model"] == m]
        complete = [r for r in pset if r["category"] == "complete"]
        partial = [r for r in pset if r["category"] == "partial"]
        complete_scores[m] = sum(r["correct"] for r in complete)
        partial_scores[m] = dict(legal_searches=sum(r["legal_missing_relation_search"] for r in partial),
                                 premature_answers=sum(r["premature_answer"] for r in partial),
                                 invalid_actions=sum(r["invalid_action"] for r in partial),
                                 repeated_queries=sum(r["repeated_query"] for r in partial))
        errors = cats["factual_error"] + cats["unsupported_by_visible_evidence"]
        scores[m] = dict(supported_success=len(success_ids), success_ids=sorted(success_ids), categories=dict(cats),
                         facts_plus_unsupported=errors,
                         complete_chain_visible=sum(r["complete_chain_visible"] for r in rset),
                         entity_relation_mismatches=sum(r["entity_relation_mismatch"] for r in rset),
                         complete_probe_correct=complete_scores[m],
                         complete_probe_binding_repairs=sum(r["correct"] and r["binding_repair"] for r in complete),
                         partial=partial_scores[m],
                         protected_losses=sorted(set(protections) - success_ids),
                         must_retain_losses=sorted(set(must_retain) - success_ids))

    a, b = scores["42-A"], scores["42-B"]
    ma, mb = metrics["42-A"], metrics["42-B"]
    probe_a, probe_b = partial_scores["42-A"], partial_scores["42-B"]
    gates = {
        "B_supported_at_least_23": b["supported_success"] >= 23,
        "B_supported_at_least_A_plus_2": b["supported_success"] - a["supported_success"] >= 2,
        "B_complete_probe_at_least_22": b["complete_probe_correct"] >= 22,
        "B_complete_probe_at_least_A_plus_2": b["complete_probe_correct"] - a["complete_probe_correct"] >= 2,
        "B_has_at_least_one_binding_repair": b["complete_probe_binding_repairs"] >= 1,
        "B_errors_not_higher_than_A": b["facts_plus_unsupported"] <= a["facts_plus_unsupported"],
        "B_errors_at_most_7": b["facts_plus_unsupported"] <= 7,
        "B_action_problem_cases_not_higher_than_A": mb["action_problem_cases"] <= ma["action_problem_cases"],
        "B_action_problem_cases_at_most_1": mb["action_problem_cases"] <= 1,
        "parse_repeats_overbudget_each_not_higher_than_A": (
            mb["parse_failures"] <= ma["parse_failures"] and
            mb["repeated_action_cases"] <= ma["repeated_action_cases"] and
            mb["overbudget_search_attempts"] <= ma["overbudget_search_attempts"]
        ),
        "B_old_protected_losses_at_most_1": len(b["protected_losses"]) <= 1,
        "B_retains_AM_Rapana_and_Duke": not b["must_retain_losses"],
        "partial_premature_invalid_repeat_not_higher": all(
            probe_b[k] <= probe_a[k] for k in ("premature_answers", "invalid_actions", "repeated_queries")
        ),
        "partial_legal_search_not_lower_than_A_or_7": (
            probe_b["legal_searches"] >= probe_a["legal_searches"] and probe_b["legal_searches"] >= 7
        ),
    }
    route = all(gates.values())
    gate = dict(seed=42, route_passed=route, baseline_supported=len(z_success), baseline_errors=z_errors,
                baseline_complete_probe_correct=20, results=scores, metrics=metrics, gates=gates,
                exact_AM_answer_reuse=52, new_unique_answer_combinations=4,
                complete_probes_reused_exactly=48, partial_probes_reviewed=16,
                seed43_executed=False, released=False)
    write_json(OUT / "seed42-gate.json", gate)

    # Add the evidence review to the already-built, fixed-order trajectory ledger.
    review_by = {(r["model"], r["case_id"]): r for r in reviews}
    paired = rows(OUT / "paired-ledger-seed42.jsonl")
    for row in paired:
        review = review_by.get((row["model"], row["case_id"]))
        if review:
            row["evidence_category"] = review["category"]
            row["complete_chain_visible"] = review["complete_chain_visible"]
            row["entity_relation_mismatch"] = review["entity_relation_mismatch"]
    write_jsonl(OUT / "paired-ledger-seed42.jsonl", paired)

    fits = {}
    targets = {r["probe_id"]: r for r in rows(OUT / "training-fit-probes.jsonl")}
    for m in ("AM", "42-A", "42-B"):
        outputs = rows(OUT / f"probe-{m}-training-fit.jsonl")
        exact = {}
        for r in outputs:
            target = targets[r["probe_id"]]["expected"]
            exact[r["probe_id"]] = (r["action"] == target["action"] and r.get("answer") == target.get("answer")
                                     and r.get("query") == target.get("query"))
        groups = {}
        for probe_id, ok in exact.items():
            groups.setdefault(targets[probe_id]["group_id"], []).append(ok)
        fits[m] = dict(correct_actions=sum(exact.values()), states=len(exact),
                       both_variants_correct=sum(all(v) for v in groups.values()), groups=len(groups))

    # One-pass report, with semantic and surface metrics kept separate.
    def ids_delta(left, right):
        return sorted(set(left) - set(right))
    gained = ids_delta(b["success_ids"], a["success_ids"])
    lost = ids_delta(a["success_ids"], b["success_ids"])
    report = f"""# Goal AO v2：同一证据、不同关系动作SFT结果

## 结论

答案泄漏材料问题已修复：Believe Tour组使用原AM轨迹中第一次真实搜索返回后的原始前缀，保留top-3正文和干扰，只避开后续assistant查询`Justin Bieber`。8组全部通过一次性正文、历史泄漏、tokenizer和评价隔离核验。没有改题、造证据或新增检索。

seed42两臂已完成训练、独立重载、64题搜索与冻结探针。**关系对照臂B未通过seed43门槛，实验在seed42收尾；不训练seed43、不发布候选。**B相对A只增加1题证据支持成功，完整证据探针没有增加，且仍有8/29题属于事实错误或实际证据不支持。当前发布产品仍为Goal Z。

## 主要结果

| 指标 | Goal Z | A：原问重复监督 | B：关系对照监督 |
|---|---:|---:|---:|
| EM /64 | {metrics['GoalZ']['EM']} | {ma['EM']} | {mb['EM']} |
| F1 /64 | {metrics['GoalZ']['F1']:.4f} | {ma['F1']:.4f} | {mb['F1']:.4f} |
| 证据支持成功 /29 | {len(z_success)} | {a['supported_success']} | {b['supported_success']} |
| A/B支持成功新增与丢失 | — | — | +{len(gained)} / −{len(lost)} |
| 完整链在最终输入可见 /29 | — | {a['complete_chain_visible']} | {b['complete_chain_visible']} |
| 事实错误＋证据不支持 /29 | {z_errors} | {a['facts_plus_unsupported']} | {b['facts_plus_unsupported']} |
| 实体/关系错配 /29 | — | {a['entity_relation_mismatches']} | {b['entity_relation_mismatches']} |
| 完整证据探针正确 /24 | 20 | {a['complete_probe_correct']} | {b['complete_probe_correct']} |
| 其中绑定修复 /24 | — | {a['complete_probe_binding_repairs']} | {b['complete_probe_binding_repairs']} |
| 部分证据合法补链查询 /8 | 6 | {probe_a['legal_searches']} | {probe_b['legal_searches']} |
| 部分探针提前回答 /8 | 0 | {probe_a['premature_answers']} | {probe_b['premature_answers']} |
| 无效/重复/预算外动作题 | {metrics['GoalZ']['action_problem_cases']} | {ma['action_problem_cases']} | {mb['action_problem_cases']} |
| 重复动作题 / 超预算搜索尝试 | {metrics['GoalZ']['repeated_action_cases']} / {metrics['GoalZ']['overbudget_search_attempts']} | {ma['repeated_action_cases']} / {ma['overbudget_search_attempts']} | {mb['repeated_action_cases']} / {mb['overbudget_search_attempts']} |
| 调用 / 真实搜索 | {metrics['GoalZ']['model_calls']} / {metrics['GoalZ']['searches']} | {ma['model_calls']} / {ma['searches']} | {mb['model_calls']} / {mb['searches']} |
| 最长实际输入 / 动作输出token | {metrics['GoalZ']['max_input_tokens']} / {metrics['GoalZ']['max_action_tokens']} | {ma['max_input_tokens']} / {ma['max_action_tokens']} | {mb['max_input_tokens']} / {mb['max_action_tokens']} |
| 正文裁剪 / 长度停止 | {metrics['GoalZ']['body_trims']} / {metrics['GoalZ']['length_stops']} | {ma['body_trims']} / {ma['length_stops']} | {mb['body_trims']} / {mb['length_stops']} |
| 平均每题 / 每生成秒 | {metrics['GoalZ']['mean_question_seconds']:.3f} / {metrics['GoalZ']['mean_generation_seconds']:.3f} | {ma['mean_question_seconds']:.3f} / {ma['mean_generation_seconds']:.3f} | {mb['mean_question_seconds']:.3f} / {mb['mean_generation_seconds']:.3f} |

Goal Z来自AL相同16384/2048入口；延迟只作描述。准入外的35题来源仍未确认，只计入固定64题EM/F1和机械轨迹统计。

## 逐题机制

B相对A唯一新增的证据支持成功是Rapana题`musique:2hop__796221_18803`。两臂都已检索并在回答输入保留Rapana→Shell以及Shell 2010年4月寻找芬兰业务买家的句子；A答成同一输入另一段出现的`service stations`，B答出完整活动与地域范围。这个个案支持**完整证据下答案选择有所修复**，不是新增检索或上下文保留收益。A/B支持成功没有丢失差异（A-only：{', '.join(lost) if lost else '无'}）。

限制这一正面信号的是Hero and the Terror题`musique:2hop__96194_78303`：两臂最终输入都同时含影片主演Chuck Norris和Walker剧集角色表，B仍答影片角色`Simon Moon`，而题目问的是Walker角色Cordell Walker。它是完整证据下的角色错绑。完整证据24题探针A/B均{a['complete_probe_correct']}/24，B相对A净增0，绑定修复0。

两臂都在`musique:2hop__418360_67370`失去AM普通SFT的支持成功：实际输入只有“The New New Rules由Bill Maher所写”的第一边，没有节目2018回归日期；模型答`October 8, 2017`，因此记证据不支持，不能把旧裁定或答案日期当作本轮可见依据。Laren题`musique:2hop__498251_126089`两臂也都没有检索到“谁负责North Holland”这条边；B恰好答中官方答案Johan Remkes，但输入只证明Laren位于North Holland/Gooi，故仍不计成功。

部分证据探针里两臂都为8/8选择search；7/8查询由可见桥接实体指向缺失关系，Crossley查询额外带入未见于当前题面/证据的`2015-16`，按冻结裁定不合法。探针不执行检索，不能算真实搜索成功。

## 准入裁定和门槛

58个准入答案组合中，52个与AM账本的同题、同实际token输入、同答案完全匹配，复用原裁定；其余4个唯一新组合（涉及Rapana、Real Time日期、Walker角色、Laren负责人3题）逐一依据本轮正文和引用裁定。成功必须有全部必要边在最终实际输入中的逐边引用。三题准入外继续保持未确认。本次只做一轮，不扩展事实或重评其他题。

冻结门槛结果：

| 条件 | 结果 |
|---|---|
| B支持成功至少23/29，且比A至少多2 | {b['supported_success']}/29，较A +{b['supported_success']-a['supported_success']}，失败 |
| B完整探针至少22/24、比A多2且有绑定修复 | {b['complete_probe_correct']}/24，较A +{b['complete_probe_correct']-a['complete_probe_correct']}；绑定修复{b['complete_probe_binding_repairs']}，失败 |
| 错误合计不高于A且不超过7 | B {b['facts_plus_unsupported']}，A {a['facts_plus_unsupported']}，绝对门槛失败 |
| 动作问题不高于A且至多1 | B {mb['action_problem_cases']}，A {ma['action_problem_cases']}，通过 |
| 保留AM的Rapana与公爵收益；旧连续搜索保护最多丢1 | B保留Rapana/公爵；旧保护丢{len(b['protected_losses'])}题，通过 |
| 部分证据动作无退化且合法补链至少7/8 | B提前回答/无效/重复 {probe_b['premature_answers']}/{probe_b['invalid_actions']}/{probe_b['repeated_queries']}；合法{probe_b['legal_searches']}/8，通过 |

路线门槛整体失败，按预注册停止，不运行seed43。B相对Goal Z只净增1道有可见完整依据的题，低于探索标准；EM的+3题并不能替代证据支持指标。

## 拟合、身份与额度

AM原普通SFT对16个训练状态为{fits['AM']['correct_actions']}/16（两问均正确{fits['AM']['both_variants_correct']}/8组）；AO A为{fits['42-A']['correct_actions']}/16（{fits['42-A']['both_variants_correct']}/8组），B为{fits['42-B']['correct_actions']}/16（{fits['42-B']['both_variants_correct']}/8组）。B的训练状态拟合略高，不构成泛化证据。

两臂均独立从AA发布Goal Z adapter加载，起点SHA-256 `{read(OUT / 'manifest.json')['expected_adapter_sha256']}`；A候选SHA-256 `{summary('42-A')['adapter_sha256']}`，B候选SHA-256 `{summary('42-B')['adapter_sha256']}`。两臂各128更新，LoRA-only、base unchanged、重载最大差值0。基础模型、BF16、不量化、optimizer与冻结计划一致。

本次共2次训练、256次更新、128条真实轨迹、{sum(1 for r in rows(OUT / 'gpu-usage.jsonl') if r.get('kind') == 'generations')}次GPU生成、{sum(1 for r in rows(OUT / 'gpu-usage.jsonl') if r.get('kind') == 'searches')}次搜索；seed43与新确认调用均为0。未超stage1的624生成/384搜索/256更新额度。候选均未发布。

主要产物：`manifest.json`、`evidence-admission.json`、`derived-states.jsonl`、`semantic-review-seed42.jsonl`、`primary-adjudications-seed42.json`、`probe-review-seed42.jsonl`、`probe-evidence-seed42.jsonl`、`paired-ledger-seed42.jsonl`、`seed42-gate.json`、两臂训练摘要及原始轨迹。全部评价材料标记为禁止训练使用。
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    (OUT / "REPORT-seed42.md").write_text(report, encoding="utf-8", newline="\n")
    write_json(OUT / "delivery-manifest.json", {
        "status": "completed_seed42_stopped_by_frozen_gate", "released": False,
        "goal_z_adapter_sha256": read(OUT / "manifest.json")["expected_adapter_sha256"],
        "seed42_A_adapter_sha256": summary("42-A")["adapter_sha256"],
        "seed42_B_adapter_sha256": summary("42-B")["adapter_sha256"],
        "goal_z_success": len(z_success), "A_supported_success": a["supported_success"],
        "B_supported_success": b["supported_success"], "route_passed": route,
        "seed43_executed": False, "new_confirmation_calls": 0,
        "manifest_sha256": hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256((ROOT / "scripts/run_v133_goal_ao_v2.py").read_bytes()).hexdigest(),
        "finalizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256((OUT / "REPORT.md").read_bytes()).hexdigest(),
        "answer_review_sha256": hashlib.sha256((OUT / "semantic-review-seed42.jsonl").read_bytes()).hexdigest(),
        "gate_sha256": hashlib.sha256((OUT / "seed42-gate.json").read_bytes()).hexdigest(),
    })
    write_json(OUT / "status.json", {"status": "completed_seed42_stopped_by_frozen_gate", "released": False,
                                      "seed43": "not_run_gate_failed", "new_confirmation_calls": 0})
    print(json.dumps({"route_passed": route, "A_supported": a["supported_success"],
                      "B_supported": b["supported_success"], "B_errors": b["facts_plus_unsupported"],
                      "B_complete_probes": b["complete_probe_correct"], "gates": gates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
