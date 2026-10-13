"""Offline, source-bound audit of the frozen AM42-A development trajectories.

Only the tokenizer is loaded. No model, retrieval service, training or network calls.
Run --prepare to extract review material, then --build after keyed adjudication.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string
import unicodedata

from am64_evidence_adjudications import NEW, PRIOR_REASONS

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
AM = ROOT / "runs/v133-goal-am-action-sft-v2"
Z = ROOT / "runs/v133-goal-z-multihop-sft"
BD = ROOT / "runs/v133-goal-bd-final-regression-v1"
OUT = ROOT / "reports/am64-evidence-visibility-20261007"
LABELS = ["证据齐，答案对", "证据齐，答案错", "证据不齐，答案对", "证据不齐，答案错"]


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(r):
    return hashlib.sha256(json.dumps([r["question"], r["trace"][-1]["prompt_token_ids"], r.get("answer")], ensure_ascii=False).encode()).hexdigest()


def write(path, value, *, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in value) if jsonl else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def normalize(text):
    # Match the frozen Goal Q/AM scorer, including its accent folding.
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def documents(prompt):
    result = []
    decoder = json.JSONDecoder()
    for block, match in enumerate(re.finditer(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", prompt, re.S)):
        payload = match.group(1)
        array = re.search(r'"result"\s*:\s*\[', payload)
        if array is None:
            continue
        position = array.end()
        while True:
            while payload[position] in " \n\r\t,":
                position += 1
            if payload[position] == "]":
                break
            doc, end = decoder.raw_decode(payload, position)
            body = re.search(r'"text"\s*:\s*("(?:[^"\\]|\\.)*")', payload[position:end], re.S)
            if body is None or json.loads(body.group(1)) != doc["text"]:
                raise ValueError("Cannot bind text field to its document object")
            start = match.start(1) + position + body.start(1) + 1
            body_end = match.start(1) + position + body.end(1) - 1
            result.append(dict(doc, tool_block=block, block_span=[match.start(), match.end()],
                               body_span=[start, body_end]))
            position = end
    return result


def inputs(tokenizer, record):
    ids = record["trace"][-1]["prompt_token_ids"]
    prompt = tokenizer.decode(ids, skip_special_tokens=False)
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    if encoded.ids != ids:
        raise ValueError(f"Tokenizer roundtrip mismatch: {record['case_id']}")
    return prompt, encoded.offsets, documents(prompt)


def citation(prompt, offsets, docs, doc_id, quote):
    for doc in docs:
        if doc["id"] != doc_id or quote not in doc["text"]:
            continue
        body = json.dumps(doc["text"], ensure_ascii=False)[1:-1]
        escaped = json.dumps(quote, ensure_ascii=False)[1:-1]
        body_start, body_end = doc["body_span"]
        if prompt[body_start:body_end] != body:
            raise ValueError("Cannot locate escaped document body")
        start = body_start + body.index(escaped)
        end = start + len(escaped)
        indices = [i for i, (s, e) in enumerate(offsets) if s < end and e > start]
        assert indices and prompt[start:end] == escaped
        return dict(document_id=doc_id, title=doc["title"], quote=quote,
                    body_char_span=[doc["text"].index(quote), doc["text"].index(quote) + len(quote)],
                    input_char_span=[start, end], input_token_span=[min(indices), max(indices) + 1],
                    tool_block=doc["tool_block"], actual_body_sha256=hashlib.sha256(doc["text"].encode()).hexdigest())
    return None


def load(tokenizer_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cases = rows(Z / "selected-dev.jsonl")
    runs = rows(AM / "eval-42-A.jsonl")
    assert len(cases) == len(runs) == 64
    assert len({r["id"] for r in cases}) == len({r["case_id"] for r in runs}) == 64
    assert {c["id"] for c in cases} == {r["case_id"] for r in runs}
    return tokenizer, cases, {r["case_id"]: r for r in runs}


def prepare(tokenizer_path):
    tokenizer, cases, runs = load(tokenizer_path)
    prior = {r["case_id"]: r for r in rows(BD / "semantic-adjudications.jsonl") if r["arm"] == "B"}
    corpus = json.loads((Z / "corpus-dev.json").read_text(encoding="utf-8"))
    material = []
    for case in cases:
        run = runs[case["id"]]
        prompt, offsets, docs = inputs(tokenizer, run)
        old = prior.get(case["id"])
        assert run["question"] == case["question"] and run["official_answer"] == case["answer"]
        if old:
            assert old["combination_sha256"] == key(run), case["id"]
        material.append(dict(case=case, answer=run["answer"], reason=run["reason"],
                             combination_sha256=key(run), visible_documents=docs,
                             labeled_support_documents=[d for d in corpus if d["title"] in case["support_titles"]],
                             queries=[s["query"] for s in run["trace"] if s.get("query")],
                             inherited_review=old))
    write(OUT / "review-material.jsonl", material, jsonl=True)
    print(json.dumps(dict(cases=64, exact_prior_review_matches=len(prior), tokenizer_roundtrips=64)))


def classify(complete, correct):
    if correct is None:
        return None
    if type(complete) is not bool or type(correct) is not bool:
        raise ValueError("Binary classification requires explicit booleans")
    return LABELS[(0 if complete else 2) + (0 if correct else 1)]


def first_occurrences(run, tokenizer, doc_id, quote):
    first_return = None
    search_number = 0
    first_visible = None
    for call_number, step in enumerate(run["trace"], 1):
        visible = documents(tokenizer.decode(step["prompt_token_ids"], skip_special_tokens=False))
        if first_visible is None and any(d["id"] == doc_id and quote in d["text"] for d in visible):
            first_visible = call_number
        if step.get("observation") is not None:
            search_number += 1
            if first_return is None and any(d["id"] == doc_id and quote in d["text"] for d in step["observation"].get("result", [])):
                first_return = search_number
    return first_return, first_visible


def summarize(audit):
    counts = {label: sum(r["classification"] == label for r in audit) for label in LABELS}
    pending = [r["case_id"] for r in audit if r["classification"] is None]
    complete = sum(r["all_necessary_evidence_visible"] for r in audit)
    strata = {}
    for visible in (True, False):
        group = [r for r in audit if r["all_necessary_evidence_visible"] is visible]
        decided = [r for r in group if r["answer_correct"] is not None]
        correct = sum(r["answer_correct"] is True for r in decided)
        strata["complete" if visible else "incomplete"] = dict(total=len(group), adjudicated=len(decided),
            pending=len(group)-len(decided), correct=correct,
            conditional_accuracy=correct / len(decided) if decided else None,
            full_group_accuracy_bounds=[correct / len(group), (correct + len(group)-len(decided)) / len(group)] if group else None)
    return dict(cases=len(audit), finalized_cross_analysis=not pending, classified=sum(counts.values()),
                pending_count=len(pending), pending_case_ids=pending, class_counts=counts,
                class_proportions_of_all_cases={k: v / len(audit) for k, v in counts.items()},
                all_necessary_evidence_visible=complete, all_necessary_evidence_visibility_rate=complete / len(audit),
                visibility_formula="complete_evidence_cases / all_64_cases; includes cases with pending answer judgment",
                conditional_answer_accuracy=strata,
                answer_correct=sum(r["answer_correct"] is True for r in audit),
                answer_incorrect=sum(r["answer_correct"] is False for r in audit),
                strict_em=sum(r["strict_exact_match"] for r in audit),
                normalized_em=sum(r["normalized_exact_match"] for r in audit),
                inherited_input_answer_matches=sum(r["review_origin"] == "exact_BD_B_reuse" for r in audit))


def build(tokenizer_path):
    tokenizer, cases, runs = load(tokenizer_path)
    prior = {r["case_id"]: r for r in rows(BD / "semantic-adjudications.jsonl") if r["arm"] == "B"}
    assert set(NEW).isdisjoint(prior)
    assert set(NEW) | set(prior) == set(runs)
    assert set(PRIOR_REASONS) == set(prior)
    material = {r["case"]["id"]: r for r in rows(OUT / "review-material.jsonl")}
    settings = json.loads((ROOT / "configs/demo-web.json").read_text(encoding="utf-8"))
    historical = json.loads((AM / "execution-freeze.json").read_text(encoding="utf-8"))
    delivery = json.loads((AM / "delivery-validation.json").read_text(encoding="utf-8"))
    expected = settings["models"]["am-research"]["sha256"]
    actual = sha(AM / "seed42-A/adapter-final/adapter_model.safetensors")
    assert actual == expected == delivery["candidates"]["A"]["adapter_sha256"]
    for field, value in historical["product"].items():
        assert settings["product"][field] == value, field
    paired = {r["case_id"]: r for r in rows(BD / "paired-ledger.jsonl") if r["split"] == "goal_z_dev"}
    assert set(paired) == set(runs)
    corpus = json.loads((Z / "corpus-dev.json").read_text(encoding="utf-8"))
    corpus_sha = sha(Z / "corpus-dev.json")
    assert corpus_sha == settings["corpora"]["goal-z-dev"]["sha256"] == historical["hashes"]["dev_corpus"]
    source_indices = {r["case_id"]: i for i, r in enumerate(rows(AM / "eval-42-A.jsonl"))}
    audit = []
    for number, case in enumerate(cases, 1):
        cid = case["id"]
        run = runs[cid]
        assert run["model"] == "42-A"
        assert run["question"] == case["question"] and run["official_answer"] == case["answer"]
        assert paired[cid]["B_answer"] == run["answer"]
        assert material[cid]["combination_sha256"] == key(run)
        assert material[cid]["case"] == case
        prompt, offsets, docs = inputs(tokenizer, run)
        assert run["em"] == int(bool(run["answer"]) and normalize(run["answer"]) == normalize(case["answer"]))
        if cid in prior:
            old = prior[cid]
            assert old["combination_sha256"] == key(run), cid
            specifications = [dict(e, title=e["citation"]["title"] if e["citation"] else None,
                                    quote=e["citation"]["quote"] if e["citation"] else None,
                                    missing_reason=None if e["actual_visible"] else PRIOR_REASONS[cid]) for e in old["edges"]]
            # The independent answer judgment is not a consequence of visibility:
            # the three inherited unsupported answers were separately checked
            # for wrong jurisdiction, wrong cup years, and wrong answer type.
            correct = old["category"] == "success"
            reason = PRIOR_REASONS[cid]
            flags = ["independent_wrong_answer_review"] if old["category"] == "unsupported_by_visible_evidence" else []
            origin = "exact_BD_B_reuse"
        else:
            review = NEW[cid]
            specifications = review["edges"]
            correct, reason, flags = review["answer_correct"], review["answer_reason"], review["annotation_flags"]
            origin = "new_primary_offline_review"
        edges = []
        for i, spec in enumerate(specifications):
            cit = None
            if spec["quote"]:
                candidates = [d for d in docs if d["title"] == spec["title"] and spec["quote"] in d["text"]]
                if not candidates:
                    raise ValueError(f"Unlocatable reviewed citation: {cid} edge{i}")
                cit = citation(prompt, offsets, docs, candidates[0]["id"], spec["quote"])
                if cid in prior:
                    assert cit == prior[cid]["edges"][i]["citation"], (cid, i)
            first_return, first_visible = first_occurrences(run, tokenizer, cit["document_id"], cit["quote"]) if cit else (None, None)
            if cit:
                assert first_return is not None and first_visible is not None
            edges.append(dict(edge_id=spec.get("edge_id", f"{cid}:edge{i}"), subject=spec["subject"],
                relation=spec["relation"], object=spec["object"], actual_visible=cit is not None,
                citation=cit, first_return_search=first_return, first_visible_model_call=first_visible,
                missing_reason=spec.get("missing_reason")))
        complete = all(e["actual_visible"] for e in edges)
        if cid in prior:
            assert complete == prior[cid]["complete_chain_visible"]
        support_docs = [d for d in corpus if d["title"] in case["support_titles"]]
        assert {d["title"] for d in support_docs} == set(case["support_titles"])
        assert material[cid]["labeled_support_documents"] == support_docs
        visible_ids = {d["id"] for d in docs}
        seen_ids = {d["id"] for s in run["trace"] for d in s.get("observation", {}).get("result", [])}
        expected_support_ids = {d["id"] for d in support_docs}
        changes = [s.get("context_usage", {}) for s in run["trace"]]
        facts = []
        if expected_support_ids <= visible_ids and not complete:
            facts.append("标注支持文档均在最终输入，但至少一条必要关系无正文证明")
        if any(s.get("repeated") for s in run["trace"]):
            facts.append("轨迹存在重复查询事件")
        if not run["answer"]:
            facts.append("无最终答案")
        if not any(c.get("prompt_truncated") or c.get("lossy_body_trimmed") for c in changes):
            facts.append("日志未记录输入截断或正文裁剪")
        if not complete:
            hypotheses = ["优先核查查询目标、返回材料的关系覆盖及停止决策；不能由此表单独确定因果"]
            if correct is True:
                hypotheses = ["可能使用参数记忆或猜测，也可能存在证据／支持标注不足；当前轨迹不能区分这些解释"]
        elif correct is False:
            hypotheses = ["优先核查实体绑定、关系对象选择及回答限定／粒度；证据齐不等于模型已正确利用"]
        else:
            hypotheses = []
        audit.append(dict(number=number, case_id=cid, question=case["question"], answer=run["answer"],
            reference_answer=case["answer"], reference_aliases=case.get("answer_aliases", []),
            model_id="am-research", adapter_sha256=actual, corpus_sha256=corpus_sha,
            retriever=historical["product"]["retriever"], reasoning_mode="off",
            source=case["source"], reason=run["reason"], queries=material[cid]["queries"],
            strict_exact_match=int(bool(run["answer"]) and run["answer"] == case["answer"]),
            normalized_exact_match=int(bool(run["answer"]) and any(normalize(run["answer"]) == normalize(g) for g in [case["answer"], *case.get("answer_aliases", [])])),
            historical_em_unchanged=run["em"], historical_f1_unchanged=run["f1"],
            answer_correct=correct, answer_correctness_basis="frozen_reference_semantics_and_question_compatible_visible_evidence; not_independent_gold_validation",
            answer_reason=reason, all_necessary_evidence_visible=complete,
            classification=classify(complete, correct), review_status="pending_answer_adjudication" if correct is None else "reviewed",
            edges=edges, missing_edge_ids=[e["edge_id"] for e in edges if not e["actual_visible"]],
            all_labeled_support_documents_returned=expected_support_ids <= seen_ids,
            all_labeled_support_documents_final_visible=expected_support_ids <= visible_ids,
            labeled_support_document_ids=sorted(expected_support_ids), reviewed_final_document_ids=sorted(visible_ids),
            labeled_source_documents=support_docs,
            annotation_flags=flags, observations=facts, diagnostic_hypotheses=hypotheses,
            review_origin=origin, combination_sha256=key(run), final_input_tokens=len(run["trace"][-1]["prompt_token_ids"]),
            source_trace_file=str((AM / "eval-42-A.jsonl").relative_to(ROOT)), source_record_index=source_indices[cid],
            actual_input_selection="last recorded decision; answer generation when an answer exists",
            review_all_visible_document_bodies=True, training_use_forbidden=True))
    summary = summarize(audit)
    assert summary["classified"] + summary["pending_count"] == 64
    source_files = [AM / "eval-42-A.jsonl", Z / "selected-dev.jsonl", Z / "corpus-dev.json",
                    BD / "semantic-adjudications.jsonl", BD / "paired-ledger.jsonl", BD / "manifest.json",
                    AM / "execution-freeze.json", AM / "delivery-validation.json", ROOT / "configs/demo-web.json",
                    Path(__file__), Path(__file__).with_name("am64_evidence_adjudications.py"),
                    OUT / "review-material.jsonl", tokenizer_path]
    summary.update(model_id="am-research", adapter_sha256=actual, corpus_sha256=corpus_sha,
                   source_hashes={str(p.resolve()): sha(p) for p in source_files},
                   historical_data_mutated=False, model_generations=0, training_updates=0,
                   all64_BD_answers_match=True, all64_BD_inputs_verified=False,
                   input_identity_scope="29 exact question/input-token/answer matches to BD B; remaining35 use original AM inputs, not claimed BD-input identical",
                   tokenizer_roundtrips_final_inputs=64)
    write(OUT / "per-case.jsonl", audit, jsonl=True)
    write(OUT / "summary.json", summary)
    render(audit, summary)
    print(json.dumps({k: summary[k] for k in ("cases", "class_counts", "pending_count", "all_necessary_evidence_visible", "strict_em", "normalized_em", "answer_correct")}, ensure_ascii=True))


def cell(value):
    return str(value if value is not None else "（无答案）").replace("|", "\\|").replace("\n", " ")


def render(audit, summary):
    n, complete = summary["cases"], summary["all_necessary_evidence_visible"]
    lines = ["# AM64 必要证据可见率与答案正确性交叉分析", "",
        "2026-10-07。固定当前配置默认 AM42-A、BM25、64道已暴露开发题；只分析历史记录，0次模型生成、0次训练。", "",
        f"**全部必要证据可见：{complete}/{n}（{complete/n:.2%}）。** 可见性已逐题核验；答案正确性交叉分类完成{summary['classified']}/64，另有{summary['pending_count']}题任职时间／答案范围歧义待裁定，因此不是完整64题的最终四分类结论。", "",
        "| 类别 | 题数 | 占全部64题 |", "|---|---:|---:|"]
    lines += [f"| {k} | {v} | {v/n:.2%} |" for k, v in summary["class_counts"].items()]
    lines += [f"| 待核验（尚未分入四类） | {summary['pending_count']} | {summary['pending_count']/n:.2%} |", "",
        "待核验是审核状态，不是第五种结果类型；四类合计与待核验合计为64，不把未知判成错误。两道待核验题的必要证据均齐，故证据齐总数为33＋8＋2＝43；答案未裁定不影响证据维度的计数。", "",
        "## 交叉指标与口径", "", "| 必要证据 | 全部题数 | 答案已裁定 | 正确 | 已裁定题正确率 | 全组正确率可确定范围 |", "|---|---:|---:|---:|---:|---:|"]
    for name, label in (("complete", "齐"), ("incomplete", "不齐")):
        s = summary["conditional_answer_accuracy"][name]
        bounds = s["full_group_accuracy_bounds"]
        lines.append(f"| {label} | {s['total']} | {s['adjudicated']} | {s['correct']} | {s['correct']}/{s['adjudicated']}（{s['conditional_accuracy']:.2%}） | {bounds[0]:.2%}–{bounds[1]:.2%} |")
    lines += ["", f"字面严格 EM：{summary['strict_em']}/64；归一化 EM：{summary['normalized_em']}/64。原AM的历史 em 实际使用去重音、去英文标点／冠词归一化，本批与字面严格EM恰好同为26；保留原值，不改写历史报告。",
        "", "语义正确性以冻结参考答案为基础，结合题目要求和可见正文接受等价人名、地名及合理机构粒度；不把缺证据本身作为答案错的依据，也不声称参考答案已获独立外部金标验证。只有年份而参考要求完整日期，本次按回答不完整处理，并单列敏感性。",
        "", "可见性以最后一次实际生成输入中的工具正文为准；没有答案则取最后一次决策输入。必要关系须有正文证明或可明确推出的关系（如同一组织第二／第三任继任顺序），不以目标词命中代替关系。标题只帮助定位实体，不单独证明关系。所有可见正文均已检查等价证据。", "",
        "## 主要发现", ""]
    body_gaps = [r for r in audit if r["all_labeled_support_documents_final_visible"] and not r["all_necessary_evidence_visible"]]
    lines += [f"- {len(body_gaps)}题的标注支持文档均已在最终输入中，但正文仍缺必要关系。文档可见率不能替代必要证据可见率。",
        "- 证据齐但答错的题包含人物／亲属对象错选、本人和配偶混淆、出生年与死亡年混淆，以及必要类别／日期精度遗漏，不能全算检索失败。",
        "- 证据不齐但答对的题中，退休到某地不能证明死于某地，电影节地点不能证明导演死亡地，生育关系不能单独证明正式配偶关系。命中答案只支持参考答案一致，不证明基于完整证据推理。",
        "- 无正文裁剪／输入截断的日志证据时，不把缺证据归因于裁剪。查询偏移、重复搜索及关系不足是观察；修复哪一环能改变答案仍需另外验证。", "",
        "## 待核验与敏感性", ""]
    for r in audit:
        if r["answer_correct"] is None:
            lines.append(f"- `{r['case_id']}`：{r['answer']} / 参考 {r['reference_answer']}。{r['answer_reason']}")
    lines += ["- 日期精度：Ippolito 题回答1688，参考29 June 1688。本表记不完整；若接受只答年份，‘证据齐，答案错’减少1，‘证据齐，答案对’增加1，可见率不变。",
        "- Hagar 的 spouse 与 union／生育关系、最高收入与最富有、Hyderabad 当地设立与机构成立等语义／来源问题保留审核标记，不修订原参考答案。", "",
        "## 资产与复现边界", "",
        f"AM adapter SHA-256：`{summary['adapter_sha256']}`。当前配置、交付验证与本地权重文件哈希一致；历史产品参数与当前默认参数的共同字段全部一致。", "",
        "64题问题、参考、AM版本标识与BD的64题答案对齐。29题完整组合（问题、最终输入token、答案）与BD B审核精确一致；其余35题审查的是原AM轨迹，不凭答案相同声称BD输入相同。BM25结果不外推到网页混合检索表现。", "",
        "本次为单个主审代理的事后审核，属于已暴露开发集诊断，不是独立泛化验证。待核验题保留原答案与两种EM；若要最终四类合计64，必须先确定两道任职题的时间及替代答案接受规则。", "",
        "逐题分类与正文引用见 [CASES.md](CASES.md)，机器明细见 [per-case.jsonl](per-case.jsonl)，源文件哈希与汇总见 [summary.json](summary.json)，核验结果见 [verification.json](verification.json)。", "",
        "复现：在项目目录执行 `python -B scripts/analyze_am_evidence_visibility.py prepare`、`build`、`verify`；可用 `--tokenizer /absolute/path/tokenizer.json` 指定同一分词器文件。只覆盖本目录派生产物，不修改原始记录。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    detail = ["# AM64 逐题分类与必要证据", "", "答案待裁定的题保留审核状态，不强行归入四类。所有引用坐标为0起始半开区间，搜索／模型调用次数为1起始。", ""]
    for label in [*LABELS, None]:
        group = [r for r in audit if r["classification"] == label]
        detail += [f"## {label or '待核验（答案正确性未裁定）'}：{len(group)}题", "", "| 序号／ID | 问题 | AM答案 | 冻结参考 | 严格／归一化EM | 判定理由／缺失项 |", "|---|---|---|---|---|---|"]
        for r in group:
            missing = "；".join(e["missing_reason"] for e in r["edges"] if not e["actual_visible"])
            detail.append(f"| {r['number']} `{r['case_id']}` | {cell(r['question'])} | {cell(r['answer'])} | {cell(r['reference_answer'])} | {r['strict_exact_match']}/{r['normalized_exact_match']} | {cell(r['answer_reason'])} {cell(missing)} |")
        detail.append("")
    detail += ["## 每题证据链与可追溯引用", ""]
    for r in audit:
        detail += [f"### {r['number']}. {r['case_id']}", "", f"{r['classification'] or '待核验'}；{r['answer_reason']}", "", f"查询：{' → '.join(r['queries'])}。停止：`{r['reason']}`。最终输入：{r['final_input_tokens']} token。", ""]
        for i, e in enumerate(r["edges"], 1):
            detail.append(f"- 必要关系{i}：{e['subject']} → {e['relation']} → {e['object']}。{'可见' if e['actual_visible'] else '不齐'}。")
            if e["citation"]:
                c = e["citation"]
                detail += [f"  文档 `{c['document_id']}`（{c['title']}）；输入token `{c['input_token_span']}`，输入字符 `{c['input_char_span']}`，正文字符 `{c['body_char_span']}`；首次搜索返回{e['first_return_search']}，首次进入模型调用{e['first_visible_model_call']}。", f"  原句：{c['quote']}"]
            else:
                detail.append(f"  缺失项：{e['missing_reason']}")
        detail += ["", f"审核标记：{', '.join(r['annotation_flags']) or '无'}。观察：{'；'.join(r['observations'])}。", "", f"诊断假设：{'；'.join(r['diagnostic_hypotheses']) or '无额外因果推断'}。", "", f"输入—答案组合 SHA-256：`{r['combination_sha256']}`。原始记录：`{r['source_trace_file']}`，行{r['source_record_index']+1}。", ""]
    (OUT / "CASES.md").write_text("\n".join(detail), encoding="utf-8")


def verify(tokenizer_path):
    tokenizer, cases, runs = load(tokenizer_path)
    audit = rows(OUT / "per-case.jsonl")
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert len(audit) == len({r["case_id"] for r in audit}) == 64
    assert {r["case_id"] for r in audit} == set(runs)
    roundtrips = citations = 0
    source_records = rows(AM / "eval-42-A.jsonl")
    for r in audit:
        run = runs[r["case_id"]]
        assert r["combination_sha256"] == key(run)
        assert r["answer"] == run["answer"]
        assert source_records[r["source_record_index"]]["case_id"] == r["case_id"]
        assert r["historical_em_unchanged"] == run["em"] and r["historical_f1_unchanged"] == run["f1"]
        for step in run["trace"]:
            ids = step["prompt_token_ids"]
            assert tokenizer.encode(tokenizer.decode(ids, skip_special_tokens=False), add_special_tokens=False).ids == ids
            roundtrips += 1
        prompt, offsets, docs = inputs(tokenizer, run)
        for edge in r["edges"]:
            if edge["actual_visible"]:
                c = edge["citation"]
                assert citation(prompt, offsets, docs, c["document_id"], c["quote"]) == c
                assert first_occurrences(run, tokenizer, c["document_id"], c["quote"]) == (edge["first_return_search"], edge["first_visible_model_call"])
                citations += 1
            else:
                assert edge["citation"] is None and edge["missing_reason"]
        assert r["all_necessary_evidence_visible"] == all(e["actual_visible"] for e in r["edges"])
        assert r["classification"] == classify(r["all_necessary_evidence_visible"], r["answer_correct"])
    recomputed = summarize(audit)
    for k, v in recomputed.items():
        assert summary[k] == v, k
    for p, digest in summary["source_hashes"].items():
        assert sha(p) == digest, p
    assert sum(summary["class_counts"].values()) + summary["pending_count"] == 64
    receipt = dict(passed=True, cases=64, unique_case_ids=64, actual_input_roundtrips=roundtrips,
                   final_inputs_checked=64, citations_checked=citations, source_hashes_checked=len(summary["source_hashes"]),
                   all64_original_scores_preserved=True, cross_counts_plus_pending_equal_64=True,
                   unknown_answer_labels_not_coerced_to_false=True, model_generations=0, training_updates=0,
                   output_hashes={name: sha(OUT / name) for name in ("per-case.jsonl", "summary.json", "REPORT.md", "CASES.md")})
    write(OUT / "verification.json", receipt)
    print(json.dumps(receipt, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=Path("D:/qwen_08b/tokenizer.json"))
    parser.add_argument("mode", choices=["prepare", "build", "verify"])
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.tokenizer)
    elif args.mode == "build":
        build(args.tokenizer)
    else:
        verify(args.tokenizer)


if __name__ == "__main__":
    main()
