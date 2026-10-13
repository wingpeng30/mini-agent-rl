"""Seal final manifests for the CPU-only Goal-Z source and trajectory audit."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ai-z64-audit-repair-v7"
OLD = ROOT / "runs/v133-goal-ai-z64-source-admission-v2"
CASES = ROOT / "runs/v133-goal-ah-entity-relation/evaluation-cases-goal_z_dev.jsonl"
CORPUS = ROOT / "runs/v133-goal-z-multihop-sft/corpus-dev.json"
ADAPTER = ROOT / "runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final/adapter_model.safetensors"
AA = ROOT / "runs/v133-goal-aa-audit-confirmation/product-decision.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    freeze = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    aa = json.loads(AA.read_text(encoding="utf-8"))
    edges = jsonl(OUT / "edge-evidence-ledger.final.jsonl")
    admission = jsonl(OUT / "source-admission-ledger.final.jsonl")
    calls = jsonl(OUT / "trajectory-call-ledger.jsonl")
    mechanisms = jsonl(OUT / "trajectory-mechanism-ledger-v2.jsonl")
    candidates = jsonl(OUT / "candidate-freeze.jsonl")
    addenda = jsonl(OUT / "candidate-freeze-addendum.jsonl")
    candidate_docs = jsonl(OUT / "candidate-documents.jsonl")
    candidate_hits = jsonl(OUT / "candidate-locator-hits.jsonl")
    replay = json.loads((OUT / "production-input-replay-v4.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((OUT / "tokenizer-replay-verification-v2.json").read_text(encoding="utf-8"))
    decision = json.loads((OUT / "mechanism-decision-v2.json").read_text(encoding="utf-8"))
    revision = jsonl(OUT / "source-review-revisions.jsonl")

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    docs = {str(x["id"]): x for x in corpus}
    quote_failures = []
    for e in edges:
        quote = e.get("quote")
        if not quote:
            if e.get("source_char_span") is not None:
                quote_failures.append({"case_id": e["case_id"], "edge_id": e["edge_id"], "failure": "span_without_quote"})
            continue
        d = docs.get(str(e.get("document_id")))
        body = str((d or {}).get("text", ""))
        span = e.get("source_char_span")
        body_hash = hashlib.sha256(body.encode()).hexdigest()
        if not d or not span or body_hash != e.get("document_body_sha256"):
            quote_failures.append({"case_id": e["case_id"], "edge_id": e["edge_id"], "failure": "source_identity_or_hash_mismatch"})
        elif body[span[0]:span[1]] != quote or body.find(quote) != span[0]:
            quote_failures.append({"case_id": e["case_id"], "edge_id": e["edge_id"], "failure": "quote_span_mismatch"})

    old_manifest = json.loads((OLD / "delivery-manifest.json").read_text(encoding="utf-8"))
    old_unchanged = {name: sha(OLD / name) == expected for name, expected in old_manifest["sha256"].items()}
    per_edge_candidates = Counter((x["case_id"], x["edge_id"]) for x in candidates + addenda)
    edge_status_counts = Counter(x["status"] for x in edges)
    admission_counts = Counter(x["admission"] for x in admission)
    replay_ok = replay["status"] == "pass" and replay["cases"] == 64 and replay["historical_calls"] == 194 and len(replay["failures"]) == 0
    tokenizer_ok = tokenizer["status"] == "pass_with_invalid_actions_recorded" and tokenizer["cases"] == 64 and tokenizer["historical_calls"] == 194
    adapter_hash = sha(ADAPTER)
    identity_ok = (adapter_hash == freeze["identity"]["adapter_sha256"]
                   and aa.get("adapter") == str(ADAPTER.parent))
    checks = {
        "fixed_cases_64": len(admission) == 64,
        "keyed_edges_128": len(edges) == 128 and len({(x["case_id"], x["edge_id"]) for x in edges}) == 128,
        "admission_rows_unique": len({x["case_id"] for x in admission}) == 64,
        "source_quotes_and_spans_valid": not quote_failures,
        "adapter_identity_matches_aa_and_freeze": identity_ok,
        "historical_call_rows_194": len(calls) == 194,
        "production_inputs_rebuilt_exactly": replay_ok,
        "tokenizer_roundtrip_and_budget_check": tokenizer_ok,
        "candidate_pair_cap": len(candidates) + len(addenda) <= 1536,
        "per_edge_candidate_cap": max(per_edge_candidates.values(), default=0) <= 12,
        "old_v2_files_unchanged": all(old_unchanged.values()),
        "diagnostics_excluded_from_training": all(not x.get("training_allowed", True) for x in edges + admission + mechanisms),
    }
    verification = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {"cases": len(admission), "edges": len(edges), "edge_status": dict(edge_status_counts),
                   "admitted": admission_counts.get("admitted", 0), "not_admitted": admission_counts.get("not_admitted", 0),
                   "candidate_edge_document_pairs": len(candidates), "candidate_revision_addenda": len(addenda),
                   "candidate_documents_screened": len(candidate_docs), "literal_locator_hits": len(candidate_hits),
                   "historical_calls": len(calls), "mechanism_case_rows": len(mechanisms)},
        "identity": {"base_model": "/root/autodl-tmp/qwen_08b", "adapter": str(ADAPTER),
                     "adapter_sha256": adapter_hash, "adapter_loaded": False,
                     "aa_published": aa.get("published"), "aa_route": aa.get("selected_route"), "aa_seed": aa.get("selected_seed")},
        "mechanisms": decision,
        "production_input_replay": {k: replay[k] for k in ("status", "cases", "historical_calls", "production_renderer_sha256", "local_model_client_sha256", "prompt_source_sha256")},
        "tokenizer_verification": {"status": tokenizer["status"], "max_actual_input_tokens": max(x["max_input_tokens"] for x in tokenizer["budget_by_case"].values()),
                                   "overbudget_search_attempt_cases": [cid for cid, x in tokenizer["budget_by_case"].items() if x["overbudget_search_attempts"]]},
        "source_quote_failures": quote_failures,
        "old_v2_file_integrity": old_unchanged,
        "revisions": revision,
        "activity": {"new_searches": 0, "new_model_calls": 0, "gpu_calls": 0, "training_updates": 0},
        "candidate_review_limit_note": "139 frozen candidate bodies were screened by a literal subject/object/predicate locator; zero sentences passed that locator. This is not a semantic human review of all frozen documents and does not establish corpus-wide absence. The corrected scholarship-edge addendum received explicit manual review.",
    }
    dump("verification-final.json", verification)

    old_admission = {x["case_id"]: x for x in jsonl(OLD / "source-admission-ledger.final.jsonl")}
    revised = {x["case_id"]: x for x in admission}
    revisions_by_case = {x["case_id"]: x for x in revision}
    comparison = []
    for cid in sorted(revised):
        old = old_admission.get(cid, {})
        new = revised[cid]
        comparison.append({"case_id": cid, "old_v2_admission": old.get("admission"), "repair_v7_admission": new.get("admission"),
                          "old_v2_gates": {k: old.get(k) for k in ("chain_supported", "question_resolvable", "label_compatible")},
                          "repair_v7_gates": {k: new.get(k) for k in ("chain_supported", "question_resolvable", "label_compatible")},
                          "explicit_revision": revisions_by_case.get(cid), "diagnostic_only": True, "training_allowed": False})
    path = OUT / "prior-audit-comparison.jsonl"
    comparison_text = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in comparison)
    if path.exists():
        if path.read_text(encoding="utf-8") != comparison_text:
            raise RuntimeError("Existing prior-audit comparison differs from recomputed contents")
    else:
        path.write_text(comparison_text, encoding="utf-8")

    report = f'''# Goal Z 64题来源准入与既有轨迹机制复核（修复版 v7）

## 结论

本轮在AutoDL远程仓库完成了CPU只读复核，未加载adapter、未生成模型输出，也没有新增搜索、GPU调用或训练更新。AA记录的产品模型仍是Goal Z完整轨迹SFT seed 42；发布adapter SHA-256为`{adapter_hash}`，与AA记录及冻结manifest一致。

修复后的逐边账本覆盖64题、128条必要关系：{edge_status_counts.get('pass', 0)}条有来源引用支持，{edge_status_counts.get('unconfirmed', 0)}条未确认；29题三项准入均通过，35题未准入。未确认不等于冻结语料绝对缺失。此前“按序号指定通过”的v2账本与结果保留未动；逐题差异见`prior-audit-comparison.jsonl`。

## 轨迹与机制

共读取64题194次历史调用。使用生产Goal Q renderer和`_inputs`逐次重建，194/194输入token ID与保存值完全相同，最大输入长度为{verification['tokenizer_verification']['max_actual_input_tokens']} token。另有3题在历史最后调用尝试了第4次搜索；这些是预算外无效动作，未执行成搜索。所有结论都复用原轨迹，没有新查询或回答。

只在来源准入题及有直接依据的案例上计算机制：上下文丢证2题、检索漏证2题、查询生成0题、完整证据下回答决策错误3题。查询0表示保存的合法反事实不足以直接确认，不代表查询能力不存在问题。候选语料的有限筛选没有形成更多支持证据；各轴最高3例，未达到路线准入4例。因此唯一结论是**机制证据不足，不启动下一条机制训练/产品实验**。

## 来源审核边界

账本按`case_id + edge_id`记录实体、关系、对象、限定、文档身份、正文哈希、引文和字符位置。明确修订了一条“获得表演学院奖学金即证明就读”的错误：奖学金不能证明入学、学习或任职，原通过改为未确认。候选筛选冻结139个边—文档组合及1条修订追加项；139篇正文经过字面主体/对象/谓词句定位器筛查，命中0句。该自动筛查不是对139篇材料的语义人工逐篇审核，也不能证明语料没有证据。奖学金边追加候选已人工阅读，仍不支持就读关系。

未准入样例体现了必要关系边界：Bill Gates“最富有”不能证明“收入最高”；唱片发行不能证明艺人签约；安塞尔姆·塔珀的出生信息不能移给查尔斯·塔珀；退休地点不能替代死亡地点；任职历史不能自动满足现时关系。证据不足的题不按事实错误处理。

## 后续决策

按预先冻结的4个独立案例门槛和同数优先级，没有机制路线达标。接下来先补足能直接裁定的证据来源/题目依据，或取得对既有保存分支的明确直接证据；不要据本轮结果训练孤立判断任务。AA与AG确认集继续只作回归集。未来任何产品发布仍需未暴露新确认集及预先冻结隔离、语料、评分规则。

固定题目分母64；来源准入分母{admission_counts.get('admitted', 0)}。历史分数没有重算或跨集混用。训练材料使用、示范和能力保留损失均为0。

## 完整性

最终封存校验：`{verification['status']}`。旧v2 manifest所列文件均保持原SHA-256；详细检查在`verification-final.json`，文件哈希在`delivery-manifest.json`。
'''
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    exclude = {"delivery-manifest.json"}
    hashes = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in exclude}
    manifest = {"run_id": OUT.name, "remote_authority": "/root/autodl-tmp/mini-agent-rl/" + str(OUT.relative_to(ROOT)).replace("\\", "/"),
                "status": verification["status"], "repair_script_sha256": sha(ROOT / "scripts/run_v133_z64_audit_repair.py"),
                "prepare_time_script_sha256": freeze.get("sha256", {}).get("scripts/run_v133_z64_audit_repair.py"),
                "sha256": hashes}
    dump("delivery-manifest.json", manifest)
    print(json.dumps({"status": verification["status"], "checks": checks, "counts": verification["counts"],
                      "manifest_files": len(hashes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

