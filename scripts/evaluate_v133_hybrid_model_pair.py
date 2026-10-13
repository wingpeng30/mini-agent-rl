"""Paired Goal Z/AM evaluation with the frozen local hybrid retriever.

This runner is intentionally inference-only.  Preparation and result auditing
can run on Windows or Linux; only ``run`` needs the CUDA environment.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
import posixpath
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from mini_agent_rl.evaluation import answer_metrics
from mini_agent_rl.retrieval.local import LocalRetrieval, config as retrieval_config, sha

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "v133-hybrid-model-pair-20261011"
OUTPUT = ROOT / "runs" / RUN_ID
EXISTING = Path("E:/AgentLightningAssets/runs/retrieval-v1")
FROZEN = EXISTING / "paired-128-frozen-0b6100db84b249cfa9a82c0861cb3852"
SUPPLEMENT = EXISTING / "supplement-reviewed-64-7011e1801e974f0eb055256abddbeab9"
OLD_HYBRID = EXISTING / "hybrid-adaptation-20261006"
FROZEN_INDEX_MANIFEST = Path("E:/AgentLightningAssets/indexes/goal-z-dev-minilm-v1/manifest.json")
DEV_ADMISSION = ROOT / "runs/v133-goal-ai-z64-audit-repair-v7/source-admission-ledger.final.jsonl"
DEV_EDGES = ROOT / "runs/v133-goal-ai-z64-audit-repair-v7/edge-evidence-ledger.final.jsonl"
CONFIG_PATH = ROOT / "configs/demo-web.json"
CASES_SHA = "7e603a7b2eff16cb11ffb26e6ac719869944f09758ab39b797309b1b81b0677f"
CORPUS_SHA = "dc2a2f843ad04bb2c38ce839e1959034b7c63f1cbbc24971de2557ce3cf62d37"
BASE_SHA = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
ADAPTERS = {
    "goal-z": {"label": "A", "sha256": "d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52",
               "relative": "runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final"},
    "am-research": {"label": "B", "sha256": "3bcd1f85382c1d7775fa674c1a6e1e68c80911a69b86de209d20f2ab6269280c",
                    "relative": "runs/v133-goal-am-action-sft-v2/seed42-A/adapter-final"},
}
STRATA = ("development", "natural_supplement", "constructed_singlehop")
MAX_GPU_SECONDS = 10_800
MAX_GENERATIONS = 1_024
REVIEW_CATEGORIES = {"correct", "partial", "factual_error", "no_answer", "indeterminate"}
CODE_FILES = (
    "scripts/evaluate_v133_hybrid_model_pair.py", "scripts/run_v133_goal_q_search.py",
    "scripts/run_v133_goal_p_variant.py", "src/mini_agent_rl/model/local.py",
    "src/mini_agent_rl/retrieval/local.py", "src/mini_agent_rl/tools/search.py",
    "src/mini_agent_rl/evaluation.py",
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def resume_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        last_complete = payload.rfind(b"\n")
        with path.open("r+b") as handle:
            handle.truncate(last_complete + 1 if last_complete >= 0 else 0)
            handle.flush()
            os.fsync(handle.fileno())
        payload = payload[:last_complete + 1] if last_complete >= 0 else b""
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_once(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl_frozen(path: Path, values: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Frozen artifact exists with different contents: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write((json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_assets(root: Path = ROOT) -> dict[str, Path]:
    return {key: root / value["relative"] for key, value in ADAPTERS.items()}


def validate_ledger(cases: list[dict[str, Any]], ledger: list[dict[str, Any]], arm: str,
                    *, require_complete: bool = False) -> set[str]:
    expected = {case["id"]: case for case in cases}
    ids = [row.get("case_id") for row in ledger]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ledger record: {arm}")
    for row in ledger:
        case = expected.get(row.get("case_id"))
        if (case is None or row.get("arm") != arm or row.get("adapter_sha256") != ADAPTERS[arm]["sha256"] or
                row.get("question") != case["question"] or row.get("stratum") != case["stratum"]):
            raise ValueError(f"Ledger identity drift: {arm}/{row.get('case_id')}")
    if require_complete and set(ids) != set(expected):
        raise ValueError(f"Incomplete ledger for {arm}: {len(set(ids))}/{len(expected)}")
    return set(ids)


def validate_complete_pair(cases: list[dict[str, Any]], before: list[dict[str, Any]],
                           after: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    validate_ledger(cases, before, "goal-z", require_complete=True)
    validate_ledger(cases, after, "am-research", require_complete=True)
    return ({row["case_id"]: row for row in before}, {row["case_id"]: row for row in after})


def support_coverage(required_doc_ids: set[str], retrieved_ids: set[str], final_visible_ids: set[str]) -> dict[str, Any]:
    return {"support_documents_expected": len(required_doc_ids),
            "retrieved_support_count": len(required_doc_ids & retrieved_ids),
            "final_visible_support_count": len(required_doc_ids & final_visible_ids),
            "complete_chain_visible": bool(required_doc_ids) and required_doc_ids.issubset(final_visible_ids)}


def engineering_acceptance(ledger: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    first_ids = [case["id"] for case in cases[:2]]
    by_id = {row["case_id"]: row for row in ledger}
    if len(first_ids) != 2 or not set(first_ids).issubset(by_id):
        raise ValueError("The two frozen engineering-acceptance questions are incomplete")
    checks = []
    for case_id in first_ids:
        row = by_id[case_id]
        checks.append({"case_id": case_id,
            "trace_saved": bool(row.get("trace")) and all(isinstance(step.get("raw"), str) for step in row["trace"]),
            "prompt_tokens_match_captured_messages": all(
                step.get("actual_prompt_sha256") == digest(step.get("prompt_token_ids", [])) and
                bool(step.get("actual_input_messages")) and
                step.get("context_usage", {}).get("actual_prompt_tokens", len(step.get("prompt_token_ids", []))) <= 16384
                for step in row.get("trace", [])),
            "within_call_budgets": 0 < row.get("model_calls", 0) <= 4 and 0 <= row.get("searches", 0) <= 3
                and not row.get("over_budget"),
            "retrieval_frozen": row.get("retrieval", {}).get("retriever") == "hybrid-rrf-v1"
                and row.get("retrieval", {}).get("candidate_k") == 20
                and row.get("retrieval", {}).get("rrf_k") == 60,
        })
    passed = all(all(value for key, value in check.items() if key != "case_id") for check in checks)
    if not passed:
        raise RuntimeError(f"Two-question engineering acceptance failed: {checks}")
    return {"passed": True, "case_ids": first_ids, "checks": checks,
            "acceptance_questions_counted_in_formal_sample": True}


def source_audit(cases: list[dict[str, Any]], corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reuse frozen source admissions; never infer source validity from model output."""
    docs = {row["id"]: row for row in corpus}
    admissions = {row["case_id"]: row for row in rows(DEV_ADMISSION)}
    edges_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in rows(DEV_EDGES):
        edges_by_case[edge["case_id"]].append(edge)
    natural_review = {row["id"]: row for row in rows(SUPPLEMENT / "semantic-review.jsonl")}
    results = []
    for case in cases:
        cid, stratum = case["id"], case["stratum"]
        edges: list[dict[str, Any]] = []
        reason = None
        if stratum == "development":
            admission = admissions.get(cid)
            if not admission:
                raise ValueError(f"Missing frozen development source-admission row: {cid}")
            candidates = edges_by_case.get(cid, [])
            for edge in candidates:
                evidence = None
                doc_id = edge.get("document_id")
                quote = edge.get("quote")
                if doc_id in docs and isinstance(quote, str) and quote in docs[doc_id]["text"]:
                    body = docs[doc_id]["text"]
                    actual_hash = hashlib.sha256(body.encode()).hexdigest()
                    if edge.get("document_body_sha256") != actual_hash:
                        raise ValueError(f"Frozen edge body hash mismatch: {cid}/{edge.get('edge_id')}")
                    evidence = {"document_id": doc_id, "title": docs[doc_id]["title"], "quote": quote,
                                "body_sha256": actual_hash, "source_status": edge.get("status")}
                edges.append({"edge_id": edge.get("edge_id"), "subject": edge.get("subject"),
                              "relation": edge.get("relation"), "object": edge.get("object"),
                              "qualifiers": edge.get("qualifiers", {}), "status": edge.get("status"),
                              "source_evidence": evidence})
            valid = (admission.get("admission") == "admitted" and candidates and
                     all(edge.get("status") == "pass" and edge.get("source_evidence") for edge in edges))
            status = "confirmed" if valid else "unresolved"
            reason = admission.get("reason")
        elif stratum == "natural_supplement":
            review = natural_review.get(cid)
            if not review or review.get("accepted") is not True:
                raise ValueError(f"Frozen natural question lacks accepted review: {cid}")
            evidence_docs = case.get("support_documents", [])
            chain = case.get("chain", [])
            support_ids = case.get("support_document_ids", [])
            supporting_facts = case.get("supporting_facts", [])
            if not evidence_docs or len(support_ids) != len(evidence_docs):
                raise ValueError(f"Frozen natural source chain/evidence mismatch: {cid}")
            prior_answer = None
            for i, fact in enumerate(chain):
                if isinstance(fact, list) and len(fact) == 3:
                    supporting_title = supporting_facts[i][0] if i < len(supporting_facts) else None
                    cited = next((item for item in evidence_docs if item.get("title") == supporting_title), None)
                    relation_subject, relation, obj = fact
                    source_index = evidence_docs.index(cited) if cited else -1
                elif isinstance(fact, dict):
                    cited = next((item for item in evidence_docs
                                  if item.get("idx") == fact.get("paragraph_support_idx")), None)
                    source_index = evidence_docs.index(cited) if cited else -1
                    question_text = fact.get("question", "")
                    if " >> " in question_text:
                        relation_subject, relation = question_text.split(" >> ", 1)
                    elif "#1" in question_text:
                        if prior_answer is None:
                            raise ValueError(f"Natural relation placeholder has no antecedent: {cid}/{i}")
                        relation_subject, relation = prior_answer, question_text.replace("#1", str(prior_answer))
                    else:
                        relation_subject, relation = None, question_text
                    obj = fact.get("answer")
                    prior_answer = obj
                else:
                    raise ValueError(f"Unsupported frozen natural relation: {cid}/{i}")
                if cited is None or source_index < 0:
                    raise ValueError(f"Natural relation has no source paragraph: {cid}/{i}")
                support_id = support_ids[source_index]
                original = docs.get(support_id)
                quote = cited.get("text")
                if (not original or original["title"] != cited.get("title") or
                        not isinstance(quote, str) or not quote or quote not in original["text"]):
                    raise ValueError(f"Supplement source body mismatch: {cid}/{support_id}")
                edges.append({"edge_id": f"relation:{i+1}", "subject": relation_subject, "relation": relation,
                              "object": obj, "qualifiers": {}, "status": "previously_reviewed",
                              "source_evidence": {
                                  "document_id": original["id"], "title": original["title"], "quote": quote,
                                  "body_sha256": hashlib.sha256(original["text"].encode()).hexdigest(),
                                  "source_status": "accepted_in_frozen_review"}})
            status, reason = "confirmed", review.get("reason")
        elif stratum == "constructed_singlehop":
            cited_docs = case.get("support_documents", [])
            if len(cited_docs) != 1:
                raise ValueError(f"Constructed diagnostic must cite one frozen source: {cid}")
            cited = cited_docs[0]
            original = docs.get(cited.get("id"))
            if not original or original["text"] != cited.get("text") or original["title"] != cited.get("title"):
                raise ValueError(f"Constructed source mismatch: {cid}/{cited.get('id')}")
            answer = case["answer"]
            if answer.casefold() not in original["text"].casefold():
                raise ValueError(f"Constructed answer is absent from frozen source: {cid}")
            quote = next(text for text in re.finditer(re.escape(answer), original["text"], re.IGNORECASE)).group(0)
            edges = [{"edge_id": "support:1", "subject": None, "relation": None, "object": answer,
                      "status": "exact_answer_span", "source_evidence": {
                          "document_id": original["id"], "title": original["title"], "quote": quote,
                          "body_sha256": hashlib.sha256(original["text"].encode()).hexdigest(),
                          "source_status": "exact_answer_substring"}}]
            status, reason = "confirmed", "Frozen diagnostic answer appears verbatim in its support document."
        else:
            raise ValueError(f"Unknown frozen stratum: {stratum}")
        results.append({"case_id": cid, "stratum": stratum, "status": status, "reason": reason,
                        "question": case["question"], "gold_answer": case["answer"],
                        "answer_aliases": case.get("answer_aliases", case.get("aliases", [])),
                        "required_edges": edges,
                        "support_documents": [edge["source_evidence"] for edge in edges if edge.get("source_evidence")],
                        "source_audit_sha256": digest(edges)})
    if len(results) != 128 or len({row["case_id"] for row in results}) != 128:
        raise ValueError("Source audit must cover all 128 unique frozen questions")
    return results


def make_manifest(folder: Path) -> dict[str, Any]:
    cases = rows(folder / "cases.jsonl")
    corpus = json.loads((folder / "corpus.json").read_text(encoding="utf-8"))
    audit = source_audit(cases, corpus)
    audit_path = folder / "source-audit.jsonl"
    write_jsonl_once(audit_path, audit)
    if sum(row["status"] == "unresolved" for row in audit) != 35:
        raise ValueError("Expected to preserve the 35 previously unadmitted development questions")
    if sum(row["status"] == "confirmed" for row in audit) != 93:
        raise ValueError("Expected 93 previously reviewed/source-verified questions")
    case_ids = [row["id"] for row in cases]
    if len(cases) != 128 or len(set(case_ids)) != 128 or sha(folder / "cases.jsonl") != CASES_SHA:
        raise ValueError("Frozen 128-question file identity mismatch")
    if sha(folder / "corpus.json") != CORPUS_SHA or len(corpus) != 2048:
        raise ValueError("Frozen corpus identity mismatch")
    settings = read_config(folder / "settings.json")
    product = dict(settings["product"])
    product.update({"retriever": "hybrid-rrf-v1", "top_k": 3, "thinking": False,
                    "max_input_tokens": 16384, "max_new_tokens": 2048,
                    "max_model_calls_per_case": 4, "max_searches_per_case": 3,
                    "action_completion": "first-json-v1", "observation_encoding": "legacy",
                    "observation_presentation": "legacy-json", "action_constraint": "none",
                    "context_aware_decoding": None, "load_in_4bit": False,
                    "device": "cuda", "dtype": "bfloat16"})
    if settings["base_weights_sha256"] != BASE_SHA:
        raise ValueError("Frozen base model hash mismatch in local settings")
    manifest = {
        "run_id": RUN_ID, "status": "prepared", "kind": "exposed frozen paired diagnostic; not independent generalization",
        "case_count": 128, "strata": dict(Counter(row["stratum"] for row in cases)),
        "cases_sha256": sha(folder / "cases.jsonl"), "corpus_sha256": sha(folder / "corpus.json"),
        "corpus_documents": len(corpus), "source_audit_sha256": sha(audit_path),
        "source_status_counts": dict(Counter(row["status"] for row in audit)),
        "source_provenance": {"development_admission_ledger_sha256": sha(DEV_ADMISSION),
            "development_edge_evidence_ledger_sha256": sha(DEV_EDGES),
            "natural_question_review_sha256": sha(SUPPLEMENT / "semantic-review.jsonl"),
            "frozen_index_manifest_sha256": sha(folder / "index-manifest.json")},
        "base_weights_sha256": BASE_SHA, "adapters": ADAPTERS,
        "code_hashes": {relative: sha(ROOT / relative) for relative in CODE_FILES},
        "product": product,
        "retrieval": {"mode": "hybrid-rrf-v1", "candidate_k": 20, "rrf_k": 60, "top_k": 3,
                      "model_revision": "fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9",
                      "index_manifest_sha256": sha(folder / "index-manifest.json")},
        "seeds": {"policy": 42, "per_call": "42 + step"}, "arms_order": ["goal-z", "am-research"],
        "gpu_seconds_cap": MAX_GPU_SECONDS, "generation_attempt_cap": MAX_GENERATIONS,
        "training": False, "external_confirmation": False,
        "runner_sha256": sha(Path(__file__)),
    }
    write_json(folder / "manifest.json", manifest)
    return manifest


def prepare(folder: Path, *, from_remote: bool = False) -> dict[str, Any]:
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    if from_remote:
        cases_path, corpus_path = folder / "cases.jsonl", folder / "corpus.json"
        if not cases_path.is_file() or not corpus_path.is_file():
            raise FileNotFoundError("Remote prepare expects transferred cases.jsonl and corpus.json")
    else:
        import shutil
        shutil.copy2(FROZEN / "cases.jsonl", folder / "cases.jsonl")
        shutil.copy2(ROOT / "runs/v133-goal-z-multihop-sft/corpus-dev.json", folder / "corpus.json")
        shutil.copy2(CONFIG_PATH, folder / "settings.json")
        shutil.copy2(FROZEN_INDEX_MANIFEST, folder / "index-manifest.json")
    write_review_guide(folder)
    return make_manifest(folder)


def write_review_guide(folder: Path) -> None:
    text = """# 盲审操作说明

每个 JSONL 行是一个去重后的输出包。只依据包中的问题、实际最终输入、模型答案、结束原因、冻结参考关系和正文判断；不要推测或补入模型身份、EM/F1或汇总分数。相同问题/实际输入token/答案/结束原因只评一次，映射回两臂时复用同一票。

语义标签：`correct` 表示准确、充分回答，允许等价表达；`partial` 表示有正确部分但缺少必要答案或关系；`factual_error` 表示主要结论错误；`no_answer` 表示没有给出答案；`indeterminate` 表示题目来源或题义无法可靠裁定。来源状态是 `unresolved` 时必须用 `indeterminate`，不得当作错题。

证据支持必须比语义正确更严格：每个被判为支持的必要关系都要出现在 `covered_edge_ids`，并在 `citations` 中逐关系给出同一 edge ID、冻结来源文档ID和实际最终输入中可逐字找到的短引文。只看见文档标题、曾被搜索返回但未进入最终输入、或引用了无关句子都不算支持。答案语义不依赖检索证据；没有足够证据时仍可判 `correct`，但不要覆盖关系或添加引用。

每行输出一个 JSON 对象，不要改 review_id，不要省略任何包：

```json
{"review_id":"原样复制","semantic_category":"correct|partial|factual_error|no_answer|indeterminate","reason":"简短说明实体/关系/时间/限定条件","covered_edge_ids":["relation:1"],"citations":[{"edge_id":"relation:1","document_id":"文档ID","quote":"最终输入中逐字可见的短引文"}]}
```

审阅文件必须为 `review-a.jsonl` 或 `review-b.jsonl`，一行一个结果、UTF-8编码。完成后检查行数与review_id唯一性；不要修改原始review-packets.jsonl。
"""
    (folder / "REVIEW_GUIDE.md").write_text(text, encoding="utf-8")


class Budget:
    def __init__(self, folder: Path):
        self.path = folder / "gpu-budget.json"
        self.prior_seconds = read_config(self.path).get("charged_seconds", 0.0) if self.path.exists() else 0.0
        self.prior_generations = read_config(self.path).get("generation_attempts", 0) if self.path.exists() else 0
        self.started = time.monotonic()
        self.attempts = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.heartbeat = threading.Thread(target=self._heartbeat, name="gpu-budget-heartbeat", daemon=True)
        self.heartbeat.start()

    def _heartbeat(self) -> None:
        while not self.stop_event.wait(1.0):
            self.snapshot()

    def close(self) -> None:
        self.snapshot()
        self.stop_event.set()
        self.heartbeat.join(timeout=2)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            charged = self.prior_seconds + time.monotonic() - self.started
            value = {"charged_seconds": charged, "cap_seconds": MAX_GPU_SECONDS,
                     "generation_attempts": self.prior_generations + self.attempts,
                     "generation_attempt_cap": MAX_GENERATIONS, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            write_json(self.path, value)
            return value

    def check(self, *, generation: bool = False) -> None:
        value = self.snapshot()
        if value["charged_seconds"] >= MAX_GPU_SECONDS - 45:
            raise RuntimeError("Frozen 3-hour GPU wall-clock cap reached")
        if generation:
            if value["generation_attempts"] >= MAX_GENERATIONS:
                raise RuntimeError("Frozen 1024-generation cap reached")
            self.attempts += 1
            self.snapshot()


def model_input_capture(client):
    original = client._inputs
    events: list[dict[str, Any]] = []
    def capture(messages, *args, **kwargs):
        encoded = original(messages, *args, **kwargs)
        ids = encoded["input_ids"][0].tolist()
        actual = [message.model_dump(mode="json") for message in encoded._budgeted_messages]
        events.append({"prompt_token_ids": ids, "prompt_sha256": digest(ids), "messages": actual,
                       "context_usage": encoded._context_usage})
        return encoded
    client._inputs = capture
    return events


def finite_hook(torch):
    def hook(_module, _inputs, output):
        logits = getattr(output, "logits", None)
        if logits is not None and not torch.isfinite(logits[:, -1, :]).all().item():
            raise FloatingPointError("non-finite final-step model logits")
    return hook


def visible_documents_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        try:
            payload = json.loads(message["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        for doc in (payload.get("result", []) if isinstance(payload, dict) else []):
            if not all(isinstance(doc.get(key), str) for key in ("id", "title", "text")):
                continue
            documents.append({"document_id": doc["id"], "title": doc["title"], "text": doc["text"],
                "body_sha256": hashlib.sha256(doc["text"].encode()).hexdigest(), "message_index": message_index})
    return documents


def model_visible_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop orchestration metadata that the processor does not render."""
    return [{"role": message["role"], "content": message["content"]} for message in messages]


def append_result(folder: Path, arm: str, result: dict[str, Any], case: dict[str, Any], events: list[dict[str, Any]],
                  engine: LocalRetrieval, torch, client, started: float) -> dict[str, Any]:
    trace = result["trace"]
    if len(events) != len(trace):
        raise RuntimeError(f"Actual-input count differs from trace: {case['id']}")
    for step, event in zip(trace, events):
        if list(step.get("prompt_token_ids") or []) != event["prompt_token_ids"]:
            raise RuntimeError(f"Saved prompt token IDs do not match captured input: {case['id']}")
        step["actual_input_messages"] = event["messages"]
        step["actual_prompt_sha256"] = event["prompt_sha256"]
        step["context_usage"] = event["context_usage"]
    audit = read_audit(folder, case["id"])
    observed_ids = {doc["id"] for step in trace for doc in step.get("observation", {}).get("result", [])}
    final_docs = visible_documents_from_messages(events[-1]["messages"]) if events else []
    final_visible_ids = {doc["document_id"] for doc in final_docs}
    required_doc_ids = ({edge["source_evidence"]["document_id"] for edge in audit["required_edges"]
                        if edge.get("source_evidence")} if audit["status"] == "confirmed" else set())
    coverage = support_coverage(required_doc_ids, observed_ids, final_visible_ids)
    aliases = case.get("answer_aliases") or case.get("aliases") or []
    scores = [answer_metrics(result.get("answer"), gold) for gold in [case["answer"]] + aliases]
    row = dict(result, case_id=case["id"], stratum=case["stratum"], arm=arm,
               model_id=arm, adapter_sha256=ADAPTERS[arm]["sha256"],
               answers_scored=[case["answer"]] + aliases,
               **{key: max(score[key] for score in scores) for key in scores[0]},
               invalid_actions=sum(bool(step.get("parse_error")) for step in trace),
               repeated_actions=sum(bool(step.get("repeated")) for step in trace),
               over_budget=result["model_calls"] > 4 or result["searches"] > 3,
               retrieved_document_ids=sorted(observed_ids), final_visible_document_ids=sorted(final_visible_ids),
               **coverage,
               gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
               gpu_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
               gpu_question_seconds=time.monotonic() - started,
               retrieval=engine.metadata("hybrid-rrf-v1"))
    return row


def read_audit(folder: Path, case_id: str) -> dict[str, Any]:
    # The small audit file is stable and read during result materialization.
    path = folder / "source-audit.jsonl"
    for row in rows(path):
        if row["case_id"] == case_id:
            return row
    raise KeyError(case_id)


def code_hashes() -> dict[str, str]:
    return {relative: sha(ROOT / relative) for relative in CODE_FILES}


def runtime_identity() -> dict[str, Any]:
    import importlib.metadata
    import subprocess
    import torch
    packages = {dist.metadata["Name"].lower().replace("_", "-"): dist.version
                for dist in importlib.metadata.distributions() if dist.metadata.get("Name")}
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], check=True,
                            capture_output=True, text=True).stdout.splitlines()
    return {"python": sys.version, "packages": dict(sorted(packages.items())),
            "pip_freeze": sorted(freeze), "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0))}


def preflight(folder: Path) -> dict[str, Any]:
    import torch
    manifest = read_config(folder / "manifest.json")
    if manifest.get("runner_sha256") != sha(Path(__file__)) or code_hashes() != manifest["code_hashes"]:
        raise RuntimeError("Frozen evaluation code differs from the prepared code hashes")
    if sha(folder / "cases.jsonl") != CASES_SHA or sha(folder / "corpus.json") != CORPUS_SHA:
        raise RuntimeError("Frozen case/corpus hash mismatch")
    if sha(ROOT / "configs/demo-web.json") != sha(folder / "settings.json"):
        raise RuntimeError("Frozen product config drift")
    frozen_product = read_config(folder / "settings.json")["product"]
    if (frozen_product.get("max_input_tokens") != 16384 or frozen_product.get("max_new_tokens") != 2048 or
            frozen_product.get("max_model_calls_per_case") != 4 or frozen_product.get("max_searches_per_case") != 3 or
            frozen_product.get("thinking") is not False or frozen_product.get("dtype") != "bfloat16" or
            frozen_product.get("device") != "cuda"):
        raise RuntimeError("Frozen inference settings differ from the approved experiment plan")
    if sha(folder / "retrieval-config.json") != manifest.get("retrieval_config_sha256"):
        raise RuntimeError("Frozen retrieval configuration missing or changed")
    base = Path(manifest["base_model_path"])
    if sha(base / "model.safetensors-00001-of-00001.safetensors") != BASE_SHA:
        raise RuntimeError("Base model weight hash mismatch")
    for arm, adapter in expected_assets().items():
        if sha(adapter / "adapter_model.safetensors") != ADAPTERS[arm]["sha256"]:
            raise RuntimeError(f"Adapter hash mismatch: {arm}")
    cfg = retrieval_config(folder / "retrieval-config.json")
    if (cfg["corpus_sha256"] != CORPUS_SHA or cfg["document_count"] != 2048 or
            cfg["candidate_k"] != 20 or cfg["rrf_k"] != 60 or cfg["top_k"] != 3 or
            cfg["model_revision"] != manifest["retrieval"]["model_revision"]):
        raise RuntimeError("Hybrid retrieval parameters drifted from the frozen configuration")
    index_manifest = Path(cfg["index_dir"]) / "manifest.json"
    if sha(index_manifest) != manifest["retrieval"]["index_manifest_sha256"]:
        raise RuntimeError("Hybrid retrieval index manifest drift")
    engine = LocalRetrieval(cfg)
    status = engine.status()
    if not status["ready"] or status["index_sha256"] != manifest["retrieval"]["index_manifest_sha256"]:
        raise RuntimeError(f"Frozen MiniLM index is not ready: {status}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The frozen CUDA/BF16 runtime is unavailable")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 20 * 1024**3:
        raise RuntimeError(f"Insufficient free GPU memory before inference: {free_bytes} bytes")
    runtime = runtime_identity()
    expected_versions = {"torch": "2.5.1+cu124", "transformers": "5.17.0",
                         "peft": runtime["packages"].get("peft")}
    if (runtime["packages"].get("torch") != expected_versions["torch"] or
            runtime["packages"].get("transformers") != expected_versions["transformers"] or
            expected_versions["peft"] is None):
        raise RuntimeError(f"Runtime model stack is unsupported or incomplete: {expected_versions}")
    lock_path = folder / "runtime-lock.json"
    if lock_path.exists():
        if read_config(lock_path) != runtime:
            raise RuntimeError("Runtime dependency/hardware lock changed after preflight")
    else:
        write_json(lock_path, runtime)
    manifest["runtime_lock_sha256"] = sha(lock_path)
    manifest["runtime_versions"] = expected_versions
    manifest["retrieval_index_sha256"] = status["index_sha256"]
    manifest["status"] = "preflight_passed"
    write_json(folder / "manifest.json", manifest)
    return {"runtime": runtime, "gpu": {"name": runtime["gpu_name"], "free_bytes": free_bytes,
                                         "total_bytes": total_bytes},
            "retrieval": status}


def run(folder: Path, *, resume: bool = False) -> None:
    import gc
    import torch
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools import ToolRegistry
    from mini_agent_rl.tools.search import SearchTool
    import run_v133_goal_q_search as q

    failure_path = folder / "failure.json"
    if resume and failure_path.exists() and read_config(failure_path).get("kind") in {"oom", "non_finite", "hash_drift"}:
        raise RuntimeError("This frozen evaluation stopped on OOM, non-finite values, or weight/code drift; inference cannot resume")
    manifest = read_config(folder / "manifest.json")
    if manifest.get("runner_sha256") != sha(Path(__file__)):
        raise RuntimeError("Runner hash differs from the frozen manifest")
    if manifest.get("status") not in {"preflight_passed", "running", "stopped", "inference_complete"}:
        raise RuntimeError("Run requires a successful frozen preflight")
    if code_hashes() != manifest["code_hashes"]:
        raise RuntimeError("A production loop, model, retriever, or scorer file changed after preparation")
    if sha(folder / "retrieval-config.json") != manifest.get("retrieval_config_sha256"):
        raise RuntimeError("Frozen hybrid retrieval configuration changed")
    if sha(folder / "runtime-lock.json") != manifest.get("runtime_lock_sha256"):
        raise RuntimeError("Frozen dependency lock is missing or changed")
    if runtime_identity() != read_config(folder / "runtime-lock.json"):
        raise RuntimeError("Runtime dependencies or GPU identity drifted")
    if sha(folder / "cases.jsonl") != CASES_SHA or sha(folder / "corpus.json") != CORPUS_SHA:
        raise RuntimeError("Frozen case/corpus hash mismatch")
    settings = read_config(folder / "settings.json")
    if sha(ROOT / "configs/demo-web.json") != sha(folder / "settings.json"):
        # Product settings are copied before retriever overrides in the manifest;
        # local project settings themselves must remain identical.
        raise RuntimeError("Product config drift")
    base = Path(manifest["base_model_path"])
    base_weights = base / "model.safetensors-00001-of-00001.safetensors"
    if sha(base_weights) != BASE_SHA:
        raise RuntimeError("Base model weight hash mismatch")
    for arm, adapter in expected_assets().items():
        adapter_file = adapter / "adapter_model.safetensors"
        if sha(adapter_file) != ADAPTERS[arm]["sha256"]:
            raise RuntimeError(f"Adapter hash mismatch: {arm}")
    cases = rows(folder / "cases.jsonl")
    cfg = retrieval_config(folder / "retrieval-config.json")
    if (cfg["corpus_sha256"] != CORPUS_SHA or cfg["document_count"] != 2048 or
            cfg["candidate_k"] != 20 or cfg["rrf_k"] != 60 or cfg["top_k"] != 3 or
            cfg["model_revision"] != manifest["retrieval"]["model_revision"]):
        raise RuntimeError("Hybrid retrieval config identity mismatch")
    if sha(Path(cfg["index_dir"]) / "manifest.json") != manifest["retrieval"]["index_manifest_sha256"]:
        raise RuntimeError("Hybrid index manifest drift")
    engine = LocalRetrieval(cfg)
    if not engine.status()["ready"]:
        raise RuntimeError("Frozen hybrid index is unavailable or mismatched")
    # Warm the CPU query encoder before timing the fixed-weight GPU run.
    engine.search("index warmup", "hybrid-rrf-v1")
    budget = Budget(folder)
    write_json(folder / "run-state.json", {"status": "running", "resumed": resume,
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "current_arm": "goal-z"})
    product = manifest["product"]
    if (product.get("thinking") is not False or product.get("dtype") != "bfloat16" or
            product.get("max_input_tokens") != 16384 or product.get("max_new_tokens") != 2048 or
            product.get("max_model_calls_per_case") != 4 or product.get("max_searches_per_case") != 3):
        raise RuntimeError("Frozen generation settings drift")
    for arm in ("goal-z", "am-research"):
        target = folder / f"{ADAPTERS[arm]['label']}-ledger.jsonl"
        if target.exists() and not resume:
            raise FileExistsError(target)
        existing = resume_rows(target)
        done = validate_ledger(cases, existing, arm)
        if len(done) == len(cases):
            loading_path = folder / f"{ADAPTERS[arm]['label']}-loading.json"
            if not loading_path.exists():
                raise RuntimeError(f"Completed arm is missing its checkpointed model-load record: {arm}")
            load_record = read_config(loading_path)
            load_record["completed"] = True
            load_record["completed_case_count"] = len(cases)
            write_json(loading_path, load_record)
            write_json(folder / "run-state.json", {"status": "running", "resumed": resume,
                "completed_arm": arm, "current_arm": "am-research" if arm == "goal-z" else None})
            continue
        try:
            budget.check()
        except RuntimeError as exc:
            write_json(folder / "failure.json", {"arm": arm, "kind": "budget_exhausted",
                "stage": "before_model_load", "error": repr(exc),
                "completed_case_ids": sorted(done), "gpu_budget": budget.snapshot()})
            write_json(folder / "run-state.json", {"status": "stopped", "arm": arm,
                "failure": "budget_exhausted"})
            raise
        adapter = expected_assets()[arm]
        config = LocalModelConfig(model_id=str(base), adapter_path=str(adapter),
            max_input_tokens=product["max_input_tokens"], max_new_tokens=product["max_new_tokens"],
            enable_thinking=False, deterministic=True, device="cuda", dtype="bfloat16", load_in_4bit=False,
            action_completion=product["action_completion"], observation_encoding=product["observation_encoding"],
            observation_presentation=product["observation_presentation"], action_constraint="none",
            context_aware_decoding=None, recompute_old_logprobs=False)
        client = TransformersModelClient(config)
        load_started = time.monotonic()
        try:
            client._load()
        except BaseException as exc:
            error_text = str(exc).lower()
            kind = "oom" if "out of memory" in error_text else "non_finite" if isinstance(exc, FloatingPointError) else "runtime"
            write_json(folder / "failure.json", {"arm": arm, "kind": kind, "stage": "model_load",
                "error": repr(exc), "completed_case_ids": sorted(done), "gpu_budget": budget.snapshot()})
            write_json(folder / "run-state.json", {"status": "stopped", "arm": arm, "failure": kind})
            raise
        loading_seconds = time.monotonic() - load_started
        loading_path = folder / f"{ADAPTERS[arm]['label']}-loading.json"
        load_record = read_config(loading_path) if loading_path.exists() else {"load_events": []}
        load_record.setdefault("load_events", []).append({"model_loading_seconds": loading_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        load_record["total_model_loading_seconds"] = sum(event["model_loading_seconds"] for event in load_record["load_events"])
        load_record["completed"] = False
        write_json(loading_path, load_record)
        original_generate = client.generate
        generation_outputs: list[dict[str, Any]] = []
        async def budgeted_generate(*args, **kwargs):
            budget.check(generation=True)
            response = await original_generate(*args, **kwargs)
            generation_outputs.append({"raw_text": response.raw_text, "content": response.content,
                "parse_error": response.parse_error, "tool_call": (response.tool_call.model_dump(mode="json")
                    if response.tool_call else None), "usage": response.usage})
            return response
        client.generate = budgeted_generate
        hook = client._model.register_forward_hook(finite_hook(torch))
        torch.cuda.reset_peak_memory_stats()
        wrapped = q.SearchTextAliasClient(client)
        registry = ToolRegistry()
        registry.register(SearchTool(engine.documents, "hybrid-rrf-v1", retrieval=engine))
        for index, case in enumerate(cases):
            if case["id"] in done:
                continue
            question_started = time.monotonic()
            events = model_input_capture(client)
            generation_outputs.clear()
            try:
                budget.check()
                result = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False,
                    seed=42, product_config=product))
                budget.check()
                if result["model_calls"] > 4 or result["searches"] > 3 or result["auxiliary_calls"] != 0:
                    raise RuntimeError(f"Per-question frozen budget violated: {case['id']}")
                row = append_result(folder, arm, result, case, events, engine, torch, client, question_started)
                append_jsonl(target, row)
                done.add(case["id"])
                budget.snapshot()
                print(json.dumps({"event": "case_complete", "arm": arm, "case": case["id"],
                                  "done": len(done), "total": len(cases), "reason": row["reason"],
                                  "normalized_em": row["normalized_exact_match"]}), flush=True)
                acceptance_path = folder / f"{ADAPTERS[arm]['label']}-first-two.json"
                if not acceptance_path.exists() and {case["id"] for case in cases[:2]}.issubset(done):
                    acceptance = engineering_acceptance(resume_rows(target), cases)
                    acceptance["actual_generations"] = budget.prior_generations + budget.attempts
                    write_json(acceptance_path, acceptance)
                    print(json.dumps({"event": "engineering_acceptance_passed", "arm": arm,
                                      "case_ids": acceptance["case_ids"]}), flush=True)
            except BaseException as exc:
                error_text = str(exc).lower()
                kind = ("non_finite" if isinstance(exc, FloatingPointError) else "oom" if "out of memory" in error_text
                        else "budget_exhausted" if "cap reached" in error_text else "runtime")
                write_json(folder / "failure.json", {"arm": arm, "case_id": case["id"], "kind": kind,
                    "error": repr(exc), "completed_case_ids": sorted(done), "gpu_budget": budget.snapshot(),
                    "partial_actual_inputs": events, "completed_generation_outputs": list(generation_outputs)})
                write_json(folder / "run-state.json", {"status": "stopped", "arm": arm, "failure": kind})
                raise
        if len(done) != 128:
            raise RuntimeError(f"Incomplete arm {arm}: {len(done)}/128")
        acceptance_path = folder / f"{ADAPTERS[arm]['label']}-first-two.json"
        if not acceptance_path.exists():
            acceptance = engineering_acceptance(resume_rows(target), cases)
            acceptance["actual_generations"] = budget.prior_generations + budget.attempts
            write_json(acceptance_path, acceptance)
        load_record["completed"] = True
        load_record["completed_case_count"] = len(done)
        write_json(loading_path, load_record)
        hook.remove()
        del client, wrapped, registry, hook
        gc.collect()
        torch.cuda.empty_cache()
        budget.snapshot()
        write_json(folder / "run-state.json", {"status": "running", "completed_arm": arm,
                    "current_arm": "am-research" if arm == "goal-z" else None})
    budget.close()
    write_json(folder / "run-state.json", {"status": "inference_complete", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "gpu_budget": budget.snapshot()})


def make_review_packets(folder: Path) -> list[dict[str, Any]]:
    from transformers import AutoProcessor
    manifest = read_config(folder / "manifest.json")
    cases = {row["id"]: row for row in rows(folder / "cases.jsonl")}
    audits = {row["case_id"]: row for row in rows(folder / "source-audit.jsonl")}
    if any(not (folder / f"{ADAPTERS[arm]['label']}-ledger.jsonl").exists() for arm in ADAPTERS):
        raise FileNotFoundError("Both inference ledgers are required before packet creation")
    processor = AutoProcessor.from_pretrained(manifest["base_model_path"], local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    seen: dict[str, dict[str, Any]] = {}
    mapping = []
    for arm in ("goal-z", "am-research"):
        for result in rows(folder / f"{ADAPTERS[arm]['label']}-ledger.jsonl"):
            final = result["trace"][-1] if result.get("trace") else {}
            ids = final.get("prompt_token_ids")
            if not ids:
                raise ValueError(f"Missing actual final input token IDs: {result['case_id']}/{arm}")
            actual_messages = final.get("actual_input_messages")
            if not actual_messages:
                raise ValueError(f"Missing actual final input messages: {result['case_id']}/{arm}")
            template_messages = model_visible_messages(actual_messages)
            prompt = processor.apply_chat_template(template_messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            rendered = processor.apply_chat_template(template_messages, tokenize=True,
                return_dict=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False)
            rendered_ids = rendered["input_ids"][0].tolist()
            if list(rendered_ids) != list(ids):
                raise ValueError(f"Actual saved messages do not reproduce captured prompt tokens: {result['case_id']}/{arm}")
            if result["question"].casefold() not in prompt.casefold():
                raise ValueError(f"Question absent from saved actual input: {result['case_id']}/{arm}")
            case, audit = cases[result["case_id"]], audits[result["case_id"]]
            prompt_sha = digest(list(ids))
            key = digest([case["question"], prompt_sha, result.get("answer"), result["reason"]])
            docs = visible_documents_from_messages(actual_messages)
            item = {"review_id": key, "question": case["question"], "answer": result.get("answer"),
                    "termination_reason": result["reason"], "source_status": audit["status"],
                    "source_reason": audit["reason"], "gold_answer": case["answer"],
                    "answer_aliases": case.get("answer_aliases") or case.get("aliases") or [],
                    "source_chain": case.get("chain", []), "required_edges": audit["required_edges"],
                    "actual_prompt": prompt, "actual_input_messages": template_messages,
                    "prompt_token_ids": ids, "prompt_sha256": prompt_sha,
                    "visible_documents": docs,
                    "visible_document_ids": sorted({doc["document_id"] for doc in docs}),
                    "review_status": "pending"}
            existing = seen.get(key)
            if existing and digest(existing) != digest(item):
                raise ValueError(f"Review deduplication key collision: {key}")
            seen[key] = item
            mapping.append({"review_id": key, "case_id": result["case_id"], "arm": arm,
                            "adapter_sha256": result["adapter_sha256"]})
    packet_path = folder / "review-packets.jsonl"
    map_path = folder / "review-map.jsonl"
    write_jsonl_frozen(packet_path, sorted(seen.values(), key=lambda row: row["review_id"]))
    write_jsonl_frozen(map_path, mapping)
    manifest["review_packet_count"] = len(seen)
    manifest["review_packet_sha256"] = sha(packet_path)
    manifest["review_map_sha256"] = sha(map_path)
    write_json(folder / "manifest.json", manifest)
    return sorted(seen.values(), key=lambda row: row["review_id"])


def paired_delta(before: dict[str, Any], after: dict[str, Any], field: str, ids: list[str], draws: int = 2000) -> dict[str, Any]:
    lookup_a, lookup_b = before, after
    deltas = [float(lookup_b[cid].get(field, 0)) - float(lookup_a[cid].get(field, 0)) for cid in ids]
    rng = random.Random(42)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(draws)) if deltas else []
    return {"pairs": len(deltas), "improved": sum(delta > 0 for delta in deltas),
            "regressed": sum(delta < 0 for delta in deltas), "unchanged": sum(delta == 0 for delta in deltas),
            "mean_delta": sum(deltas)/len(deltas) if deltas else None,
            "bootstrap_95": [samples[int(draws*.025)], samples[int(draws*.975)-1]] if samples else None,
            "bootstrap_draws": draws if samples else 0, "bootstrap_seed": 42}


def validate_citation(packet: dict[str, Any], citation: dict[str, Any], tokenizer,
                      encoded_prompt=None) -> dict[str, Any] | None:
    doc = next((row for row in packet["visible_documents"] if row["document_id"] == citation.get("document_id")), None)
    quote = citation.get("quote")
    if doc is None or not isinstance(quote, str) or not quote or quote not in doc["text"]:
        return None
    prompt = packet["actual_prompt"]
    actual_ids = packet["prompt_token_ids"]
    encoded = encoded_prompt or tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) != list(actual_ids):
        raise ValueError(f"Actual prompt cannot be re-tokenized exactly: {packet['review_id']}")
    message = packet["actual_input_messages"][doc["message_index"]]
    message_content = message.get("content", "")
    escaped_body = json.dumps(doc["text"], ensure_ascii=False)[1:-1]
    escaped_quote = json.dumps(quote, ensure_ascii=False)[1:-1]
    body_start = message_content.find(escaped_body)
    if body_start < 0:
        return None
    prompt_message_start = message_span(packet, doc["message_index"])
    if prompt_message_start < 0:
        return None
    quote_start = message_content.find(escaped_quote, body_start)
    if quote_start < 0:
        return None
    char_start = prompt_message_start + quote_start
    char_end = char_start + len(escaped_quote)
    token_indices = [i for i, (start, end) in enumerate(encoded["offset_mapping"]) if start < char_end and end > char_start]
    if not token_indices:
        return None
    return {"document_id": doc["document_id"], "title": doc["title"], "quote": quote,
            "body_sha256": doc["body_sha256"], "input_char_span": [char_start, char_end],
            "input_token_span": [min(token_indices), max(token_indices)+1], "message_index": doc["message_index"]}


def message_span(packet: dict[str, Any], target_index: int) -> int:
    cursor = 0
    for index, message in enumerate(packet["actual_input_messages"]):
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError(f"Unexpected non-text message in frozen input: {packet['review_id']}")
        start = packet["actual_prompt"].find(content, cursor)
        if start < 0:
            raise ValueError(f"Saved input message cannot be located in rendered prompt: {packet['review_id']}")
        if index == target_index:
            return start
        cursor = start + len(content)
    raise ValueError(f"Message index outside saved actual prompt: {packet['review_id']}")


def validate_vote(packet: dict[str, Any], vote: dict[str, Any], tokenizer, encoded_prompt) -> dict[str, Any]:
    key = packet["review_id"]
    category = vote.get("semantic_category")
    if category not in REVIEW_CATEGORIES:
        raise ValueError(f"Invalid semantic category: {key}")
    if packet["source_status"] == "unresolved" and category != "indeterminate":
        raise ValueError(f"Unresolved source cannot receive a definitive semantic label: {key}")
    if packet["termination_reason"] != "answer" and category not in {"no_answer", "indeterminate"}:
        raise ValueError(f"Non-answer termination must be recorded as no-answer/indeterminate: {key}")
    required = {edge["edge_id"]: edge for edge in packet["required_edges"]}
    covered = set(vote.get("covered_edge_ids", []))
    if not covered.issubset(required):
        raise ValueError(f"Reviewer marks an unknown required edge: {key}")
    citations = []
    for citation in vote.get("citations", []):
        edge_id = citation.get("edge_id")
        if edge_id not in required:
            raise ValueError(f"Citation lacks a valid required edge ID: {key}")
        edge_doc_id = required[edge_id].get("source_evidence", {}).get("document_id")
        if citation.get("document_id") != edge_doc_id:
            raise ValueError(f"Citation document does not match its frozen source edge: {key}/{edge_id}")
        found = validate_citation(packet, citation, tokenizer, encoded_prompt)
        if found is None:
            raise ValueError(f"Citation is absent from actual final input: {key}")
        citations.append(dict(found, edge_id=edge_id))
    citation_edges = {citation["edge_id"] for citation in citations}
    if not citation_edges.issubset(covered):
        raise ValueError(f"Citation supplied for an edge not marked covered: {key}")
    vote["validated_citations"] = citations
    vote["evidence_supported"] = (category == "correct" and bool(required) and covered == set(required)
        and citation_edges == set(required)
        and all(edge.get("status") != "unresolved" for edge in required.values()))
    return vote


def validate_review_rows(folder: Path, reviews_a: list[dict[str, Any]], reviews_b: list[dict[str, Any]],
                         adjudications: list[dict[str, Any]], tokenizer) -> dict[str, Any]:
    packets = rows(folder / "review-packets.jsonl")
    packet_ids = {packet["review_id"] for packet in packets}
    if len(packet_ids) != len(packets):
        raise ValueError("Duplicate packet IDs")
    for name, values in (("A", reviews_a), ("B", reviews_b)):
        ids = [row["review_id"] for row in values]
        if set(ids) != packet_ids or len(ids) != len(packet_ids):
            raise ValueError(f"Reviewer {name} has missing/duplicate review IDs")
    left, right = ({row["review_id"]: row for row in vals} for vals in (reviews_a, reviews_b))
    packet_map = {row["review_id"]: row for row in packets}
    encoded_cache = {}
    def encoded_for(packet):
        key = packet["review_id"]
        if key not in encoded_cache:
            encoded = tokenizer(packet["actual_prompt"], add_special_tokens=False, return_offsets_mapping=True)
            if list(encoded["input_ids"]) != list(packet["prompt_token_ids"]):
                raise ValueError(f"Actual prompt cannot be re-tokenized exactly: {key}")
            encoded_cache[key] = encoded
        return encoded_cache[key]
    for reviewer in (left, right):
        for key, vote in reviewer.items():
            validate_vote(packet_map[key], vote, tokenizer, encoded_for(packet_map[key]))
    def vote_fingerprint(vote):
        citations = sorted((row.get("edge_id"), row.get("document_id"), row.get("quote"))
                           for row in vote.get("citations", []))
        return vote.get("semantic_category"), sorted(vote.get("covered_edge_ids", [])), citations
    disputed = {key for key in packet_ids if vote_fingerprint(left[key]) != vote_fingerprint(right[key])}
    root_review_required = disputed | {key for key in packet_ids
        if left[key].get("evidence_supported") or right[key].get("evidence_supported")}
    adjudication_map = {row["review_id"]: row for row in adjudications}
    if set(adjudication_map) != root_review_required or len(adjudications) != len(root_review_required):
        raise ValueError("Root adjudications must cover every disagreement and every claimed evidence success")
    accepted = {}
    for packet in packets:
        key = packet["review_id"]
        vote = adjudication_map.get(key, left[key])
        if key in adjudication_map:
            vote = dict(vote)
            validate_vote(packet, vote, tokenizer, encoded_for(packet))
        accepted[key] = vote
    return {"packets": packet_map, "reviewer_a": left, "reviewer_b": right,
            "accepted": accepted, "disputed_count": len(disputed),
            "root_review_count": len(root_review_required)}


def prepare_root_review(folder: Path) -> list[dict[str, Any]]:
    from transformers import AutoProcessor
    packets = rows(folder / "review-packets.jsonl")
    a_rows, b_rows = rows(folder / "review-a.jsonl"), rows(folder / "review-b.jsonl")
    packet_map = {packet["review_id"]: packet for packet in packets}
    if len(packet_map) != len(packets):
        raise ValueError("Duplicate blinded packet ID")
    left = {row["review_id"]: row for row in a_rows}
    right = {row["review_id"]: row for row in b_rows}
    if len(left) != len(a_rows) or len(right) != len(b_rows) or set(left) != set(packet_map) or set(right) != set(packet_map):
        raise ValueError("Both independent reviewers must cover every unique blinded packet exactly once")
    manifest = read_config(folder / "manifest.json")
    processor = AutoProcessor.from_pretrained(manifest["base_model_path"], local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded_cache = {}
    for key, packet in packet_map.items():
        encoded_cache[key] = tokenizer(packet["actual_prompt"], add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded_cache[key]["input_ids"]) != list(packet["prompt_token_ids"]):
            raise ValueError(f"Captured prompt token identity mismatch: {key}")
        validate_vote(packet, left[key], tokenizer, encoded_cache[key])
        validate_vote(packet, right[key], tokenizer, encoded_cache[key])
    def fingerprint(vote):
        citations = sorted((item.get("edge_id"), item.get("document_id"), item.get("quote"))
                           for item in vote.get("citations", []))
        return vote.get("semantic_category"), sorted(vote.get("covered_edge_ids", [])), citations
    targets = {key for key in packet_map if fingerprint(left[key]) != fingerprint(right[key]) or
               left[key].get("evidence_supported") or right[key].get("evidence_supported")}
    root_packets = [{"review_id": key, "root_action": "resolve_disagreement" if fingerprint(left[key]) != fingerprint(right[key])
                    else "independently_verify_claimed_evidence_success", "packet": packet_map[key],
                    "reviewer_a": left[key], "reviewer_b": right[key]}
                    for key in sorted(targets)]
    output = folder / "root-review-packets.jsonl"
    write_jsonl_frozen(output, root_packets)
    manifest["root_review_packet_count"] = len(root_packets)
    manifest["root_review_packet_sha256"] = sha(output)
    write_json(folder / "manifest.json", manifest)
    return root_packets


def freeze_postprocessing(folder: Path) -> dict[str, Any]:
    manifest = read_config(folder / "manifest.json")
    state = read_config(folder / "run-state.json")
    if state.get("status") != "inference_complete":
        raise RuntimeError("Postprocessing code may only be updated after both inference arms are complete")
    cases = rows(folder / "cases.jsonl")
    before, after = validate_complete_pair(cases, rows(folder / "A-ledger.jsonl"), rows(folder / "B-ledger.jsonl"))
    if set(before) != set(after) or len(before) != 128:
        raise RuntimeError("Cannot freeze postprocessing before validating all 256 paired outputs")
    manifest.setdefault("inference_code_hashes", manifest["code_hashes"])
    manifest.setdefault("inference_runner_sha256", manifest["runner_sha256"])
    current_hashes = code_hashes()
    current_runner = sha(Path(__file__))
    manifest["postprocessing_code_hashes"] = current_hashes
    manifest["postprocessing_runner_sha256"] = current_runner
    manifest["postprocessing_freeze_reason"] = (
        "Use the same AutoProcessor chat template as inference, exclude non-rendered random tool-call IDs from "
        "blind-packet deduplication, and report completion, budget, latency, and GPU-memory diagnostics."
    )
    manifest["inference_reproducibility_note"] = (
        "Greedy decoding and seed42 were fixed for both arms, but run.log records CUDA nondeterminism warnings "
        "for cuBLAS and cumsum_cuda; bitwise repeatability is therefore not guaranteed."
    )
    if (folder / "run.log").exists():
        manifest["run_log_sha256"] = sha(folder / "run.log")
    manifest["code_hashes"] = current_hashes
    manifest["runner_sha256"] = current_runner
    write_json(folder / "manifest.json", manifest)
    return manifest


def aggregate(rows_: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows_)
    if not count:
        return {"count": 0}
    semantic_correct = sum(row.get("semantic_category") == "correct" for row in rows_)
    evidence_success = sum(row.get("evidence_supported", False) for row in rows_)
    if evidence_success > semantic_correct:
        raise ValueError("Evidence-supported success exceeds semantically correct answers")
    latencies = sorted(row["elapsed_seconds"] for row in rows_)
    p95 = latencies[max(0, math.ceil(.95 * count) - 1)]
    total_support_docs = sum(row.get("support_documents_expected", 0) for row in rows_)
    visible_support_docs = sum(row.get("final_visible_support_count", 0) for row in rows_)
    return {"count": count, "strict_em": sum(row["strict_exact_match"] for row in rows_),
            "strict_em_rate": sum(row["strict_exact_match"] for row in rows_)/count,
            "normalized_em": sum(row["normalized_exact_match"] for row in rows_),
            "normalized_em_rate": sum(row["normalized_exact_match"] for row in rows_)/count,
            "mean_token_f1": sum(row["token_f1"] for row in rows_)/count,
            "answer_completion": sum(row["reason"] == "answer" for row in rows_),
            "answer_completion_rate": sum(row["reason"] == "answer" for row in rows_)/count,
            "semantic_correct": semantic_correct,
            "semantic_partial": sum(row.get("semantic_category") == "partial" for row in rows_),
            "semantic_incorrect": sum(row.get("semantic_category") == "factual_error" for row in rows_),
            "semantic_no_answer": sum(row.get("semantic_category") == "no_answer" for row in rows_),
            "semantic_indeterminate": sum(row.get("semantic_category") == "indeterminate" for row in rows_),
            "evidence_supported_success": evidence_success,
            "complete_chain_visible": sum(row.get("complete_chain_visible", False) for row in rows_),
            "support_documents_expected": total_support_docs,
            "support_documents_retrieved": sum(row.get("retrieved_support_count", 0) for row in rows_),
            "support_documents_visible_at_decision": visible_support_docs,
            "support_document_final_visibility_rate": visible_support_docs/total_support_docs if total_support_docs else None,
            "invalid_actions": sum(row.get("invalid_actions", 0) > 0 for row in rows_),
            "repeated_actions": sum(row.get("repeated_actions", 0) > 0 for row in rows_),
            "over_budget": sum(row.get("over_budget", False) for row in rows_),
            "model_call_limit_exhausted": sum(row["model_calls"] >= 4 and row["reason"] != "answer" for row in rows_),
            "search_limit_reached": sum(row["searches"] >= 3 for row in rows_),
            "total_model_calls": sum(row["model_calls"] for row in rows_),
            "total_searches": sum(row["searches"] for row in rows_),
            "total_prompt_tokens": sum(len(step.get("prompt_token_ids") or []) for row in rows_ for step in row.get("trace", [])),
            "total_generated_tokens": sum(len(step.get("action_token_ids") or []) for row in rows_ for step in row.get("trace", [])),
            "mean_calls": sum(row["model_calls"] for row in rows_)/count,
            "mean_searches": sum(row["searches"] for row in rows_)/count,
            "mean_seconds": sum(row["elapsed_seconds"] for row in rows_)/count,
            "p50_seconds": latencies[(count - 1)//2], "p95_seconds": p95,
            "mean_gpu_question_seconds": sum(row["gpu_question_seconds"] for row in rows_)/count,
            "max_gpu_peak_allocated_bytes": max(row.get("gpu_peak_allocated_bytes", 0) for row in rows_),
            "max_gpu_peak_reserved_bytes": max(row.get("gpu_peak_reserved_bytes", 0) for row in rows_)}


def summarize(folder: Path, adjudications: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoProcessor
    a_reviews, b_reviews = rows(folder / "review-a.jsonl"), rows(folder / "review-b.jsonl")
    processor = AutoProcessor.from_pretrained(read_config(folder / "manifest.json")["base_model_path"], local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    validated = validate_review_rows(folder, a_reviews, b_reviews, adjudications, tokenizer)
    write_jsonl_frozen(folder / "semantic-review-adjudicated.jsonl", [
        {"review_id": key, "semantic_category": vote["semantic_category"],
         "reason": vote.get("reason"), "covered_edge_ids": vote.get("covered_edge_ids", []),
         "validated_citations": vote.get("validated_citations", []),
         "evidence_supported": vote.get("evidence_supported", False)}
        for key, vote in sorted(validated["accepted"].items())])
    mapping = rows(folder / "review-map.jsonl")
    by_arm_case = {(row["arm"], row["case_id"]): validated["accepted"][row["review_id"]] for row in mapping}
    source = {row["case_id"]: row for row in rows(folder / "source-audit.jsonl")}
    case_rows = rows(folder / "cases.jsonl")
    all_results = {arm: rows(folder / f"{ADAPTERS[arm]['label']}-ledger.jsonl") for arm in ADAPTERS}
    before, after = validate_complete_pair(case_rows, all_results["goal-z"], all_results["am-research"])
    report: dict[str, Any] = {"run_id": RUN_ID, "status": "complete", "kind": "exposed paired diagnostic; not independent generalization",
        "source_status_counts": dict(Counter(row["status"] for row in source.values())),
        "review_packets": len(validated["packets"]), "reviewed_outputs": len(mapping),
        "identical_outputs_deduplicated": len(mapping) - len(validated["packets"]),
        "review_disagreements_adjudicated": validated["disputed_count"],
        "review_successes_root_checked": validated["root_review_count"] - validated["disputed_count"],
        "gpu_budget": read_config(folder / "gpu-budget.json"), "by_arm": {}, "by_stratum": {}, "paired": {}}
    ids = [row["id"] for row in case_rows]
    for arm in ADAPTERS:
        decorated = []
        for item in all_results[arm]:
            vote = by_arm_case[arm, item["case_id"]]
            decorated.append(dict(item, semantic_category=vote["semantic_category"],
                                  evidence_supported=vote["evidence_supported"],
                                  semantic_reason=vote.get("reason"),
                                  validated_citations=vote.get("validated_citations", [])))
        all_results[arm] = decorated
        report["by_arm"][arm] = {"all_128": aggregate(decorated), "confirmed_correct_all128_rate":
            sum(row["semantic_category"] == "correct" for row in decorated)/128,
            "common_adjudicable_denominator": None}
        for stratum in STRATA:
            report["by_stratum"].setdefault(stratum, {})[arm] = aggregate([row for row in decorated if row["stratum"] == stratum])
        for group, predicate in (("natural_104", lambda row: row["stratum"] != "constructed_singlehop"),):
            report["by_arm"][arm][group] = aggregate([row for row in decorated if predicate(row)])
    before = {row["case_id"]: row for row in all_results["goal-z"]}
    after = {row["case_id"]: row for row in all_results["am-research"]}
    common = [cid for cid in ids if source[cid]["status"] == "confirmed" and
              by_arm_case["goal-z", cid]["semantic_category"] != "indeterminate" and
              by_arm_case["am-research", cid]["semantic_category"] != "indeterminate"]
    natural_ids = [cid for cid in ids if before[cid]["stratum"] != "constructed_singlehop"]
    natural_common = [cid for cid in natural_ids if source[cid]["status"] == "confirmed" and
                      by_arm_case["goal-z", cid]["semantic_category"] != "indeterminate" and
                      by_arm_case["am-research", cid]["semantic_category"] != "indeterminate"]
    for arm in ADAPTERS:
        rows_by_id = {row["case_id"]: row for row in all_results[arm]}
        report["by_arm"][arm]["common_adjudicable_denominator"] = len(common)
        report["by_arm"][arm]["common_adjudicable_semantic_correct"] = sum(rows_by_id[cid]["semantic_category"] == "correct" for cid in common)
        report["by_arm"][arm]["common_adjudicable_semantic_rate"] = (
            report["by_arm"][arm]["common_adjudicable_semantic_correct"]/len(common) if common else None)
        report["by_arm"][arm]["natural_104_common_adjudicable_denominator"] = len(natural_common)
        report["by_arm"][arm]["natural_104_common_adjudicable_semantic_correct"] = sum(
            rows_by_id[cid]["semantic_category"] == "correct" for cid in natural_common)
        report["by_arm"][arm]["natural_104_common_adjudicable_semantic_rate"] = (
            report["by_arm"][arm]["natural_104_common_adjudicable_semantic_correct"]/len(natural_common)
            if natural_common else None)
    report["paired"] = {"strict_em": paired_delta(before, after, "strict_exact_match", ids),
        "normalized_em": paired_delta(before, after, "normalized_exact_match", ids),
        "token_f1": paired_delta(before, after, "token_f1", ids)}
    # Paired bootstrap helper reads row fields; keep temporary binary outcomes out of source ledgers.
    for cid in common:
        before[cid]["_semantic_correct"] = float(before[cid]["semantic_category"] == "correct")
        after[cid]["_semantic_correct"] = float(after[cid]["semantic_category"] == "correct")
    report["paired"]["semantic_correct_common"] = paired_delta(before, after, "_semantic_correct", common)
    report["paired"]["strict_em_natural_104"] = paired_delta(before, after, "strict_exact_match", natural_ids)
    report["paired"]["normalized_em_natural_104"] = paired_delta(before, after, "normalized_exact_match", natural_ids)
    report["paired"]["token_f1_natural_104"] = paired_delta(before, after, "token_f1", natural_ids)
    report["paired"]["semantic_correct_natural_104_common"] = paired_delta(
        before, after, "_semantic_correct", natural_common)
    for cid in ids:
        before[cid]["_evidence_supported"] = float(before[cid].get("evidence_supported", False))
        after[cid]["_evidence_supported"] = float(after[cid].get("evidence_supported", False))
    report["paired"]["evidence_supported_all128"] = paired_delta(before, after, "_evidence_supported", ids)
    report["paired"]["evidence_supported_common_adjudicable"] = paired_delta(before, after, "_evidence_supported", common)
    report["paired"]["evidence_supported_natural_104"] = paired_delta(
        before, after, "_evidence_supported", natural_ids)
    report["paired"]["evidence_supported_natural_104_common"] = paired_delta(
        before, after, "_evidence_supported", natural_common)
    for arm_rows in (before, after):
        for group_rows in [arm_rows[cid] for cid in ids]:
            if group_rows.get("evidence_supported") and group_rows.get("semantic_category") != "correct":
                raise ValueError("Evidence-supported count cannot exceed semantic-correct count")
    # Per-case stable output is the single source for every aggregate table.
    for item in (*before.values(), *after.values()):
        item.pop("_semantic_correct", None)
        item.pop("_evidence_supported", None)
    write_jsonl_frozen(folder / "paired-results.jsonl", [
        {"case_id": cid, "stratum": before[cid]["stratum"], "question": before[cid]["question"],
         "source_status": source[cid]["status"], "goal_z": before[cid], "am": after[cid]}
        for cid in ids])
    loading = {arm: read_config(folder / f"{ADAPTERS[arm]['label']}-loading.json") for arm in ADAPTERS}
    report["model_loading"] = loading
    write_chinese_report(folder, report)
    write_json(folder / "report.json", report)
    manifest = read_config(folder / "manifest.json")
    manifest["status"] = "complete"
    manifest["report_sha256"] = sha(folder / "report.json")
    manifest["paired_results_sha256"] = sha(folder / "paired-results.jsonl")
    manifest["chinese_report_sha256"] = sha(folder / "对比报告.md")
    write_json(folder / "manifest.json", manifest)
    return report


def write_chinese_report(folder: Path, report: dict[str, Any]) -> None:
    labels = {"goal-z": "Goal Z seed42", "am-research": "AM seed42-A"}
    def number(x, digits=3):
        return "—" if x is None else f"{x:.{digits}f}"
    def pct(x):
        return "—" if x is None else f"{100*x:.1f}%"
    def metric_line(name, group):
        lines = []
        for arm, title in labels.items():
            item = report["by_arm"][arm][group] if group in report["by_arm"][arm] else report["by_stratum"][group][arm]
            lines.append(f"| {title} | {item['count']} | {item['strict_em']}/{item['count']} ({pct(item['strict_em_rate'])}) | "
                         f"{item['normalized_em']}/{item['count']} ({pct(item['normalized_em_rate'])}) | "
                         f"{number(item['mean_token_f1'])} | {item['semantic_correct']}/{item['count']} | "
                         f"{item['semantic_indeterminate']} | {item['evidence_supported_success']}/{item['count']} | "
                         f"{item['answer_completion']}/{item['count']} ({pct(item['answer_completion_rate'])}) | "
                         f"{item['final_visible_support_documents'] if 'final_visible_support_documents' in item else item['support_documents_visible_at_decision']}/"
                         f"{item['support_documents_expected']} ({pct(item['support_document_final_visibility_rate'])}) | "
                         f"{item['total_model_calls']} / {item['total_searches']} | {number(item['p95_seconds'], 1)} s |")
        return lines
    lines = ["# AM seed42-A 与 Goal Z seed42：混合检索完整对比", "",
        "本轮在冻结的已暴露题集上做配对诊断，不代表独立泛化测试，也不直接决定产品选模。两臂使用同一批128题、2048篇语料、MiniLM索引及 hybrid RRF（双路各20篇、RRF k=60、Top 3）。",
        "", f"来源状态：已确认 {report['source_status_counts'].get('confirmed', 0)} 题；无法裁定 {report['source_status_counts'].get('unresolved', 0)} 题。无法裁定项保留在EM/F1与全题分母中，语义标签记为‘无法判定’，不当作模型答错。",
        "固定使用 greedy 和 seed42；运行日志记录了 cuBLAS 与 cumsum_cuda 的 CUDA 非确定性警告，因此协议一致，但不承诺逐位复现。",
        "推理固定配置为 BF16、thinking 关闭、每题输入/输出上限 16384/2048 tokens，最多4次模型调用和3次搜索；Goal Z 先运行、AM 随后运行，每个适配器仅加载一次。",
        "", "语义正确要求回答准确且充分，允许等价表达；证据支持还要求答案语义正确，并且每条必要关系都有逐关系引用，引用文字实际出现在最终决策输入中。仅被搜索返回但未进入最终输入的文档不算支持。",
        "", "## 指标总表", "", "| 模型 | 题数 | 严格EM | 归一化EM | Token F1均值 | 语义正确数 | 无法判定数 | 证据支持成功数 | 完成回答 | 最终可见支持文档 | 调用/搜索 | P95耗时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines.extend(metric_line("all_128", "all_128"))
    lines.extend(["", "## 分组结果", "", "| 分组 / 模型 | 题数 | 严格EM | 归一化EM | Token F1均值 | 语义正确 | 无法判定 | 证据支持 | 完成回答 | 最终可见支持文档 | 调用/搜索 | P95耗时 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for group, title in (("natural_104", "自然多跳104题"), ("development", "开发题64题"),
                         ("natural_supplement", "自然补充40题"), ("constructed_singlehop", "人工单跳诊断24题")):
        for line in metric_line(title, group):
            lines.append(line.replace("| Goal Z seed42 |", f"| {title} / Goal Z seed42 |", 1)
                             .replace("| AM seed42-A |", f"| {title} / AM seed42-A |", 1))
    goal_sem = report["by_arm"]["goal-z"]
    am_sem = report["by_arm"]["am-research"]
    common_n = goal_sem["common_adjudicable_denominator"]
    natural_common_n = goal_sem["natural_104_common_adjudicable_denominator"]
    lines.extend(["", f"全128题共同可裁定语义子集：{common_n}/128 题（来源已确认，且两臂均非‘无法判定’）；"
        f"Goal Z 正确 {goal_sem['common_adjudicable_semantic_correct']}/{common_n} "
        f"（{pct(goal_sem['common_adjudicable_semantic_rate'])}），AM 正确 "
        f"{am_sem['common_adjudicable_semantic_correct']}/{common_n} "
        f"（{pct(am_sem['common_adjudicable_semantic_rate'])}）。",
        f"主要自然多跳104题共同可裁定语义子集：{natural_common_n}/104 题；"
        f"Goal Z 正确 {goal_sem['natural_104_common_adjudicable_semantic_correct']}/{natural_common_n} "
        f"（{pct(goal_sem['natural_104_common_adjudicable_semantic_rate'])}），AM 正确 "
        f"{am_sem['natural_104_common_adjudicable_semantic_correct']}/{natural_common_n} "
        f"（{pct(am_sem['natural_104_common_adjudicable_semantic_rate'])}）。"])
    lines.extend(["", "## AM 相对 Goal Z 的配对差值", "", "正值表示AM更高；EM、语义率和证据支持率以百分点展示，Token F1以分值展示。区间为2000次配对bootstrap、seed42的95%百分位区间。", "",
                  "| 指标 / 比较范围 | AM改进 | AM退步 | 持平 | 平均差值 | 95%区间 | 配对题数 |", "|---|---:|---:|---:|---:|---:|---:|"])
    paired_labels = (("严格EM / 全128", "strict_em", 100), ("归一化EM / 全128", "normalized_em", 100),
                     ("Token F1 / 全128", "token_f1", 1), ("语义正确 / 全128共同可裁定", "semantic_correct_common", 100),
                     ("证据支持 / 全128", "evidence_supported_all128", 100),
                     ("证据支持 / 全128共同可裁定", "evidence_supported_common_adjudicable", 100),
                     ("严格EM / 自然多跳104", "strict_em_natural_104", 100),
                     ("归一化EM / 自然多跳104", "normalized_em_natural_104", 100),
                     ("Token F1 / 自然多跳104", "token_f1_natural_104", 1),
                     ("语义正确 / 自然104共同可裁定", "semantic_correct_natural_104_common", 100),
                     ("证据支持 / 自然多跳104", "evidence_supported_natural_104", 100),
                     ("证据支持 / 自然104共同可裁定", "evidence_supported_natural_104_common", 100))
    for title, key, scale in paired_labels:
        result = report["paired"][key]
        ci = result["bootstrap_95"]
        mean = result["mean_delta"]
        if mean is not None:
            mean *= scale
        interval = "—" if ci is None else f"[{ci[0]*scale:.3f}, {ci[1]*scale:.3f}]"
        lines.append(f"| {title} | {result['improved']} | {result['regressed']} | {result['unchanged']} | {number(mean)} | {interval} | {result['pairs']} |")
    lines.extend(["", "## 运行与审核", "",
        f"盲审覆盖 {report['reviewed_outputs']} 条模型输出，按问题、实际输入token、答案和结束原因去重为 {report['review_packets']} 个独立包（去重 {report['identical_outputs_deduplicated']} 条）；双审分歧 {report['review_disagreements_adjudicated']} 项由主审裁定，另主审复核了 {report['review_successes_root_checked']} 项声称证据成功的结果。",
        f"GPU累计预算 {report['gpu_budget']['charged_seconds']:.1f}/{report['gpu_budget']['cap_seconds']} 秒，模型生成调用 {report['gpu_budget']['generation_attempts']}/{report['gpu_budget']['generation_attempt_cap']} 次。模型加载单列：Goal Z {report['model_loading']['goal-z'].get('total_model_loading_seconds', 0):.1f}秒，AM {report['model_loading']['am-research'].get('total_model_loading_seconds', 0):.1f}秒。",
        "", "## 完成情况、预算与资源", "",
        "| 模型 | 完成回答 | 非回答结束 | 调用触顶未答 | 搜索达3次 | 非法动作题数 | 重复动作题数 | 题级超预算 | 平均 / P50 / P95耗时 | 峰值分配显存 / 保留显存 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for arm, title in labels.items():
        item = report["by_arm"][arm]["all_128"]
        gib = 1024 ** 3
        lines.append(f"| {title} | {item['answer_completion']}/128 | {128-item['answer_completion']} | "
            f"{item['model_call_limit_exhausted']} | {item['search_limit_reached']} | {item['invalid_actions']} | "
            f"{item['repeated_actions']} | {item['over_budget']} | "
            f"{number(item['mean_seconds'], 1)} / {number(item['p50_seconds'], 1)} / {number(item['p95_seconds'], 1)}秒 | "
            f"{number(item['max_gpu_peak_allocated_bytes']/gib, 2)} / {number(item['max_gpu_peak_reserved_bytes']/gib, 2)} GiB |")
    lines.extend([
        "", "逐题原始记录见 `A-ledger.jsonl`、`B-ledger.jsonl`；配对账本见 `paired-results.jsonl`；来源与逐关系证据审计见 `source-audit.jsonl`、`semantic-review-adjudicated.jsonl`。",
        ""])
    (folder / "对比报告.md").write_text("\n".join(lines), encoding="utf-8")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "preflight", "run", "resume", "post-freeze", "packets", "review-plan", "summary"))
    parser.add_argument("--run-dir", type=Path, default=OUTPUT)
    parser.add_argument("--base-model-path", type=str)
    parser.add_argument("--retrieval-model-path", type=str)
    parser.add_argument("--index-dir", type=str)
    parser.add_argument("--remote-run-dir", type=str)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    if args.mode == "prepare":
        if not all((args.base_model_path, args.retrieval_model_path, args.index_dir, args.remote_run_dir)):
            parser.error("prepare requires remote model/index paths and --remote-run-dir")
        if not args.remote_run_dir.startswith("/"):
            parser.error("--remote-run-dir must be an absolute POSIX path")
        manifest = prepare(folder)
        manifest["base_model_path"] = args.base_model_path
        manifest["runner_sha256"] = sha(Path(__file__))
        retrieval = read_config(ROOT / "configs/retrieval-local.json")
        retrieval["corpus"] = posixpath.join(args.remote_run_dir.rstrip("/"), "corpus.json")
        retrieval["corpus_sha256"] = CORPUS_SHA
        retrieval["model_path"] = args.retrieval_model_path
        retrieval["index_dir"] = args.index_dir
        write_json(folder / "retrieval-config.json", retrieval)
        manifest["retrieval_config_sha256"] = sha(folder / "retrieval-config.json")
        write_json(folder / "manifest.json", manifest)
        print(json.dumps({"folder": str(folder), "manifest": manifest["source_status_counts"]}, ensure_ascii=False))
    elif args.mode == "preflight":
        result = preflight(folder)
        print(json.dumps({"preflight": "passed", "runtime_lock_sha256": sha(folder / "runtime-lock.json"),
                          "gpu": result["gpu"], "retrieval": result["retrieval"]}, ensure_ascii=False))
    elif args.mode in ("run", "resume"):
        run(folder, resume=args.mode == "resume")
    elif args.mode == "post-freeze":
        manifest = freeze_postprocessing(folder)
        print(json.dumps({"inference_runner_sha256": manifest["inference_runner_sha256"],
                          "postprocessing_runner_sha256": manifest["postprocessing_runner_sha256"]}, ensure_ascii=False))
    elif args.mode == "packets":
        packets = make_review_packets(folder)
        print(json.dumps({"packets": len(packets), "run_dir": str(folder)}, ensure_ascii=False))
    elif args.mode == "review-plan":
        packets = prepare_root_review(folder)
        print(json.dumps({"root_review_packets": len(packets), "run_dir": str(folder)}, ensure_ascii=False))
    else:
        adjudications = rows(folder / "root-adjudications.jsonl")
        report = summarize(folder, adjudications)
        print(json.dumps({"status": report["status"], "run_dir": str(folder),
                          "source_status_counts": report["source_status_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    cli()
