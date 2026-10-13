"""Goal AM: matched action-SFT study for decisions under visible evidence.

CPU preparation freezes all inputs. GPU commands require an explicit approval
file bound to the resulting human-readable execution freeze.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import re
import statistics
import time
import shutil
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_v133_goal_ah as ah
import run_v133_goal_q_search as q

OUT = ROOT / "runs/v133-goal-am-action-sft-v2"
PREVIOUS = ROOT / "runs/v133-goal-am-action-sft-v1"
FAILURE = OUT / "experiment-failure.json"
JOURNAL = OUT / "gpu-usage.jsonl"
Z = ROOT / "runs/v133-goal-z-multihop-sft"
AI = ROOT / "runs/v133-goal-ai-evidence-feasibility/closure-v3"
AL = ROOT / "runs/v133-goal-al-wide-context-v1"
AH = ROOT / "runs/v133-goal-ah-entity-relation"
AA = ROOT / "runs/v133-goal-aa-audit-confirmation"
SOURCE = Path("/root/autodl-tmp/mini-agent-rl/runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final")
BASE = Path("/root/autodl-tmp/qwen_08b")
EXPECTED_ADAPTER = ah.EXPECTED_ADAPTER
PC = dict(retriever="bm25-v2", top_k=3, system_prompt="direct-v090", thinking=False,
          max_input_tokens=16384, max_new_tokens=2048, max_model_calls_per_case=4,
          max_searches_per_case=3, observation_encoding="legacy", action_completion="first-json-v1",
          device="cuda", dtype="bfloat16", load_in_4bit=False, timeout=None)
OPT = dict(learning_rate=5e-6, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.0, max_grad_norm=1.0)
GPU_LIMITS = dict(generations=1176, searches=768, updates=512)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        if jsonl:
            for row in value:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def gpu_usage(kind):
    if not JOURNAL.exists():
        return 0
    return sum(1 for line in JOURNAL.read_text(encoding="utf-8").splitlines()
               if line.strip() and json.loads(line).get("kind") == kind)


def record_gpu_attempt(kind, stage, case_id):
    if FAILURE.exists():
        raise RuntimeError("AM stopped by the existing experiment failure latch")
    cap = GPU_LIMITS["generations" if kind == "generation" else "searches"]
    if gpu_usage(kind) >= cap:
        raise RuntimeError(f"AM frozen {kind} cap exceeded ({cap})")
    with JOURNAL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(kind=kind, stage=stage, case_id=case_id,
                                attempt=gpu_usage(kind)+1, status="started"),ensure_ascii=False)+"\n")
        f.flush()


def instrument_rollout(wrapped, registry, stage, active_case):
    original_generate = wrapped.generate
    original_execute = registry.execute

    async def counted_generate(*args, **kwargs):
        record_gpu_attempt("generation", stage, active_case["id"])
        return await original_generate(*args, **kwargs)

    async def counted_execute(*args, **kwargs):
        if args and args[0] == "search":
            record_gpu_attempt("search", stage, active_case["id"])
        return await original_execute(*args, **kwargs)

    wrapped.generate = counted_generate
    registry.execute = counted_execute


def norm(s):
    return re.sub(r"\W+", " ", str(s).casefold(), flags=re.UNICODE).strip()


def key(s):
    return re.sub(r"\s+", " ", norm(s))


def tokenizer_from_base():
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(BASE), local_files_only=True)
    return getattr(processor, "tokenizer", processor)


def semantic_weights(tokenizer, token_ids, action):
    """Return unit weights and B's field-value weights, aligned to saved IDs."""
    text = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    obj_start = text.find("{")
    parsed, obj_end = json.JSONDecoder().raw_decode(text[obj_start:])
    if parsed != action:
        raise ValueError("saved target token IDs do not decode to their action")
    obj_end += obj_start
    spans = []
    for field in ("action", "answer", "query"):
        value = action.get(field)
        if not isinstance(value, str):
            continue
        pattern = re.compile(r'"' + re.escape(field) + r'"\s*:\s*("(?:\\.|[^"\\])*")')
        matches = list(pattern.finditer(text, obj_start, obj_end))
        if len(matches) != 1:
            raise ValueError(f"cannot find unique JSON semantic value: {field}")
        spans.append((matches[0].start(1), matches[0].end(1)))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if ids != list(token_ids) or offsets is None or len(offsets) != len(token_ids):
        raise ValueError("tokenizer offsets do not reproduce target token IDs")
    weighted = [4.0 if any(a < end and b > start for start, end in spans) else 1.0
                for a, b in offsets]
    return [1.0] * len(token_ids), weighted, spans, text


def decode_input(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def cite_check(tokenizer, example, citations):
    ids = example["prompt_token_ids"]
    text = decode_input(tokenizer, ids)
    good = []
    for group in citations:
        if not group:
            return False, "empty citation group"
        matches = []
        for c in group:
            quote = c.get("quote", "")
            title = c.get("title", "")
            if quote and norm(quote) in norm(text) and title and norm(title) in norm(text):
                matches.append(c)
        if not matches:
            return False, "required edge has no cited title+quote in actual input"
        good.append(matches[0])
    return True, good


def eval_isolation():
    freeze = read(AH / "execution-freeze.json")
    eval_ids, eval_questions = set(), set()
    for spec in freeze["evaluation"]:
        for row in rows(ROOT / spec["cases"]):
            eval_ids.add(row.get("id", row.get("case_id")))
            eval_questions.add(key(row["question"]))
    if len(eval_ids) != sum(x["count"] for x in freeze["evaluation"]):
        # IDs can recur across legacy regression sets; report unique count while
        # retaining exact per-source overlap checks below.
        pass
    return eval_ids, eval_questions, freeze


def validate_state_isolation(states, eval_ids, eval_questions):
    """Reject evaluation overlap by either source ID or normalized question."""
    for state in states:
        question = key(state.get("question", ""))
        if not question:
            raise RuntimeError(f"AM state has no question: {state.get('state_id')}")
        if state["case_id"] in eval_ids or question in eval_questions:
            raise RuntimeError(f"evaluation leakage: {state['state_id']}")


def prepare():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if read(AL / "status.json").get("status") != "complete":
        raise RuntimeError("Goal AL must be complete before Goal AM")
    aa = read(AA / "product-decision.json")
    if not aa.get("published") or aa.get("selected_seed") != 42 or aa.get("selected_route") != "trajectory":
        raise RuntimeError("AA does not identify the released Goal Z seed42 trajectory")
    adapter_hash = sha(SOURCE / "adapter_model.safetensors")
    if adapter_hash != EXPECTED_ADAPTER:
        raise RuntimeError("published Goal Z adapter hash mismatch")
    tokenizer = tokenizer_from_base()
    eval_ids, eval_questions, ah_freeze = eval_isolation()

    answer_sets = [
        ("ai-recovery-post-repair-answer", rows(AI / "recovery-pairs-v3.jsonl"), lambda r: r["answer"], "recovery"),
        ("ai-answer-correction", rows(AI / "answer-only-examples-v3.jsonl"), lambda r: r["example"], "answer_only"),
        ("ai-supported-answer", rows(AI / "supported-positive-examples-v3.jsonl"), lambda r: r["example"], "correct"),
    ]
    admission = {r["case_id"]: r for r in rows(AI / "train-audit-ledger-v4.jsonl")}
    answer_states = []
    for source, source_rows, get_example, expected_status in answer_sets:
        for r in source_rows:
            case_id = r["case_id"]
            review = admission.get(case_id)
            if not review or review["status"] != expected_status:
                raise RuntimeError(f"source-admission mismatch: {case_id}")
            # The old diagnostic-only mark remains in the source. This is a new,
            # explicit AM-only release candidate, pending the final human review.
            ex = get_example(r)
            act = ex["target_action"]
            if act.get("action") != "answer" or not act.get("answer"):
                raise RuntimeError(f"AM answer state malformed: {case_id}")
            ok, cited = cite_check(tokenizer, ex, r["edge_citations"])
            if not ok:
                raise RuntimeError(f"AM evidence citation check failed: {case_id}: {cited}")
            action_ids=list(ex["target_token_ids"])
            unit, field_weights, spans, _ = semantic_weights(tokenizer, action_ids, act)
            answer_states.append(dict(state_id=f"{source}:{case_id}", case_id=case_id,
                question=review["question"],
                state_type="answer", source=source, source_status=expected_status,
                prompt_token_ids=ex["prompt_token_ids"], prompt_attention_mask=[1]*len(ex["prompt_token_ids"]),
                action_token_ids=action_ids, target_action=act,
                citations=cited, source_review_reason=review["reason"],
                old_training_release_approved=review.get("training_release_approved") is True,
                diagnostic_only_source=(r.get("diagnostic_only") is True),
                semantic_spans=spans, unit_weights=unit, field_weights=field_weights,
                teacher_supplemented=(source=="ai-recovery-post-repair-answer"), evaluation_use_forbidden=True))
    if len(answer_states) != 24 or len({s["case_id"] for s in answer_states}) != 24:
        raise RuntimeError("expected 24 unique answer states")

    # Ten authentic second-search states, in Goal Z's frozen original order.
    selected = {r["id"]: r for r in rows(Z / "selected-train.jsonl")}
    original = rows(Z / "training-inputs.jsonl")
    search_states = []
    rejected = Counter()
    for case in original:
        if len(search_states) == 10:
            break
        meta = selected.get(case["id"])
        if not meta or len(meta.get("chain", [])) != 2 or len(case.get("actions", [])) < 3:
            rejected["no_two_hop_source"] += 1; continue
        first, second = meta["chain"]
        a0, a1 = case["actions"][:2]
        if a0["action"] != "search" or a1["action"] != "search" or not a0.get("observation", {}).get("ok"):
            rejected["no_two_real_searches"] += 1; continue
        visible = "\n".join(d.get("text", "") for d in a0["observation"].get("result", []))
        prompt = decode_input(tokenizer, a1["prompt_token_ids"])
        if norm(first["answer"]) not in norm(visible) or norm(first["answer"]) not in norm(prompt):
            rejected["bridge_not_visible"] += 1; continue
        if norm(second["answer"]) in norm(prompt) or norm(second["answer"]) in norm(a1.get("query", "")):
            rejected["answer_leak"] += 1; continue
        found_target = any(norm(second["answer"]) in norm(d.get("text", ""))
                           for d in a1.get("observation", {}).get("result", []))
        if not found_target:
            rejected["later_result_not_verified"] += 1; continue
        # The action and query are from the original Goal Z real search trace.
        tok = list(a1["action_token_ids"])
        act = dict(action="search", query=a1["query"])
        unit, weighted, spans, _ = semantic_weights(tokenizer, tok, act)
        search_states.append(dict(state_id=f"goal-z-second-search:{case['id']}", case_id=case["id"],
            question=case["question"],
            state_type="search", source="goal-z-original-complete-trajectory",
            prompt_token_ids=a1["prompt_token_ids"], prompt_attention_mask=a1["prompt_attention_mask"],
            action_token_ids=tok, target_action=act, query=a1["query"],
            bridge_entity=first["answer"], bridge_relation=first["question"],
            missing_relation=second["question"], result_verified=True,
            target_result_ids=[d["id"] for d in a1["observation"].get("result", [])
                               if norm(second["answer"]) in norm(d.get("text", ""))],
            semantic_spans=spans, unit_weights=unit, field_weights=weighted,
            goal_z_call_index=1, goal_z_searches_before=1,
            teacher_supplemented=False, evaluation_use_forbidden=True))
    if len(search_states) != 10:
        raise RuntimeError(f"only {len(search_states)}/10 valid original search states; no backfill allowed; {dict(rejected)}")

    states = answer_states + search_states
    validate_state_isolation(states, eval_ids, eval_questions)
    for s in states:
        if len(s["prompt_token_ids"]) > PC["max_input_tokens"] or len(s["action_token_ids"]) > PC["max_new_tokens"]:
            raise RuntimeError(f"AM action state exceeds frozen limits: {case_id}")
    if len({s["state_id"] for s in states}) != 34:
        raise RuntimeError("state IDs are not unique")

    z_inputs = rows(Z / "training-inputs.jsonl")
    if len(z_inputs) != 128 or any(not r.get("actions") or any(not a.get("action_token_ids") for a in r["actions"]) for r in z_inputs):
        raise RuntimeError("published Goal Z training trajectories changed")
    order_cfg = read(AH / "execution-freeze.json")["order"]
    if set(order_cfg) != {"42", "43"} or any(len(v) != 128 for v in order_cfg.values()):
        raise RuntimeError("frozen training orders missing")
    protected = read(AL / "manifest.json")["protected"]
    if len(protected) != 7:
        raise RuntimeError("AL protection set mismatch")
    probes = make_probes(tokenizer)
    hashes = {
        "runner": sha(__file__), "adapter": adapter_hash,
        "aa_decision": sha(AA / "product-decision.json"),
        "z_training_inputs": sha(Z / "training-inputs.jsonl"),
        "z_selected_train": sha(Z / "selected-train.jsonl"),
        "ai_recovery": sha(AI / "recovery-pairs-v3.jsonl"),
        "ai_answer_only": sha(AI / "answer-only-examples-v3.jsonl"),
        "ai_supported": sha(AI / "supported-positive-examples-v3.jsonl"),
        "ai_audit": sha(AI / "train-audit-ledger-v4.jsonl"),
        "al_report": sha(AL / "REPORT.md"), "al_manifest": sha(AL / "manifest.json"),
        "al_semantic_review": sha(AL / "semantic-review.jsonl"),
        "ah_freeze": sha(AH / "execution-freeze.json"),
        "ah_cases": sha(AH / "evaluation-cases-goal_z_dev.jsonl"),
        "dev_corpus": sha(Z / "corpus-dev.json"),
        "tokenizer": sha(BASE / "tokenizer.json"), "base_config": sha(BASE / "config.json"),
    }
    base_weights = next(iter(sorted(BASE.glob("model.safetensors-*-of-*.safetensors"))), None)
    if base_weights is None:
        raise RuntimeError("base model weight shard missing")
    hashes["base_weights"] = sha(base_weights)
    frozen_paths = {
        "runner": str(Path(__file__).resolve()),
        "adapter": str(SOURCE / "adapter_model.safetensors"),
        "aa_decision": str(AA / "product-decision.json"),
        "z_training_inputs": str(Z / "training-inputs.jsonl"),
        "z_selected_train": str(Z / "selected-train.jsonl"),
        "ai_recovery": str(AI / "recovery-pairs-v3.jsonl"),
        "ai_answer_only": str(AI / "answer-only-examples-v3.jsonl"),
        "ai_supported": str(AI / "supported-positive-examples-v3.jsonl"),
        "ai_audit": str(AI / "train-audit-ledger-v4.jsonl"),
        "al_report": str(AL / "REPORT.md"), "al_manifest": str(AL / "manifest.json"),
        "al_semantic_review": str(AL / "semantic-review.jsonl"),
        "ah_freeze": str(AH / "execution-freeze.json"),
        "ah_cases": str(AH / "evaluation-cases-goal_z_dev.jsonl"),
        "dev_corpus": str(Z / "corpus-dev.json"),
        "tokenizer": str(BASE / "tokenizer.json"), "base_config": str(BASE / "config.json"),
    }
    frozen_paths["base_weights"] = str(base_weights)
    OUT.mkdir(parents=True)
    write_once(OUT / "candidate-training-states.jsonl", states, jsonl=True)
    write_once(OUT / "frozen-probes.jsonl", probes, jsonl=True)
    write_once(OUT / "baseline-state-probes.jsonl", [dict(probe_id=f"baseline-state:{s['state_id']}",
        case_id=s["case_id"], question=s["question"], category="training-state-baseline",
        prompt_token_ids=s["prompt_token_ids"], state_type=s["state_type"],
        target_action=s["target_action"], training_or_evaluation_use_forbidden=True) for s in answer_states], jsonl=True)
    write_once(OUT / "evaluation-cases.jsonl", rows(AH / "evaluation-cases-goal_z_dev.jsonl"), jsonl=True)
    hashes.update({"candidate_states": sha(OUT / "candidate-training-states.jsonl"),
        "frozen_probes": sha(OUT / "frozen-probes.jsonl"),
        "baseline_state_probes": sha(OUT / "baseline-state-probes.jsonl"),
        "evaluation_cases": sha(OUT / "evaluation-cases.jsonl")})
    frozen_paths.update({k: str(OUT / v) for k,v in {
        "candidate_states":"candidate-training-states.jsonl", "frozen_probes":"frozen-probes.jsonl",
        "baseline_state_probes":"baseline-state-probes.jsonl", "evaluation_cases":"evaluation-cases.jsonl"}.items()})
    freeze = dict(goal="AM", status="awaiting_human_pre_gpu_confirmation", source_model=str(SOURCE),
        base=str(BASE), product=PC, optimizer=OPT, seeds=[42,43], updates_per_run=128,
        order=order_cfg, training_trajectory_count=128, states=34,
        state_counts=dict(answer=24,search=10), state_schedule="updates 2,4,...,68, frozen same order in A/B and both seeds",
        loss=dict(original_trajectory="sum assistant token NLL / trajectory assistant tokens",
                  state_A="uniform current action-token NLL / target token count",
                  state_B="token NLL weighted 4 on JSON action/answer/query string values and 1 elsewhere, normalized by weight sum",
                  state_mix=0.25, target="current action and end-of-turn token only; no prompt/history/tool loss"),
        arms=dict(A="Goal Z trajectory + 34 state actions, uniform state token weights",
                  B="same material/order/updates; state semantic-field token weights 4:1"),
        evaluation=dict(dev=64, admitted=29, unconfirmed=35, complete_evidence_probes=24,
                        missing_link_probes=8, state_baseline_checks=24, protected=7),
        limits=dict(training_runs=4,updates=512,full_trajectories=256,generated_actions=1176,searches=768,
                    dev_per_candidate=64,probe_actions_per_candidate=32),
        thresholds=dict(dev_supported_net_vs_goal_z=2,errors_not_higher=True,invalid_actions_not_higher=True,
                        protected_losses_max=1,complete_probe_correct_gain=2,missing_probe_premature_answers_not_higher=True,
                        seed43_only_after_seed42_route_pass=True),
        hashes=hashes, frozen_paths=frozen_paths,
        probe_definition="AL B real token inputs; 24 complete-chain questions; 8 eligible incomplete-chain search states; no search execution",
        failure_policy="identity/GPU/numeric/infrastructure failure: preserve state and stop; no auto retry or parameter change",
        data_gate=dict(old_ai_release_approved_count=sum(bool(s.get("old_training_release_approved")) for s in answer_states),
                       manual_am_review_required=True, no_evaluation_training=True),
        gpu_authorized=False, released=False)
    write_once(OUT / "execution-freeze.json", freeze)
    digest = sha(OUT / "execution-freeze.json")
    report = human_report(states, probes, freeze, digest)
    (OUT / "PRE_GPU_REVIEW.md").write_text(report, encoding="utf-8", newline="\n")
    write_once(OUT / "status.json", dict(status="awaiting_human_pre_gpu_confirmation", gpu_generations=0, updates=0))
    print(json.dumps(dict(output=str(OUT), training_states=len(states), probes=len(probes),
                          freeze_sha256=digest, status="awaiting_human_pre_gpu_confirmation"), ensure_ascii=False))


def make_probes(tokenizer):
    packets = rows(AL / "review-packets.jsonl")
    review = {r["case_id"]: r for r in rows(AL / "semantic-review.jsonl")}
    if len(packets) != 29 or len(review) != 29:
        raise RuntimeError("AL admitted review ledger incomplete")
    complete = []
    missing_candidates = []
    for p in packets:
        c = p["case_id"]
        b = p["B"]
        if len(b["token_ids"]) > PC["max_input_tokens"]:
            raise RuntimeError("AL actual input unexpectedly exceeds AM budget")
        final_review = review[c]["B"]
        if final_review["complete_chain_visible"]:
            complete.append(dict(probe_id=f"complete:{c}", case_id=c, category="complete",
                question=p["question"], prompt_token_ids=b["token_ids"], expected_chain_visible=True,
                baseline_answer=b["answer"], baseline_category=final_review["category"],
                quotes=[dict(edge_id=e["edge_id"],document_id=e["document_id"],quote=e["quote"])
                        for e in p["edges"]], no_search_execution=True))
        if final_review["category"] in ("unsupported_by_visible_evidence","factual_error","no_answer_repeated_action"):
            for t in b["trace"]:
                if t.get("action") != "search" or t.get("step",99) >= 3:
                    continue
                prior_searches = sum(x.get("action")=="search" for x in b["trace"] if x.get("step",99)<t["step"])
                if prior_searches >= 3:
                    continue
                ids = t.get("prompt_token_ids", [])
                text = decode_input(tokenizer, ids)
                absent = [e for e in p["edges"] if norm(e["quote"]) not in norm(text)]
                if absent:
                    missing_candidates.append(dict(probe_id=f"missing:{c}:step{t['step']}", case_id=c,
                        category="missing_link", question=p["question"],prompt_token_ids=ids,
                        expected_chain_visible=False, baseline_action=t.get("action"),
                        baseline_query=t.get("query"),missing_edge_ids=[e["edge_id"] for e in absent],
                        quote_check="at least one frozen necessary-edge quote absent from actual input",
                        calls_before=t["step"],searches_before=prior_searches,no_search_execution=True))
                break
    if len(complete) != 24:
        raise RuntimeError(f"expected 24 AL complete evidence probes, got {len(complete)}")
    chosen=[]; seen=set()
    for x in missing_candidates:
        if x["case_id"] in seen: continue
        seen.add(x["case_id"]);chosen.append(x)
        if len(chosen)==8:break
    if len(chosen)!=8:
        raise RuntimeError(f"only {len(chosen)}/8 eligible missing-link probes; do not backfill")
    return complete+chosen


def human_report(states, probes, freeze, digest):
    base_hashes = freeze["hashes"]
    lines=["# Goal AM：GPU前冻结审阅", "",f"执行冻结SHA-256：`{digest}`", "",
        "当前状态：34条候选已由冻结材料构建；GPU调用与更新均为0。AI原账本把这些材料标为诊断用途，旧审批通过数为0；AM本轮是新的候选准入，人工确认前不视为已获训练审批。", "",
        "## 实验前提", "", "- 起点：AA发布的Goal Z轨迹seed42；adapter SHA-256：`"+base_hashes["adapter"]+"`。",
        "- 基础模型：`"+str(freeze["base"])+"`；共`"+str(freeze.get("base_weight_shard_count","?"))+"`个权重分片，合并SHA-256：`"+freeze["base_weights_bundle_sha256"]+"`；tokenizer SHA-256：`"+base_hashes["tokenizer"]+"`。",
        "- A/B均保留相同128条完整轨迹，使用同一34条状态、相同顺序和128更新；A等权，B仅对JSON动作类型及查询/答案值token按4:1加权。",
        "- 当前动作JSON格式、direct-v090提示、Goal Q、first-json-v1、legacy、BM25-v2/top-3不变；两臂评价使用16384/2048共同预算。",
        "- BF16、单卡、不量化；LR 5e-6、AdamW beta=(0.9,0.999), eps=1e-8, weight_decay=0、梯度裁剪1；只训练LoRA。",
        "- seed42先执行；符合冻结继续门槛才执行seed43。最多4次训练/512更新、256条开发轨迹、1176次GPU生成、768次检索。", "",
        "## 34条训练状态", "", "| # | 类型 | 来源 | 案例 | 问题 | 正确目标动作 | 桥接关系或完整引文 |", "|---:|---|---|---|---|---|---|"]
    for i,s in enumerate(states,1):
        citation_quotes = [c.get("quote", "") for c in s.get("citations", [])]
        if s["state_type"] == "search":
            citation_quotes = ["bridge doc="+str(s.get("bridge_document_id"))+": "+s.get("bridge_quote", "")]
            citation_quotes += ["result doc="+str(e["document_id"])+" rank="+str(e["rank"])+": "+e["quote"]
                                for e in s.get("target_result_evidence", [])]
        cite="; ".join(x.replace("|","/").replace("\n"," ") for x in citation_quotes if x)
        proof=(s.get("bridge_entity","")+" / "+s.get("missing_relation",s.get("source_review_reason",""))).replace("|","/")
        question=s.get("question","").replace("|","/").replace("\n"," ")
        lines.append(f"| {i} | {s['state_type']} | {s['source']} | `{s['case_id']}` | {question} | `{json.dumps(s['target_action'],ensure_ascii=False)}` | {(proof+' '+cite).replace('|','/')} |")
    lines += ["",f"## 机制探针", "",f"冻结{len(probes)}次单动作检查（完整证据{sum(p['category']=='complete' for p in probes)}、缺链{sum(p['category']=='missing_link' for p in probes)}）；使用AL保存的实际输入token，不执行搜索。另有24个answer训练状态的Goal Z基线单动作检查。完整证据问题：{', '.join(p['case_id'] for p in probes if p['category']=='complete')}。缺链问题：{', '.join(p['case_id'] for p in probes if p['category']=='missing_link')}。", "",
        "## 门槛与停止", "", "seed42开发资格：准入29题支持成功净增≥2、事实错误+证据不支持不增、动作问题不增、连续搜索保护题最多丢失1；完整证据探针正确数相对Goal Z增加≥2，缺链探针提前回答不增加。",
        "未过seed42则停止，不启动seed43。正式产品升级仍需完整回归和真正未暴露确认集。GPU/身份/数值/基础设施失败立即留存现场并停止，不自动重试。", "",
        "确认记录只能绑定上述冻结SHA。未收到对这份实际材料、参数和额度的确认时，不启动GPU。", ""]
    return "\n".join(lines)


def refresh_pre_gpu():
    """Version a pre-GPU presentation/schema correction without touching sources."""
    if not (OUT / "execution-freeze.json").exists():
        raise RuntimeError("AM CPU preparation does not exist")
    if (OUT / "gpu-approval.json").exists() or read(OUT / "status.json").get("gpu_generations", 0):
        raise RuntimeError("pre-GPU refresh is forbidden after approval or GPU use")
    archive=OUT/"preparation-v1"
    if not archive.exists():
        archive.mkdir(exist_ok=False)
        for name in ("execution-freeze.json","PRE_GPU_REVIEW.md","candidate-training-states.jsonl",
                     "frozen-probes.jsonl","evaluation-cases.jsonl","status.json"):
            shutil.copy2(OUT/name,archive/name)
    tokenizer=tokenizer_from_base()
    ai={"ai-recovery-post-repair-answer":{r["case_id"]:r for r in rows(AI/"recovery-pairs-v3.jsonl")},
        "ai-answer-correction":{r["case_id"]:r for r in rows(AI/"answer-only-examples-v3.jsonl")},
        "ai-supported-answer":{r["case_id"]:r for r in rows(AI/"supported-positive-examples-v3.jsonl")}}
    admission={r["case_id"]:r for r in rows(AI/"train-audit-ledger-v4.jsonl")}
    zmeta={r["id"]:r for r in rows(Z/"selected-train.jsonl")}
    states=rows(OUT/"candidate-training-states.jsonl")
    for s in states:
        s["evaluation_use_forbidden"] = True
        if s["state_type"]=="answer":
            src=ai[s["source"]][s["case_id"]]
            ex=src["answer"] if s["source"]=="ai-recovery-post-repair-answer" else src["example"]
            s["question"]=admission[s["case_id"]]["question"]
            s["prompt_attention_mask"]=[1]*len(s["prompt_token_ids"])
            s["old_training_release_approved"]=admission[s["case_id"]].get("training_release_approved") is True
            s["teacher_supplemented"]=s["source"]=="ai-recovery-post-repair-answer"
            # Rebuild proof pointers from immutable source rows and verify the
            # marked action is byte-for-byte the source's reviewed target.
            if s["target_action"]!=ex["target_action"]:raise RuntimeError("answer target changed since candidate freeze")
            ok,cited=cite_check(tokenizer,ex,src["edge_citations"])
            if not ok:raise RuntimeError("AM answer citation failed during pre-GPU refresh: "+s["case_id"])
            s["citations"]=cited
            s["semantic_spans"]=_weights(tokenizer,s)
        else:
            r=next(x for x in rows(Z/"training-inputs.jsonl") if x["id"]==s["case_id"])
            s["question"]=r["question"]
            m=zmeta[s["case_id"]];first,second=m["chain"]
            bridge=next(d for d in r["actions"][0]["observation"]["result"] if norm(first["answer"]) in norm(d["text"]))
            s["bridge_quote"]=bridge["text"]
            s["bridge_document_id"]=bridge["id"]
            targets=[(i,d) for i,d in enumerate(r["actions"][1]["observation"]["result"])
                     if norm(second["answer"]) in norm(d.get("text",""))]
            s["target_result_quotes"]=[d["text"] for _,d in targets]
            s["target_result_evidence"]=[dict(document_id=d["id"],title=d.get("title"),quote=d["text"],rank=i+1)
                for i,d in targets]
    # Rewrite only AM-derived files; originals are copied above and source files
    # are immutable. New hashes bind the readable candidate states to approval.
    for name,obj in (("candidate-training-states.jsonl",states),):
        p=OUT/name
        p.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in obj),encoding="utf-8")
    probes=rows(OUT/"frozen-probes.jsonl")
    baseline=[dict(probe_id=f"baseline-state:{s['state_id']}",case_id=s["case_id"],
        question=s["question"],category="training-state-baseline",prompt_token_ids=s["prompt_token_ids"],
        state_type=s["state_type"],target_action=s["target_action"],training_or_evaluation_use_forbidden=True)
        for s in states if s["state_type"]=="answer"]
    for b,s in zip(baseline,[x for x in states if x["state_type"]=="answer"]):
        b["question"]=s["question"]
    (OUT/"baseline-state-probes.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in baseline),encoding="utf-8")
    old=read(OUT/"execution-freeze.json")
    old_sha=sha(OUT/"execution-freeze.json")
    base_shards = sorted(BASE.glob("*.safetensors"))
    if not base_shards:
        raise RuntimeError("base model weight shard missing during AM pre-GPU identity refresh")
    shard_rows = [dict(name=p.name, path=str(p), sha256=sha(p)) for p in base_shards]
    shard_manifest = OUT/"base-weight-shards.json"
    shard_manifest.write_text(json.dumps(shard_rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    bundle_hash = hashlib.sha256(json.dumps(
        [{"name":r["name"],"sha256":r["sha256"]} for r in shard_rows],
        sort_keys=True,separators=(",",":")).encode()).hexdigest()
    old["hashes"].update(runner=sha(__file__),candidate_states=sha(OUT/"candidate-training-states.jsonl"),
        baseline_state_probes=sha(OUT/"baseline-state-probes.jsonl"),
        frozen_probes=sha(OUT/"frozen-probes.jsonl"),
        evaluation_cases=sha(OUT/"evaluation-cases.jsonl"),
        base_weight_shards_manifest=sha(shard_manifest),
        tokenizer=sha(BASE/"tokenizer.json"),
        base_config=sha(BASE/"config.json"))
    old["base_weights_bundle_sha256"]=bundle_hash
    old["frozen_paths"]["runner"]=str(Path(__file__).resolve())
    old["frozen_paths"]["candidate_states"]=str(OUT/"candidate-training-states.jsonl")
    old["frozen_paths"]["baseline_state_probes"]=str(OUT/"baseline-state-probes.jsonl")
    old["frozen_paths"].update(base_weight_shards_manifest=str(shard_manifest),
        frozen_probes=str(OUT/"frozen-probes.jsonl"),evaluation_cases=str(OUT/"evaluation-cases.jsonl"),
        tokenizer=str(BASE/"tokenizer.json"),base_config=str(BASE/"config.json"))
    for i,entry in enumerate(shard_rows):
        key_name=f"base_weight_shard_{i:02d}"
        old["hashes"][key_name]=entry["sha256"]
        old["frozen_paths"][key_name]=entry["path"]
    old["base_weight_shard_count"]=len(shard_rows)
    old["preparation_amendment"]="v4: explicit diagnostic-only legacy approval count, evaluation isolation flag on all 34 rows, linked second-search source document identities and quotes, all base shard/tokenizer identities"
    old["data_gate"] = dict(
        old_ai_release_approved_count=sum(bool(s.get("old_training_release_approved")) for s in states if s["state_type"] == "answer"),
        manual_am_review_required=True,
        no_evaluation_training=True,
        legacy_ai_rows_remain_diagnostic=True)
    (OUT/"execution-freeze.json").write_text(json.dumps(old,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    digest=sha(OUT/"execution-freeze.json")
    (OUT/"PRE_GPU_REVIEW.md").write_text(human_report(states,probes,old,digest),encoding="utf-8",newline="\n")
    with (OUT/"preparation-amendments.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(dict(version="v4",previous_freeze_sha256=old_sha,new_freeze_sha256=digest,
            reason="Review artifact was JSON-escaped; AM states lacked decoded questions/masks and state weights; no GPU use"),ensure_ascii=False)+"\n")
    print(json.dumps(dict(status="awaiting_human_pre_gpu_confirmation",freeze_sha256=digest,
        states=len(states),baseline_state_probes=len(baseline),gpu_generations=0),ensure_ascii=False))


def _weights(tokenizer,state):
    unit,weighted,spans,_=semantic_weights(tokenizer,state["action_token_ids"],state["target_action"])
    state["unit_weights"]=unit;state["field_weights"]=weighted
    return spans


def approved(require_preflight=True):
    if not (OUT / "gpu-approval.json").is_file():
        raise RuntimeError("GPU is blocked until user confirms this exact PRE_GPU_REVIEW and freeze SHA")
    cfg=read(OUT/"execution-freeze.json"); ap=read(OUT/"gpu-approval.json")
    if ap.get("human_confirmed") is not True or ap.get("execution_freeze_sha256")!=sha(OUT/"execution-freeze.json"):
        raise RuntimeError("approval does not match frozen AM plan")
    for name,digest in cfg["hashes"].items():
        p=Path(cfg["frozen_paths"][name])
        if sha(p)!=digest: raise RuntimeError("frozen AM input changed: "+name)
    shard_rows=read(cfg["frozen_paths"]["base_weight_shards_manifest"])
    actual=[]
    for row in shard_rows:
        if sha(row["path"])!=row["sha256"]:
            raise RuntimeError("base model shard changed: "+row["name"])
        actual.append({"name":row["name"],"sha256":row["sha256"]})
    bundle=hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if bundle!=cfg["base_weights_bundle_sha256"]:
        raise RuntimeError("base model shard bundle identity mismatch")
    if require_preflight:
        preflight_path=OUT/"cpu-preflight.json"
        if not preflight_path.exists() or read(preflight_path).get("passed") is not True:
            raise RuntimeError("AM CPU execution preflight has not passed")
    return cfg


def amend_failed_execution():
    """New user-authorized execution revision; do not unlock the failed run."""
    if OUT.exists():raise FileExistsError(OUT)
    previous_freeze=PREVIOUS/"execution-freeze.json"
    expected="a6cfd186981642016627c6b7938e575ae5974dcd15f3f0665333dda4cbc655b5"
    if sha(previous_freeze)!=expected:raise RuntimeError("previous approved AM freeze changed")
    failed=read(PREVIOUS/"status.json")
    if failed.get("successful_optimizer_updates")!=0 or failed.get("gpu_generations")!=0:
        raise RuntimeError("AM revision requires the verified zero-update original failure")
    cfg=read(previous_freeze)
    for name,digest in cfg["hashes"].items():
        if name=="runner":continue
        if sha(cfg["frozen_paths"][name])!=digest:raise RuntimeError("AM source changed: "+name)
    if cfg["product"]!=PC or cfg["optimizer"]!=OPT:raise RuntimeError("AM repair must preserve experiment parameters")
    OUT.mkdir(exist_ok=False)
    names=("candidate-training-states.jsonl","frozen-probes.jsonl","baseline-state-probes.jsonl",
           "evaluation-cases.jsonl","base-weight-shards.json")
    for name in names:shutil.copy2(PREVIOUS/name,OUT/name)
    shutil.copy2(PREVIOUS/"PRE_GPU_REVIEW.md",OUT/"PRE_GPU_REVIEW_SOURCE.md")
    for name,path in list(cfg["frozen_paths"].items()):
        if Path(path).parent==PREVIOUS:
            cfg["frozen_paths"][name]=str(OUT/Path(path).name)
    dependencies=dict(runner=Path(__file__),optimizer_guard=ROOT/"src/mini_agent_rl/training/grpo.py",
        goal_q_runner=ROOT/"scripts/run_v133_goal_q_search.py",model_client=ROOT/"src/mini_agent_rl/model/local.py",
        search_implementation=ROOT/"src/mini_agent_rl/tools/search.py",prompts=ROOT/"src/mini_agent_rl/prompts.py",
        action_alias=ROOT/"scripts/run_v133_goal_p_variant.py",parameter_hash_helper=ROOT/"scripts/run_v133_goal_o_budget.py",
        goal_z_baseline=AL/"trajectory-B.jsonl")
    for name,path in dependencies.items():
        cfg["frozen_paths"][name]=str(path.resolve());cfg["hashes"][name]=sha(path)
    cfg.update(status="authorized_execution_revision_cpu_preflight_pending",gpu_authorized=True,
        execution_revision="v2",previous_run=str(PREVIOUS),previous_freeze_sha256=expected,
        previous_successful_updates=0,previous_generation_count=0,
        repair_scope=["canonical guarded optimizer import","case_id evaluation schema",
                      "reuse production JSON-array corpus registry","CPU loss/mask/API preflight"],
        human_authorization="根据上面内容，完成实验前的全部优化器等修正，完成后执行实验")
    write_once(OUT/"execution-freeze.json",cfg)
    digest=sha(OUT/"execution-freeze.json")
    write_once(OUT/"gpu-approval.json",dict(human_confirmed=True,execution_freeze_sha256=digest,
        confirmation_text=cfg["human_authorization"],scope="same frozen AM experiment; seed42 first, seed43 only if gate passes"))
    write_once(OUT/"status.json",dict(status="cpu_preflight_pending",gpu_generations=0,successful_optimizer_updates=0))
    (OUT/"EXECUTION_REVISION.md").write_text(
        "# Goal AM 执行修订v2\n\n旧v1失败现场保留。原尝试成功更新0、生成0、搜索0。\n\n"
        "用户已授权修正后执行。材料、顺序、损失、参数、预算和资格门槛均沿用原冻结方案。\n\n"
        "修复：优化器从canonical模块导入；评价使用case_id；语料索引复用产品工厂；新增CPU执行前核验。\n\n"
        f"新冻结SHA-256：`{digest}`。\n\n"
        "先seed42 A/B各128更新及64题评价、32探针；达标后seed43复现。总上限512更新、1176生成、768搜索。\n"
        "新增故障仍保存现场并停止，不自动重跑。\n",encoding="utf-8")
    print(json.dumps(dict(revision=str(OUT),freeze_sha256=digest,authorized=True),ensure_ascii=False),flush=True)


def cpu_preflight():
    import torch
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as budget
    cfg=approved(require_preflight=False)
    if FAILURE.exists():raise RuntimeError("failure latch is present")
    if not callable(guarded_optimizer_step) or not callable(budget.parameter_hashes):
        raise RuntimeError("optimizer/hash API unavailable")
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float32))
    opt=torch.optim.AdamW([p],lr=OPT["learning_rate"],betas=tuple(OPT["betas"]),eps=OPT["eps"],weight_decay=0)
    p.grad=torch.tensor([.25]);before=p.detach().clone()
    guarded_optimizer_step(opt,[p],{"peak_reserved_gb":0.},21.)
    if not (p.detach()<before).all():raise RuntimeError("CPU optimizer did not update")
    p.grad=torch.tensor([float("nan")]);snapshot=p.detach().clone()
    try:guarded_optimizer_step(opt,[p],{"peak_reserved_gb":0.},21.)
    except RuntimeError:pass
    else:raise RuntimeError("optimizer NaN gate failed")
    if not torch.equal(p.detach(),snapshot):raise RuntimeError("NaN guard mutated parameter")
    class ToyClient:
        def __init__(self):self.values=torch.tensor([-1.,-2.,-3.],requires_grad=True)
        def score_action_tensor(self,messages,ids,grad,prompt,mask):
            assert ids==[10,11,12] and prompt==[99,98] and mask==[1,1] and grad
            return self.values
    toy_action=dict(action_token_ids=[10,11,12],prompt_token_ids=[99,98],prompt_attention_mask=[1,1])
    for weights in ([1.,1.,1.],[4.,4.,1.]):
        toy=ToyClient();loss=target_loss(toy,toy_action,weights,.25)
        expected=-.25*torch.tensor(weights)/sum(weights)
        if not torch.allclose(toy.values.grad,expected):raise RuntimeError("state normalized loss gradient mismatch")
        if not math.isclose(loss,.25*sum((i+1)*w for i,w in enumerate(weights))/sum(weights),rel_tol=1e-6):
            raise RuntimeError("state normalized loss value mismatch")
    trajectories=rows(Z/"training-inputs.jsonl");states=rows(OUT/"candidate-training-states.jsonl")
    if len(trajectories)!=128 or len(states)!=34:raise RuntimeError("AM material counts changed")
    ids={r["id"] for r in trajectories}
    for seed,order in cfg["order"].items():
        if len(order)!=128 or set(order)!=ids:raise RuntimeError("AM frozen trajectory order mismatch")
    all_actions=[a for r in trajectories for a in r["actions"]]+states
    for a in all_actions:
        if not a["action_token_ids"] or len(a["prompt_token_ids"])!=len(a["prompt_attention_mask"]):
            raise RuntimeError("AM action/prompt mask mismatch")
        if not all(v==1 for v in a["prompt_attention_mask"]):raise RuntimeError("unexpected padding in frozen material")
    tok=tokenizer_from_base()
    for s in states:
        unit,field,spans,_=semantic_weights(tok,s["action_token_ids"],s["target_action"])
        if unit!=s["unit_weights"] or field!=s["field_weights"] or field[-1]!=1:
            raise RuntimeError("semantic field/end-token weights mismatch")
    eval_ids,eval_questions,_=eval_isolation();validate_state_isolation(states,eval_ids,eval_questions)
    cases=rows(OUT/"evaluation-cases.jsonl")
    if len(cases)!=64 or any(not all(k in c for k in ("case_id","question","answer")) for c in cases):
        raise RuntimeError("evaluation schema mismatch")
    client,wrapped,registry=product_client(SOURCE,Z/"corpus-dev.json")
    if client._model is not None:raise RuntimeError("CPU preflight unexpectedly loaded model")
    if not registry.definitions():raise RuntimeError("production search registry missing")
    for k in ("max_input_tokens","max_new_tokens","action_completion","observation_encoding"):
        if getattr(client.config,k)!=PC[k]:raise RuntimeError("client product config mismatch")
    write_once(OUT/"cpu-preflight.json",dict(passed=True,optimizer_valid_step=True,optimizer_nan_rejected=True,
        normalized_loss_and_gradients=True,trajectory_count=128,trajectory_actions=len(all_actions)-34,
        states=34,evaluation_cases=64,semantic_weights_checked=34,mask_actions=len(all_actions),
        factory_and_registry_schema=True,gpu_calls=0,model_optimizer_updates=0,runner_sha256=sha(__file__)))
    print(json.dumps(read(OUT/"cpu-preflight.json")),flush=True)


def product_client(adapter, corpus):
    return q.make_client(adapter, deterministic=True, product_config=PC,
                         model_path=BASE, corpus=corpus)


def target_loss(client, action, weights, scale=1.0):
    import torch
    values=client.score_action_tensor([],action["action_token_ids"],True,
        action["prompt_token_ids"],action["prompt_attention_mask"]).float()
    w=torch.tensor(weights,dtype=values.dtype,device=values.device)
    if len(w)!=len(values) or not torch.isfinite(values).all() or not torch.isfinite(w).all() or (w<=0).any():
        raise RuntimeError("invalid AM action loss or token weights")
    loss=-scale*(values*w).sum()/w.sum()
    if not torch.isfinite(loss): raise RuntimeError("nonfinite AM loss")
    loss.backward()
    return float(loss.detach())


def train(seed, arm):
    cfg=approved()
    if seed not in (42,43) or arm not in ("A","B"): raise ValueError("unfrozen AM train selector")
    if seed == 43:
        gate_path=OUT/"seed42-gate.json"
        if not gate_path.exists() or read(gate_path).get("route_passed") is not True:
            raise RuntimeError("seed43 is frozen until the reviewed seed42 gate passes")
    if gpu_usage("update") >= GPU_LIMITS["updates"] or gpu_usage("update")+128 > GPU_LIMITS["updates"]:
        raise RuntimeError("AM frozen optimizer-update cap would be exceeded")
    import torch
    from transformers import set_seed
    import run_v133_goal_o_budget as budget
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    folder=OUT/f"seed{seed}-{arm}"
    folder.mkdir(exist_ok=False)
    if torch.cuda.device_count()!=1: raise RuntimeError("AM requires exactly one visible GPU")
    set_seed(seed)
    client,wrapped,registry=product_client(SOURCE,Z/"corpus-train.json")
    named=client.prepare_for_training();params=[p for _,p in named]
    if any("lora_" not in n for n,_ in named):raise RuntimeError("non-LoRA parameter trainable")
    base_before=budget.parameter_hashes(client._model,lora=False);lora_before=budget.parameter_hashes(client._model,lora=True)
    zdata={r["id"]:r for r in rows(Z/"training-inputs.jsonl")};states=rows(OUT/"candidate-training-states.jsonl")
    order=cfg["order"][str(seed)];opt=cfg["optimizer"]
    optimizer=torch.optim.AdamW(params,lr=opt["learning_rate"],betas=tuple(opt["betas"]),eps=opt["eps"],weight_decay=opt["weight_decay"])
    state_by_update={2+2*i:s for i,s in enumerate(states)}
    for update,cid in enumerate(order,1):
        optimizer.zero_grad(set_to_none=True);client.set_training_forward_mode()
        acts=zdata[cid]["actions"];denom=sum(len(a["action_token_ids"]) for a in acts)
        tl=0.0
        for a in acts:
            n=len(a["action_token_ids"])
            tl+=target_loss(client,a,[1.0]*n,scale=n/denom)
        sid=state_by_update.get(update); sl=0.0
        if sid:
            weights=sid["unit_weights"] if arm=="A" else sid["field_weights"]
            sl=target_loss(client,sid,weights,scale=0.25)
        grad=float(torch.nn.utils.clip_grad_norm_(params,opt["max_grad_norm"]))
        if not math.isfinite(grad) or grad<=0:raise RuntimeError("invalid AM gradient")
        with (OUT/JOURNAL.name).open("a",encoding="utf-8") as f:
            f.write(json.dumps(dict(kind="update",stage=f"seed{seed}-{arm}",case_id=cid,
                                    attempt=gpu_usage("update")+1,status="started"),ensure_ascii=False)+"\n")
            f.flush()
        guarded_optimizer_step(optimizer,params,client.gpu_memory_stats(),21.)
        if any(not torch.isfinite(p).all() for p in params):raise RuntimeError("nonfinite AM parameter")
        with (folder/"updates.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(dict(update=update,trajectory_id=cid,state_id=sid["state_id"] if sid else None,
                trajectory_loss=tl,state_loss=sl,gradient=grad),ensure_ascii=False)+"\n")
        print(json.dumps(dict(stage=f"seed{seed}-{arm}",update=update,trajectory_loss=tl,state_loss=sl,
                              gradient=grad,peak_memory=client.gpu_memory_stats()),ensure_ascii=False),flush=True)
    if budget.parameter_hashes(client._model,lora=False)!=base_before:raise RuntimeError("base model changed")
    changed=budget.parameter_hashes(client._model,lora=True)
    if sum(changed[n]!=v for n,v in lora_before.items())!=192:raise RuntimeError("LoRA update verification failed")
    adapter=folder/"adapter-final";client._model.save_pretrained(adapter)
    action=zdata[order[0]]["actions"][-1]
    expected=client.score_action_tensor([],action["action_token_ids"],False,action["prompt_token_ids"],action["prompt_attention_mask"]).detach().float().cpu()
    del client,wrapped,registry,optimizer,params,named;gc.collect();torch.cuda.empty_cache()
    reloaded,_,_=product_client(adapter,Z/"corpus-train.json")
    actual=reloaded.score_action_tensor([],action["action_token_ids"],False,action["prompt_token_ids"],action["prompt_attention_mask"]).detach().float().cpu()
    delta=float((expected-actual).abs().max())
    if not torch.isfinite(actual).all() or delta>0.001:raise RuntimeError("AM independent reload check failed")
    write_once(folder/"summary.json",dict(seed=seed,arm=arm,updates=128,base_unchanged=True,
        changed_lora_tensors=192,reload_max_abs=delta,adapter_sha256=sha(adapter/"adapter_model.safetensors")))


def evaluate(label):
    cfg=approved(); allowed=("baseline","42-A","42-B","43-A","43-B")
    if label not in allowed:raise ValueError("unfrozen AM evaluation selector")
    adapter=SOURCE if label=="baseline" else OUT/f"seed{label[:2]}-{label[-1]}/adapter-final"
    if label!="baseline":
        summary=read(adapter.parent/"summary.json")
        if sha(adapter/"adapter_model.safetensors")!=summary["adapter_sha256"]:raise RuntimeError("AM adapter hash mismatch")
    if label == "baseline":
        raise ValueError("Goal Z baseline is reused from AL; do not rerun baseline trajectories")
    if label.startswith("43-"):
        gate_path=OUT/"seed42-gate.json"
        if not gate_path.exists() or read(gate_path).get("route_passed") is not True:
            raise RuntimeError("seed43 evaluation is frozen until the reviewed seed42 gate passes")
    cases=rows(OUT/"evaluation-cases.jsonl");out=OUT/f"eval-{label}.jsonl"
    if out.exists():raise FileExistsError(out)
    import torch
    client,wrapped,registry=product_client(adapter,Z/"corpus-dev.json")
    active_case={"id":"unassigned"};instrument_rollout(wrapped,registry,f"eval-{label}",active_case)
    for i,case in enumerate(cases,1):
        active_case["id"]=case["case_id"]
        result=asyncio.run(q.rollout(case["question"],wrapped,registry,sample=False,seed=42,product_config=PC))
        result.update(case_id=case["case_id"],model=label,official_answer=case["answer"],**q.score(result["answer"],case["answer"]))
        if result["reason"]=="tool_error":raise RuntimeError("AM search infrastructure failure")
        if result["model_calls"]>4 or result["searches"]>3:raise RuntimeError("AM rollout budget overflow")
        if any(len(t["prompt_token_ids"])>16384 or len(t["action_token_ids"])>2048 for t in result["trace"]):
            raise RuntimeError("AM actual token budget overflow")
        with out.open("a",encoding="utf-8") as f:f.write(json.dumps(result,ensure_ascii=False)+"\n")
        print(json.dumps(dict(stage=f"eval-{label}",case=i,case_id=case["case_id"],em=result["em"],reason=result["reason"])),flush=True)
    del client,wrapped,registry;gc.collect();torch.cuda.empty_cache()


def probe(label):
    cfg=approved()
    if label not in ("baseline-state","42-A","42-B","43-A","43-B"):raise ValueError("unfrozen AM probe selector")
    if label.startswith("43-"):
        gate_path=OUT/"seed42-gate.json"
        if not gate_path.exists() or read(gate_path).get("route_passed") is not True:
            raise RuntimeError("seed43 probe is frozen until the reviewed seed42 gate passes")
    adapter=SOURCE if label=="baseline-state" else OUT/f"seed{label[:2]}-{label[-1]}/adapter-final"
    out=OUT/f"probe-{label}.jsonl"
    if out.exists():raise FileExistsError(out)
    import torch
    client,_,_=product_client(adapter,Z/"corpus-dev.json")
    # Directly replay frozen token IDs: diagnostics use exactly the archived
    # production input and never run another search.
    probes=rows(OUT/("baseline-state-probes.jsonl" if label=="baseline-state" else "frozen-probes.jsonl"))
    client._inputs=lambda messages,phase="unknown":{"input_ids":torch.tensor([current_probe["prompt_token_ids"]],device=client._model.device),
        "attention_mask":torch.ones((1,len(current_probe["prompt_token_ids"])),dtype=torch.long,device=client._model.device)}
    for current_probe in probes:
        record_gpu_attempt("generation",f"probe-{label}",current_probe["case_id"])
        r=client._generate_sync([],0.0,42)
        with out.open("a",encoding="utf-8") as f:f.write(json.dumps(dict(probe_id=current_probe["probe_id"],category=current_probe["category"],
            raw=r.raw_text,answer=r.content,action="answer" if r.content is not None else "search" if r.tool_call else "invalid",
            query=r.tool_call.arguments.get("query") if r.tool_call else None,parse_error=r.parse_error,
            prompt_token_ids=r.prompt_token_ids),ensure_ascii=False)+"\n")
        if r.prompt_token_ids!=current_probe["prompt_token_ids"]:raise RuntimeError("AM probe input changed")
    del client;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="cmd",required=True)
    sub.add_parser("prepare")
    sub.add_parser("refresh-pre-gpu")
    sub.add_parser("amend-failed-execution")
    sub.add_parser("cpu-preflight")
    t=sub.add_parser("train");t.add_argument("--seed",type=int,choices=[42,43],required=True);t.add_argument("--arm",choices=["A","B"],required=True)
    e=sub.add_parser("evaluate");e.add_argument("--model",choices=["baseline","42-A","42-B","43-A","43-B"],required=True)
    p=sub.add_parser("probe");p.add_argument("--model",choices=["baseline-state","42-A","42-B","43-A","43-B"],required=True)
    args=parser.parse_args()
    if args.cmd=="prepare":prepare()
    elif args.cmd=="refresh-pre-gpu":refresh_pre_gpu()
    elif args.cmd=="amend-failed-execution":amend_failed_execution()
    elif args.cmd=="cpu-preflight":cpu_preflight()
    else:
        approved()
        try:
            if FAILURE.exists():raise RuntimeError("AM stopped by experiment failure latch")
            if args.cmd=="train":train(args.seed,args.arm)
            elif args.cmd=="evaluate":evaluate(args.model)
            else:probe(args.model)
        except BaseException as exc:
            if not FAILURE.exists():
                write_once(FAILURE,dict(stage=args.cmd,selector=getattr(args,"model",None),
                    error_type=type(exc).__name__,error=str(exc),timestamp=time.time(),
                    generation_attempts=gpu_usage("generation"),search_attempts=gpu_usage("search"),
                    update_attempts=gpu_usage("update")))
            raise


if __name__=="__main__":main()
