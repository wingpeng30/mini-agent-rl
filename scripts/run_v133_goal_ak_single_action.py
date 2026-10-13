"""Goal AK: isolate first-action completion and compare relation prompts."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import run_v133_goal_aj_relation_prompt as aj

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ak-single-action-v1"
AJ_OUT = ROOT / "runs/v133-goal-aj-relation-prompt-v1"
AK_PROMPTS = {"A": "direct-v090", "B": "direct-v090-relation-v1"}
EXPECTED_ENGINEER = 8


def fail(stage: str, exc: Exception) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    aj.write(OUT / "failure.json", {"stage": stage, "error_type": type(exc).__name__,
        "error": str(exc), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "automatic_retry": False})
    aj.write(OUT / "status.json", {"status": "stopped_on_failure", "stage": stage})


def first_action(text: str):
    from mini_agent_rl.model.local import first_json_object_end, parse_agent_action
    end = first_json_object_end(text)
    if end is None:
        return None, None
    try:
        answer, call = parse_agent_action(text[:end])
        return ("search", str(call.arguments["query"]).strip()) if call else ("answer", answer), end
    except Exception:
        return None, end


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"run directory exists; refusing overwrite: {OUT}")
    manifest = aj.read(AJ_OUT / "manifest.json")
    if manifest.get("model", {}).get("adapter_sha256") != aj.EXPECTED_ADAPTER:
        raise RuntimeError("Goal AJ manifest adapter identity differs from published Goal Z")
    for stage_file in ("trajectory-A.jsonl", "trajectory-B.jsonl", "probes-ab.jsonl",
                       "protection-frozen.jsonl", "manifest.json"):
        if not (AJ_OUT / stage_file).is_file():
            raise FileNotFoundError(AJ_OUT / stage_file)
    if aj.sha(aj.ADAPTER / "adapter_model.safetensors") != aj.EXPECTED_ADAPTER:
        raise RuntimeError("published Goal Z adapter hash mismatch")
    if aj.sha(aj.CASES) != aj.EXPECTED_CASES or aj.sha(aj.CORPUS) != aj.EXPECTED_CORPUS:
        raise RuntimeError("frozen Goal Z data identity mismatch")
    historical_a, historical_b = aj.rows(AJ_OUT / "trajectory-A.jsonl"), aj.rows(AJ_OUT / "trajectory-B.jsonl")
    probes = aj.rows(AJ_OUT / "probes-ab.jsonl")
    if len(historical_a) != 64 or len(historical_b) != 64 or len(probes) != 32:
        raise RuntimeError("Goal AJ archives are incomplete")
    raw_a = [step for row in historical_a for step in row.get("trace", [])]
    raw_b = [step for row in historical_b for step in row.get("trace", [])]
    raw_probe = probes
    first_valid_a = sum(first_action(step.get("raw", ""))[0] is not None for step in raw_a)
    first_valid_b = sum(first_action(step.get("raw", ""))[0] is not None for step in raw_b)
    first_valid_probe = sum(first_action(step.get("raw", ""))[0] is not None for step in raw_probe)
    if (len(raw_a), len(raw_b), len(raw_probe), first_valid_a, first_valid_b) != (194, 79, 32, 194, 78):
        raise RuntimeError("AJ raw first-action audit no longer matches frozen findings")
    eligible = []
    for row in historical_b:
        trace = row.get("trace", [])
        if not trace:
            continue
        step = trace[0]
        raw = step.get("raw", "")
        parsed, _ = first_action(raw)
        if step.get("parse_error") and parsed and raw.count('{"action"') > 1:
            eligible.append({"case_id": row["case_id"], "question": row["question"],
                "historical_A": historical_a[next(i for i, x in enumerate(historical_a) if x["case_id"] == row["case_id"])]["trace"][0],
                "historical_B": step, "expected_A_first_action": first_action(
                    historical_a[next(i for i, x in enumerate(historical_a) if x["case_id"] == row["case_id"])]["trace"][0].get("raw", ""))[0],
                "expected_B_first_action": parsed})
        if len(eligible) == EXPECTED_ENGINEER:
            break
    if len(eligible) != EXPECTED_ENGINEER:
        raise RuntimeError(f"expected 8 eligible paired engineering cases, found {len(eligible)}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(aj.BASE), local_files_only=True, trust_remote_code=False)
    token_counts = {"A": len(tokenizer.encode(manifest["arms"]["A"]["system_prompt"], add_special_tokens=False)),
                    "B": len(tokenizer.encode(manifest["arms"]["B"]["system_prompt"], add_special_tokens=False))}
    if token_counts != {"A": 21, "B": 74}:
        raise RuntimeError(f"prompt token counts changed: {token_counts}")
    OUT.mkdir(parents=True)
    import shutil
    for source, archived_name in ((ROOT / "src/mini_agent_rl/model/local.py", "source-local-model-ak.py"),
        (Path(__file__), "source-ak-runner.py"), (ROOT / "scripts/run_v133_goal_q_search.py", "source-q-runner.py"),
        (ROOT / "scripts/run_v133_goal_aj_relation_prompt.py", "source-aj-runner.py"),
        (ROOT / "src/mini_agent_rl/prompts.py", "source-prompts.py")):
        shutil.copyfile(source, OUT / archived_name)
    prior_local = Path("/tmp/goal-ak-local-model-before-ak.py")
    if prior_local.is_file():
        shutil.copyfile(prior_local, OUT / "source-local-model-before-ak.py")
    for index, row in enumerate(eligible):
        aj.append(OUT / "engineer-cases.jsonl", {"index": index, **row})
    source_hashes = {"runner": aj.sha(Path(__file__)), "local_model": aj.sha(ROOT / "src/mini_agent_rl/model/local.py"),
        "q_runner": aj.sha(ROOT / "scripts/run_v133_goal_q_search.py"), "prompt_source": aj.sha(ROOT / "src/mini_agent_rl/prompts.py"),
        "aj_runner": aj.sha(ROOT / "scripts/run_v133_goal_aj_relation_prompt.py"),
        "aj_manifest": aj.sha(AJ_OUT / "manifest.json"), "aj_A": aj.sha(AJ_OUT / "trajectory-A.jsonl"),
        "aj_B": aj.sha(AJ_OUT / "trajectory-B.jsonl"), "aj_probes": aj.sha(AJ_OUT / "probes-ab.jsonl"),
        "adapter": aj.sha(aj.ADAPTER / "adapter_model.safetensors"), "cases": aj.sha(aj.CASES),
        "corpus": aj.sha(aj.CORPUS)}
    gpu = aj.check_gpu()
    if not gpu.get("bf16_supported"):
        raise RuntimeError("CUDA device does not support BF16")
    result = {"goal": "AK", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": manifest["model"], "data": manifest["data"], "product": {**manifest["product"],
            "action_completion": "first-json-v1", "max_model_calls_per_case": 4, "max_searches_per_case": 3,
            "max_input_tokens": 2048, "max_new_tokens": 128}, "arms": manifest["arms"],
        "limits": {"engineering_generations": 16, "trajectory_cases": 128, "trajectory_generation_max": 512,
            "generation_max_total": 528, "search_max": 384, "training_updates": 0, "new_confirmation_calls": 0,
            "automatic_retry": False}, "gpu_preflight": gpu, "hashes": source_hashes,
        "historical_first_action_audit": {"A_generations": len(raw_a), "A_first_action_valid": first_valid_a,
            "B_generations": len(raw_b), "B_first_action_valid": first_valid_b,
            "paired_probe_generations": len(raw_probe), "probe_first_action_valid": first_valid_probe,
            "total_generations_reviewed": len(raw_a)+len(raw_b)+len(raw_probe)},
        "scope": "development-only; behavior experiment; no model release"}
    aj.write(OUT / "manifest.json", result)
    aj.write(OUT / "approval.json", {"authorized_by_user": True, "source": "explicit PLEASE IMPLEMENT THIS PLAN",
        "limits_frozen": True, "gpu_run_allowed": True})
    aj.write(OUT / "status.json", {"status": "ready", "stage": "prepare", "gpu_generations": 0})
    print(json.dumps({"prepared": str(OUT), "engineering_cases": len(eligible), "first_action_A": first_valid_a,
        "first_action_B": first_valid_b, "prompt_tokens": token_counts, "gpu": gpu}, ensure_ascii=False))


def require_ready(stage: str) -> None:
    if not (OUT / "approval.json").is_file():
        raise RuntimeError("no frozen run authorization record")
    status = aj.read(OUT / "status.json").get("status")
    if status not in ("ready", "stage_complete"):
        raise RuntimeError(f"stage {stage} blocked by state {status}")


def load_runtime():
    gpu = aj.check_gpu()
    if not gpu.get("bf16_supported"):
        raise RuntimeError("GPU BF16 support missing")
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    import run_v133_goal_q_search as q
    config = LocalModelConfig(model_id=str(aj.BASE), adapter_path=str(aj.ADAPTER), device="cuda", dtype="bfloat16",
        load_in_4bit=False, max_input_tokens=2048, max_new_tokens=128, enable_thinking=False,
        deterministic=True, observation_encoding="legacy", recompute_old_logprobs=False,
        action_completion="first-json-v1")
    client = TransformersModelClient(config)
    client._load()
    identity = client.policy_fingerprint()
    if identity.get("adapter_weight_sha256") != aj.EXPECTED_ADAPTER or identity.get("gpu") != gpu["name"]:
        raise RuntimeError("loaded model/adapter identity mismatch")
    registry = ToolRegistry()
    registry.register(SearchTool(aj.read(aj.CORPUS), "bm25-v2"))
    return q, client, q.SearchTextAliasClient(client), registry


def _parsed(response):
    if response.tool_call:
        return "search", str(response.tool_call.arguments.get("query", "")).strip()
    return "answer", response.content


def engineer() -> None:
    require_ready("engineer")
    target = OUT / "engineer-results.jsonl"
    if target.exists():
        raise FileExistsError(target)
    try:
        q, client, wrapped, registry = load_runtime()
        from mini_agent_rl.domain import Message
        from mini_agent_rl.prompts import system_prompt
        for case in aj.rows(OUT / "engineer-cases.jsonl"):
            for arm, variant in AK_PROMPTS.items():
                archived = case["historical_A" if arm == "A" else "historical_B"]
                messages = [Message(role="system", content=system_prompt(False, variant)),
                    Message(role="user", content=case["question"])]
                encoded = client._inputs(messages, phase="goal_ak_engineer_replay")
                ids = encoded["input_ids"][0].tolist()
                if ids != archived.get("prompt_token_ids"):
                    raise RuntimeError(f"{arm} engineering input replay mismatch for {case['case_id']}")
                response = asyncio.run(wrapped.generate(messages, registry.definitions(), temperature=0., seed=42))
                expected = case["expected_A_first_action" if arm == "A" else "expected_B_first_action"]
                actual = _parsed(response)
                record = {"index": case["index"], "case_id": case["case_id"], "arm": arm,
                    "expected_first_action": expected, "actual_action": actual, "action_matches": actual == expected,
                    "prompt_token_ids_match": True, "usage": response.usage, "raw": response.raw_text,
                    "parse_error": response.parse_error, "action_token_ids": response.action_token_ids}
                aj.append(target, record)
                print(json.dumps({"engineering": len(aj.rows(target)), "case_id": case["case_id"], "arm": arm,
                    "match": record["action_matches"], "parse_error": response.parse_error}, ensure_ascii=False), flush=True)
        results = aj.rows(target)
        if len(results) != 16 or any(not r["action_matches"] or not r["prompt_token_ids_match"] or
            r["parse_error"] or r["usage"].get("first_json_stop_offset") is None for r in results):
            raise RuntimeError("16-call first-action engineering gate failed")
        aj.write(OUT / "status.json", {"status": "stage_complete", "stage": "engineer", "gpu_generations": 16,
            "model_identity": client.policy_fingerprint()})
        del client, wrapped, registry
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except Exception as exc:
        fail("engineer", exc)
        raise


def evaluate(arm: str) -> None:
    stage = f"evaluate-{arm.lower()}"
    require_ready(stage)
    if not (OUT / "engineer-results.jsonl").is_file() or len(aj.rows(OUT / "engineer-results.jsonl")) != 16:
        raise RuntimeError("engineering gate is incomplete")
    target = OUT / f"trajectory-{arm}.jsonl"
    if target.exists():
        raise FileExistsError(target)
    try:
        q, client, wrapped, registry = load_runtime()
        archived_path = AJ_OUT / f"trajectory-{arm}.jsonl"
        archived = {row["case_id"]: row for row in aj.rows(archived_path)}
        cases = aj.rows(aj.CASES)
        for i, row in enumerate(cases):
            prompt = AK_PROMPTS[arm]
            pc = {**aj.PRODUCT, "system_prompt": prompt}
            value = asyncio.run(q.rollout(row["question"], wrapped, registry, sample=False, seed=42, product_config=pc))
            score = q.score(value["answer"], row["answer"])
            value.update(case_id=row["case_id"], split=row.get("split", "goal_z_dev"), arm=arm,
                official_answer=row["answer"], support_titles=row.get("support_titles", []), **score)
            aj.append(target, value)
            if arm == "A":
                ref = archived.get(row["case_id"])
                if not ref:
                    raise RuntimeError(f"historical A baseline absent for {row['case_id']}")
                current = value.get("trace", [])
                old = ref.get("trace", [])
                if len(current) != len(old):
                    raise RuntimeError(f"A baseline call count changed for {row['case_id']}")
                for call_index, (new_call, old_call) in enumerate(zip(current, old)):
                    if new_call.get("prompt_token_ids") != old_call.get("prompt_token_ids"):
                        raise RuntimeError(f"A input replay mismatch at {row['case_id']} call {call_index}")
                    if (new_call.get("action"), new_call.get("query"), new_call.get("answer")) != \
                       (old_call.get("action"), old_call.get("query"), old_call.get("answer")):
                        raise RuntimeError(f"A first-action baseline mismatch at {row['case_id']} call {call_index}")
            elif i == 0:
                ref = archived.get(row["case_id"], {}).get("trace", [{}])[0]
                if not value.get("trace") or value["trace"][0].get("prompt_token_ids") != ref.get("prompt_token_ids"):
                    raise RuntimeError("B initial input replay mismatch")
            if value["model_calls"] > 4 or value["searches"] > 3:
                raise RuntimeError(f"product budget exceeded for {row['case_id']}")
            print(json.dumps({"arm": arm, "index": i + 1, "case_id": row["case_id"], "em": value["em"],
                "calls": value["model_calls"], "searches": value["searches"]}, ensure_ascii=False), flush=True)
        if len(aj.rows(target)) != 64:
            raise RuntimeError(f"incomplete {arm} trajectory")
        aj.write(OUT / "status.json", {"status": "stage_complete", "stage": stage, "cases": 64,
            "generations": sum(r["model_calls"] for r in aj.rows(target)), "model_identity": client.policy_fingerprint()})
        del client, wrapped, registry
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except Exception as exc:
        fail(stage, exc)
        raise


def freeze_protection() -> None:
    require_ready("freeze-protection")
    a_path, freeze_path = OUT / "trajectory-A.jsonl", OUT / "protection-frozen.jsonl"
    if not a_path.is_file() or freeze_path.exists():
        raise RuntimeError("A must finish and protection may be frozen only once")
    aj_frozen = {r["case_id"] for r in aj.rows(AJ_OUT / "protection-frozen.jsonl")}
    a = {r["case_id"]: r for r in aj.rows(a_path)}
    frozen = []
    for item in aj.rows(AJ_OUT / "protection-frozen.jsonl"):
        case_id = item["case_id"]
        if a.get(case_id, {}).get("em") == 1:
            aj.append(freeze_path, {**item, "frozen_before_B": True, "source": "Goal AJ frozen 7-case protection"})
            frozen.append(case_id)
    if set(frozen) != aj_frozen or len(frozen) != 7:
        raise RuntimeError("A did not preserve the seven frozen sequential-search cases")
    aj.write(OUT / "status.json", {"status": "stage_complete", "stage": "freeze-protection", "cases": len(frozen)})
    print(json.dumps({"protected": frozen}, ensure_ascii=False))


def report() -> None:
    require_ready("report")
    a, b = aj.rows(OUT / "trajectory-A.jsonl"), aj.rows(OUT / "trajectory-B.jsonl")
    if len(a) != 64 or len(b) != 64:
        raise RuntimeError("both 64-case trajectories are required")
    admission = {r["case_id"]: r for r in aj.rows(aj.ADMISSION)}
    edge_map = {}
    for edge in aj.rows(aj.EDGES):
        edge_map.setdefault(edge["case_id"], []).append(edge)
    amap, bmap = {r["case_id"]: r for r in a}, {r["case_id"]: r for r in b}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(aj.BASE), local_files_only=True, trust_remote_code=False)
    cases = []
    for case_id in amap:
        pa, pb = amap[case_id], bmap[case_id]
        admitted = admission.get(case_id, {}).get("admission") == "admitted"
        def edge_visibility(row):
            trace = row.get("trace", [])
            actual = trace[-1].get("prompt_token_ids", []) if trace else []
            text = ""
            if actual:
                text = tokenizer.decode(actual, skip_special_tokens=False)
            required = [e for e in edge_map.get(case_id, []) if e.get("status") == "pass" and e.get("quote")]
            visible = [e for e in required if e["quote"].casefold() in text.casefold()]
            return {"required_edges": len(required), "visible_edges": len(visible),
                "quotes_visible": len(required) > 0 and len(visible) == len(required),
                "visible_edge_ids": [e.get("edge_id") for e in visible], "actual_input_tokens": len(actual)}
        va, vb = edge_visibility(pa), edge_visibility(pb)
        cases.append({"case_id": case_id, "admitted": admitted, "question": pa["question"],
            "A_answer": pa.get("answer"), "A_em": pa["em"], "A_reason": pa["reason"], "A_searches": pa["searches"],
            "A_evidence": va, "B_answer": pb.get("answer"), "B_em": pb["em"], "B_reason": pb["reason"],
            "B_searches": pb["searches"], "B_evidence": vb, "B_new_success_candidate": admitted and pb["em"] and vb["quotes_visible"]})
    metrics = {}
    for arm, data in (("A", a), ("B", b)):
        traces = [call for row in data for call in row["trace"]]
        metrics[arm] = {"cases": len(data), "em": sum(r["em"] for r in data), "mean_f1": statistics.mean(r["f1"] for r in data),
            "parse_invalid_cases": sum(r["reason"] == "invalid_action" for r in data),
            "repeated_action_cases": sum(r["reason"] == "repeated_action" for r in data),
            "invalid_or_repeated_cases": sum(r["reason"] in ("invalid_action", "repeated_action", "tool_error") for r in data),
            "searches": sum(r["searches"] for r in data), "model_calls": sum(r["model_calls"] for r in data),
            "mean_searches_per_case": statistics.mean(r["searches"] for r in data),
            "mean_generation_seconds": statistics.mean(t["model_seconds"] for t in traces),
            "mean_question_seconds": statistics.mean(r["elapsed_seconds"] for r in data)}
    gained_em = sorted(k for k in amap if bmap[k]["em"] and not amap[k]["em"])
    lost_em = sorted(k for k in amap if amap[k]["em"] and not bmap[k]["em"])
    admitted_ids = {k for k, v in admission.items() if v.get("admission") == "admitted"}
    support_a = {x["case_id"] for x in cases if x["admitted"] and x["A_em"] and x["A_evidence"]["quotes_visible"]}
    support_b = {x["case_id"] for x in cases if x["admitted"] and x["B_em"] and x["B_evidence"]["quotes_visible"]}
    protected = {r["case_id"] for r in aj.rows(OUT / "protection-frozen.jsonl")}
    protected_lost = sorted(k for k in protected if amap[k]["em"] and not bmap[k]["em"])
    result = {"goal": "AK", "scope": "fixed Goal Z seed42 development behavior experiment; no release",
        "metrics": metrics, "admitted_cases": len(admitted_ids), "unconfirmed_cases": 64-len(admitted_ids),
        "source_quote_supported_EM": {"A": len(support_a), "B": len(support_b), "net": len(support_b)-len(support_a),
            "gained": sorted(support_b-support_a), "lost": sorted(support_a-support_b)},
        "all64_em_gained": gained_em, "all64_em_lost": lost_em, "protected_cases": sorted(protected),
        "protected_A_success_lost": protected_lost, "case_ledger": cases,
        "counts": {"engineering_generations": 16, "trajectory_generations": metrics["A"]["model_calls"]+metrics["B"]["model_calls"],
            "total_generations": 16+metrics["A"]["model_calls"]+metrics["B"]["model_calls"],
            "searches": metrics["A"]["searches"]+metrics["B"]["searches"], "training_updates": 0},
        "interpretation_limits": ["Unconfirmed 35 cases are not counted as semantic errors.",
            "Source-quote visibility and EM are conservative proxies; manually adjudicate new answer/evidence pairs once before gate."]}
    support_gained = support_b-support_a
    result["exploration_gate_pre_adjudication"] = {"at_least_two_new_supported_cases": len(support_gained)>=2,
        "admitted_support_net_plus_two": len(support_b)-len(support_a)>=2,
        "factual_error_total": "requires one-pass primary adjudication of up to 58 answer/evidence pairs",
        "invalid_actions_not_higher": metrics["B"]["invalid_or_repeated_cases"]<=metrics["A"]["invalid_or_repeated_cases"],
        "protected_losses_at_most_one": len(protected_lost)<=1,
        "status": "pending_semantic_adjudication"}
    aj.write(OUT / "same-question-ab-ledger.json", cases)
    aj.write(OUT / "report-pre-adjudication.json", result)
    lines = ["# Goal AK：首个JSON动作边界与关系提示A/B", "", "## 原始失败复核", "",
        "AJ缓存的305次生成显示A 194/194、B 78/79首个JSON动作可解析；本轮先核对生成边界，再看关系提示是否改善搜索。", "",
        "## 同题指标", "", "| 指标 | A 原提示 | B 关系提示 |", "|---|---:|---:|"]
    for key, label in (("em","EM / 64"),("mean_f1","平均F1"),("parse_invalid_cases","解析无效题"),
        ("repeated_action_cases","重复动作题"),("model_calls","模型调用"),("searches","真实搜索"),
        ("mean_searches_per_case","每题搜索"),("mean_generation_seconds","每次生成秒"),("mean_question_seconds","每题秒")):
        x,y=metrics["A"][key],metrics["B"][key]
        lines.append(f"| {label} | {x:.3f} | {y:.3f} |" if isinstance(x,float) else f"| {label} | {x} | {y} |")
    lines += ["", f"EM赢回{len(gained_em)}题，丢失{len(lost_em)}题。准入层有{len(support_a)}/29和{len(support_b)}/29题同时满足EM及必要原文引用可见。B新候选{len(support_gained)}题。", "",
        f"连续搜索保护题丢失{len(protected_lost)}/{len(protected)}。共完成{result['counts']['total_generations']}次生成、{result['counts']['searches']}次真实搜索；上限分别528、384。", "",
        "尚需对准入题的新答案/可见证据组合做一次有限裁定，最多58对；完整逐题信息在same-question-ab-ledger.json。" ]
    (OUT/"REPORT-pre-adjudication.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    aj.write(OUT / "status.json", {"status":"stage_complete","stage":"report-pre-adjudication",
        "generations":result["counts"]["total_generations"],"searches":result["counts"]["searches"]})
    print(json.dumps({"A_EM":metrics["A"]["em"],"B_EM":metrics["B"]["em"],
        "support_A":len(support_a),"support_B":len(support_b),"new_supported_candidates":len(support_gained),
        "protected_lost":len(protected_lost),"counts":result["counts"]},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","engineer","evaluate-a","freeze-protection","evaluate-b","report"))
    stage=parser.parse_args().stage
    try:
        {"prepare":prepare,"engineer":engineer,"evaluate-a":lambda:evaluate("A"),
         "freeze-protection":freeze_protection,"evaluate-b":lambda:evaluate("B"),"report":report}[stage]()
    except Exception as exc:
        if OUT.exists() and not (OUT/"failure.json").exists():
            fail(stage,exc)
        raise


if __name__=="__main__":
    main()
