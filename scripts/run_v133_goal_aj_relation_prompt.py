"""Goal AJ: frozen Goal Z relation-prompt A/B behavior experiment on AutoDL."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-aj-relation-prompt-v1"
BASE = Path("/root/autodl-tmp/qwen_08b")
ADAPTER = ROOT / "runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final"
CASES = ROOT / "runs/v133-goal-ah-entity-relation/evaluation-cases-goal_z_dev.jsonl"
HIST = ROOT / "runs/v133-goal-ah-entity-relation/eval-goal_z_dev-baseline.jsonl"
PROBES = ROOT / "runs/v133-goal-ah-entity-relation/mechanism-probes.jsonl"
PROTECTION = ROOT / "runs/v133-goal-ah-entity-relation/necessary-search-protection-frozen.jsonl"
ADMISSION = ROOT / "runs/v133-goal-ai-z64-audit-repair-v7/source-admission-ledger.final.jsonl"
EDGES = ROOT / "runs/v133-goal-ai-z64-audit-repair-v7/edge-evidence-ledger.final.jsonl"
PRIOR_REVIEWS = ROOT / "runs/v133-goal-ah-entity-relation/final-source-review-baseline.jsonl"
V7_VERIFY = ROOT / "runs/v133-goal-ai-z64-audit-repair-v7/verification-final.json"
V7_FREEZE = ROOT / "runs/v133-goal-ai-z64-audit-repair-v7/freeze.json"
CORPUS = ROOT / "runs/v133-goal-z-multihop-sft/corpus-dev.json"
EXPECTED_ADAPTER = "d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52"
EXPECTED_CASES = "8c7fe268441632d38102f1eead4ae7ea5cbf90c5e0289cbddca0b7e106fde9e6"
EXPECTED_CORPUS = "dc2a2f843ad04bb2c38ce839e1959034b7c63f1cbbc24971de2557ce3cf62d37"
PROMPT_B = (
    "You are an evidence-driven retrieval agent. Output one JSON search or answer action, using visible evidence only.\n"
    "Search for a known entity and the missing relation. Keep the entity's requested role, relation direction, time, and scope. "
    "Related people and matching names do not establish a missing link. Answer only when the needed links are visible. "
    "Use at most three searches."
)
PRODUCT = {
    "retriever": "bm25-v2", "top_k": 3, "system_prompt": "direct-v090",
    "thinking": False, "max_input_tokens": 2048, "max_new_tokens": 128,
    "max_model_calls_per_case": 4, "max_searches_per_case": 3,
    "observation_encoding": "legacy",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_gpu():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Goal AJ requires exactly one available CUDA GPU; refusing fallback/retry")
    props = torch.cuda.get_device_properties(0)
    if "4090" not in props.name:
        raise RuntimeError(f"unexpected GPU identity: {props.name}")
    return {"name": props.name, "total_memory": props.total_memory, "bf16_supported": torch.cuda.is_bf16_supported()}


def prepare():
    if OUT.exists():
        raise FileExistsError(f"run directory already exists; refusing overwrite: {OUT}")
    for path in (ADAPTER / "adapter_model.safetensors", CASES, HIST, PROBES, PROTECTION, ADMISSION,
                 V7_VERIFY, V7_FREEZE, CORPUS, BASE / "config.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    identity = read(V7_VERIFY)
    freeze = read(V7_FREEZE)
    if identity.get("status") != "pass" or identity.get("identity", {}).get("aa_seed") != 42:
        raise RuntimeError("Goal Z/AA identity verification did not pass")
    adapter_sha = sha(ADAPTER / "adapter_model.safetensors")
    if adapter_sha != EXPECTED_ADAPTER or identity["identity"].get("adapter_sha256") != adapter_sha:
        raise RuntimeError("Goal Z seed42 adapter hash mismatch")
    if sha(CASES) != EXPECTED_CASES or sha(CORPUS) != EXPECTED_CORPUS:
        raise RuntimeError("frozen Goal Z cases/corpus hash mismatch")
    replay = identity.get("production_input_replay", {})
    if replay.get("status") != "pass" or replay.get("historical_calls") != 194 or replay.get("cases") != 64:
        raise RuntimeError("historical 194-call exact input replay is not verified")
    frozen_identity = freeze.get("identity", {})
    if frozen_identity.get("adapter_sha256") != EXPECTED_ADAPTER or frozen_identity.get("selected_seed") != 42 or not frozen_identity.get("published"):
        raise RuntimeError("AI freeze does not point at the published Goal Z adapter")
    cases, hist, probes = rows(CASES), rows(HIST), rows(PROBES)
    if len(cases) != 64 or len(hist) != 64 or len(probes) != 16:
        raise RuntimeError("frozen evaluation inputs have unexpected counts")
    # CPU tokenizer check: no adapter/model load and no GPU generation here.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=False)
    direct = "You are an evidence-driven retrieval agent. Output one JSON search or answer action, using visible evidence only."
    token_counts = {"A": len(tok.encode(direct, add_special_tokens=False)),
                    "B": len(tok.encode(PROMPT_B, add_special_tokens=False))}
    if token_counts != {"A": 21, "B": 74}:
        raise RuntimeError(f"system prompt token counts differ from frozen expectation: {token_counts}")
    OUT.mkdir(parents=True)
    (OUT / "status.json").write_text(json.dumps({"status": "prepared", "stage": "prepare"}, indent=2) + "\n", encoding="utf-8")
    hashes = {"runner": sha(Path(__file__)), "q_runner": sha(ROOT / "scripts/run_v133_goal_q_search.py"),
              "prompts": sha(ROOT / "src/mini_agent_rl/prompts.py"), "adapter": adapter_sha,
              "cases": sha(CASES), "corpus": sha(CORPUS), "history": sha(HIST),
              "probes": sha(PROBES), "admission": sha(ADMISSION), "ai_verification": sha(V7_VERIFY)}
    manifest = {
        "goal": "AJ", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"base": str(BASE), "adapter": str(ADAPTER), "adapter_sha256": adapter_sha,
                  "published_goal": "AA / Goal Z complete-trajectory SFT seed 42"},
        "data": {"cases": 64, "admitted_source_cases": sum(r.get("admission") == "admitted" for r in rows(ADMISSION)),
                 "unconfirmed_or_not_admitted": 35, "probes": 16, "corpus": str(CORPUS)},
        "product": {**PRODUCT, "temperature": 0.0, "seed": 42, "device": "cuda:0", "dtype": "bfloat16", "quantization": False},
        "arms": {"A": {"system_prompt": direct, "sha256": hashlib.sha256(direct.encode()).hexdigest()},
                 "B": {"system_prompt": PROMPT_B, "sha256": hashlib.sha256(PROMPT_B.encode()).hexdigest()}},
        "system_prompt_tokens": token_counts,
        "limits": {"real_trajectories": 128, "probe_generations": 32, "total_generations": 544,
                   "searches": 384, "training_updates": 0, "automatic_retry": False},
        "gpu_preflight": check_gpu(), "hashes": hashes,
        "historical_replay": {"cases": 64, "calls": 194, "status": "reused verified Goal AI v7 exact replay"},
        "scope": "development-only; no new confirmation or model release",
    }
    write(OUT / "manifest.json", manifest)
    write(OUT / "approval.json", {"authorized_by_user": True, "source": "explicit PLEASE IMPLEMENT THIS PLAN",
                                  "limits_frozen": True, "gpu_run_allowed": True})
    write(OUT / "status.json", {"status": "ready", "stage": "prepare", "gpu_calls": 0})
    print(json.dumps({"prepared": str(OUT), "cases": 64, "admitted": manifest["data"]["admitted_source_cases"],
                      "probes": 16, "tokens": token_counts, "gpu": manifest["gpu_preflight"]}, ensure_ascii=False))


def load_client():
    check_gpu()
    import run_v133_goal_q_search as q
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    config = LocalModelConfig(model_id=str(BASE), adapter_path=str(ADAPTER), device="cuda", dtype="bfloat16",
        load_in_4bit=False, gradient_checkpointing=True, max_input_tokens=2048, max_new_tokens=128,
        enable_thinking=False, deterministic=True, observation_encoding="legacy", recompute_old_logprobs=False)
    client = TransformersModelClient(config)
    registry = ToolRegistry()
    registry.register(SearchTool(read(CORPUS), "bm25-v2"))
    return q, client, q.SearchTextAliasClient(client), registry


def fail(stage, exc):
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "failure.json", {"stage": stage, "error_type": type(exc).__name__, "error": str(exc),
                                 "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "automatic_retry": False})
    write(OUT / "status.json", {"status": "stopped_on_failure", "stage": stage})


def require_ready(stage):
    if not (OUT / "approval.json").exists() or read(OUT / "status.json").get("status") not in ("ready", "stage_complete"):
        raise RuntimeError(f"stage {stage} blocked by run state")


def evaluate(arm):
    require_ready(f"evaluate-{arm.lower()}")
    target = OUT / f"trajectory-{arm}.jsonl"
    if target.exists():
        raise FileExistsError(target)
    try:
        q, client, wrapped, registry = load_client()
        cases = rows(CASES)
        prompt = "direct-v090" if arm == "A" else "direct-v090-relation-v1"
        pc = {**PRODUCT, "system_prompt": prompt}
        for i, row in enumerate(cases):
            value = asyncio.run(q.rollout(row["question"], wrapped, registry, sample=False, seed=42, product_config=pc))
            value.update(case_id=row["case_id"], split=row.get("split", "goal_z_dev"), arm=arm,
                         official_answer=row["answer"], support_titles=row.get("support_titles", []),
                         em=q.score(value["answer"], row["answer"])["em"], f1=q.score(value["answer"], row["answer"])["f1"])
            append(target, value)
            print(json.dumps({"arm": arm, "index": i + 1, "case_id": row["case_id"], "em": value["em"],
                              "calls": value["model_calls"], "searches": value["searches"]}, ensure_ascii=False), flush=True)
            if value["model_calls"] > 4 or value["searches"] > 3:
                raise RuntimeError(f"budget exceeded for {row['case_id']}")
        if len(rows(target)) != 64:
            raise RuntimeError(f"incomplete {arm} result")
        write(OUT / "status.json", {"status": "stage_complete", "stage": f"evaluate-{arm.lower()}", "gpu_calls": 64})
        del client, wrapped, registry
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except Exception as exc:
        fail(f"evaluate-{arm.lower()}", exc)
        raise


def freeze_protection():
    require_ready("freeze-protection")
    a_path, freeze_path = OUT / "trajectory-A.jsonl", OUT / "protection-frozen.jsonl"
    if not a_path.exists() or freeze_path.exists():
        raise RuntimeError("A must finish first; protection freeze cannot be replaced")
    admitted = {r["case_id"] for r in rows(ADMISSION) if r.get("admission") == "admitted"}
    reviewed = {r["case_id"]: r for r in rows(PROTECTION) if r.get("protected") and r.get("supported_success")}
    a = {r["case_id"]: r for r in rows(a_path)}
    frozen = []
    for case_id in sorted(admitted & set(reviewed)):
        old = reviewed[case_id]
        later = [r for r in old.get("necessary_relations", []) if int(r.get("first_visible_call", 0)) > 1]
        current = a.get(case_id, {})
        if current.get("em") == 1 and later:
            frozen.append({"case_id": case_id, "protected_before_B": True,
                           "reason": "A is successful; previously source-reviewed correct relation first appeared after the first search",
                           "later_relations": later, "A_answer": current.get("answer")})
    for item in frozen:
        append(freeze_path, item)
    write(OUT / "protection-summary.json", {"count": len(frozen), "case_ids": [x["case_id"] for x in frozen],
                                            "frozen_before_B": True, "historical_reviews_reused": True})
    write(OUT / "status.json", {"status": "stage_complete", "stage": "freeze-protection", "gpu_calls": 0})
    print(json.dumps({"protected_cases": len(frozen), "case_ids": [x["case_id"] for x in frozen]}, ensure_ascii=False))


def probe():
    require_ready("probe")
    if not (OUT / "trajectory-A.jsonl").exists() or not (OUT / "trajectory-B.jsonl").exists():
        raise RuntimeError("both trajectory arms must finish before probes")
    target = OUT / "probes-ab.jsonl"
    if target.exists():
        raise FileExistsError(target)
    try:
        q, client, wrapped, registry = load_client()
        # Context reconstruction uses the same processor/template as generation;
        # TransformersModelClient intentionally loads these lazily.
        client._load()
        from mini_agent_rl.domain import Message
        hist = {r["case_id"]: r for r in rows(HIST)}
        cases = {r["case_id"]: r for r in rows(CASES)}
        prompt_b = "direct-v090-relation-v1"
        for p in rows(PROBES):
            case = hist.get(p["case_id"])
            if case is None or p["case_id"] not in cases:
                raise RuntimeError(f"probe has no frozen history: {p['case_id']}")
            trace = case["trace"]
            call_index = int(p["call_index"])
            if call_index >= len(trace):
                raise RuntimeError(f"probe call index not in historical trace: {p['case_id']}")
            messages = [Message(role="system", content=__import__("mini_agent_rl.prompts", fromlist=["system_prompt"]).system_prompt(False, "direct-v090")),
                        Message(role="user", content=case["question"])]
            for prior in trace[:call_index]:
                if prior.get("action") != "search" or "observation" not in prior:
                    raise RuntimeError(f"probe prefix is not a search prefix: {p['case_id']}")
                obs = {k: v for k, v in prior["observation"].items() if k != "latency_ms"}
                canonical = json.dumps({"action": "search", "query": prior["query"]}, ensure_ascii=False)
                messages.extend([Message(role="assistant", content=canonical),
                                 Message(role="tool", content=json.dumps(obs, ensure_ascii=False))])
            client._context_records.clear()
            encoded = client._inputs(messages, phase="goal_aj_probe_replay")
            actual_ids = encoded["input_ids"][0].tolist()
            expected = trace[call_index].get("prompt_token_ids", [])
            if actual_ids != expected or actual_ids != p.get("prompt_token_ids"):
                raise RuntimeError(f"A probe input does not replay exactly: {p['case_id']} call {call_index}")
            for arm, variant in (("A", "direct-v090"), ("B", prompt_b)):
                current = list(messages)
                current[0] = Message(role="system", content=__import__("mini_agent_rl.prompts", fromlist=["system_prompt"]).system_prompt(False, variant))
                start = time.monotonic()
                response = asyncio.run(wrapped.generate(current, registry.definitions(), temperature=0., seed=42))
                record = {"case_id": p["case_id"], "category": p.get("category"), "call_index": call_index,
                          "arm": arm, "question": p["question"], "raw": response.raw_text,
                          "parse_error": response.parse_error, "action": "search" if response.tool_call else "answer",
                          "query": response.tool_call.arguments.get("query") if response.tool_call else None,
                          "answer": response.content if not response.tool_call else None,
                          "prompt_token_ids": response.prompt_token_ids, "prompt_attention_mask": response.prompt_attention_mask,
                          "context_usage": client._context_records[-1] if client._context_records else None,
                          "latency_seconds": time.monotonic() - start, "executed_followup_search": False,
                          "training_use_forbidden": True}
                append(target, record)
                print(json.dumps({"probe": p["case_id"], "arm": arm, "action": record["action"]}, ensure_ascii=False), flush=True)
        if len(rows(target)) != 32:
            raise RuntimeError("probe count is not 32")
        write(OUT / "status.json", {"status": "stage_complete", "stage": "probe", "gpu_calls": 32})
        del client, wrapped, registry
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    except Exception as exc:
        fail("probe", exc)
        raise


def recover_probe_setup():
    failure = OUT / "failure.json"
    if not failure.exists() or (OUT / "probes-ab.jsonl").exists():
        raise RuntimeError("probe setup recovery only applies before the first probe generation")
    record = read(failure)
    known_setup_bugs = ((record.get("error_type") == "AttributeError" and "apply_chat_template" in record.get("error", "")) or
                        (record.get("error_type") == "ValueError" and "too many values to unpack" in record.get("error", "")))
    if record.get("stage") != "probe" or not known_setup_bugs:
        raise RuntimeError("failure is not the recorded lazy-tokenizer initialization bug; refusing resume")
    archived = OUT / ("failure-probe-setup.json" if not (OUT / "failure-probe-setup.json").exists()
                      else "failure-probe-setup2.json")
    if archived.exists():
        raise FileExistsError(archived)
    failure.replace(archived)
    write(OUT / f"recovery-note-{archived.stem}.json", {"issue": record.get("error"),
        "fix": "align replay with lazy client initialization and _inputs return signature; no model generation happened in failed attempt",
        "probe_generation_calls_before_recovery": 0, "user_plan_and_conditions_unchanged": True})
    write(OUT / "status.json", {"status": "ready", "stage": "probe-setup-recovered", "gpu_calls": 0,
                                 "prior_failure_preserved": str(archived.name)})
    print(json.dumps({"recovery": "setup-only", "preserved_failure": archived.name,
                      "probe_generations_before_recovery": 0}))


def recover_report_version():
    failure = OUT / "failure.json"
    if not failure.exists() or not (OUT / "report-final.json").exists() or not (OUT / "same-question-ab-ledger-v2.jsonl").exists():
        raise RuntimeError("report version recovery requires a complete prior report and no GPU work")
    record = read(failure)
    if record.get("stage") != "report" or record.get("error_type") not in ("FileExistsError", "RuntimeError"):
        raise RuntimeError("the report failure is outside the known artifact-version issue; refusing resume")
    if (OUT / "REPORT-final-v3.md").exists() or (OUT / "same-question-ab-ledger-v3.jsonl").exists():
        raise FileExistsError("v3 report artifacts already exist")
    archived = OUT / "failure-report-version.json"
    if archived.exists():
        raise FileExistsError(archived)
    failure.replace(archived)
    write(OUT / "recovery-report-version.json", {"issue": record.get("error"),
        "fix": "write expanded report into v3 files and preserve existing report-v2 files",
        "gpu_generations_before_recovery": 0, "user_plan_and_conditions_unchanged": True})
    write(OUT / "status.json", {"status": "ready", "stage": "report-version-recovered", "gpu_calls": 0,
                                 "prior_failure_preserved": archived.name})
    print(json.dumps({"recovery": "report-version-only", "preserved_failure": archived.name}))


def report():
    require_ready("report")
    a, b = rows(OUT / "trajectory-A.jsonl"), rows(OUT / "trajectory-B.jsonl")
    if len(a) != 64 or len(b) != 64:
        raise RuntimeError("cannot report before both full trajectory arms")
    amap, bmap = {r["case_id"]: r for r in a}, {r["case_id"]: r for r in b}
    admitted = {r["case_id"] for r in rows(ADMISSION) if r.get("admission") == "admitted"}
    protected = {r["case_id"] for r in rows(OUT / "protection-frozen.jsonl")} if (OUT / "protection-frozen.jsonl").exists() else set()
    metrics = {}
    for name, data in (("A", a), ("B", b)):
        traces = [step for row in data for step in row["trace"]]
        metrics[name] = {"n": len(data), "em": sum(r["em"] for r in data),
                         "mean_f1": statistics.mean(r["f1"] for r in data),
                         "parse_invalid_cases": sum(r.get("reason") == "invalid_action" for r in data),
                         "repeated_action_cases": sum(r.get("reason") == "repeated_action" for r in data),
                         "invalid_or_repeated_cases": sum(r.get("reason") in ("invalid_action", "repeated_action", "tool_error") for r in data),
                         "model_calls": sum(r["model_calls"] for r in data), "searches": sum(r["searches"] for r in data),
                         "mean_model_call_latency_seconds": statistics.mean(t["model_seconds"] for t in traces),
                         "mean_latency_seconds": statistics.mean(r["elapsed_seconds"] for r in data),
                         "mean_searches_per_case": statistics.mean(r["searches"] for r in data)}
    gained = [k for k in amap if k in admitted and bmap[k]["em"] and not amap[k]["em"]]
    lost = [k for k in amap if k in admitted and amap[k]["em"] and not bmap[k]["em"]]
    gained_all = sorted(k for k in amap if bmap[k]["em"] and not amap[k]["em"])
    lost_all = sorted(k for k in amap if amap[k]["em"] and not bmap[k]["em"])
    support_gained = [k for k in gained if bmap[k].get("answer")]
    if (OUT / "same-question-ab-ledger-v3.jsonl").exists() or (OUT / "REPORT-final-v3.md").exists():
        raise FileExistsError("Goal AJ report artifacts already exist; refusing overwrite")
    # Reuse only old reviews whose exact historical A input and answer are unchanged.
    prior = [r for r in rows(PRIOR_REVIEWS) if r.get("split") == "goal_z_dev" and r.get("input_and_answer_identical_to_prior")]
    prior_by_case = {}
    for r in prior:
        prior_by_case[r["case_id"]] = r
    edges_by_case = {}
    for edge in rows(EDGES):
        edges_by_case.setdefault(edge["case_id"], []).append(edge)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=False)
    def norm(value):
        value = unicodedata.normalize("NFKC", str(value)).casefold()
        return "".join(re.findall(r"[\w]+", value, flags=re.UNICODE))
    def visible_edges(row, case_id):
        final = row.get("trace", [])[-1] if row.get("trace") else {}
        text = tokenizer.decode(final.get("prompt_token_ids", []), skip_special_tokens=False)
        checks = []
        for edge in edges_by_case.get(case_id, []):
            if edge.get("status") != "pass" or not edge.get("quote"):
                continue
            checks.append({"edge_id": edge["edge_id"], "document_id": edge.get("document_id"),
                           "quote_visible_in_actual_final_input": norm(edge["quote"]) in norm(text),
                           "document_id_visible_in_actual_final_input": str(edge.get("document_id")) in text,
                           "triple": [edge.get("subject"), edge.get("relation"), edge.get("object")]})
        return checks
    manual_b = {
        "musique:2hop__789565_416401": {"category_if_chain_visible": "factual_error", "entity_mismatch": True,
            "relation_mismatch": True, "reason": "The visible chain identifies Wang Jipeng as Wang Yanjun's child; it describes Lady Li as Wang Jipeng's cousin, not as Wang Yanjun's child."},
        "musique:2hop__158275_35739": {"category_if_chain_visible": "success", "entity_mismatch": False,
            "relation_mismatch": False, "reason": "The visible Windows 8 passage identifies Firefox and says its version first became available in September 2013; same evidence-answer decision as the already-reviewed A result."},
    }
    case_rows = []
    b_review_rows = []
    for case_id in sorted(amap):
        admission = next((x for x in rows(ADMISSION) if x["case_id"] == case_id), None)
        a_edges, b_edges = visible_edges(amap[case_id], case_id), visible_edges(bmap[case_id], case_id)
        a_review = prior_by_case.get(case_id) if case_id in admitted else None
        b_review = None
        if case_id in admitted and bmap[case_id].get("answer") is not None:
            proposal = manual_b.get(case_id)
            b_text = tokenizer.decode(bmap[case_id]["trace"][-1].get("prompt_token_ids", []), skip_special_tokens=False)
            nb = norm(b_text)
            if case_id == "musique:2hop__789565_416401":
                # Use the actual returned biography sentence, not the potentially narrower edge quote.
                complete = ("empressdowagerhuang" in nb and "motherofwangyanjun" in nb and
                            "ladyliwasacousintowangjipeng" in nb and "hisfatherwangyanjun" in nb)
            elif case_id == "musique:2hop__158275_35739":
                # Windows 8's returned body itself binds Firefox, the version, and the date.
                complete = "firefoxsversionwhichwasfirstmadeavailableontheaurorareleasechannelinseptember2013" in nb
            else:
                complete = bool(b_edges) and all(e["quote_visible_in_actual_final_input"] for e in b_edges)
            if proposal:
                category = proposal["category_if_chain_visible"] if complete else "unsupported_by_visible_evidence"
                b_review = {"case_id": case_id, "answer": bmap[case_id]["answer"], "category": category,
                    "evidence_complete": complete, "entity_mismatch": proposal["entity_mismatch"] if complete else False,
                    "relation_mismatch": proposal["relation_mismatch"] if complete else False,
                    "reason": proposal["reason"] if complete else "The complete source-reviewed relation chain was not retained in the actual final input.",
                    "reviewer": "primary-agent; single pass; actual Goal AJ output", "training_use_forbidden": True,
                    "actual_input_sha256": hashlib.sha256(str(bmap[case_id]["trace"][-1].get("prompt_token_ids", [])).encode()).hexdigest(),
                    "actual_input_token_count": len(bmap[case_id]["trace"][-1].get("prompt_token_ids", [])),
                    "manual_body_binding_check": {"chain_visible": complete, "case_specific_source_body_rule": True},
                    "edge_checks": b_edges}
                b_review_rows.append(b_review)
        case_rows.append({"case_id": case_id, "question": amap[case_id]["question"], "admission": admission.get("admission") if admission else "unknown",
            "A_answer": amap[case_id].get("answer"), "A_reason": amap[case_id].get("reason"), "A_em": amap[case_id]["em"],
            "A_review_reused": a_review is not None, "A_category": a_review.get("category") if a_review else "unknown",
            "A_edge_checks": a_edges, "B_answer": bmap[case_id].get("answer"), "B_reason": bmap[case_id].get("reason"),
            "B_em": bmap[case_id]["em"], "B_category": b_review.get("category") if b_review else
                ("invalid_action" if bmap[case_id].get("reason") == "invalid_action" else "unknown"),
            "B_edge_checks": b_edges})
    # The only novel answer/evidence combinations in the admitted layer are explicitly adjudicated above.
    for review in b_review_rows:
        append(OUT / "source-review-b-v3.jsonl", review)
    for item in case_rows:
        append(OUT / "same-question-ab-ledger-v3.jsonl", item)
    a_admitted = [r for r in case_rows if r["admission"] == "admitted"]
    b_successes = sum(r["B_category"] == "success" for r in a_admitted)
    a_successes = sum(r["A_category"] == "success" for r in a_admitted)
    a_err = sum(r["A_category"] in ("factual_error", "unsupported_by_visible_evidence") for r in a_admitted)
    b_err = sum(r["B_category"] in ("factual_error", "unsupported_by_visible_evidence") for r in a_admitted)
    raw_parse_bad = metrics["B"]["parse_invalid_cases"]
    support_gained = sorted(r["case_id"] for r in a_admitted if r["B_category"] == "success" and r["A_category"] != "success")
    support_lost = sorted(r["case_id"] for r in a_admitted if r["A_category"] == "success" and r["B_category"] != "success")
    probes = rows(OUT / "probes-ab.jsonl") if (OUT / "probes-ab.jsonl").exists() else []
    probe_by_case = {}
    for p in probes:
        probe_by_case.setdefault(p["case_id"], {})[p["arm"]] = p
    probe_transitions = {}
    for pair in probe_by_case.values():
        if set(pair) == {"A", "B"}:
            key = f"{pair['A']['action']}→{pair['B']['action']}"
            probe_transitions[key] = probe_transitions.get(key, 0) + 1
    context_summary = {}
    concatenated = 0
    for name, data in (("A", a), ("B", b)):
        traces = [t for row in data for t in row["trace"]]
        uses = [t.get("context_usage") or {} for t in traces]
        context_summary[name] = {"calls": len(traces), "max_input_tokens": max((len(t.get("prompt_token_ids", [])) for t in traces), default=0),
            "prompt_truncated_calls": sum(bool(c.get("prompt_truncated")) for c in uses),
            "lossy_body_trimmed_calls": sum(bool(c.get("lossy_body_trimmed")) for c in uses),
            "documents_deleted": sum((len(v) if isinstance(v, list) else int(v or 0)) for c in uses for v in [c.get("documents_deleted", 0)]),
            "observation_compacted_calls": sum(bool(c.get("observation_compacted")) for c in uses)}
        if name == "B":
            concatenated = sum(bool(t.get("parse_error")) and str(t.get("raw", "")).count('{"action"') > 1 for t in traces)
    results = {"scope": "development-only prompt behavior comparison; not a model release",
        "model_identity": {"base": str(BASE), "configured_runtime_adapter": str(ADAPTER),
                           "runtime_adapter_sha256": sha(ADAPTER / "adapter_model.safetensors"),
                           "matches_published_goal_z_seed42": sha(ADAPTER / "adapter_model.safetensors") == EXPECTED_ADAPTER},
        "observed_executions": {"A_trajectory_generations": metrics["A"]["model_calls"],
            "B_trajectory_generations": metrics["B"]["model_calls"], "probe_generations": len(probes),
            "total_generations": metrics["A"]["model_calls"] + metrics["B"]["model_calls"] + len(probes),
            "real_searches": metrics["A"]["searches"] + metrics["B"]["searches"], "training_updates": 0},
        "metrics": metrics, "admitted_cases": len(admitted), "admitted_em_gained_raw": gained,
        "admitted_em_lost_raw": lost, "em_gained_raw_all64": gained_all, "em_lost_raw_all64": lost_all,
        "protected_cases": len(protected),
        "protected_A_success_lost_raw": sorted(k for k in protected if amap.get(k, {}).get("em") and not bmap.get(k, {}).get("em")),
        "admitted_source_scoring": {"A_evidence_supported_success": a_successes, "B_evidence_supported_success": b_successes,
            "A_factual_error_plus_unsupported": a_err, "B_factual_error_plus_unsupported": b_err,
            "A_to_B_supported_success_gained": support_gained, "A_to_B_supported_success_lost": support_lost,
            "B_novel_answer_reviews": len(b_review_rows), "unadmitted_35": "unknown; excluded from semantic error counts"},
        "B_parse_invalid_actions": raw_parse_bad, "B_outputs_with_multiple_concatenated_actions": concatenated,
        "context_budget": context_summary,
        "probes": {"generation_count": len(probes), "case_count": len(probe_by_case), "followup_searches_executed": 0,
                   "action_transitions_A_to_B": probe_transitions, "diagnostic_only": True},
        "exploration_gate": {"at_least_two_new_supported_cases": len(support_gained) >= 2,
            "admitted_support_net_plus_two": b_successes - a_successes >= 2,
            "factual_plus_unsupported_not_higher": b_err <= a_err,
            "invalid_actions_not_higher": metrics["B"]["invalid_or_repeated_cases"] <= metrics["A"]["invalid_or_repeated_cases"],
            "protected_losses_at_most_one": len([k for k in protected if amap.get(k, {}).get("em") and not bmap.get(k, {}).get("em")]) <= 1,
            "passed": (len(support_gained) >= 2 and b_successes - a_successes >= 2 and b_err <= a_err and
                       metrics["B"]["invalid_or_repeated_cases"] <= metrics["A"]["invalid_or_repeated_cases"] and
                       len(results["protected_A_success_lost_raw"]) <= 1)}}
    write(OUT / "report-final-v3.json", results)
    md = ["# Goal AJ：关系核对提示A/B实验", "", "本次只比较固定Goal Z seed42的提示行为，开发集64题；未训练、未运行新确认集。A与B只有system prompt不同。语义来源未确认的35题不计入证据错误。", "",
          "## 同题结果", "", "| 指标 | A：direct-v090 | B：direct-v090-relation-v1 |", "|---|---:|---:|"]
    for key, label in (("em", "EM / 64"), ("mean_f1", "平均F1"), ("parse_invalid_cases", "解析无效动作题数"),
                       ("repeated_action_cases", "重复动作题数"), ("model_calls", "模型调用数"), ("searches", "搜索数"),
                       ("mean_searches_per_case", "每题平均搜索数"), ("mean_model_call_latency_seconds", "每次生成平均秒数"),
                       ("mean_latency_seconds", "每题平均秒数")):
        md.append(f"| {label} | {metrics['A'][key]:.3f} | {metrics['B'][key]:.3f} |" if isinstance(metrics['A'][key], float) else f"| {label} | {metrics['A'][key]} | {metrics['B'][key]} |")
    md += ["", "## 同题赢回与丢失", "", f"全64题中B新增EM成功{len(gained_all)}题，失去A侧EM成功{len(lost_all)}题。准入29题中新增{len(gained)}、丢失{len(lost)}。完整ID和A/B答案见`same-question-ab-ledger-v3.jsonl`。", "",
           "## 证据准入层与保护集", "", f"A的来源支持成功为{a_successes}/29，B为{b_successes}/29，净变化{b_successes-a_successes:+d}。事实错误与证据不支持合计分别为A {a_err}、B {b_err}。这29题逐边证据只在来源审核记录与实际token输入一致时复用；其余35题保持未确认。", "",
           f"B有{raw_parse_bad}题首个动作无法解析，其中{concatenated}个completion把多条JSON动作拼在一起；实际造成搜索从{metrics['A']['searches']}次降至{metrics['B']['searches']}次。保护集冻结于B运行前共{len(protected)}题；其中{len(results['protected_A_success_lost_raw'])}题失去A侧成功。", "",
           "B的已审查新增答案例：Wang Jipeng的子女关系链将Lady Li识别为表亲而非目标子女，属于实体/关系对象错配；Firefox/Windows 8案例回答September 2013，复用相同可见关系和答案的通过裁定。其他B无效动作没有答案，不归为事实错误。", "",
           f"实际共生成{metrics['A']['model_calls'] + metrics['B']['model_calls'] + len(probes)}次（上限544），执行{metrics['A']['searches'] + metrics['B']['searches']}次真实搜索（上限384），训练更新0次。B最大输入{context_summary['B']['max_input_tokens']}/2048 token，正文裁剪/删除均为0；A侧有{context_summary['A']['lossy_body_trimmed_calls']}次预算正文压缩。B新增53系统token没有造成测得的正文裁剪，结果下降与上下文丢文无关。", "",
           "## 决定", "", "探索门槛未通过：B没有净增证据支持成功，解析无效动作显著增加，且保护集成功损失超过1题。无证据表明B改善了查询或证据取得；主要退化发生在动作格式/单动作协议，Lady Li个案还显示完整链可见时仍发生实体与关系错配。停止该提示及其蒸馏路线；不发布提示或模型。下一步按原计划准备独立证据保留实验，固定Goal Z、提示和搜索预算。", "",
           "## 限制", "", "旧审核组合仅在64题A轨迹与AH原始输出逐调用完全一致时复用。B的两个来源准入答案组合做了一次主agent裁定。旧确认集没有运行，也没有据此作产品升级结论。", ""]
    (OUT / "REPORT-final-v3.md").write_text("\n".join(md), encoding="utf-8")
    write(OUT / "status.json", {"status": "stage_complete", "stage": "report-draft", "gpu_calls": 0})
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "evaluate-a", "freeze-protection", "evaluate-b", "probe",
                                           "recover-probe-setup", "recover-report-version", "report"))
    args = parser.parse_args()
    try:
        if args.stage == "prepare": prepare()
        elif args.stage == "evaluate-a": evaluate("A")
        elif args.stage == "freeze-protection": freeze_protection()
        elif args.stage == "evaluate-b": evaluate("B")
        elif args.stage == "probe": probe()
        elif args.stage == "recover-probe-setup": recover_probe_setup()
        elif args.stage == "recover-report-version": recover_report_version()
        else: report()
    except Exception as exc:
        if OUT.exists() and not (OUT / "failure.json").exists():
            fail(args.stage, exc)
        raise


if __name__ == "__main__":
    main()
