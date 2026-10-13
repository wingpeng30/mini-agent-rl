"""Goal AL: frozen Goal Z, two token budgets, bounded remote-only inference."""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import re
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import run_v133_goal_aj_relation_prompt as assets
import run_v133_goal_q_search as q
from mini_agent_rl.domain import Message
from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
from mini_agent_rl.prompts import system_prompt

ROOT = assets.ROOT
OUT = ROOT / "runs/v133-goal-al-wide-context-v1"
AK = ROOT / "runs/v133-goal-ak-single-action-v1"
AI = ROOT / "runs/v133-goal-ai-evidence-feasibility"
AA = ROOT / "runs/v133-goal-aa-audit-confirmation"
read, rows, sha, write, append = assets.read, assets.rows, assets.sha, assets.write, assets.append
COMMON = dict(retriever="bm25-v2", top_k=3, system_prompt="direct-v090", thinking=False,
              max_model_calls_per_case=4, max_searches_per_case=3, observation_encoding="legacy",
              action_completion="first-json-v1", device="cuda", dtype="bfloat16", load_in_4bit=False,
              timeout=None)
# The sole source of the two experimental token limits.
ARMS = {"A": dict(COMMON, max_input_tokens=2048, max_new_tokens=128),
        "B": dict(COMMON, max_input_tokens=16384, max_new_tokens=2048)}
LIMITS = dict(generations=522, engineering_generations=10, searches=384, training_updates=0, confirmation_calls=0)


def initial(question):
    return [Message(role="system", content=system_prompt(False, "direct-v090")), Message(role="user", content=question)]


def extend(messages, step):
    obs = {k: v for k, v in step["observation"].items() if k != "latency_ms"}
    return messages + [Message(role="assistant", content=json.dumps(dict(action="search", query=step["query"]), ensure_ascii=False)),
                       Message(role="tool", content=json.dumps(obs, ensure_ascii=False))]


def prefix(row, index):
    messages = initial(row["question"])
    for step in row["trace"][:index]:
        if "observation" not in step:
            raise RuntimeError("non-search prefix")
        messages = extend(messages, step)
    return messages


def source_history():
    """AH preserves production JSON key order; AK sorted its archive keys.

    Equal JSON values do not imply equal tokenizer inputs. Never reconstruct
    production text from a sorted archival rendering of tool objects.
    """
    originals = rows(assets.HIST)
    archive = {r["case_id"]: r for r in rows(AK / "trajectory-A.jsonl")}
    if len(originals) != 64 or len(archive) != 64:
        raise RuntimeError("source/archive population mismatch")
    for row in originals:
        archived = archive[row["case_id"]]
        if row["question"] != archived["question"] or len(row["trace"]) != len(archived["trace"]):
            raise RuntimeError("source/archive trajectory mismatch")
        for old, saved in zip(row["trace"], archived["trace"]):
            if old["prompt_token_ids"] != saved["prompt_token_ids"]:
                raise RuntimeError("source/archive actual input mismatch")
            for key in ("query", "action", "answer"):
                if old.get(key) != saved.get(key):
                    raise RuntimeError(f"source/archive action mismatch: {key}")
            if "observation" in old:
                left = {k: v for k, v in old["observation"].items() if k != "latency_ms"}
                right = {k: v for k, v in saved["observation"].items() if k != "latency_ms"}
                if left != right:
                    raise RuntimeError("source/archive tool payload mismatch")
    return originals, archive


def config(pc):
    return LocalModelConfig(model_id=str(assets.BASE), adapter_path=str(assets.ADAPTER), deterministic=True,
        max_input_tokens=pc["max_input_tokens"], max_new_tokens=pc["max_new_tokens"], enable_thinking=pc["thinking"],
        observation_encoding=pc["observation_encoding"], action_completion=pc["action_completion"],
        device=pc["device"], dtype=pc["dtype"], load_in_4bit=pc["load_in_4bit"], timeout=pc["timeout"],
        recompute_old_logprobs=False)


def renderer(pc, processor):
    client = object.__new__(TransformersModelClient)
    client.config = config(pc)
    client._processor = processor
    client._model = SimpleNamespace(device="cpu")
    client._context_records = []
    return client


def recover_prepare():
    """One explicit user-authorized recovery of the zero-call replay failure."""
    failure = read(OUT / "failure.json")
    original_failure = (failure["stage"] == "prepare" and failure["error"].startswith("input replay/retention mismatch"))
    optional_config_failure = (failure["stage"] == "recover-prepare" and failure["type"] == "FileNotFoundError" and
        failure["error"] == "[Errno 2] No such file or directory: '/root/autodl-tmp/qwen_08b/generation_config.json'")
    if (read(OUT / "status.json") != dict(status="failed", stage=failure["stage"]) or
        failure["counts"] != dict(generations=0, searches=0) or not (original_failure or optional_config_failure) or
        any((OUT / p).exists() for p in ("manifest.json", "engineer-results.jsonl", "generation-journal.jsonl"))):
        raise RuntimeError("not the authorized zero-call prepare failure")
    archived = OUT / ("failed-prepare-v1" if original_failure else "failed-prepare-v2")
    archived.mkdir()
    hashes = {}
    for name in ("failure.json", "status.json", "failed-runner-source.py", "REPORT.md", "replay-diagnosis.json", "recovery.json"):
        file = OUT / name
        if file.exists():
            hashes[name] = sha(file)
            file.rename(archived / name)
    write(OUT / "recovery.json", dict(authorized_by_user="修复完就继续之前的任务", archived_hashes=hashes,
        source_sha256=sha(Path(__file__)), prior_generations=0, prior_searches=0, limits_unchanged=LIMITS,
        correction="production-order AH replay; generation config file is optional and falls back to model config"))
    prepare(recover=True)


def prepare(*, recover=False):
    if OUT.exists() and not recover:
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True, exist_ok=recover)
    write(OUT / "status.json", dict(status="preparing"))
    gpu = assets.check_gpu()
    if not gpu["bf16_supported"]:
        raise RuntimeError("BF16 unavailable")
    decision = read(AA / "product-decision.json")
    if not decision["published"] or decision["selected_seed"] != 42 or decision["adapter"] != str(assets.ADAPTER):
        raise RuntimeError("AA published identity mismatch")
    if sha(assets.ADAPTER / "adapter_model.safetensors") != assets.EXPECTED_ADAPTER:
        raise RuntimeError("adapter identity mismatch")
    if sha(assets.CASES) != assets.EXPECTED_CASES or sha(assets.CORPUS) != assets.EXPECTED_CORPUS:
        raise RuntimeError("data identity mismatch")
    base_config = read(assets.BASE / "config.json")
    max_positions = base_config.get("text_config", base_config)["max_position_embeddings"]
    if any(p["max_input_tokens"] + p["max_new_tokens"] > max_positions for p in ARMS.values()):
        raise RuntimeError("model context cannot accommodate input+output")
    from transformers import AutoProcessor, GenerationConfig
    processor = AutoProcessor.from_pretrained(str(assets.BASE), local_files_only=True)
    cpu = {arm: renderer(pc, processor) for arm, pc in ARMS.items()}
    historical, archive = source_history()
    if len(historical) != 64 or len(read(assets.CORPUS)) != 2048:
        raise RuntimeError("frozen population mismatch")
    checks, probes = [], []
    for row in historical:
        for i, step in enumerate(row["trace"]):
            messages = prefix(row, i)
            a = cpu["A"]._inputs(messages, "mechanical_A")["input_ids"][0].tolist()
            b = cpu["B"]._inputs(messages, "mechanical_B")["input_ids"][0].tolist()
            raw = cpu["B"]._render_chat(messages)["input_ids"][0].tolist()
            if a != step["prompt_token_ids"] or b != raw:
                raise RuntimeError(f"input replay/retention mismatch {row['case_id']}:{i}")
            checks.append(dict(case_id=row["case_id"], step=i, A_match=True, B_lossless=True, A_tokens=len(a), B_tokens=len(b)))
            saved_usage = archive[row["case_id"]]["trace"][i]["context_usage"]
            if saved_usage.get("original_prompt_tokens", 0) > ARMS["A"]["max_input_tokens"]:
                probes.append(dict(case_id=row["case_id"], step=i, messages=[m.model_dump(mode="json") for m in messages],
                                   A_ids=a, B_ids=b, diagnostic_only=True))
    if len(checks) != 194 or len(probes) != 4:
        raise RuntimeError("historical count mismatch")
    sentinel = next(r for r in rows(AI / "closure-v3/repair-query-results.jsonl") if r["plan"]["case_id"] == "musique:2hop__776776_90536")
    train = next(r for r in rows(AI / "closure-v3/train-review-packets.jsonl") if r["case_id"] == sentinel["plan"]["case_id"])
    original = next(r for r in rows(AI / "gpu-train-rollouts.jsonl") if r["case_id"] == train["case_id"])
    k = sentinel["plan"]["call_index"]
    messages = prefix(dict(original, question=train["question"]), k)
    norm = lambda text: re.sub(r"\W+", " ", str(text).casefold()).strip()
    seen = {norm(t["query"]) for t in original["trace"][:k] if "observation" in t}
    query = sentinel["plan"]["query"]
    obs = dict(ok=True, result=sentinel["result"], searched_queries=sorted(seen | {norm(query)}))
    messages = extend(messages, dict(query=query, observation=obs))
    ids = {arm: c._inputs(messages, "chelsea")["input_ids"][0].tolist() for arm, c in cpu.items()}
    if ids["A"] != sentinel["answer_context"]["prompt_token_ids"] or len(ids["A"]) != 245:
        raise RuntimeError("Chelsea replay mismatch")
    if ids["B"] != cpu["B"]._render_chat(messages)["input_ids"][0].tolist():
        raise RuntimeError("Chelsea B lossy")
    probes.append(dict(case_id=train["case_id"], step=k+1, messages=[m.model_dump(mode="json") for m in messages],
                       A_ids=ids["A"], B_ids=ids["B"], diagnostic_only=True, sentinel=True))
    admitted = [r["case_id"] for r in rows(assets.ADMISSION) if r["admission"] == "admitted"]
    protected = read(AK / "decision.json")["protected_cases"]
    if len(admitted) != 29 or len(protected) != 7:
        raise RuntimeError("admission/protection count mismatch")
    paths = [Path(__file__), ROOT / "scripts/run_v133_goal_q_search.py", ROOT / "scripts/run_v133_goal_p_variant.py",
             *sorted((ROOT / "src/mini_agent_rl/model").glob("*.py")), ROOT / "src/mini_agent_rl/prompts.py",
             ROOT / "src/mini_agent_rl/tools/search.py", assets.CASES, assets.CORPUS, assets.HIST, assets.ADMISSION, assets.EDGES,
             AA / "product-decision.json", AK / "trajectory-A.jsonl", AK / "decision.json",
             AI / "closure-v3/repair-query-results.jsonl", AI / "closure-v3/train-review-packets.jsonl", AI / "gpu-train-rollouts.jsonl",
             assets.ADAPTER / "adapter_model.safetensors", assets.ADAPTER / "adapter_config.json"]
    paths += sorted(p for p in assets.BASE.iterdir() if p.is_file() and p.suffix in (".json", ".jinja", ".safetensors"))
    generation_file = assets.BASE / "generation_config.json"
    generation_config = read(generation_file) if generation_file.exists() else dict(
        source="Transformers GenerationConfig.from_model_config at actual load (no standalone file)",
        eos_token_id=base_config.get("text_config", base_config).get("eos_token_id"),
        explicit_max_new_tokens={arm: pc["max_new_tokens"] for arm, pc in ARMS.items()})
    manifest = dict(goal="AL", arms=ARMS, limits=LIMITS, gpu=gpu, max_position_embeddings=max_positions,
        model=dict(base=str(assets.BASE), adapter=str(assets.ADAPTER), adapter_sha256=assets.EXPECTED_ADAPTER),
        admitted=admitted, protected=protected, hashes={str(p): sha(p) for p in paths},
        tokenizer_model_max_length=getattr(getattr(processor, "tokenizer", processor), "model_max_length", None),
        generation_config=generation_config, authorized_by="user PLEASE IMPLEMENT THIS PLAN; 修复完就继续之前的任务")
    write(OUT / "config.json", ARMS)
    write(OUT / "manifest.json", manifest)
    write(OUT / "mechanical-replay.json", checks)
    for probe in probes:
        append(OUT / "engineer-cases.jsonl", probe)
    write(OUT / "configuration-audit.json", dict(shared_source="config.json", explicit_generation_max_new_tokens=True,
        no_tokenizer_truncation=True, no_secondary_input_cap=True, default_product_unchanged=True,
        legacy_char_caps_unchanged=True, historical_AJ_AK_duplicate_literals_not_on_this_run_path=True,
        cli_and_training_defaults_not_on_this_run_path=True, maximum_saved_B_input=max(c["B_tokens"] for c in checks)))
    (OUT / "PLAN.md").write_text("# Goal AL：扩大输入与输出预算\n\n固定Goal Z完整轨迹seed42、direct-v090、thinking=false、first-json-v1、legacy、BM25-v2/top-3、greedy、单卡BF16、不量化。A=2048输入/128输出，B=16384输入/2048输出。每题4调用/3搜索。\n\n复用64题、2048篇冻结语料、29题准入层及7题保护集；194次输入CPU重建，5个前缀各两次工程回放，A/B各64题真实搜索。最多522次生成、384次搜索，无训练或新确认。只进行一次最多58组合语义审核，未确认不算事实错误，允许有依据的等价答案。\n\n继续条件：准入层支持成功净增至少2，事实错与证据不支持合计不增加，无效或重复动作不增加，保护题最多损失1。通过仅准备回归，不发布。失败停止本配置路线。GPU、数值、身份、哈希或基础设施失败保存现场、不自动重跑或改参。\n\n本轮是输入输出预算联合对照，不能仅由总分归因输入长度。配置、模型、数据及代码身份见manifest.json。\n", encoding="utf-8")
    write(OUT / "counts.json", dict(generations=0, searches=0))
    write(OUT / "status.json", dict(status="prepared"))
    print(json.dumps(dict(stage="prepare", replay=194, engineering_cases=5, gpu=gpu, max_positions=max_positions)), flush=True)


def guard(expected):
    if read(OUT / "status.json")["status"] != expected:
        raise RuntimeError("stage order/state mismatch")
    manifest = read(OUT / "manifest.json")
    if read(OUT / "config.json") != manifest["arms"]:
        raise RuntimeError("configuration changed")
    for path, digest in manifest["hashes"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"frozen hash changed: {path}")
    assets.check_gpu()
    return manifest


class CountedClient:
    def __init__(self, wrapped, stage):
        self.wrapped, self.base, self.stage = wrapped, wrapped.base, stage

    async def generate(self, *args, **kwargs):
        count = read(OUT / "counts.json")
        if count["generations"] >= LIMITS["generations"]:
            raise RuntimeError("generation cap reached")
        count["generations"] += 1
        write(OUT / "counts.json", count)
        started = time.monotonic()
        response = await self.wrapped.generate(*args, **kwargs)
        elapsed = time.monotonic() - started
        append(OUT / "generation-journal.jsonl", dict(stage=self.stage, ordinal=count["generations"],
            raw=response.raw_text, parse_error=response.parse_error, prompt_token_ids=response.prompt_token_ids,
            action_token_ids=response.action_token_ids, usage=response.usage, elapsed_seconds=elapsed))
        if len(response.prompt_token_ids) > self.base.config.max_input_tokens or len(response.action_token_ids) > self.base.config.max_new_tokens:
            raise RuntimeError("actual token budget violation")
        return response


def load(arm, stage):
    pc = read(OUT / "config.json")[arm]
    started = time.monotonic()
    client, wrapped, registry = q.make_client(assets.ADAPTER, deterministic=True, product_config=pc,
        model_path=assets.BASE, corpus=assets.CORPUS)
    index_setup_seconds = time.monotonic() - started
    client._load()
    identity = client.policy_fingerprint()
    if identity["adapter_weight_sha256"] != assets.EXPECTED_ADAPTER:
        raise RuntimeError("actual loaded identity mismatch")
    append(OUT / "runtime-loads.jsonl", dict(stage=stage, arm=arm, identity=identity,
        model_load_seconds=client.load_seconds, total_model_and_index_setup_seconds=time.monotonic()-started,
        index_and_client_setup_seconds=index_setup_seconds,
        config=client.config.model_dump()))
    return client, CountedClient(wrapped, stage), registry


def engineer():
    guard("prepared")
    for arm in ("A", "B"):
        client, wrapped, registry = load(arm, "engineer")
        for case in rows(OUT / "engineer-cases.jsonl"):
            messages = [Message.model_validate(m) for m in case["messages"]]
            response = asyncio.run(wrapped.generate(messages, registry.definitions(), temperature=0., seed=42+case["step"]))
            record = dict(case_id=case["case_id"], arm=arm, step=case["step"], raw=response.raw_text,
                answer=response.content, query=response.tool_call.arguments if response.tool_call else None,
                parse_error=response.parse_error, prompt_token_ids=response.prompt_token_ids,
                action_token_ids=response.action_token_ids, usage=response.usage, diagnostic_only=True)
            append(OUT / "engineer-results.jsonl", record)
            if response.prompt_token_ids != case[f"{arm}_ids"]:
                raise RuntimeError("engineering actual input mismatch")
            if arm == "B" and (response.usage["lossy_body_trimmed"] or response.usage["documents_deleted"]):
                raise RuntimeError("engineering B unexpectedly lost evidence")
            if response.usage.get("first_json_stop_offset") is not None and response.usage["output_truncated"]:
                raise RuntimeError("complete first JSON incorrectly marked truncated")
            print(json.dumps(dict(engineering=arm, case=case["case_id"], input=len(response.prompt_token_ids), raw=response.raw_text)), flush=True)
        del client, wrapped, registry
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    if len(rows(OUT / "engineer-results.jsonl")) != LIMITS["engineering_generations"]:
        raise RuntimeError("engineering count mismatch")
    write(OUT / "status.json", dict(status="engineered"))


def finalize_case(result, case, arm):
    result.update(case_id=case["case_id"], arm=arm, official_answer=case["answer"],
                  **q.score(result["answer"], case["answer"]))
    return result


def evaluate(arm):
    guard("engineered" if arm == "A" else "A_complete")
    target = OUT / f"trajectory-{arm}.jsonl"
    completed = rows(target) if target.exists() else []
    if completed and not (arm == "A" and len(completed) == 1 and (OUT / "answer-field-recovery.json").exists()):
        raise FileExistsError(target)
    client, wrapped, registry = load(arm, f"evaluate-{arm}")
    pc = read(OUT / "config.json")[arm]
    cases = rows(assets.CASES)
    if completed and completed[0]["case_id"] != cases[0]["case_id"]:
        raise RuntimeError("recovered first case identity mismatch")
    for case in cases[len(completed):]:
        gold = case["answer"]  # Validate the source schema before generating.
        result = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False, seed=42, product_config=pc))
        finalize_case(result, case, arm)
        append(target, result)
        count = read(OUT / "counts.json")
        count["searches"] += result["searches"]
        write(OUT / "counts.json", count)
        if result["model_calls"] > pc["max_model_calls_per_case"] or result["searches"] > pc["max_searches_per_case"] or count["searches"] > LIMITS["searches"]:
            raise RuntimeError("trajectory/search cap violation")
        print(json.dumps(dict(arm=arm, case=case["case_id"], em=result["em"], answer=result["answer"])), flush=True)
    write(OUT / f"runtime-{arm}.json", dict(peak_gpu_memory_gb=client.gpu_memory_peak_gb(), model_load_seconds=client.load_seconds))
    write(OUT / "status.json", dict(status=f"{arm}_complete"))


def report():
    guard("B_complete")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(assets.BASE), local_files_only=True)
    manifest = read(OUT / "manifest.json")
    data = {arm: rows(OUT / f"trajectory-{arm}.jsonl") for arm in ("A", "B")}
    summary = {}
    for arm, items in data.items():
        if len(items) != 64:
            raise RuntimeError("incomplete evaluation")
        calls = [t for r in items for t in r["trace"]]
        summary[arm] = dict(cases=len(items), EM=sum(r["em"] for r in items), F1=statistics.mean(r["f1"] for r in items),
            calls=len(calls), searches=sum(r["searches"] for r in items),
            invalid_or_repeated=sum(r["reason"] in ("invalid_action", "repeated_action") or any(t.get("action")=="search" and "observation" not in t for t in r["trace"]) for r in items),
            parse_invalid=sum(r["reason"]=="invalid_action" for r in items), repeated=sum(r["reason"]=="repeated_action" for r in items),
            overbudget_search_attempts=sum(t.get("action")=="search" and "observation" not in t and not t.get("repeated") for t in calls),
            body_trimmed=sum(t["usage"]["lossy_body_trimmed"] for t in calls), deleted=sum(t["usage"]["documents_deleted"] for t in calls),
            length_stops=sum(t["usage"].get("generation_stop_reason")=="length" for t in calls),
            max_input=max(len(t["prompt_token_ids"]) for t in calls), max_output=max(len(t["action_token_ids"]) for t in calls),
            mean_question_seconds=statistics.mean(r["elapsed_seconds"] for r in items if r.get("elapsed_seconds") is not None),
            mean_generation_seconds=statistics.mean(t["model_seconds"] for t in calls if t.get("model_seconds") is not None),
            question_latency_denominator=sum(r.get("elapsed_seconds") is not None for r in items),
            generation_latency_denominator=sum(t.get("model_seconds") is not None for t in calls))
    packets = []
    edges = rows(assets.EDGES)
    for i, a in enumerate(data["A"]):
        b = data["B"][i]
        if a["case_id"] != b["case_id"]:
            raise RuntimeError("case order changed")
        changed = next((j for j in range(min(len(a["trace"]),len(b["trace"]))) if a["trace"][j]["raw"] != b["trace"][j]["raw"] or a["trace"][j]["prompt_token_ids"] != b["trace"][j]["prompt_token_ids"]), None)
        ledger = dict(case_id=a["case_id"], question=a["question"], A_answer=a["answer"], B_answer=b["answer"], A_em=a["em"], B_em=b["em"], first_divergence=changed,
            admitted=a["case_id"] in manifest["admitted"], protected=a["case_id"] in manifest["protected"], A_reason=a["reason"], B_reason=b["reason"])
        append(OUT / "paired-ledger.jsonl", ledger)
        if ledger["admitted"]:
            packet = dict(ledger, edges=[e for e in edges if e["case_id"]==a["case_id"]])
            for arm, row in (("A", a), ("B", b)):
                final = row["trace"][-1]
                packet[arm] = dict(answer=row["answer"], final_action=final.get("action"), reason=row["reason"],
                    actual_input=processor.decode(final["prompt_token_ids"], skip_special_tokens=False),
                    queries=[t.get("query") for t in row["trace"] if t.get("query")], token_ids=final["prompt_token_ids"],
                    context_usage=final["context_usage"], trace=row["trace"])
            packets.append(packet)
    for packet in packets:
        append(OUT / "review-packets.jsonl", packet)
    write(OUT / "metrics.json", dict(metrics=summary, counts=read(OUT / "counts.json"), review_combinations=len(packets)*2))
    write(OUT / "status.json", dict(status="awaiting_one_pass_semantic_review"))
    print(json.dumps(read(OUT / "metrics.json")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "recover-prepare", "engineer", "evaluate-a", "evaluate-b", "report"])
    args = parser.parse_args()
    try:
        {"prepare": prepare, "recover-prepare": recover_prepare, "engineer": engineer, "evaluate-a": lambda: evaluate("A"),
         "evaluate-b": lambda: evaluate("B"), "report": report}[args.stage]()
    except Exception as exc:
        if OUT.exists() and not (OUT / "failure.json").exists():
            write(OUT / "failure.json", dict(stage=args.stage, type=type(exc).__name__, error=str(exc),
                  counts=read(OUT / "counts.json") if (OUT / "counts.json").exists() else dict(generations=0,searches=0)))
            write(OUT / "status.json", dict(status="failed", stage=args.stage))
        raise


if __name__ == "__main__":
    main()
