"""Goal AQ: diagnostic-only, single-generation grounded-action A/B."""
from __future__ import annotations
import argparse, asyncio, gc, hashlib, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v133_goal_ap as ap
from mini_agent_rl.domain import Message
from mini_agent_rl.prompts import system_prompt
from mini_agent_rl.model.grounded_action_audit import audit_grounded_action, map_documents

OUT = ROOT / "runs/v133-goal-aq-grounded-action-v1"
PRIOR = ROOT / "runs/v133-goal-ap-evidence-presentation-v2"
CAPS = {"generations": 576, "searches": 384}
sha, read, rows, write = ap.sha, ap.read, ap.rows, ap.write_once


def config(arm):
    if arm not in ("A", "B"):
        raise ValueError(arm)
    return dict(ap.PC, observation_presentation="legacy-json",
        system_prompt="direct-v090" if arm == "A" else "direct-v090-grounded-action-v1")


def make_client(model, arm):
    return ap.q.make_client(ap.AMADAPTER if model == "AM" else ap.ZADAPTER, deterministic=True,
        product_config=config(arm), model_path=ap.BASE, corpus=ap.Z / "corpus-dev.json")


def count(kind):
    return sum(r["kind"] == kind for r in rows(OUT / "gpu-usage.jsonl"))


def record(kind, stage, case):
    if count(kind) >= CAPS[kind]:
        raise RuntimeError("AQ frozen quota exceeded")
    ap.append(OUT / "gpu-usage.jsonl", dict(kind=kind, stage=stage, case_id=case,
        attempt=count(kind)+1, time=time.time(), status="started"))


def no_clipping(usage):
    if any(usage.get(k) for k in ("context_budget_error", "lossy_body_trimmed", "documents_deleted", "history_dropped", "prompt_truncated")):
        raise RuntimeError("AQ actual input required clipping/deletion")


def prepare():
    if not OUT.exists() or {p.name for p in OUT.iterdir()} != {"PLAN.md"}:
        raise FileExistsError("AQ requires an independent directory containing only PLAN.md")
    if sha(ap.ZADAPTER / "adapter_model.safetensors") != ap.Z_HASH or sha(ap.AMADAPTER / "adapter_model.safetensors") != ap.AM_HASH:
        raise RuntimeError("released/candidate adapter identity mismatch")
    if not read(ap.AA / "product-decision.json").get("published"):
        raise RuntimeError("Goal Z AA publication identity missing")
    prior = read(PRIOR / "manifest.json")
    if prior["arms"]["A"] != config("A"):
        raise RuntimeError("AP A generation/product settings cannot be reused")
    for key in ("model_client", "q_runner", "tokenizer", "base_config"):
        if sha(prior["paths"][key]) != prior["hashes"][key]:
            raise RuntimeError(f"AP A generation dependency changed: {key}")
    for name, digest in prior["expected_hashes"].items():
        if digest != {"GoalZ": ap.Z_HASH, "AM42A": ap.AM_HASH}[name]:
            raise RuntimeError("AP baseline model mismatch")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(ap.BASE), local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    probes = rows(PRIOR / "frozen-probes.jsonl")
    cases = rows(PRIOR / "evaluation-cases.jsonl")
    if len(probes) != 32 or len(cases) != 64 or sum(p["category"] == "complete" for p in probes) != 24:
        raise RuntimeError("AQ fixed inventory mismatch")
    inputs, provenance = [], []
    archived_inputs = {(r["model"], r["probe_id"]): r for r in rows(PRIOR / "probe-inputs.jsonl") if r["arm"] == "A"}
    for model in ("AM", "Z"):
        old = rows(PRIOR / f"probe-{model}-A.jsonl")
        if len(old) != 32 or len({r["probe_id"] for r in old}) != 32:
            raise RuntimeError("AP A reuse coverage mismatch")
        baseline = {r["probe_id"]: r for r in old}
        for arm in ("A", "B"):
            client, _, _ = make_client(model, arm)
            client._processor = processor
            client._model = SimpleNamespace(device="cpu")  # CPU _inputs; never loads weights.
            for p in probes:
                messages = ap.restore_tool_messages(ap.parse_archived_messages(tokenizer, p["prompt_token_ids"]))
                if not messages or messages[0]["role"] != "system" or messages[0]["content"] != system_prompt(False):
                    raise RuntimeError("unexpected archived system prompt")
                messages[0] = dict(messages[0], content=system_prompt(False, config(arm)["system_prompt"]))
                encoded = client._inputs([Message(**m) for m in messages], "aq_cpu_preflight")
                ids = encoded["input_ids"][0].tolist()
                usage = encoded._context_usage
                no_clipping(usage)
                if arm == "A" and (ids != p["prompt_token_ids"] or ids != baseline[p["probe_id"]]["prompt_token_ids"] or ids != archived_inputs[(model,p["probe_id"])]["prompt_token_ids"]):
                    raise RuntimeError("A production input does not exactly match archived baseline")
                if arm == "B" and ids == p["prompt_token_ids"]:
                    raise RuntimeError("AQ B system prompt was not applied")
                docs, _ = map_documents(tokenizer, ids, messages)
                inputs.append(dict(model=model, arm=arm, probe_id=p["probe_id"], case_id=p["case_id"],
                    category=p["category"], messages=messages, prompt_token_ids=ids, documents=docs,
                    context_usage=usage, training_use_forbidden=True, no_search_execution=True))
            del client
        write(OUT / f"probe-{model}-A.jsonl", old, jsonl=True)
        provenance.append(dict(model=model, source=str(PRIOR / f"probe-{model}-A.jsonl"),
            source_sha256=sha(PRIOR / f"probe-{model}-A.jsonl"), exact_input_verified=32,
            original_greedy_not_bitwise_guaranteed=True))
    by = {(r["model"], r["arm"], r["probe_id"]): r for r in inputs}
    for model in ("AM", "Z"):
        for p in probes:
            a, b = (by[(model, arm, p["probe_id"])] for arm in ("A", "B"))
            if a["messages"][1:] != b["messages"][1:] or [(d["document_id"],d["body_sha256"]) for d in a["documents"]] != [(d["document_id"],d["body_sha256"]) for d in b["documents"]]:
                raise RuntimeError("AQ changed evidence or historical messages")
    write(OUT / "frozen-probes.jsonl", probes, jsonl=True)
    write(OUT / "evaluation-cases.jsonl", cases, jsonl=True)
    write(OUT / "probe-inputs.jsonl", inputs, jsonl=True)
    write(OUT / "reused-a-provenance.json", provenance)
    write(OUT / "gpu-usage.jsonl", [], jsonl=True)
    paths = ap.source_paths()
    paths.update(runner=Path(__file__), prompts=ROOT / "src/mini_agent_rl/prompts.py",
        auditor=ROOT / "src/mini_agent_rl/model/grounded_action_audit.py",
        ap_baseline_review=PRIOR / "probe-adjudications-revision.jsonl", ap_baseline_manifest=PRIOR / "manifest.json",
        ap_model_client=ROOT / "scripts/run_v133_goal_ap.py")
    for model in ("AM", "Z"):
        paths[f"baseline_{model}"] = PRIOR / f"probe-{model}-A.jsonl"
        paths[f"adapter_config_{model}"] = (ap.AMADAPTER if model == "AM" else ap.ZADAPTER) / "adapter_config.json"
    paths["action_alias"] = ROOT / "scripts/run_v133_goal_p_variant.py"
    paths["budget"] = ROOT / "src/mini_agent_rl/model/context.py"
    for name in ("tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
        if (ap.BASE / name).exists(): paths[name] = ap.BASE / name
    for name in ("PLAN.md", "frozen-probes.jsonl", "evaluation-cases.jsonl", "probe-inputs.jsonl", "reused-a-provenance.json"):
        paths[name] = OUT / name
    manifest = dict(goal="AQ", arms={arm:config(arm) for arm in ("A","B")}, limits=CAPS,
        paths={k:str(v) for k,v in paths.items()}, hashes={k:sha(v) for k,v in paths.items()},
        expected_hashes={"GoalZ":ap.Z_HASH,"AM42A":ap.AM_HASH}, training_updates=0, new_confirmation_calls=0,
        probe_decisions=128, reused_A=64, new_B=64, primary_model="AM42A", GoalZ="descriptive_only",
        runtime_audit="diagnostic_only_no_dispatch_changes", gpu_authorization="existing user authorization",
        failure_policy="preserve and stop; no rerun or tuning")
    write(OUT / "manifest.json", manifest)
    write(OUT / "cpu-preflight.json", dict(passed=True, production_inputs=128, A_exact_reuse=64,
        B_only_system_changed=True, no_clipping=True, document_occurrences=sum(len(r["documents"]) for r in inputs),
        max_input_tokens=max(len(r["prompt_token_ids"]) for r in inputs), manifest_sha256=sha(OUT / "manifest.json")))
    print(json.dumps(read(OUT / "cpu-preflight.json")))


def frozen():
    if (OUT / "experiment-failure.json").exists():
        raise RuntimeError("AQ failure latch; stop")
    m = read(OUT / "manifest.json")
    for key, path in m["paths"].items():
        if sha(path) != m["hashes"][key]:
            raise RuntimeError(f"AQ frozen file changed: {key}")
    if not read(OUT / "cpu-preflight.json")["passed"]:
        raise RuntimeError("AQ CPU preflight failed")
    return m


def probe(model):
    frozen()
    path = OUT / f"probe-{model}-B.jsonl"
    if path.exists():
        raise FileExistsError(path)
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("AQ requires one visible GPU")
    client, wrapped, _ = make_client(model, "B")
    for item in [r for r in rows(OUT / "probe-inputs.jsonl") if r["model"] == model and r["arm"] == "B"]:
        record("generations", f"probe-{model}-B", item["case_id"])
        start = time.monotonic()
        response = asyncio.run(wrapped.generate([Message(**m) for m in item["messages"]], [], temperature=0., seed=42))
        if response.prompt_token_ids != item["prompt_token_ids"]:
            raise RuntimeError("AQ actual GPU input differs from production preflight")
        no_clipping(response.usage)
        if len(response.prompt_token_ids)>16384 or len(response.action_token_ids)>2048 or any(not math.isfinite(x) for x in response.old_logprobs):
            raise RuntimeError("AQ token/numeric failure")
        tokenizer=getattr(client._processor,"tokenizer",client._processor)
        docs, offsets = map_documents(tokenizer, response.prompt_token_ids, item["messages"])
        audit = audit_grounded_action(response.raw_text, docs, offsets)
        ap.append(path, dict(model=model, arm="B", probe_id=item["probe_id"], case_id=item["case_id"],
            category=item["category"], action=ap.an.parsed_action(response), answer=response.content,
            query=response.tool_call.arguments.get("query") if response.tool_call else None,
            parse_error=response.parse_error, raw=response.raw_text, prompt_token_ids=response.prompt_token_ids,
            action_token_ids=response.action_token_ids, usage=response.usage, grounded_audit=audit,
            elapsed_seconds=time.monotonic()-start, no_search_execution=True, training_use_forbidden=True))
        print(json.dumps(dict(model=model, probe=item["probe_id"], action=ap.an.parsed_action(response),
            schema=audit["schema_valid"], quotes=audit["exact_quotes_valid"])), flush=True)
    del client, wrapped
    gc.collect(); torch.cuda.empty_cache()


def gate():
    frozen()
    rulings = rows(OUT / "probe-adjudications.jsonl")
    expected = {(model,arm,p["probe_id"]) for model in ("AM","Z") for arm in ("A","B") for p in rows(OUT/"frozen-probes.jsonl")}
    by = {(r["model"],r["arm"],r["probe_id"]):r for r in rulings}
    if len(rulings) != 128 or set(by) != expected:
        raise RuntimeError("AQ adjudication coverage mismatch")
    def metric(arm,category,key):
        return sum(r.get(key) is True for r in rulings if r["model"]=="AM" and r["arm"]==arm and r["category"]==category)
    required = ("supported_correct","grounded_links_correct","binding_repair")
    partial = ("legal_bridge_search","premature_answer","invalid_action","repeated_query")
    for r in rulings:
        if r["model"]=="AM" and (not r.get("adjudication_reason") or any(r.get(k) is None for k in (required if r["category"]=="complete" else partial))):
            raise RuntimeError("AQ primary adjudication missing; cannot infer pass")
    a,b = (metric(arm,"complete","supported_correct") for arm in ("A","B"))
    gains=[r for r in rulings if r["model"]=="AM" and r["arm"]=="B" and r["category"]=="complete" and r["supported_correct"] is True and by[("AM","A",r["probe_id"])]["supported_correct"] is not True]
    losses=[r["probe_id"] for r in rulings if r["model"]=="AM" and r["arm"]=="A" and r["category"]=="complete" and r["supported_correct"] is True and by[("AM","B",r["probe_id"])]["supported_correct"] is not True]
    grounded_gains=[r for r in gains if r["grounded_links_correct"] is True]
    pa,pb=(metric(arm,"partial","legal_bridge_search") for arm in ("A","B"))
    passed=b>=22 and b-a>=2 and len(grounded_gains)>=2 and any(r["binding_repair"] for r in grounded_gains) and len(losses)<=1 and pb>=7 and pb>=pa and all(metric("B","partial",k)<=metric("A","partial",k) for k in partial[1:])
    result=dict(passed=passed, A_complete=a, B_complete=b, grounded_gains=[r["probe_id"] for r in grounded_gains],
        losses=losses, A_partial=pa, B_partial=pb, review_sha256=sha(OUT/"probe-adjudications.jsonl"),
        GoalZ="descriptive_only", decision="proceed_to_real_search" if passed else "stop_no_training_no_search")
    write(OUT / "probe-gate.json", result)
    print(json.dumps(result))


def evaluate(arm):
    frozen()
    if not read(OUT/"probe-gate.json")["passed"]:
        raise RuntimeError("AQ probe gate failed")
    if arm=="B" and len(rows(OUT/"eval-AM-A.jsonl"))!=64:
        raise RuntimeError("A must complete before B")
    path=OUT/f"eval-AM-{arm}.jsonl"
    if path.exists():raise FileExistsError(path)
    client,wrapped,registry=make_client("AM",arm)
    original=wrapped.generate;execute=registry.execute
    active={"id":None}
    async def generate(messages,*args,**kwargs):
        record("generations",f"eval-{arm}",active["id"])
        response=await original(messages,*args,**kwargs)
        no_clipping(response.usage)
        if arm=="B":
            tokenizer=getattr(client._processor,"tokenizer",client._processor)
            docs,offsets=map_documents(tokenizer,response.prompt_token_ids,[m.model_dump() for m in messages])
            usage=dict(response.usage,grounded_audit=audit_grounded_action(response.raw_text,docs,offsets))
            response=response.model_copy(update={"usage":usage})
        return response
    async def search(*args,**kwargs):
        if args and args[0]=="search":record("searches",f"eval-{arm}",active["id"])
        return await execute(*args,**kwargs)
    wrapped.generate=generate;registry.execute=search
    for case in rows(OUT/"evaluation-cases.jsonl"):
        active["id"]=case["case_id"]
        result=asyncio.run(ap.q.rollout(case["question"],wrapped,registry,sample=False,seed=42,product_config=config(arm)))
        if result["reason"]=="tool_error":raise RuntimeError("AQ search infrastructure failure")
        result.update(case_id=case["case_id"],official_answer=case["answer"],**ap.q.score(result["answer"],case["answer"]))
        ap.append(path,result)
        print(json.dumps(dict(arm=arm,case=case["case_id"],em=result["em"])),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["prepare","probe","gate","evaluate"])
    parser.add_argument("--model",choices=["AM","Z"],default="AM");parser.add_argument("--arm",choices=["A","B"],default="A")
    args=parser.parse_args()
    try:
        if args.stage=="prepare":prepare()
        elif args.stage=="probe":probe(args.model)
        elif args.stage=="gate":gate()
        else:evaluate(args.arm)
    except BaseException as exc:
        if OUT.exists() and not (OUT/"experiment-failure.json").exists():
            write(OUT/"experiment-failure.json",dict(stage=args.stage,error_type=type(exc).__name__,error=str(exc),
                generations=count("generations") if (OUT/"gpu-usage.jsonl").exists() else 0, searches=0,training_updates=0))
        raise


if __name__=="__main__":main()
