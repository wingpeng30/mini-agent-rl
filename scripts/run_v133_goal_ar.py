"""Bounded Goal AR constrained-selection study; no automatic semantic scoring."""
from __future__ import annotations
import argparse, asyncio, hashlib, importlib.metadata, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v133_goal_ap as ap
from mini_agent_rl.domain import Message
from mini_agent_rl.prompts import system_prompt
from mini_agent_rl.model.constrained_evidence import audit_selection, action_grammar
from mini_agent_rl.model.grounded_action_audit import map_documents

OUT = ROOT / "runs/v133-goal-ar-constrained-evidence-v1"
PRIOR = ROOT / "runs/v133-goal-ap-evidence-presentation-v2"
CAPS = dict(generations=840, searches=576)
sha, rows, read, write = ap.sha, ap.rows, ap.read, ap.write_once


def config(arm):
    if arm not in ("A", "C", "B"): raise ValueError(arm)
    return dict(ap.PC, action_constraint={"A":"none","C":"json-action-v1","B":"evidence-action-v1"}[arm],
        observation_presentation="legacy-json" if arm=="A" else "evidence-addresses-v1",
        system_prompt="direct-v090-evidence-address-v1" if arm=="B" else "direct-v090")


def client_for(arm):
    return ap.q.make_client(ap.AMADAPTER, deterministic=True, product_config=config(arm),
        model_path=ap.BASE, corpus=ap.Z / "corpus-dev.json")


def count(kind):
    path=OUT/"gpu-usage.jsonl"
    return sum(r["kind"]==kind for r in rows(path)) if path.exists() else 0


def record(kind,stage,case):
    if count(kind)>=CAPS[kind]:raise RuntimeError("AR quota exceeded")
    ap.append(OUT/"gpu-usage.jsonl",dict(kind=kind,stage=stage,case_id=case,
        attempt=count(kind)+1,time=time.time(),status="started"))


def no_clipping(usage):
    if any(usage.get(k) for k in ("context_budget_error","lossy_body_trimmed","documents_deleted","history_dropped","prompt_truncated")):
        raise RuntimeError("AR input required clipping; stop presentation experiment")


def prepare():
    if (OUT/"manifest.json").exists():raise FileExistsError("AR already frozen")
    if not (OUT/"PLAN.md").exists():raise FileNotFoundError("AR PLAN.md")
    if sha(ap.AMADAPTER/"adapter_model.safetensors")!=ap.AM_HASH or sha(ap.ZADAPTER/"adapter_model.safetensors")!=ap.Z_HASH:
        raise RuntimeError("AM/GoalZ identity mismatch")
    if not read(ap.AA/"product-decision.json")["published"]:raise RuntimeError("AA publication missing")
    for name,version in (("xgrammar","0.2.8"),("apache-tvm-ffi","0.1.14.post1")):
        if importlib.metadata.version(name)!=version:raise RuntimeError("dependency lock mismatch")
    from transformers import AutoProcessor
    import xgrammar as xgr
    processor=AutoProcessor.from_pretrained(str(ap.BASE),local_files_only=True)
    tokenizer=getattr(processor,"tokenizer",processor)
    probes=rows(PRIOR/"frozen-probes.jsonl")
    old=rows(PRIOR/"probe-AM-A.jsonl")
    by={r["probe_id"]:r for r in old}
    if len(probes)!=32 or len(old)!=32 or sum(p["category"]=="complete" for p in probes)!=24:
        raise RuntimeError("AR fixed probe inventory mismatch")
    actual_config=read(ap.BASE/"config.json")
    vocab=actual_config.get("vocab_size",actual_config.get("text_config",{}).get("vocab_size"))
    from mini_agent_rl.model.local import generation_terminator_ids
    compiler=xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer,vocab_size=vocab,
        stop_token_ids=generation_terminator_ids(tokenizer)))
    input_rows=[]
    for arm in ("A","C","B"):
        client,_,_=client_for(arm);client._processor=processor;client._model=SimpleNamespace(device="cpu")
        for p in probes:
            messages=ap.restore_tool_messages(ap.parse_archived_messages(tokenizer,p["prompt_token_ids"]))
            messages[0]=dict(messages[0],content=system_prompt(False,config(arm)["system_prompt"]))
            encoded=client._inputs([Message(**m) for m in messages],"ar_cpu_prepare")
            ids=encoded["input_ids"][0].tolist();usage=encoded._context_usage;no_clipping(usage)
            if arm=="A" and (ids!=p["prompt_token_ids"] or ids!=by[p["probe_id"]]["prompt_token_ids"]):
                raise RuntimeError("A archived token mismatch")
            if arm!="A" and ids==p["prompt_token_ids"]:raise RuntimeError("indexing did not change input")
            spans=usage["presentation"].get("document_bodies",[])
            if arm=="A":docs,_=map_documents(tokenizer,ids,messages)
            else:
                docs=[]
                from mini_agent_rl.model.grounded_action_audit import visible_documents
                for d in visible_documents(messages):
                    matching=[s for s in spans if s["message_index"]==d["message_index"] and s["rank"]==d["rank"]]
                    if ''.join(s["quote"] for s in matching)!=d["text"]:raise RuntimeError("body not fully indexed")
                    docs.append(d)
                compiler.compile_grammar(action_grammar(config(arm)["action_constraint"],[s["address"] for s in spans]))
            input_rows.append(dict(arm=arm,probe_id=p["probe_id"],case_id=p["case_id"],category=p["category"],
                messages=messages,prompt_token_ids=ids,documents=docs,context_usage=usage,
                training_use_forbidden=True,no_search_execution=True))
        del client
    by_inputs={(r["arm"],r["probe_id"]):r for r in input_rows}
    for p in probes:
        c,b=(by_inputs[(arm,p["probe_id"])] for arm in ("C","B"))
        if c["messages"][1:]!=b["messages"][1:] or c["documents"]!=b["documents"]:
            raise RuntimeError("C/B evidence/history differ")
    cases=rows(PRIOR/"evaluation-cases.jsonl")
    if len(cases)!=64:raise RuntimeError("case inventory mismatch")
    write(OUT/"frozen-probes.jsonl",probes,jsonl=True)
    write(OUT/"evaluation-cases.jsonl",cases,jsonl=True)
    write(OUT/"probe-inputs.jsonl",input_rows,jsonl=True)
    write(OUT/"probe-A.jsonl",old,jsonl=True)
    write(OUT/"gpu-usage.jsonl",[],jsonl=True)
    paths=ap.source_paths()
    paths.update(runner=Path(__file__),constraint=ROOT/"src/mini_agent_rl/model/constrained_evidence.py",
        prompts=ROOT/"src/mini_agent_rl/prompts.py",baseline=PRIOR/"probe-AM-A.jsonl",
        baseline_review=PRIOR/"probe-adjudications-revision.jsonl",baseline_manifest=PRIOR/"manifest.json",
        budget=ROOT/"src/mini_agent_rl/model/context.py",model_adapter_config=ap.AMADAPTER/"adapter_config.json")
    for p in [OUT/"PLAN.md",OUT/"probe-inputs.jsonl",OUT/"frozen-probes.jsonl",OUT/"evaluation-cases.jsonl",
              ap.BASE/"chat_template.jinja",ap.BASE/"tokenizer_config.json",ap.BASE/"generation_config.json"]:
        if p.exists():paths[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]=p
    for p in (OUT/"dependency-wheels").glob("*.whl"):paths[p.name]=p
    manifest=dict(goal="AR",research_parent="AM42A",arms={arm:config(arm) for arm in ("A","C","B")},
        caps=CAPS,paths={k:str(v) for k,v in paths.items()},hashes={k:sha(v) for k,v in paths.items()},
        dependencies={n:importlib.metadata.version(n) for n in ("torch","transformers","xgrammar","apache-tvm-ffi")},
        baseline_reuse=dict(count=32,exact_actual_input_verified=True,default_generation_no_constraint=True,
            default_path_verified_by_tests=True,source_hash=sha(PRIOR/"probe-AM-A.jsonl")),
        training_updates=0,new_confirmation_calls=0,default_product_changed=False)
    write(OUT/"manifest.json",manifest)
    write(OUT/"cpu-preflight.json",dict(passed=True,A_exact_reuse=32,production_inputs=96,
        C_B_same_evidence=True,grammars_compiled=64,no_clipping=True,
        max_input_tokens=max(len(r["prompt_token_ids"]) for r in input_rows)))
    print(json.dumps(read(OUT/"cpu-preflight.json")),flush=True)


def frozen():
    if (OUT/"experiment-failure.json").exists():raise RuntimeError("AR failure latch")
    m=read(OUT/"manifest.json")
    for key,path in m["paths"].items():
        if sha(path)!=m["hashes"][key]:raise RuntimeError(f"AR frozen asset drift: {key}")
    return m


def generate_one(client,wrapped,item,stage):
    record("generations",stage,item["case_id"])
    start=time.monotonic()
    response=asyncio.run(wrapped.generate([Message(**m) for m in item["messages"]],[],temperature=0.,seed=42))
    if response.prompt_token_ids!=item["prompt_token_ids"]:raise RuntimeError("AR GPU input drift")
    no_clipping(response.usage)
    if len(response.prompt_token_ids)>16384 or len(response.action_token_ids)>2048:
        raise RuntimeError("AR length cap failure")
    if any(not math.isfinite(x) for x in response.old_logprobs):raise FloatingPointError("nonfinite score")
    selection=audit_selection(response.raw_text,response.usage["presentation"]["document_bodies"]) if item["arm"]=="B" and not response.parse_error else None
    return dict(arm=item["arm"],probe_id=item["probe_id"],case_id=item["case_id"],category=item["category"],
        action=ap.an.parsed_action(response),answer=response.content,query=response.tool_call.arguments.get("query") if response.tool_call else None,
        parse_error=response.parse_error,raw=response.raw_text,prompt_token_ids=response.prompt_token_ids,
        action_token_ids=response.action_token_ids,usage=response.usage,selection_audit=selection,
        elapsed_seconds=time.monotonic()-start,no_search_execution=True,training_use_forbidden=True)


def release(client,wrapped):
    import gc,torch
    del client,wrapped;gc.collect();torch.cuda.empty_cache()


def engineer():
    frozen()
    if (OUT/"engineer.jsonl").exists():raise FileExistsError("engineering already attempted")
    import torch
    if torch.cuda.device_count()!=1:raise RuntimeError("one GPU required")
    probes=rows(OUT/"frozen-probes.jsonl")
    selected=[p["probe_id"] for category in ("complete","partial") for p in [x for x in probes if x["category"]==category][:2]]
    for arm in ("C","B"):
        client,wrapped,_=client_for(arm)
        for item in [r for r in rows(OUT/"probe-inputs.jsonl") if r["arm"]==arm and r["probe_id"] in selected]:
            r=generate_one(client,wrapped,item,"engineer");ap.append(OUT/"engineer.jsonl",r)
            if r["parse_error"] or r["usage"]["generation_stop_reason"]!="first_json" or (arm=="B" and not r["selection_audit"]["addresses_valid"]):
                raise RuntimeError("engineering protocol/stop failure")
        release(client,wrapped);del client,wrapped
    write(OUT/"engineer-complete.json",dict(passed=True,generations=8,searches=0))


def probe(arm):
    frozen()
    if arm not in ("C","B"):raise ValueError(arm)
    if not read(OUT/"engineer-complete.json")["passed"]:raise RuntimeError("engineering gate")
    path=OUT/f"probe-{arm}.jsonl"
    if path.exists():raise FileExistsError(path)
    client,wrapped,_=client_for(arm)
    for item in [r for r in rows(OUT/"probe-inputs.jsonl") if r["arm"]==arm]:
        r=generate_one(client,wrapped,item,f"probe-{arm}");ap.append(path,r)
        print(json.dumps(dict(arm=arm,probe=r["probe_id"],action=r["action"],answer=r["answer"],query=r["query"])),flush=True)
    release(client,wrapped)


def gate():
    frozen();reviews=rows(OUT/"probe-adjudications.jsonl")
    expected={(arm,p["probe_id"]) for arm in ("A","C","B") for p in rows(OUT/"frozen-probes.jsonl")}
    by={(r["arm"],r["probe_id"]):r for r in reviews}
    if len(by)!=96 or len(reviews)!=96 or set(by)!=expected:raise RuntimeError("AR review coverage")
    for r in reviews:
        required=("supported_correct","binding_repair") if r["category"]=="complete" else ("legal_bridge_search","premature_answer","invalid_action","repeated_query")
        if not r.get("adjudication_reason") or any(not isinstance(r.get(k),bool) for k in required):raise RuntimeError("unconfirmed review; cannot pass")
        if r["arm"]=="B" and not isinstance(r.get("selected_evidence_correct"),bool):raise RuntimeError("B selected evidence unreviewed")
    def n(arm,category,key):return sum(r.get(key) is True for r in reviews if r["arm"]==arm and r["category"]==category)
    summary={arm:dict(complete=n(arm,"complete","supported_correct"),partial=n(arm,"partial","legal_bridge_search")) for arm in ("A","C","B")}
    eligibility={}
    for arm in ("C","B"):
        gains=[r for r in reviews if r["arm"]==arm and r["category"]=="complete" and r["supported_correct"] and not by[("A",r["probe_id"])]["supported_correct"]]
        losses=[r["probe_id"] for r in reviews if r["arm"]=="A" and r["category"]=="complete" and r["supported_correct"] and not by[(arm,r["probe_id"])]["supported_correct"]]
        ok=summary[arm]["complete"]>=22 and summary[arm]["complete"]-summary["A"]["complete"]>=2 and any(r["binding_repair"] for r in gains) and len(losses)<=1 and summary[arm]["partial"]>=7 and summary[arm]["partial"]>=summary["A"]["partial"] and all(n(arm,"partial",k)<=n("A","partial",k) for k in ("premature_answer","invalid_action","repeated_query"))
        if arm=="B":ok=ok and summary["B"]["complete"]-summary["C"]["complete"]>=2 and sum(r.get("selected_evidence_correct") is True for r in gains)>=2
        eligibility[arm]=dict(passed=ok,gains=[r["probe_id"] for r in gains],losses=losses)
    write(OUT/"probe-gate.json",dict(metrics=summary,eligibility=eligibility,
        real_search_allowed=any(v["passed"] for v in eligibility.values()),review_sha256=sha(OUT/"probe-adjudications.jsonl")))
    print(json.dumps(read(OUT/"probe-gate.json")),flush=True)


def evaluate(arm):
    frozen();g=read(OUT/"probe-gate.json")
    if not g["real_search_allowed"]:raise RuntimeError("AR probe gate failed")
    allowed=["A","C"]+(["B"] if g["eligibility"]["B"]["passed"] else [])
    if arm not in allowed:raise ValueError("not an eligible comparison arm")
    for preceding in allowed[:allowed.index(arm)]:
        if len(rows(OUT/f"eval-{preceding}.jsonl"))!=64:raise RuntimeError("fixed arm order violated")
    path=OUT/f"eval-{arm}.jsonl"
    if path.exists():raise FileExistsError(path)
    client,wrapped,registry=client_for(arm);original=wrapped.generate;execute=registry.execute;active={"id":None}
    async def generate(*args,**kwargs):
        record("generations",f"eval-{arm}",active["id"]);response=await original(*args,**kwargs);no_clipping(response.usage)
        if arm=="B" and not response.parse_error:
            response=response.model_copy(update={"usage":dict(response.usage,selection_audit=audit_selection(response.raw_text,response.usage["presentation"]["document_bodies"]))})
        return response
    async def search(*args,**kwargs):
        if args and args[0]=="search":record("searches",f"eval-{arm}",active["id"])
        return await execute(*args,**kwargs)
    wrapped.generate=generate;registry.execute=search
    for case in rows(OUT/"evaluation-cases.jsonl"):
        active["id"]=case["case_id"]
        r=asyncio.run(ap.q.rollout(case["question"],wrapped,registry,sample=False,seed=42,product_config=config(arm)))
        if r["reason"]=="tool_error":raise RuntimeError("search infrastructure failure")
        r.update(case_id=case["case_id"],official_answer=case["answer"],**ap.q.score(r["answer"],case["answer"]))
        ap.append(path,r)
    release(client,wrapped)


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","engineer","probe","gate","evaluate"])
    p.add_argument("--arm",choices=["A","C","B"],default="C");args=p.parse_args()
    try:
        if args.stage in ("prepare","engineer","gate"):globals()[args.stage]()
        elif args.stage=="probe":probe(args.arm)
        else:evaluate(args.arm)
    except BaseException as exc:
        if OUT.exists() and not (OUT/"experiment-failure.json").exists():
            write(OUT/"experiment-failure.json",dict(stage=args.stage,error_type=type(exc).__name__,error=str(exc),
                generations=count("generations"),searches=count("searches"),time=time.time(),stop_no_auto_rerun=True))
        raise

if __name__=="__main__":main()
