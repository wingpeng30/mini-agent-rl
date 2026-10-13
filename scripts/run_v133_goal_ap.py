"""Goal AP: lossless evidence-presentation A/B on fixed AM and Goal Z weights."""
from __future__ import annotations
import argparse, asyncio, gc, hashlib, json, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_v133_goal_am as am
import run_v133_goal_an as an
import run_v133_goal_ao_v2 as ao
import run_v133_goal_q_search as q
from mini_agent_rl.model.observation_presentation import DOCUMENT_BLOCKS_V1, parse_document_blocks, present_observation

OUT=ROOT/os.environ.get("GOAL_AP_RUN_DIR","runs/v133-goal-ap-evidence-presentation-v1")
AM=ROOT/"runs/v133-goal-am-action-sft-v2"
AN=ROOT/"runs/v133-goal-an-hard-state-sft-v2"
Z=ROOT/"runs/v133-goal-z-multihop-sft"
AA=ROOT/"runs/v133-goal-aa-audit-confirmation"
BASE=am.BASE; ZADAPTER=am.SOURCE; AMADAPTER=AM/"seed42-A/adapter-final"
Z_HASH=am.EXPECTED_ADAPTER
AM_HASH="3bcd1f85382c1d7775fa674c1a6e1e68c80911a69b86de209d20f2ab6269280c"
PC=dict(am.PC,observation_presentation="legacy-json")
CAPS={"generations":640,"searches":384}
PREPARE_CREATED=False

sha=am.sha; read=am.read; rows=am.rows; write_once=am.write_once

def append(path,value):
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(value,ensure_ascii=False,sort_keys=True)+"\n");f.flush()

def count(kind):
    p=OUT/"gpu-usage.jsonl"
    return sum(json.loads(x).get("kind")==kind for x in p.read_text(encoding="utf-8").splitlines()) if p.exists() else 0

def record(kind,stage,case):
    if count(kind)>=CAPS[kind]:raise RuntimeError(f"Goal AP {kind} cap exceeded")
    append(OUT/"gpu-usage.jsonl",dict(kind=kind,stage=stage,case_id=case,attempt=count(kind)+1,status="started",time=time.time()))

def config(arm):
    if arm not in ("A","B"):raise ValueError(arm)
    return dict(PC,observation_presentation="legacy-json" if arm=="A" else "document-blocks-v1")

def make_client(adapter,arm,corpus):
    return q.make_client(adapter,deterministic=True,product_config=config(arm),model_path=BASE,corpus=corpus)

def parse_archived_messages(tokenizer,ids):
    return ao.parse_messages(tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False))

def restore_tool_messages(messages):
    """Restore Qwen's rendered user/<tool_response> wrapper to the runtime tool role."""
    restored=[]
    for message in messages:
        value=dict(message)
        content=value.get("content","")
        prefix="<tool_response>\n";suffix="\n</tool_response>"
        if value.get("role")=="user" and content.startswith(prefix) and content.endswith(suffix):
            body=content[len(prefix):-len(suffix)]
            payload=json.loads(body)
            if not isinstance(payload,dict) or not isinstance(payload.get("result",[]),list):
                raise RuntimeError("rendered tool wrapper does not contain a search observation")
            value={"role":"tool","content":body}
        restored.append(value)
    return restored

def input_ids(client,messages):
    from mini_agent_rl.domain import Message
    rendered=client._render_chat([Message(**m) for m in messages])
    return rendered["input_ids"][0].tolist()

def source_paths():
    return {
      "runner":Path(__file__),"model_client":ROOT/"src/mini_agent_rl/model/local.py",
      "presentation_serializer":ROOT/"src/mini_agent_rl/model/observation_presentation.py",
      "q_runner":ROOT/"scripts/run_v133_goal_q_search.py",
      "ao_runner":ROOT/"scripts/run_v133_goal_ao_v2.py",
      "am_runner":ROOT/"scripts/run_v133_goal_am.py",
      "an_runner":ROOT/"scripts/run_v133_goal_an.py",
      "aa_release":AA/"product-decision.json",
      "z_adapter":ZADAPTER/"adapter_model.safetensors",
      "am_adapter":AMADAPTER/"adapter_model.safetensors",
      "am_summary":AM/"seed42-A/summary.json",
      "am_freeze":AM/"execution-freeze.json",
      "evaluation":AM/"evaluation-cases.jsonl",
      "complete_probes":AM/"frozen-probes.jsonl",
      "partial_probes":AN/"partial-probes.jsonl",
      "probe_rulings":ao.OUT/"probe-review-seed42.jsonl",
      "probe_evidence":ao.OUT/"probe-evidence-seed42.jsonl",
      "semantic_rulings":AM/"semantic-review-seed42.jsonl",
      "dev_corpus":Z/"corpus-dev.json",
      "tokenizer":BASE/"tokenizer.json","base_config":BASE/"config.json",
      "aa_manifest":AA/"product-decision.json",
    }

def prepare():
    global PREPARE_CREATED
    if OUT.exists():
        existing={p.name for p in OUT.iterdir()}
        if existing!={"PLAN.md"}:raise FileExistsError(f"refusing to overwrite existing AP run: {sorted(existing)}")
    else:
        OUT.mkdir(parents=True,exist_ok=False)
    PREPARE_CREATED=True
    if not (OUT/"PLAN.md").exists():raise FileNotFoundError("write frozen PLAN.md before prepare")
    if sha(ZADAPTER/"adapter_model.safetensors")!=Z_HASH:raise RuntimeError("Goal Z released adapter hash mismatch")
    if sha(AMADAPTER/"adapter_model.safetensors")!=AM_HASH:raise RuntimeError("AM A adapter hash mismatch")
    release=read(AA/"product-decision.json")
    if not release.get("published"):raise RuntimeError("AA Goal Z release decision is not published")
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(BASE),local_files_only=True)
    tokenizer=getattr(processor,"tokenizer",processor)
    complete=[p for p in rows(AM/"frozen-probes.jsonl") if p["category"]=="complete"]
    partial=rows(AN/"partial-probes.jsonl")
    cases=rows(AM/"evaluation-cases.jsonl")
    if len(complete)!=24 or len(partial)!=8 or len(cases)!=64:raise RuntimeError("frozen AP evaluation inventory changed")
    all_probes=complete+partial
    write_once(OUT/"frozen-probes.jsonl",all_probes,jsonl=True)
    write_once(OUT/"evaluation-cases.jsonl",cases,jsonl=True)
    input_records=[]
    for label,adapter in (("AM",AMADAPTER),("Z",ZADAPTER)):
        for arm in ("A","B"):
            client,_,_=make_client(adapter,arm,Z/"corpus-dev.json")
            if client._model is not None:raise RuntimeError("CPU prepare unexpectedly loaded model weights")
            client._processor=processor
            for p in all_probes:
                archived_messages=parse_archived_messages(tokenizer,p["prompt_token_ids"])
                messages=archived_messages if arm=="A" else restore_tool_messages(archived_messages)
                ids=input_ids(client,messages)
                if arm=="A" and ids!=p["prompt_token_ids"]:
                    raise RuntimeError(f"legacy input does not reproduce archived tokens: {p['probe_id']}")
                if arm=="B" and len(ids)>PC["max_input_tokens"]:
                    raise RuntimeError(f"document presentation exceeds input budget: {p['probe_id']}")
                if arm=="B" and ids==p["prompt_token_ids"]:
                    raise RuntimeError(f"B presentation did not change the archived input: {p['probe_id']}")
                audit={}
                if arm=="B":
                    encoded,presentation=client._render_chat(
                        [__import__("mini_agent_rl.domain",fromlist=["Message"]).Message(**m) for m in messages],
                        return_presentation=True)
                    if encoded["input_ids"][0].tolist()!=ids:raise RuntimeError("presentation preflight ids drift")
                    audit=presentation
                    if not audit.get("document_bodies"):
                        raise RuntimeError(f"B did not find any wrapped search documents: {p['probe_id']}")
                    for doc in audit["document_bodies"]:
                        if doc["input_token_span"][0]>=doc["input_token_span"][1] and doc["body_chars"]:
                            raise RuntimeError("nonempty document has empty input token span")
                input_records.append(dict(model=label,arm=arm,probe_id=p["probe_id"],case_id=p["case_id"],
                    category=p["category"],messages=messages,prompt_token_ids=ids,input_sha256=hashlib.sha256(
                    json.dumps(ids,separators=(",",":")).encode()).hexdigest(),presentation=audit,
                    no_search_execution=True,training_use_forbidden=True))
            del client
    write_once(OUT/"probe-inputs.jsonl",input_records,jsonl=True)
    for path in (AM/"seed42-A/summary.json",):
        if read(path).get("adapter_sha256")!=AM_HASH:raise RuntimeError("AM summary/hash mismatch")
    files=source_paths();hashes={k:sha(p) for k,p in files.items()}
    paths={k:str(p.resolve()) for k,p in files.items()}
    manifest=dict(goal="AP",status="prepared",models={"AM":str(AMADAPTER),"Z":str(ZADAPTER)},
      expected_hashes={"GoalZ":Z_HASH,"AM42A":AM_HASH},hashes=hashes,paths=paths,
      probe_counts={"complete":24,"partial":8,"per_model_arm":32,"models":2,"arms":2},dev_cases=64,
      arms={"A":config("A"),"B":config("B")},limits=CAPS,
      stages={"probe_generations":128,"probe_searches":0,"dev_trajectories":128,"dev_generations_max":512,"dev_searches_max":384},
      gate={"complete_B_at_least":22,"complete_gain_vs_A":2,"binding_repair_min":1,"complete_losses_max":1,
        "partial_legal_search_min":7,"partial_legal_search_not_below_A":True,"partial_bad_actions_no_increase":True,
        "supported_B_at_least":23,"supported_gain_vs_A":2,"errors_no_higher_than_A_and_at_most":7,
        "action_problems_no_higher_than_A":True,"protect_Rapana_Duke":True,"old_protected_losses_max":1},
      products=dict(base="/root/autodl-tmp/qwen_08b",device="cuda",dtype="bfloat16",load_in_4bit=False,
        temperature=0,thinking=False,system_prompt="direct-v090",action_completion="first-json-v1",
        observation_encoding="legacy",retriever="bm25-v2",top_k=3,max_input_tokens=16384,max_new_tokens=2048,
        max_model_calls_per_case=4,max_searches_per_case=3),
      failure_policy="model identity, token mapping, GPU, numeric or infrastructure failure: preserve and stop; no retries/tuning",
      gpu_authorization_basis="用户已明确授权GPU实验并说明以后无需再次确认",released_model_remains="Goal Z AA seed42")
    manifest["frozen_output_hashes"]={name:sha(OUT/name) for name in ("frozen-probes.jsonl","evaluation-cases.jsonl","probe-inputs.jsonl")}
    manifest["frozen_output_paths"]={name:str((OUT/name).resolve()) for name in manifest["frozen_output_hashes"]}
    write_once(OUT/"manifest.json",manifest)
    write_once(OUT/"gpu-usage.jsonl",[],jsonl=True)
    write_once(OUT/"status.json",dict(stage="prepared",generations=0,searches=0,training_updates=0))
    print(json.dumps(dict(prepared=True,manifest_sha256=sha(OUT/"manifest.json"),probe_inputs=len(input_records))))

def frozen():
    m=read(OUT/"manifest.json")
    recovery_path=OUT/"recovery-amendment.json"
    recovered=recovery_path.exists() and read(recovery_path).get("status")=="remainder_completed"
    if (OUT/"experiment-failure.json").exists() and not recovered:
        raise RuntimeError("AP failure latch present; only remainder-only software recovery can resume")
    for key,path in m["paths"].items():
        if recovered and key=="runner":
            approved_hash=read(recovery_path).get("new_runner_sha256")
            post_path=OUT/"postprocessing-amendment.json"
            if post_path.exists():
                post=read(post_path)
                if post.get("status")!="cpu_scoring_only" or post.get("parent_manifest_sha256")!=sha(OUT/"manifest.json"):
                    raise RuntimeError("invalid post-generation scoring amendment")
                approved_hash=post.get("new_runner_sha256")
            if sha(path)!=approved_hash:
                raise RuntimeError("AP recovery runner hash mismatch")
            continue
        if sha(path)!=m["hashes"][key]:raise RuntimeError(f"frozen AP input changed: {key}")
    for name,digest in m["frozen_output_hashes"].items():
        if sha(OUT/name)!=digest:raise RuntimeError(f"frozen AP output changed: {name}")
    if not read(OUT/"cpu-preflight.json").get("passed"):raise RuntimeError("AP CPU preflight missing")
    return m

def cpu_preflight():
    m=read(OUT/"manifest.json")
    if sha(ZADAPTER/"adapter_model.safetensors")!=Z_HASH or sha(AMADAPTER/"adapter_model.safetensors")!=AM_HASH:
        raise RuntimeError("adapter identity changed after prepare")
    records=rows(OUT/"probe-inputs.jsonl")
    if len(records)!=128:raise RuntimeError("expected exact 128 probe inputs")
    by={(r["model"],r["arm"],r["probe_id"]):r for r in records}
    for p in rows(OUT/"frozen-probes.jsonl"):
        for model in ("AM","Z"):
            a=by[(model,"A",p["probe_id"])];b=by[(model,"B",p["probe_id"])]
            if a["prompt_token_ids"]!=p["prompt_token_ids"]:raise RuntimeError("A probe ids differ from archive")
            amsg=restore_tool_messages(a["messages"]);bmsg=b["messages"]
            if len(amsg)!=len(bmsg):raise RuntimeError("A/B message count changed")
            for i,(x,y) in enumerate(zip(amsg,bmsg)):
                if x["role"]!=y["role"]:raise RuntimeError("restored A/B message roles differ")
                if x["role"]!="tool" and x!=y:raise RuntimeError("A/B changed non-observation message")
                if x["role"]=="tool" and json.loads(x["content"])!=json.loads(y["content"]):
                    raise RuntimeError("A/B changed tool payload content")
            docs=b["presentation"].get("document_bodies",[])
            tool_messages=[msg["content"] for msg in amsg if msg["role"]=="tool"]
            if len(docs)!=sum(len(json.loads(content).get("result",[])) for content in tool_messages):
                raise RuntimeError("document count changed in presentation")
            flattened=[]
            for content in tool_messages:
                rendered,_=present_observation(content,DOCUMENT_BLOCKS_V1)
                recovered=parse_document_blocks(rendered)
                if recovered!=json.loads(content):raise RuntimeError("document presentation changed source evidence")
                flattened.extend(recovered["result"])
            if [d["body_sha256"] for d in docs]!=[
                hashlib.sha256(str(d.get("text","")).encode()).hexdigest() for d in flattened]:
                raise RuntimeError("document body hash/order differs from tool observations")
            if any(x["body_chars"]<0 or len(x["input_token_span"])!=2 for x in docs):raise RuntimeError("bad body token mapping")
            if b["prompt_token_ids"]==a["prompt_token_ids"]:
                raise RuntimeError("document-block presentation did not change the rendered input")
            if len(b["prompt_token_ids"])>m["products"]["max_input_tokens"]:
                raise RuntimeError("document-block presentation exceeds the frozen input budget")
    write_once(OUT/"cpu-preflight.json",dict(passed=True,manifest_sha256=sha(OUT/"manifest.json"),
      adapter_hashes={"GoalZ":Z_HASH,"AM42A":AM_HASH},probe_inputs=128,legacy_replay_exact=True,
      document_body_token_mappings=sum(len(r["presentation"].get("document_bodies",[])) for r in records if r["arm"]=="B"),
      gpu_calls=0,searches=0,updates=0))
    print(json.dumps(dict(cpu_preflight=True,probes=128,manifest_sha256=sha(OUT/"manifest.json"))))

def probe(model,arm,resume=False):
    m=None if resume else frozen();adapter=AMADAPTER if model=="AM" else ZADAPTER
    if model not in ("AM","Z"):raise ValueError(model)
    out=OUT/f"probe-{model}-{arm}.jsonl"
    if out.exists() and not resume:raise FileExistsError(out)
    import torch
    client,_,_=make_client(adapter,arm,Z/"corpus-dev.json")
    records=[r for r in rows(OUT/"probe-inputs.jsonl") if r["model"]==model and r["arm"]==arm]
    if len(records)!=32:raise RuntimeError("probe set incomplete")
    completed={r["probe_id"] for r in rows(out)} if resume else set()
    for item in records:
        if item["probe_id"] in completed:continue
        from mini_agent_rl.domain import Message
        messages=[Message(**x) for x in item["messages"]]
        record("generations",f"probe-{model}-{arm}",item["case_id"])
        started=time.monotonic();response=client._generate_sync(messages,0.,42);elapsed=time.monotonic()-started
        if response.prompt_token_ids!=item["prompt_token_ids"]:raise RuntimeError(f"actual probe input mismatch: {item['probe_id']}")
        action=an.parsed_action(response)
        append(out,dict(model=model,arm=arm,probe_id=item["probe_id"],case_id=item["case_id"],category=item["category"],
          action=action,answer=response.content,query=response.tool_call.arguments.get("query") if response.tool_call else None,
          parse_error=response.parse_error,raw=response.raw_text,prompt_token_ids=response.prompt_token_ids,
          action_token_ids=response.action_token_ids,usage=response.usage,elapsed_seconds=elapsed,no_search_execution=True,
          training_use_forbidden=True))
        print(json.dumps(dict(stage=f"probe-{model}-{arm}",probe=item["probe_id"],action=action if action else "invalid")),flush=True)
    del client;gc.collect();torch.cuda.empty_cache()

def resume_probe(model,arm):
    failure=read(OUT/"experiment-failure.json")
    if failure.get("error_type")!="TypeError" or failure.get("stage")!="probe":
        raise RuntimeError("only the recorded probe-output TypeError is eligible for remainder-only recovery")
    if failure.get("generations")!=1 or failure.get("searches")!=0 or failure.get("updates")!=0:
        raise RuntimeError("unexpected work occurred before the software failure")
    m=read(OUT/"manifest.json")
    for key,path in m["paths"].items():
        if key=="runner":continue
        if sha(path)!=m["hashes"][key]:raise RuntimeError(f"frozen input changed during recovery: {key}")
    out=OUT/f"probe-{model}-{arm}.jsonl"
    saved=rows(out)
    source=[r for r in rows(OUT/"probe-inputs.jsonl") if r["model"]==model and r["arm"]==arm]
    journal=[json.loads(x) for x in (OUT/"gpu-usage.jsonl").read_text().splitlines()
             if x.strip() and json.loads(x).get("stage")==f"probe-{model}-{arm}"]
    if len(saved)!=1 or len(journal)!=1 or saved[0]["probe_id"]!=source[0]["probe_id"]:
        raise RuntimeError("partial probe output is not one completed first state")
    if saved[0].get("action") not in ("answer","search") or saved[0].get("no_search_execution") is not True:
        raise RuntimeError("first probe result is not safely reusable")
    if len(source)!=32 or saved[0]["prompt_token_ids"]!=source[0]["prompt_token_ids"]:
        raise RuntimeError("completed inference differs from frozen first input")
    amendment_path=OUT/"recovery-amendment.json"
    if amendment_path.exists():raise FileExistsError(amendment_path)
    amendment=dict(status="remainder_running",failure_latch_preserved=True,prior_generation_reused=True,
      prior_generation_count=1,prior_search_count=0,prior_update_count=0,old_runner_sha256=m["hashes"]["runner"],
      new_runner_sha256=sha(Path(__file__)),
      correction="Use parsed_action string directly in progress logging; request, prompt, weights and budgets unchanged.",
      skipped_probe_ids=[saved[0]["probe_id"]],remaining_probe_count=31)
    write_once(amendment_path,amendment)
    probe(model,arm,resume=True)
    amendment["status"]="remainder_completed";amendment["completed_probe_count"]=32
    amendment_path.write_text(json.dumps(amendment,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def instrument(wrapped,registry,stage,active):
    gen=wrapped.generate;execute=registry.execute
    async def counted(*args,**kwargs):record("generations",stage,active["id"]);return await gen(*args,**kwargs)
    async def counted_execute(*args,**kwargs):
        if args and args[0]=="search":record("searches",stage,active["id"])
        return await execute(*args,**kwargs)
    wrapped.generate=counted;registry.execute=counted_execute

def evaluate(arm):
    m=frozen()
    if not read(OUT/"probe-gate.json").get("passed"):raise RuntimeError("probe gate not passed")
    out=OUT/f"eval-AM-{arm}.jsonl"
    if out.exists():raise FileExistsError(out)
    client,wrapped,registry=make_client(AMADAPTER,arm,Z/"corpus-dev.json")
    active={"id":"unassigned"};instrument(wrapped,registry,f"eval-AM-{arm}",active)
    for i,case in enumerate(rows(OUT/"evaluation-cases.jsonl"),1):
        active["id"]=case["case_id"]
        result=asyncio.run(q.rollout(case["question"],wrapped,registry,sample=False,seed=42,product_config=config(arm)))
        result.update(case_id=case["case_id"],model=f"AM-{arm}",official_answer=case["answer"],**q.score(result["answer"],case["answer"]))
        if result["reason"]=="tool_error":raise RuntimeError("AP search infrastructure error")
        if result["model_calls"]>4 or result["searches"]>3:raise RuntimeError("AP action/search budget exceeded")
        for call in result["trace"]:
            if len(call["prompt_token_ids"])>16384 or len(call["action_token_ids"])>2048:raise RuntimeError("AP token budget exceeded")
            usage=call.get("context_usage",{})
            if usage.get("context_budget_error") or usage.get("lossy_body_trimmed") or usage.get("documents_deleted"):
                raise RuntimeError("AP presentation required evidence clipping/deletion; stop without changing conditions")
        append(out,result)
        print(json.dumps(dict(stage=f"eval-AM-{arm}",case=i,em=result["em"],searches=result["searches"])),flush=True)
    del client,wrapped,registry;gc.collect();torch.cuda.empty_cache()

def gate_probes():
    expected={(model,arm,pid) for model in ("AM","Z") for arm in ("A","B")
              for pid in (r["probe_id"] for r in rows(OUT/"frozen-probes.jsonl"))}
    observed=[]
    for model in ("AM","Z"):
        for arm in ("A","B"):
            path=OUT/f"probe-{model}-{arm}.jsonl"
            if not path.exists():raise RuntimeError(f"probe stage incomplete: {path.name}")
            observed += rows(path)
    if len(observed)!=128 or len({(r["model"],r["arm"],r["probe_id"]) for r in observed})!=128:
        raise RuntimeError("probe outputs do not cover the frozen 128 states exactly once")
    adj_path=OUT/"probe-adjudications.jsonl"
    if not adj_path.exists():
        template=[]
        for r in observed:
            template.append(dict(model=r["model"],arm=r["arm"],probe_id=r["probe_id"],case_id=r["case_id"],
              category=r["category"],action=r["action"],answer=r["answer"],query=r["query"],
              supported_correct=None,binding_repair=None,legal_bridge_search=None,premature_answer=None,
              invalid_action=None,repeated_query=None,adjudication_reason=None))
        write_once(adj_path,template,jsonl=True)
        raise RuntimeError("created one-pass probe adjudication template; fill source-grounded fields and rerun gate-probes")
    adjudications=rows(adj_path);bykey={(r["model"],r["arm"],r["probe_id"]):r for r in adjudications}
    if set(bykey)!=expected or len(adjudications)!=128:raise RuntimeError("probe adjudication coverage/identity mismatch")
    complete_required=("supported_correct","binding_repair","adjudication_reason")
    partial_required=("legal_bridge_search","premature_answer","invalid_action","repeated_query","adjudication_reason")
    if any(any(r.get(k) is None for k in (complete_required if r["category"]=="complete" else partial_required))
           for r in adjudications if r["model"]=="AM"):
        raise RuntimeError("primary-model probe adjudication incomplete")
    def metrics(model,arm,category):
        rr=[r for r in adjudications if r["model"]==model and r["arm"]==arm and r["category"]==category]
        return dict(correct=sum(r["supported_correct"] is True for r in rr),
          binding_repairs=sum(r["binding_repair"] is True for r in rr),
          legal_search=sum(r["legal_bridge_search"] is True for r in rr),
          premature=sum(r["premature_answer"] is True for r in rr),
          invalid=sum(r["invalid_action"] is True for r in rr),
          repeated=sum(r["repeated_query"] is True for r in rr),
          count=sum(r.get("supported_correct") is not None for r in rr))
    a=metrics("AM","A","complete");b=metrics("AM","B","complete")
    pa=metrics("AM","A","partial");pb=metrics("AM","B","partial")
    passed=(b["correct"]>=22 and b["correct"]-a["correct"]>=2 and b["binding_repairs"]>=1
      and a["correct"]-sum(x["supported_correct"] is True for x in adjudications if x["model"]=="AM" and x["arm"]=="A" and x["category"]=="complete" and bykey[("AM","B",x["probe_id"])]["supported_correct"] is not True)<=1
      and pb["legal_search"]>=7 and pb["legal_search"]>=pa["legal_search"]
      and pb["premature"]<=pa["premature"] and pb["invalid"]<=pa["invalid"] and pb["repeated"]<=pa["repeated"])
    result=dict(passed=passed,am={"A_complete":a,"B_complete":b,"A_partial":pa,"B_partial":pb},
      z={arm:{kind:metrics("Z",arm,kind) for kind in ("complete","partial")} for arm in ("A","B")},
      gate_rules=read(OUT/"manifest.json")["gate"],adjudications_sha256=sha(adj_path))
    write_once(OUT/"probe-gate.json",result)
    print(json.dumps(result,ensure_ascii=False))

def report():
    frozen();path=OUT/"REPORT.md"
    if path.exists():raise FileExistsError(path)
    import statistics
    comparisons=[]
    for arm in ("A","B"):
        results=rows(OUT/f"eval-AM-{arm}.jsonl")
        if len(results)!=64:raise RuntimeError(f"incomplete development set for {arm}")
        comparisons.append((arm,results))
    metrics={}
    for arm,results in comparisons:
        metrics[arm]={"em":sum(r["em"] for r in results),"f1":sum(r["f1"] for r in results)/64,
          "calls":sum(r["model_calls"] for r in results),"searches":sum(r["searches"] for r in results),
          "actions":sum(r["reason"] in ("invalid_action","repeated_action","max_steps") for r in results),
          "max_input":max((len(t["prompt_token_ids"]) for r in results for t in r["trace"]),default=0),
          "mean_latency":statistics.mean(r["elapsed_seconds"] for r in results)}
    paired={r["case_id"]:r for r in comparisons[0][1]};paired_b={r["case_id"]:r for r in comparisons[1][1]}
    text=["# Goal AP：证据文档呈现 A/B 结果","","本报告只在64题开发集评估。Goal Z仍为唯一发布模型。","",
      "| 条件 | EM/64 | F1 | 调用 | 搜索 | 动作问题题数 | 最大输入token | 平均题延迟秒 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("A","B"):
        x=metrics[arm];text.append(f"| {arm} | {x['em']} | {x['f1']:.4f} | {x['calls']} | {x['searches']} | {x['actions']} | {x['max_input']} | {x['mean_latency']:.3f} |")
    changed=[cid for cid in paired if paired[cid]["em"]!=paired_b[cid]["em"]]
    text += ["",f"EM不同案例：{', '.join(changed) if changed else '无'}。","",
      "支持成功、逐边引用及保护集必须从一次性语义裁定账本汇总；机械EM不代表证据充分。探针判断见probe-gate.json。",
      "","逐题完整轨迹见eval-AM-A.jsonl与eval-AM-B.jsonl。Body与token区间保存在每次调用context_usage.presentation中。"]
    path.write_text("\n".join(text)+"\n",encoding="utf-8",newline="\n")
    write_once(OUT/"mechanical-summary.json",metrics)
    print(json.dumps(dict(report=str(path),metrics=metrics),ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="cmd",required=True)
    sub.add_parser("prepare");sub.add_parser("cpu-preflight");sub.add_parser("gate-probes");sub.add_parser("report")
    p=sub.add_parser("probe");p.add_argument("--model",choices=["AM","Z"],required=True);p.add_argument("--arm",choices=["A","B"],required=True)
    rp=sub.add_parser("resume-probe");rp.add_argument("--model",choices=["AM","Z"],required=True);rp.add_argument("--arm",choices=["A","B"],required=True)
    e=sub.add_parser("evaluate");e.add_argument("--arm",choices=["A","B"],required=True)
    args=parser.parse_args()
    try:
        if args.cmd=="prepare":prepare()
        elif args.cmd=="cpu-preflight":cpu_preflight()
        elif args.cmd=="gate-probes":gate_probes()
        elif args.cmd=="report":report()
        else:
            import torch
            if torch.cuda.device_count()!=1:raise RuntimeError("Goal AP requires one visible GPU")
            if args.cmd=="resume-probe":resume_probe(args.model,args.arm)
            elif args.cmd=="probe":probe(args.model,args.arm)
            else:evaluate(args.arm)
    except BaseException as exc:
        if OUT.exists() and (OUT/"manifest.json").exists() or (args.cmd=="prepare" and PREPARE_CREATED):
            if (OUT/"experiment-failure.json").exists():raise
            write_once(OUT/"experiment-failure.json",dict(stage=args.cmd,error_type=type(exc).__name__,error=str(exc),
              time=time.time(),generations=count("generations"),searches=count("searches"),updates=0))
        raise

if __name__=="__main__":main()
