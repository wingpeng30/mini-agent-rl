"""Goal AN: frozen sample-level hard-state SFT, with an independent GPU gate.

All AM sources are read-only. Preparation/preflight are CPU-only; GPU commands
require a new human approval tied to the AN freeze, not the old AM approval.
"""
from __future__ import annotations
import argparse
import asyncio
import gc
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
import run_v133_goal_am as am
import run_v133_goal_q_search as q
import finalize_v133_goal_al_report as prior

ROOT=am.ROOT
OUT=ROOT/"runs/v133-goal-an-hard-state-sft-v2"
AM=am.OUT
SOURCE=am.SOURCE
BASE=am.BASE
PC=dict(am.PC)
OPT=dict(am.OPT)
CAPS=dict(update=384,generation=896,search=576)
HARD={
 "musique:2hop__407314_80178":("Jaden Smith","Justin Bieber","co-performer versus bridge performer"),
 "2wiki:b2addc160bdc11eba7f7acde48001122":("University of Helsinki","professor","work institution versus occupation"),
 "2wiki:a874fa560bda11eba7f7acde48001122":("Christiansburg, Virginia","June 28, 1898","birthplace versus birth date"),
 "2wiki:fa9654640bdb11eba7f7acde48001122":("September 3, 1922","September 3","birth date requires year"),
}
sha,read,rows,write=am.sha,am.read,am.rows,am.write_once


def usage(kind):
    f=OUT/"gpu-usage.jsonl"
    return sum(r["kind"]==kind for r in rows(f)) if f.exists() else 0


def record(kind,stage,case):
    if (OUT/"experiment-failure.json").exists():raise RuntimeError("AN failure latch present")
    if usage(kind)>=CAPS[kind]:raise RuntimeError("AN frozen quota exhausted: "+kind)
    with (OUT/"gpu-usage.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(dict(kind=kind,stage=stage,case_id=case,attempt=usage(kind)+1))+"\n")
        f.flush()


def normalized_state_weight(case_id,arm):
    if arm=="A":return 1.0
    if arm!="B":raise ValueError(arm)
    return (4.0 if case_id in HARD else 1.0)*34.0/46.0


def parsed_action(response):
    if response.parse_error:return "invalid"
    if response.tool_call is not None:return "search"
    if response.content and response.content.strip():return "answer"
    return "invalid"


def frozen(require_approval=False):
    if (OUT/"experiment-failure.json").exists():raise RuntimeError("AN failure latch present")
    cfg=read(OUT/"manifest.json")
    for name,r in cfg["files"].items():
        if sha(r["path"])!=r["sha256"]:raise RuntimeError("AN identity/archive changed: "+name)
    # Includes AA identity, all model shards, AM sources and tokenizer.
    am.approved()
    if require_approval:
        pre=read(OUT/"cpu-preflight.json")
        if not pre["passed"] or pre["manifest_sha256"]!=sha(OUT/"manifest.json"):
            raise RuntimeError("AN CPU preflight does not match freeze")
        approval=OUT/"gpu-approval.json"
        if not approval.exists():raise RuntimeError("AN requires confirmation of PRE_GPU_REVIEW.md")
        a=read(approval)
        if not a.get("human_confirmed") or a.get("manifest_sha256")!=sha(OUT/"manifest.json"):
            raise RuntimeError("AN GPU approval does not match freeze")
    return cfg


def prepare():
    amcfg=am.approved()
    if read(AM/"seed42-gate.json")["route_passed"]:raise RuntimeError("unexpected AM route status")
    if not read(AM/"delivery-validation.json")["passed"]:raise RuntimeError("AM delivery not verified")
    OUT.mkdir(exist_ok=False)
    tok=am.tokenizer_from_base()
    states=rows(AM/"candidate-training-states.jsonl")
    if len(states)!=34 or Counter(s["state_type"] for s in states)!=dict(answer=24,search=10):
        raise RuntimeError("AN requires unchanged 34 AM states")
    eval_ids,eval_questions,_=am.eval_isolation()
    am.validate_state_isolation(states,eval_ids,eval_questions)
    historical={r["probe_id"]:r for r in rows(AM/"probe-baseline-state.jsonl")}
    hard_review=[]
    for s in states:
        if s["case_id"] not in HARD:continue
        target,wrong,reason=HARD[s["case_id"]]
        if s["target_action"]!=dict(action="answer",answer=target):raise RuntimeError("hard target drift")
        h=historical["baseline-state:"+s["state_id"]]
        if h["prompt_token_ids"]!=s["prompt_token_ids"] or h["answer"]!=wrong:
            raise RuntimeError("hard-state baseline archive mismatch")
        citations=[]
        for c in s["citations"]:
            found=prior.cite(s["prompt_token_ids"],tok,c["document_id"],c["quote"])
            if found is None:raise RuntimeError("hard-state exact body quote missing")
            citations.append(dict(triple=c["triple"],**found))
        hard_review.append(dict(case_id=s["case_id"],state_id=s["state_id"],question=s["question"],
            target_action=s["target_action"],goal_z_actual_answer=wrong,reason=reason,
            citations=citations,evaluation_isolated=True,raw_weight=4,
            normalized_weight=normalized_state_weight(s["case_id"],"B")))
    if len(hard_review)!=4:raise RuntimeError("four frozen hard states required")
    weights=[dict(state_id=s["state_id"],case_id=s["case_id"],update=2+2*i,
        raw_weight=4 if s["case_id"] in HARD else 1,A=1.0,B=normalized_state_weight(s["case_id"],"B"))
        for i,s in enumerate(states)]
    if not math.isclose(sum(r["B"] for r in weights),34):raise RuntimeError("AN normalization mismatch")
    packets={r["case_id"]:r for r in rows(AM/"review-packets-seed42-corrected.jsonl") if r["model"]=="42-A"}
    partial=[]
    eligible_count=0
    baseline=rows(am.AL/"trajectory-B.jsonl")
    for case in baseline:
        if case["case_id"] not in packets:continue
        edges=packets[case["case_id"]]["edges"]
        for t in case["trace"]:
            step=t["step"]
            if not 1<=step<=2:continue
            visible=[prior.cite(t["prompt_token_ids"],tok,e["citation"]["document_id"],e["citation"]["quote"])
                if e.get("citation") else None for e in edges]
            previous=[x["query"] for x in case["trace"][:step] if x.get("query") and "observation" in x]
            if len(visible)!=2 or visible[0] is None or visible[1] is not None or len(previous)>2:continue
            eligible_count+=1
            if len(partial)<8:
                partial.append(dict(probe_id=f"partial:{case['case_id']}:step{step}",case_id=case["case_id"],
                    category="partial",question=case["question"],step=step,
                    prompt_token_ids=t["prompt_token_ids"],remaining_calls=4-step,remaining_searches=3-len(previous),
                    previous_queries=previous,bridge_entity=edges[0]["object"],missing_relation=edges[1]["relation"],
                    first_edge_citation=visible[0],second_edge_visible=False,
                    required_second_edge=edges[1],original_action=t["action"],
                    no_search_execution=True,training_use_forbidden=True))
            break
    if len(partial)!=8:raise RuntimeError("fewer than 8 valid partial probes; no replacement allowed")
    complete=[r for r in rows(AM/"frozen-probes.jsonl") if r["category"]=="complete"]
    if len(complete)!=24:raise RuntimeError("24 complete probes required")
    fit=[dict(probe_id="fit:"+s["state_id"],case_id=s["case_id"],question=s["question"],category="training_fit",
        prompt_token_ids=s["prompt_token_ids"],target_action=s["target_action"],
        no_search_execution=True,generalization_use_forbidden=True) for s in states if s["case_id"] in HARD]
    write(OUT/"hard-state-admission.json",hard_review)
    write(OUT/"state-schedule.json",weights)
    write(OUT/"partial-probes.jsonl",partial,True)
    write(OUT/"complete-probes.jsonl",complete,True)
    write(OUT/"training-fit-probes.jsonl",fit,True)
    # Sources remain references rather than rewritten AM datasets.
    paths={"am_manifest":AM/"execution-freeze.json","am_gate":AM/"seed42-gate.json",
        "am_validation":AM/"delivery-validation.json","am_candidate_states":AM/"candidate-training-states.jsonl",
        "am_A42_weights":AM/"seed42-A/adapter-final/adapter_model.safetensors",
        "am_A42_summary":AM/"seed42-A/summary.json","am_A42_trajectories":AM/"eval-42-A.jsonl",
        "am_A42_probes":AM/"probe-42-A.jsonl","am_A42_probe_review":AM/"probe-review-seed42.jsonl",
        "am_semantics":AM/"semantic-review-seed42.jsonl","am_citation_packets":AM/"review-packets-seed42-corrected.jsonl",
        "am_baseline_states":AM/"probe-baseline-state.jsonl","evaluation_cases":AM/"evaluation-cases.jsonl",
        "runner":Path(__file__),"reviewer":ROOT/"scripts/analyze_v133_goal_an.py",
        "tests":ROOT/"tests/test_v133_goal_an.py"}
    for name in ("hard-state-admission.json","state-schedule.json","partial-probes.jsonl","complete-probes.jsonl","training-fit-probes.jsonl"):
        paths[name]=OUT/name
    al=read(am.AL/"manifest.json")
    cfg=dict(goal="AN",status="awaiting_preflight_and_human_confirmation",product=PC,optimizer=OPT,
        source_model=str(SOURCE),source_sha256=am.EXPECTED_ADAPTER,base_model=str(BASE),
        source_am_freeze_sha256=sha(AM/"execution-freeze.json"),
        files={n:dict(path=str(p),sha256=sha(p)) for n,p in paths.items()},
        state_weight=dict(hard_raw=4,other_raw=1,normalizer="34/46",sum=34,mix=.25),
        training=dict(trajectories=128,states=34,updates_per_run=128,state_updates=list(range(2,69,2)),order=amcfg["order"]),
        reuse_A42=True,reused_A42_weights_sha256=read(AM/"seed42-A/summary.json")["adapter_sha256"],
        protection=al["protected"],must_retain=["musique:2hop__796221_18803","2wiki:f299dd600bdd11eba7f7acde48001122"],
        caps=CAPS,training_runs=3,full_trajectories=192,probe_generations=128,new_confirmation_calls=0,
        partial_probe_candidates=eligible_count,hard_case_ids=list(HARD),
        thresholds=dict(supported=23,complete=22,entity_binding_repair_min=1,errors_max=7,
            action_problem_cases_max=1,protected_losses_max=1,relative_A_supported_gain=2,relative_A_complete_gain=2),
        limits=dict(new_answer_combinations=87),released=False)
    write(OUT/"manifest.json",cfg)
    write(OUT/"status.json",dict(status="cpu_preflight_pending",gpu_generations=0,successful_updates=0))
    print(json.dumps(dict(prepared=True,hard_states=4,partial_candidates=eligible_count,manifest_sha256=sha(OUT/"manifest.json"))))


def cpu_preflight():
    cfg=frozen()
    import torch
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    from mini_agent_rl.domain.models import ModelResponse,ToolCall
    states=rows(AM/"candidate-training-states.jsonl")
    schedule=read(OUT/"state-schedule.json")
    assert len(schedule)==34 and math.isclose(sum(x["B"] for x in schedule),34)
    class Toy:
        def __init__(self):self.x=torch.tensor([-1.,-2.,-3.],requires_grad=True)
        def score_action_tensor(self,*args,**kwargs):return self.x
    for arm in ("A","B"):
        for case in (next(iter(HARD)),"easy"):
            toy=Toy();w=normalized_state_weight(case,arm)
            loss=am.target_loss(toy,dict(action_token_ids=[1,2,3],prompt_token_ids=[4],prompt_attention_mask=[1]),[1.]*3,scale=.25*w)
            assert math.isclose(loss,.5*w,rel_tol=1e-6)
            assert torch.allclose(toy.x.grad,torch.full_like(toy.x,-.25*w/3))
    p=torch.nn.Parameter(torch.tensor([1.]));opt=torch.optim.AdamW([p],lr=5e-6,betas=(.9,.999),eps=1e-8,weight_decay=0)
    before=p.detach().clone();p.grad=torch.ones_like(p)
    guarded_optimizer_step(opt,[p],{"peak_reserved_gb":0.},21.)
    assert torch.all(p.detach()<before)
    before=p.detach().clone();p.grad=torch.full_like(p,float("nan"))
    try:guarded_optimizer_step(opt,[p],{"peak_reserved_gb":0.},21.)
    except RuntimeError:pass
    else:raise RuntimeError("NaN guard did not reject")
    assert torch.equal(before,p.detach())
    assert parsed_action(ModelResponse(tool_calls=[ToolCall(name="search",arguments={"query":"test"})]))=="search"
    assert parsed_action(ModelResponse(content="answer"))=="answer"
    assert parsed_action(ModelResponse())=="invalid"
    trajectories=rows(am.Z/"training-inputs.jsonl")
    all_actions=[a for r in trajectories for a in r["actions"]]+states
    for a in all_actions:
        assert len(a["prompt_token_ids"])==len(a["prompt_attention_mask"])
        assert len(a["prompt_token_ids"])+len(a["action_token_ids"])<=16384+2048
    factory,_,registry=q.make_client(SOURCE,deterministic=True,product_config=PC,model_path=BASE,corpus=am.Z/"corpus-dev.json")
    assert factory._model is None and registry.definitions()
    for k in ("max_input_tokens","max_new_tokens","observation_encoding","action_completion"):
        assert getattr(factory.config,k)==PC[k]
    assert factory.config.enable_thinking==PC["thinking"]
    write(OUT/"cpu-preflight.json",dict(passed=True,manifest_sha256=sha(OUT/"manifest.json"),
        weighted_loss_and_gradient=True,ordinary_loss_matches_AM=True,weight_sum=34,
        optimizer_valid_and_nan_guard=True,action_classification=True,mask_actions=len(all_actions),
        factory_not_loaded=True,gpu_calls=0,model_updates=0))
    lines=["# Goal AN GPU前实际前提", "", "准备和CPU核验已完成；本轮GPU尚未授权。", "",
        f"冻结manifest SHA-256：`{sha(OUT/'manifest.json')}`。",
        f"模型：AA发布Goal Z seed42，权重SHA `{am.EXPECTED_ADAPTER}`；基础模型`{BASE}`。",
        f"复用A42：AM普通SFT权重SHA `{cfg['reused_A42_weights_sha256']}`，不重跑64题与24完整探针。", "",
        "| 训练难例 | Goal Z实际错误 | 正确当前动作答案 | B状态系数 |", "|---|---|---|---:|"]
    for h in read(OUT/"hard-state-admission.json"):
        lines.append(f"| {h['question']} | {h['goal_z_actual_answer']} | {h['target_action']['answer']} | {h['normalized_weight']:.9f} |")
    lines += ["", "其余30状态系数0.739130435；4难例系数2.956521739；总和34。各状态仍只出现一次。",
        "34状态与128轨迹均来自AM冻结材料，未改变目标或历史输入。仅四难例逐边复核；详见hard-state-admission.json的原句及token位置。",
        "每臂128更新，第2、4…68次加入状态；L=轨迹均值NLL+0.25×状态系数×当前动作均值NLL。当前JSON与终止符监督，其余历史mask。",
        "LR5e-6；AdamW(0.9,0.999),eps1e-8,weight_decay0；clip1；只训练LoRA，BF16，单卡，不量化。",
        "共同评价：direct-v090、first-json-v1、thinking=false、legacy、BM25-v2/top3、greedy、16384/2048、4调用/3搜索。", "",
        "部分链探针按AL题序冻结前8题，第一边原文可见、第二边未见；已有23候选，不采集新资料。", ""]
    for r in rows(OUT/"partial-probes.jsonl"):
        lines.append(f"- `{r['case_id']}`，调用{r['step']}，桥接`{r['bridge_entity']}`，剩余{r['remaining_calls']}调用/{r['remaining_searches']}搜索。")
    lines += ["", "seed42只训练B；B完整评价64题及24完整/8部分链探针；Goal Z和复用A各8部分链；A/B各4训练拟合检查。第一阶段最多128更新、64轨迹、312生成、192搜索。",
        "B需支持≥23/29、完整探针≥22/24、至少一项实体/角色/关系对象修复，错误≤7、动作问题≤1且各项不高于A，保留AM两道新增成功，7保护题最多丢1，部分链动作不退步。",
        "通过才独立训练seed43 A/B；B相同门槛且较同seed A至少增加2题支持成功和2个完整探针正确。",
        "全轮最多3训练/384更新、192轨迹、896生成、576搜索；87个新增准入答案组合；不运行新确认，不发布。",
        "GPU/数值/身份/哈希/基础设施失败保存现场并停止，不自动重跑或改参；语义失败正常计分。未达门槛不运行seed43。", "",
        "收到针对这份实际前提的确认后，写入绑定manifest哈希的新gpu-approval.json，再执行。旧AM授权不能用于AN。", ""]
    (OUT/"PRE_GPU_REVIEW.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(cpu_preflight=True,mask_actions=len(all_actions),gpu_calls=0)))


def client(adapter,corpus):
    import torch
    if torch.cuda.device_count()!=1:raise RuntimeError("AN requires exactly one visible GPU")
    return q.make_client(adapter,deterministic=True,product_config=PC,model_path=BASE,corpus=corpus)


def adapter_for(label):
    if label=="GoalZ":return SOURCE
    if label=="42-A":return AM/"seed42-A/adapter-final"
    if label not in ("42-B","43-A","43-B"):raise ValueError(label)
    p=OUT/f"seed{label}/adapter-final"
    s=read(p.parent/"summary.json")
    if sha(p/"adapter_model.safetensors")!=s["adapter_sha256"]:raise RuntimeError("AN candidate weights mismatch")
    return p


def stage43(label):
    if label.startswith("43-") and not read(OUT/"seed42-gate.json").get("route_passed"):
        raise RuntimeError("AN seed43 gate not passed")


def train(seed,arm):
    cfg=frozen(True);label=f"{seed}-{arm}"
    if label not in ("42-B","43-A","43-B"):raise ValueError("A42 must be reused")
    stage43(label)
    if usage("update")+128>CAPS["update"]:raise RuntimeError("update cap")
    folder=OUT/f"seed{label}";folder.mkdir(exist_ok=False)
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as budget
    if torch.cuda.device_count()!=1:raise RuntimeError("AN requires exactly one GPU")
    set_seed(seed)
    c,w,r=client(SOURCE,am.Z/"corpus-train.json")
    named=c.prepare_for_training();params=[p for _,p in named]
    if any("lora_" not in n for n,_ in named):raise RuntimeError("non-LoRA trainable")
    before=budget.parameter_hashes(c._model,lora=False);lora=budget.parameter_hashes(c._model,lora=True)
    optimizer=torch.optim.AdamW(params,lr=OPT["learning_rate"],betas=tuple(OPT["betas"]),eps=OPT["eps"],weight_decay=OPT["weight_decay"])
    data={r["id"]:r for r in rows(am.Z/"training-inputs.jsonl")}
    states=rows(AM/"candidate-training-states.jsonl");schedule={2+2*i:s for i,s in enumerate(states)}
    order=cfg["training"]["order"][str(seed)]
    for update,cid in enumerate(order,1):
        optimizer.zero_grad(set_to_none=True);c.set_training_forward_mode()
        actions=data[cid]["actions"];n=sum(len(a["action_token_ids"]) for a in actions)
        tl=sum(am.target_loss(c,a,[1.]*len(a["action_token_ids"]),scale=len(a["action_token_ids"])/n) for a in actions)
        state=schedule.get(update);sl=0.;weight=None
        if state:
            weight=normalized_state_weight(state["case_id"],arm)
            sl=am.target_loss(c,state,[1.]*len(state["action_token_ids"]),scale=.25*weight)
        gradient=float(torch.nn.utils.clip_grad_norm_(params,1.))
        if not math.isfinite(gradient) or gradient<=0:raise RuntimeError("invalid gradient")
        record("update",label,cid)
        guarded_optimizer_step(optimizer,params,c.gpu_memory_stats(),21.)
        if any(not torch.isfinite(p).all() for p in params):raise RuntimeError("nonfinite parameter")
        row=dict(update=update,trajectory_id=cid,state_id=state["state_id"] if state else None,
            state_weight=weight,trajectory_loss=tl,state_loss=sl,gradient=gradient,peak_memory=c.gpu_memory_stats())
        with (folder/"updates.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(row)+"\n")
        print(json.dumps(dict(stage=label,**row)),flush=True)
    if budget.parameter_hashes(c._model,lora=False)!=before:raise RuntimeError("base model changed")
    actual_lora=budget.parameter_hashes(c._model,lora=True)
    changed=sum(v!=actual_lora[n] for n,v in lora.items())
    if changed!=192:raise RuntimeError("LoRA update count mismatch")
    adapter=folder/"adapter-final";c._model.save_pretrained(adapter)
    action=data[order[0]]["actions"][-1]
    expected=c.score_action_tensor([],action["action_token_ids"],False,action["prompt_token_ids"],action["prompt_attention_mask"]).detach().float().cpu()
    del c,w,r,named,params,optimizer;gc.collect();torch.cuda.empty_cache()
    reload,_,_=client(adapter,am.Z/"corpus-train.json")
    actual=reload.score_action_tensor([],action["action_token_ids"],False,action["prompt_token_ids"],action["prompt_attention_mask"]).detach().float().cpu()
    delta=float((expected-actual).abs().max())
    if not torch.isfinite(actual).all() or delta>.001:raise RuntimeError("independent reload failed")
    write(folder/"summary.json",dict(seed=seed,arm=arm,updates=128,base_unchanged=True,
        changed_lora_tensors=changed,reload_max_abs=delta,adapter_sha256=sha(adapter/"adapter_model.safetensors")))


def evaluate(label):
    frozen(True);stage43(label)
    if label not in ("42-B","43-A","43-B"):raise ValueError("reuse A42/GoalZ evaluation")
    out=OUT/f"eval-{label}.jsonl"
    if out.exists():raise FileExistsError(out)
    c,w,r=client(adapter_for(label),am.Z/"corpus-dev.json")
    active={"case":"unassigned"};generate=w.generate;execute=r.execute
    async def counted_generate(*args,**kwargs):
        record("generation","eval-"+label,active["case"]);return await generate(*args,**kwargs)
    async def counted_execute(*args,**kwargs):
        if args and args[0]=="search":record("search","eval-"+label,active["case"])
        return await execute(*args,**kwargs)
    w.generate=counted_generate;r.execute=counted_execute
    for i,case in enumerate(rows(AM/"evaluation-cases.jsonl"),1):
        active["case"]=case["case_id"]
        result=asyncio.run(q.rollout(case["question"],w,r,sample=False,seed=42,product_config=PC))
        result.update(case_id=case["case_id"],model=label,official_answer=case["answer"],**q.score(result["answer"],case["answer"]))
        if result["reason"]=="tool_error":raise RuntimeError("retrieval infrastructure failed")
        if result["model_calls"]>4 or result["searches"]>3:raise RuntimeError("rollout budget overflow")
        if any(len(t["prompt_token_ids"])>16384 or len(t["action_token_ids"])>2048 for t in result["trace"]):raise RuntimeError("token overflow")
        with out.open("a",encoding="utf-8") as f:f.write(json.dumps(result,ensure_ascii=False)+"\n")
        print(json.dumps(dict(stage="eval-"+label,case=i,em=result["em"],reason=result["reason"])),flush=True)


def probe(label,kind):
    frozen(True);stage43(label)
    if kind=="complete" and label not in ("42-B","43-A","43-B"):raise ValueError("reuse historical complete probes")
    if kind=="training-fit" and label=="GoalZ":raise ValueError("reuse GoalZ historical fit outputs")
    filename={"partial":"partial-probes.jsonl","complete":"complete-probes.jsonl","training-fit":"training-fit-probes.jsonl"}[kind]
    out=OUT/f"probe-{label}-{kind}.jsonl"
    if out.exists():raise FileExistsError(out)
    import torch
    c,_,_=client(adapter_for(label),am.Z/"corpus-dev.json")
    c._inputs=lambda messages,phase="unknown":dict(input_ids=torch.tensor([current["prompt_token_ids"]],device=c._model.device),
        attention_mask=torch.ones((1,len(current["prompt_token_ids"])),dtype=torch.long,device=c._model.device))
    for current in rows(OUT/filename):
        record("generation",f"probe-{label}-{kind}",current["case_id"])
        start=time.monotonic();r=c._generate_sync([],0.,42);elapsed=time.monotonic()-start
        if r.prompt_token_ids!=current["prompt_token_ids"]:raise RuntimeError("probe input drift")
        if not r.action_token_ids or len(r.action_token_ids)>2048:raise RuntimeError("probe output tokens missing/over limit")
        row=dict(model=label,probe_id=current["probe_id"],case_id=current["case_id"],category=kind,
            action=parsed_action(r),answer=r.content,query=r.tool_call.arguments.get("query") if r.tool_call else None,
            raw=r.raw_text,parse_error=r.parse_error,prompt_token_ids=r.prompt_token_ids,
            prompt_attention_mask=r.prompt_attention_mask,action_token_ids=r.action_token_ids,
            usage=r.usage,elapsed_seconds=elapsed,no_search_execution=True)
        with out.open("a",encoding="utf-8") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
        print(json.dumps(dict(stage=f"probe-{label}-{kind}",probe_id=row["probe_id"],action=row["action"])),flush=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="stage",required=True)
    sub.add_parser("prepare");sub.add_parser("cpu-preflight")
    t=sub.add_parser("train");t.add_argument("--seed",type=int,choices=[42,43],required=True);t.add_argument("--arm",choices=["A","B"],required=True)
    e=sub.add_parser("evaluate");e.add_argument("--model",choices=["42-B","43-A","43-B"],required=True)
    d=sub.add_parser("probe");d.add_argument("--model",choices=["GoalZ","42-A","42-B","43-A","43-B"],required=True)
    d.add_argument("--kind",choices=["complete","partial","training-fit"],required=True)
    a=p.parse_args()
    try:
        if a.stage=="prepare":prepare()
        elif a.stage=="cpu-preflight":cpu_preflight()
        else:
            # Missing approval is a pending state, not a failed experiment.
            frozen(True)
            if a.stage=="train":train(a.seed,a.arm)
            elif a.stage=="evaluate":evaluate(a.model)
            else:probe(a.model,a.kind)
    except BaseException as exc:
        if OUT.exists() and not (OUT/"experiment-failure.json").exists() and (
            a.stage in ("prepare","cpu-preflight") or (OUT/"gpu-approval.json").exists()):
            write(OUT/"experiment-failure.json",dict(stage=a.stage,error_type=type(exc).__name__,error=str(exc),
                timestamp=time.time(),updates=usage("update"),generations=usage("generation"),searches=usage("search")))
        raise

if __name__=="__main__":main()
