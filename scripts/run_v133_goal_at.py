"""Goal AT finite attainability check on existing, isolated training states."""
from __future__ import annotations
import argparse,asyncio,json,math,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import run_v133_goal_ap as ap
from mini_agent_rl.domain import Message
from mini_agent_rl.model.grounded_action_audit import map_documents
OUT=ROOT/"runs/v133-goal-at-supported-action-learning-v1"
sha,rows,read,write=ap.sha,ap.rows,ap.read,ap.write_once
HARD=["ai-answer-correction:musique:2hop__407314_80178","ai-answer-correction:2wiki:b2addc160bdc11eba7f7acde48001122",
      "ai-answer-correction:2wiki:a874fa560bda11eba7f7acde48001122","ai-answer-correction:2wiki:fa9654640bdb11eba7f7acde48001122"]
# Exact equivalences fixed from the existing admitted targets, not sampled outputs.
ALIASES={
 "Jaden Smith":["Jaden Smith"],"University of Helsinki":["University of Helsinki","Helsinki University"],
 "Christiansburg, Virginia":["Christiansburg, Virginia","Christiansburg"],
 "September 3, 1922":["September 3, 1922","3 September 1922","1922-09-03"],
 "Audrey Hepburn":["Audrey Hepburn"],"Parliamentary privilege":["Parliamentary privilege"],
 "English":["English"],"August 1, 1876":["August 1, 1876","1 August 1876","1876-08-01"],
 "Clark County":["Clark County","Clark"],"Emilio Del Bono":["Emilio Del Bono"],
 "February 4, 1948":["February 4, 1948","4 February 1948","1948-02-04"],
 "1963":["1963"],"Margaret Davies":["Margaret Davies"],
 "14,000,000 square kilometres":["14,000,000 square kilometres","14,000,000 square kilometers","14 million square kilometres","14 million square kilometers","14 million sq km"],
 "24 September 1953":["24 September 1953","September 24, 1953","1953-09-24"],
 "6 October 877":["6 October 877","October 6, 877","0877-10-06"],
}
PC=dict(ap.PC,observation_presentation="legacy-json",action_constraint="none")


def client_for():
    return ap.q.make_client(ap.AMADAPTER,deterministic=True,product_config=PC,model_path=ap.BASE,corpus=ap.Z/"corpus-train.json")


def prepare():
    if (OUT/"manifest.json").exists():raise FileExistsError("AT already frozen")
    ar=read(ROOT/"runs/v133-goal-ar-constrained-evidence-v1/final-status.json")
    if ar["research_parent"]!="AM42A" or ar["status"]!="completed_rejected_at_probe_gate":raise RuntimeError("AT parent must be recalculated")
    if not read(ROOT/"runs/v133-goal-as-frozen-recovery-inventory-v1/status.json")["skipped"]:raise RuntimeError("AS must close first")
    if sha(ap.AMADAPTER/"adapter_model.safetensors")!=ap.AM_HASH:raise RuntimeError("AT parent identity")
    source=rows(ap.AM/"candidate-training-states.jsonl");by={s["state_id"]:s for s in source}
    selected=[by[k] for k in HARD]+sorted([s for s in source if s["state_type"]=="answer" and s["state_id"] not in HARD],key=lambda s:s["state_id"])[:12]
    if len(selected)!=16 or len({s["case_id"] for s in selected})!=16:raise RuntimeError("AT fixed states")
    eval_ids,eval_questions,_=ap.am.eval_isolation();ap.am.validate_state_isolation(selected,eval_ids,eval_questions)
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(ap.BASE),local_files_only=True);tokenizer=getattr(processor,"tokenizer",processor)
    client,_,_=client_for();client._processor=processor;client._model=SimpleNamespace(device="cpu")
    records=[]
    for s in selected:
        messages=ap.restore_tool_messages(ap.parse_archived_messages(tokenizer,s["prompt_token_ids"]))
        encoded=client._inputs([Message(**m) for m in messages],"at_cpu_prepare")
        if encoded["input_ids"][0].tolist()!=s["prompt_token_ids"]:raise RuntimeError("training-state input reconstruction")
        if any(encoded._context_usage.get(k) for k in ("prompt_truncated","lossy_body_trimmed","history_dropped","documents_deleted")):raise RuntimeError("AT clipped input")
        docs,offsets=map_documents(tokenizer,s["prompt_token_ids"],messages)
        mapped=[]
        for c in s["citations"]:
            matches=[d for d in docs if d["document_id"]==c["document_id"] and c["quote"] in d["text"]]
            if not matches:raise RuntimeError("existing admitted citation absent from actual body")
            mapped.append(dict(citation=c,actual_document_occurrences=[{k:v for k,v in d.items() if k not in ("text","serialized_body")} for d in matches]))
        target=s["target_action"]["answer"]
        if target not in ALIASES:raise RuntimeError("unfrozen reward alias set")
        records.append(dict(s,messages=messages,actual_citations=mapped,reward_aliases=ALIASES[target],
            reward_scope="source-supported correct answer and required qualifiers; unknown alternatives require one-pass adjudication",evaluation_use_forbidden=True))
    write(OUT/"frozen-states.jsonl",records,jsonl=True)
    write(OUT/"gpu-usage.jsonl",[],jsonl=True)
    paths=ap.source_paths();paths.update(runner=Path(__file__),states=OUT/"frozen-states.jsonl",plan=OUT/"PLAN.md",
        source_states=ap.AM/"candidate-training-states.jsonl",source_semantic=ap.AM/"PRE_GPU_REVIEW_SOURCE.md",
        ar_closure=ROOT/"runs/v133-goal-ar-constrained-evidence-v1/final-status.json",
        as_closure=ROOT/"runs/v133-goal-as-frozen-recovery-inventory-v1/status.json")
    write(OUT/"manifest.json",dict(goal="AT",parent="AM42A",phase="attainability",product=PC,
        sampling=dict(temperature=.7,top_p=1.,top_k=0,candidates_per_state=4,seed_base=420000),
        limits=dict(generations=640,searches=384,updates=256,attainability_generations=64),
        paths={k:str(v) for k,v in paths.items()},hashes={k:sha(v) for k,v in paths.items()},
        reward_requires_semantics=True,no_search_during_sampling=True,training_requires_mixed_groups=4))
    write(OUT/"cpu-preflight.json",dict(passed=True,states=16,exact_inputs=16,source_citations=len([c for s in records for c in s["citations"]]),evaluation_isolation=True))
    print(json.dumps(read(OUT/"cpu-preflight.json")),flush=True)


def frozen():
    if (OUT/"experiment-failure.json").exists():raise RuntimeError("AT failure latch")
    m=read(OUT/"manifest.json")
    for k,p in m["paths"].items():
        if sha(p)!=m["hashes"][k]:raise RuntimeError("AT source drift "+k)
    return m


def sample():
    m=frozen();path=OUT/"samples.jsonl"
    if path.exists():raise FileExistsError(path)
    import torch
    if torch.cuda.device_count()!=1:raise RuntimeError("AT single GPU")
    client,wrapped,_=client_for();client._load()
    client._model.generation_config.top_p=m["sampling"]["top_p"]
    client._model.generation_config.top_k=m["sampling"]["top_k"]
    # Raw-policy logprobs are diagnostics only. A subsequent policy update must
    # recompute actual temperature-adjusted probabilities, never treat these as
    # the .7 sampling distribution.
    client.config.recompute_old_logprobs=True
    write(OUT/"actual-generation-config.json",client._model.generation_config.to_dict())
    for n,s in enumerate(rows(OUT/"frozen-states.jsonl")):
        for candidate in range(4):
            count=len(rows(OUT/"gpu-usage.jsonl"))
            if count>=64:raise RuntimeError("AT frozen sampling cap")
            seed=420000+100*n+candidate
            ap.append(OUT/"gpu-usage.jsonl",dict(kind="generation",stage="attainability",state_id=s["state_id"],candidate=candidate,seed=seed,status="started"))
            start=time.monotonic();r=asyncio.run(wrapped.generate([Message(**x) for x in s["messages"]],[],temperature=.7,seed=seed))
            if r.prompt_token_ids!=s["prompt_token_ids"]:raise RuntimeError("AT GPU input drift")
            if len(r.action_token_ids)>2048 or len(r.prompt_token_ids)>16384 or not all(math.isfinite(x) for x in r.old_logprobs):raise RuntimeError("AT numeric/length failure")
            action=ap.an.parsed_action(r)
            norm=ap.q.normalize_answer
            alias_match=action=="answer" and norm(r.content) in {norm(x) for x in s["reward_aliases"]}
            reward=1 if alias_match else 0 if action!="answer" else None
            ap.append(path,dict(state_id=s["state_id"],case_id=s["case_id"],candidate=candidate,seed=seed,
                action=action,answer=r.content,query=r.tool_call.arguments.get("query") if r.tool_call else None,
                raw=r.raw_text,parse_error=r.parse_error,prompt_token_ids=r.prompt_token_ids,prompt_attention_mask=r.prompt_attention_mask,
                action_token_ids=r.action_token_ids,raw_policy_logprobs=r.old_logprobs,usage=r.usage,preliminary_reward=reward,
                preliminary_reason="Exact predeclared supported alias" if alias_match else "Sufficient evidence but no answer action" if reward==0 else "Requires source-grounded alternate-answer review",
                elapsed_seconds=time.monotonic()-start,training_only=True,no_search_execution=True))
            print(json.dumps(dict(state=s["state_id"],candidate=candidate,action=action,answer=r.content,reward=reward)),flush=True)
    write(OUT/"sampling-complete.json",dict(generations=64,searches=0,updates=0))


def gate():
    frozen();samples=rows(OUT/"samples.jsonl");reviews=rows(OUT/"reward-adjudications.jsonl")
    by={(r["state_id"],r["candidate"]):r for r in reviews}
    if len(samples)!=64 or len(reviews)!=64 or len(by)!=64 or set(by)!={(s["state_id"],s["candidate"]) for s in samples}:raise RuntimeError("AT reward review coverage")
    for r in reviews:
        if r.get("reward") not in (0,1) or not r.get("reason") or r.get("source_supported")!=(r["reward"]==1):raise RuntimeError("AT uncertain reward")
    mixed=[];summary=[]
    for s in rows(OUT/"frozen-states.jsonl"):
        values=[by[(s["state_id"],i)]["reward"] for i in range(4)]
        summary.append(dict(state_id=s["state_id"],correct=sum(values),candidates=4))
        if len(set(values))==2:mixed.append(s["state_id"])
    write(OUT/"attainability-gate.json",dict(passed=len(mixed)>=4,mixed_groups=mixed,mixed_count=len(mixed),
        summary=summary,total_correct=sum(r["reward"] for r in reviews),reward_review_sha256=sha(OUT/"reward-adjudications.jsonl"),
        decision="prepare_matched_training" if len(mixed)>=4 else "skip_training_no_extra_sampling"))
    print(json.dumps(read(OUT/"attainability-gate.json")),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","sample","gate"]);a=p.parse_args()
    try:globals()[a.stage]()
    except BaseException as exc:
        if OUT.exists() and not (OUT/"experiment-failure.json").exists():write(OUT/"experiment-failure.json",dict(stage=a.stage,error_type=type(exc).__name__,error=str(exc),stop_no_auto_rerun=True))
        raise
if __name__=="__main__":main()
