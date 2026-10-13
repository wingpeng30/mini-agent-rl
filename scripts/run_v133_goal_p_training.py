"""Goal P: bounded temperature-1 SFT / RL / joint comparison from v118."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import run_v133_goal_n_v118_grpo as n
import run_v133_goal_o_budget as o

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-p-system/p3-training"
CFG = OUT / "config-frozen.json"
STATE = OUT / "state.json"
ARMS = ("sft", "rl", "joint")


def prepare():
    if CFG.exists() or STATE.exists():
        raise FileExistsError("P3 training already prepared")
    baseline = json.loads((o.OUT/"config-frozen.json").read_text(encoding="utf-8"))
    paths = {"runner":Path(__file__), "goal_n_runner":Path(n.__file__),
             "goal_o_runner":Path(o.__file__), "grpo":ROOT/"src/mini_agent_rl/training/grpo.py",
             "sft":ROOT/"src/mini_agent_rl/training/sft.py",
             "local":ROOT/"src/mini_agent_rl/model/local.py",
             "train":n.PATHS["train"], "new_eval":n.PATHS["new_eval"],
             "goal_G_regression":n.PATHS["goal_G_regression"],
             "v118":n.PATHS["v118"],"goal_J_sft":n.PATHS["goal_J_sft"],
             "base_config":n.BASE/"config.json", "tokenizer":n.BASE/"tokenizer.json"}
    c={**baseline,"version":"v133-goal-p-training-v1","temperature":1.0,"top_p":1.0,"top_k":0,
       "seed":42,"seeds":[42],"passes":4,"arms":list(ARMS),"sft_aux_weight":0.1,
       "max_groups_per_arm":64,"max_updates_per_arm":64,"max_train_seconds_per_arm":1800,
       "max_probe_seconds":600,"max_eval_seconds":900,
       "hashes":{key:o.sha(value) for key,value in paths.items()},
       "distribution_note":"temperature=1 top_p=1 top_k=0; sampling and raw teacher-forced logprob agree",
       "comparison_note":"SFT, RL and joint have different rollout costs and can have different effective update counts."}
    expected = {"train":"d4b41cb3413c9fca736d26b30e6d5e5292c1b47308d21c01f174f8957e0b263b",
                "new_eval":"42531fc1dbd2c3cb1fbec12e85523483c1c0b40546c55a0a6d622474d9648d29",
                "goal_G_regression":"f689f94b598a1c0d89f1b87f0b008c8a03158a7ab479028368285564244aad69",
                "v118":"ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5",
                "goal_J_sft":"21b24dc960c3915e9db89626358b2271d8e37ff8b6e0205d3ab40ddb5eca0fc2"}
    for key,value in expected.items():
        if c["hashes"][key]!=value:
            raise RuntimeError(f"frozen source mismatch: {key}")
    if json.loads((o.OUT/"status.json").read_text(encoding="utf-8"))["status"]!="EXPERIMENT_COMPLETE":
        raise RuntimeError("Goal O source did not complete")
    from transformers import AutoProcessor
    from mini_agent_rl.training.sft import encode_assistant_only
    processor=AutoProcessor.from_pretrained(n.BASE,local_files_only=True)
    counts=[]
    for row in o.rows(n.PATHS["train"]):
        encoded=encode_assistant_only(processor,row["messages"],2176,row["target_assistant_index"],False)
        prefix=row["cpu_check"]["prompt_tokens"]
        if any(x!=-100 for x in encoded["labels"][:prefix]) or sum(x!=-100 for x in encoded["labels"])!=row["cpu_check"]["target_tokens"]:
            raise RuntimeError(f"SFT target mask mismatch {row['case_id']}")
        counts.append(sum(x!=-100 for x in encoded["labels"]))
    if len(counts)!=16: raise RuntimeError("SFT training count mismatch")
    c["sft_target_token_range"]=[min(counts),max(counts)]
    o.write(CFG,c)
    o.write(STATE,{"status":"CPU_READY","stage":"P3-F1","gpu_seconds_used":0.0,"config_sha256":o.sha(CFG)})
    print(json.dumps({"status":"CPU_READY","config_sha256":o.sha(CFG)}))


def config():
    c=json.loads(CFG.read_text(encoding="utf-8"))
    paths={"runner":Path(__file__),"goal_n_runner":Path(n.__file__),"goal_o_runner":Path(o.__file__),
           "grpo":ROOT/"src/mini_agent_rl/training/grpo.py","sft":ROOT/"src/mini_agent_rl/training/sft.py",
           "local":ROOT/"src/mini_agent_rl/model/local.py","train":n.PATHS["train"],
           "new_eval":n.PATHS["new_eval"],"goal_G_regression":n.PATHS["goal_G_regression"],
           "v118":n.PATHS["v118"],"goal_J_sft":n.PATHS["goal_J_sft"],
           "base_config":n.BASE/"config.json","tokenizer":n.BASE/"tokenizer.json"}
    for key,path in paths.items():
        if o.sha(path)!=c["hashes"][key]:
            raise RuntimeError(f"P3 frozen asset changed: {key}")
    return c


def sft_loss(client,row):
    import torch
    from mini_agent_rl.training.sft import encode_assistant_only
    encoded=encode_assistant_only(client._processor,row["messages"],2176,row["target_assistant_index"],False)
    labels=encoded["labels"]
    if sum(value!=-100 for value in labels)<2 or any(value!=-100 for value in labels[:row["cpu_check"]["prompt_tokens"]]):
        raise RuntimeError("SFT prefix/target mask invalid")
    batch={name:torch.tensor([value],dtype=torch.long,device=client._model.device) for name,value in encoded.items()}
    client._model.config.use_cache=False
    loss=client._model(**batch).loss
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite SFT loss")
    return loss,int(sum(value!=-100 for value in labels))


def probe():
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    c=config(); path=OUT/"probe.json"
    if path.exists(): raise FileExistsError("probe already started")
    started=time.monotonic();set_seed(c["seed"])
    client=n.load_client(n.V118,c,training=True)
    named=client.prepare_for_training();params=[p for _,p in named]
    before=o.parameter_hashes(client._model,lora=True)
    base=o.parameter_hashes(client._model,lora=False)
    optimizer=torch.optim.AdamW(params,lr=c["learning_rate"],betas=tuple(c["betas"]),
                                eps=c["optimizer_eps"],weight_decay=c["weight_decay"])
    data=o.rows(n.PATHS["train"]);samples=[]
    for row in data[:2]:
        if time.monotonic()-started>=c["max_probe_seconds"]: raise TimeoutError("probe cap")
        group=[n.generate(client,row,c,sample=True) for _ in range(4)]
        samples.append({"case_id":row["case_id"],"em":[x["em"] for x in group],
                        "reward":[x["reward"] for x in group],"summary":n.group_summary(group)})
    loss_before,target_tokens=sft_loss(client,data[0])
    optimizer.zero_grad(set_to_none=True);loss_before.backward()
    norm=float(torch.nn.utils.clip_grad_norm_(params,c["max_grad_norm"]))
    if not math.isfinite(norm) or norm<=0: raise RuntimeError("probe SFT gradient invalid")
    guarded_optimizer_step(optimizer,params,client.gpu_memory_stats(),c["max_reserved_memory_gb"])
    loss_after,_=sft_loss(client,data[0])
    lora_after=o.parameter_hashes(client._model,lora=True)
    change=sum(before[name]!=lora_after[name] for name in before)
    if change<=0 or o.parameter_hashes(client._model,lora=False)!=base:
        raise RuntimeError("probe LoRA/base gate failed")
    result={"status":"PASSED","seed":42,"prompts":2,"candidates":8,"updates":1,
            "samples":samples,"sft_loss_before":float(loss_before.detach()),
            "sft_loss_after":float(loss_after.detach()),"sft_target_tokens":target_tokens,
            "grad_norm_preclip":norm,"changed_lora_tensors":change,
            "trainable_dtypes":sorted({str(p.dtype) for p in params}),
            "gpu_memory":client.gpu_memory_stats(),"elapsed_seconds":round(time.monotonic()-started,3)}
    o.write(path,result);print(json.dumps({"status":"PASSED","seconds":result["elapsed_seconds"],
                                            "loss_before":result["sft_loss_before"],
                                            "loss_after":result["sft_loss_after"]}))


def train_arm(arm):
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import clipped_grpo_loss,guarded_optimizer_step,require_initial_ratio,ratio_statistics
    c=config()
    if arm not in ARMS: raise ValueError("unknown arm")
    folder=OUT/arm
    if folder.exists(): raise FileExistsError(f"arm {arm} already started")
    folder.mkdir(parents=True)
    set_seed(c["seed"]);started=time.monotonic()
    client=n.load_client(n.V118,c,training=True)
    named=client.prepare_for_training();params=[p for _,p in named]
    reference={name:value.detach().clone() for name,value in named}
    ref_before=o.reference_hashes(reference)
    base_before=o.parameter_hashes(client._model,lora=False)
    lora_before=o.parameter_hashes(client._model,lora=True)
    optimizer=torch.optim.AdamW(params,lr=c["learning_rate"],betas=tuple(c["betas"]),
                                eps=c["optimizer_eps"],weight_decay=c["weight_decay"])
    data=o.rows(n.PATHS["train"])
    if len(data)!=16: raise RuntimeError("training length changed")
    groups=0;updates=0;sample_count=0;check_item=None
    for epoch in range(1,5):
        for row in data:
            if time.monotonic()-started>=c["max_train_seconds_per_arm"]:
                raise TimeoutError("per-arm training cap")
            optimizer.zero_grad(set_to_none=True)
            item={"arm":arm,"pass":epoch,"case_id":row["case_id"],"group_index":groups+1}
            do_rl=False;rl_loss_value=0.0;kl_value=0.0
            if arm!="sft":
                client._model.eval()
                group=[]
                for candidate in range(4):
                    sample=n.generate(client,row,c,sample=True)
                    sample["candidate"]=candidate
                    sample["pass"]=epoch
                    sample["arm"]=arm
                    sample["old_logprobs"]=client.score_action_tensor([],sample["action_token_ids"],False,
                        sample["prompt_token_ids"],sample["prompt_attention_mask"]).detach().float().cpu().tolist()
                    o.append(folder/"rollouts.jsonl",sample)
                    group.append(sample)
                sample_count+=4
                check_item=group[0]
                summary=n.group_summary(group)
                item.update(summary)
                do_rl=bool(summary["reward_variable"])
                if do_rl:
                    olds=[];refs=[]
                    for sample in group:
                        old=client.score_action_tensor([],sample["action_token_ids"],False,
                            sample["prompt_token_ids"],sample["prompt_attention_mask"]).detach()
                        ref=client.reference_logprobs([],sample["action_token_ids"],reference,
                            sample["prompt_token_ids"],sample["prompt_attention_mask"]).detach()
                        o.check_replay(client,sample,old,ref)
                        olds.append(old);refs.append(ref)
                    client.set_training_forward_mode()
                    initial=[];losses=[];kls=[]
                    for sample,old,ref in zip(group,olds,refs):
                        current=client.score_action_tensor([],sample["action_token_ids"],True,
                            sample["prompt_token_ids"],sample["prompt_attention_mask"])
                        initial.append(torch.exp(current.detach()-old).float().cpu())
                        advantage=(sample["reward"]-summary["mean_reward"])/max(summary["reward_std_population"],1e-8)
                        loss,kl,_=clipped_grpo_loss(current,old,ref,advantage,c["clip_epsilon"],c["kl_beta"])
                        if not torch.isfinite(loss): raise RuntimeError("nonfinite RL loss")
                        (loss/4).backward()
                        losses.append(float(loss.detach()));kls.append(float(kl))
                    require_initial_ratio(ratio_statistics(initial),label=f"{arm} epoch {epoch}")
                    rl_loss_value=sum(losses)/4;kl_value=sum(kls)/4
            sft_value=None;target_tokens=0
            if arm in ("sft","joint"):
                sft,target_tokens=sft_loss(client,row)
                (sft if arm=="sft" else c["sft_aux_weight"]*sft).backward()
                sft_value=float(sft.detach())
            updated=arm!="rl" or do_rl
            if updated:
                grad_norm=float(torch.nn.utils.clip_grad_norm_(params,c["max_grad_norm"]))
                if not math.isfinite(grad_norm) or grad_norm<=0:
                    raise RuntimeError("nonfinite or zero training gradient")
                memory=client.gpu_memory_stats()
                guarded_optimizer_step(optimizer,params,memory,c["max_reserved_memory_gb"])
                updates+=1
                item.update(grad_norm=grad_norm,gpu_memory=memory)
            item.update(updated=updated,rl_updated=do_rl,sft_loss=sft_value,
                        sft_target_tokens=target_tokens,rl_loss=rl_loss_value,kl=kl_value)
            groups+=1
            if groups>64 or updates>64 or sample_count>(0 if arm=="sft" else 256):
                raise RuntimeError("arm budget exceeded")
            o.append(folder/"groups.jsonl",item)
        print(json.dumps({"arm":arm,"pass":epoch,"updates":updates,"groups":groups}),flush=True)
    if groups!=64 or updates<=0: raise RuntimeError("arm incomplete")
    if o.parameter_hashes(client._model,lora=False)!=base_before or o.reference_hashes(reference)!=ref_before:
        raise RuntimeError("base/reference changed")
    lora_after=o.parameter_hashes(client._model,lora=True)
    changed=sum(lora_before[name]!=lora_after[name] for name in lora_before)
    if changed<=0: raise RuntimeError("LoRA unchanged")
    checkpoint=folder/"adapter-final"
    client._model.save_pretrained(checkpoint)
    if check_item is None:
        # For SFT-only, the exact frozen training prompt and target answer are
        # generated by the validated tokenizer, then scored after reload.
        encoded=client._processor.apply_chat_template(data[0]["messages"][:data[0]["target_assistant_index"]],
            tokenize=True,add_generation_prompt=True,enable_thinking=False)
        prompt=list(encoded[0] if encoded and isinstance(encoded[0],list) else encoded)
        target=client._processor.tokenizer.encode(data[0]["messages"][data[0]["target_assistant_index"]]["content"],
            add_special_tokens=False)
        check_item={"prompt_token_ids":prompt,"prompt_attention_mask":[1]*len(prompt),"action_token_ids":target}
    before=client.score_action_tensor([],check_item["action_token_ids"],False,
        check_item["prompt_token_ids"],check_item["prompt_attention_mask"]).detach().float().cpu()
    del client,optimizer,params,reference
    gc.collect();torch.cuda.empty_cache()
    loaded=n.load_client(checkpoint,c)
    after=loaded.score_action_tensor([],check_item["action_token_ids"],False,
        check_item["prompt_token_ids"],check_item["prompt_attention_mask"]).detach().float().cpu()
    delta=float((before-after).abs().max())
    if delta>0.001: raise RuntimeError(f"{arm} reload logprob mismatch {delta}")
    result={"status":"PASSED","arm":arm,"groups":groups,"updates":updates,
            "sample_count":sample_count,"changed_lora_tensors":changed,"base_unchanged":True,
            "reference_unchanged":True,"reload_max_abs":delta,"checkpoint_sha256":o.sha(checkpoint/"adapter_model.safetensors"),
            "elapsed_seconds":round(time.monotonic()-started,3)}
    o.write(folder/"summary.json",result);print(json.dumps(result),flush=True)


def evaluate():
    import torch
    from transformers import set_seed
    c=config();path=OUT/"evaluation.jsonl"
    if path.exists(): raise FileExistsError("P3 evaluation already started")
    for arm in ARMS:
        if json.loads((OUT/arm/"summary.json").read_text(encoding="utf-8"))["status"]!="PASSED":
            raise RuntimeError(f"{arm} training incomplete")
    started=time.monotonic();models=[("v118",n.V118),("goal_J_sft",n.J_ADAPTER)]
    models.extend((arm,OUT/arm/"adapter-final") for arm in ARMS)
    for name,adapter in models:
        set_seed(42);client=n.load_client(adapter,c)
        for split in ("train","new_eval","goal_G_regression"):
            for row in o.rows(n.PATHS[split]):
                if time.monotonic()-started>=c["max_eval_seconds"]:
                    raise TimeoutError("P3 evaluation cap")
                value=n.generate(client,row,c,sample=False)
                o.append(path,{"arm":name,"split":split,**{key:value[key] for key in
                    ("case_id","raw","action","prediction","em","f1","prompt_tokens","action_tokens","generation_seconds")}})
        del client;gc.collect();torch.cuda.empty_cache()
    values=o.rows(path)
    if len(values)!=240: raise RuntimeError("P3 evaluation incomplete")
    result={"status":"PASSED","rows":len(values),"elapsed_seconds":round(time.monotonic()-started,3),"sets":{}}
    for name,_ in models:
        result["sets"][name]={}
        for split in ("train","new_eval","goal_G_regression"):
            part=[x for x in values if x["arm"]==name and x["split"]==split]
            result["sets"][name][split]={"n":len(part),"em":sum(x["em"] for x in part),
                "f1":sum(x["f1"] for x in part)/len(part),
                "actions":{action:sum(x["action"]==action for x in part) for action in ("answer","search","invalid")}}
    o.write(OUT/"evaluation-summary.json",result)
    print(json.dumps({"status":"PASSED","rows":240,"seconds":result["elapsed_seconds"]}),flush=True)


def run():
    c=config();state=json.loads(STATE.read_text(encoding="utf-8"))
    if state["status"]!="CPU_READY": raise RuntimeError("P3 run already started")
    started=time.monotonic();state.update(status="RUNNING",stage="P3-F1")
    o.write(STATE,state)
    stages=[("probe",["probe"],c["max_probe_seconds"])]
    stages.extend((arm,["train-arm","--arm",arm],c["max_train_seconds_per_arm"]) for arm in ARMS)
    stages.append(("evaluate",["evaluate"],c["max_eval_seconds"]))
    for stage,args,cap in stages:
        state["stage"]=stage;o.write(STATE,state)
        try:
            done=subprocess.run([sys.executable,str(Path(__file__)),*args],cwd=ROOT,timeout=cap,check=False)
        except subprocess.TimeoutExpired:
            state.update(status="TECHNICAL_BLOCKED",error=f"{stage} timeout");break
        if done.returncode:
            state.update(status="TECHNICAL_BLOCKED",error=f"{stage} exited {done.returncode}");break
        expected=(OUT/"probe.json" if stage=="probe" else OUT/"evaluation-summary.json"
                  if stage=="evaluate" else OUT/stage/"summary.json")
        if not expected.exists() or json.loads(expected.read_text(encoding="utf-8")).get("status")!="PASSED":
            state.update(status="TECHNICAL_BLOCKED",error=f"{stage} completion record missing");break
    else:
        state.update(status="EXPERIMENT_COMPLETE",stage="P3-F3")
    state["gpu_seconds_used"]=round(time.monotonic()-started,3)
    o.write(STATE,state);print(json.dumps(state),flush=True)
    if state["status"]!="EXPERIMENT_COMPLETE": raise RuntimeError(state["error"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","probe","run","train-arm","evaluate"])
    parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    if args.command=="train-arm":
        if not args.arm: parser.error("--arm required")
        train_arm(args.arm)
    else:
        if args.arm: parser.error("--arm only for train-arm")
        globals()[args.command]()


if __name__=="__main__": main()
