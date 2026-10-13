"""CPU-only score audit, paired error analysis, and curves for Goal P."""
from __future__ import annotations

import json
import hashlib
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/v133-goal-p-system"
TRAIN=OUT/"p3-training"
sys.path.insert(0,str(ROOT/"scripts"))


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(value) for value in Path(path).read_text(encoding="utf-8").splitlines() if value]


def delta(left,right):
    a={row["case_id"]:row for row in left}
    b={row["case_id"]:row for row in right}
    if a.keys()!=b.keys(): raise RuntimeError("paired IDs differ")
    return {"gained":[key for key in sorted(a) if not a[key]["em"] and b[key]["em"]],
            "lost":[key for key in sorted(a) if a[key]["em"] and not b[key]["em"]]}


def main():
    from evaluate_v133_goal_c_sft import normalize_answer,f1_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases=lines(OUT/"eval-only-cases.jsonl")
    gold={(row["split"],row["case_id"]):row["official_answer"] for row in cases}
    gold.update({("train",row["case_id"]):row["official_answer"]
                 for row in lines(ROOT/"runs/v133-goal-j-transfer/train-frozen.jsonl")})
    baseline=lines(OUT/"p1-baseline.jsonl")
    candidate=lines(OUT/"p2-candidate-1.jsonl")
    if len(baseline)!=32 or len(candidate)!=32: raise RuntimeError("system rows incomplete")
    for values in (baseline,candidate):
        for row in values:
            answer=row["answer"] or ""
            em=int(bool(answer) and normalize_answer(answer)==normalize_answer(gold[(row["split"],row["case_id"])]))
            f1=f1_score(answer,gold[(row["split"],row["case_id"])]) if answer else 0.0
            if row["em"]!=em or abs(row["f1"]-f1)>1e-9:
                raise RuntimeError("system score mismatch")
    p2={"baseline":{"em":sum(x["em"] for x in baseline),"f1":statistics.mean(x["f1"] for x in baseline),
                    "invalid":sum(x["reason"]=="invalid_action" for x in baseline),
                    "both_support":sum(x["both_support_retrieved"] for x in baseline)},
        "candidate":{"em":sum(x["em"] for x in candidate),"f1":statistics.mean(x["f1"] for x in candidate),
                    "invalid":sum(x["reason"]=="invalid_action" for x in candidate),
                    "both_support":sum(x["both_support_retrieved"] for x in candidate)},
        "paired":delta(baseline,candidate)}
    p2["retained"]=(p2["candidate"]["invalid"]<p2["baseline"]["invalid"]
        and p2["candidate"]["em"]>=p2["baseline"]["em"]
        and p2["candidate"]["f1"]>=p2["baseline"]["f1"])
    evaluation=lines(TRAIN/"evaluation.jsonl")
    if len(evaluation)!=240: raise RuntimeError("training evaluation incomplete")
    by={}
    for row in evaluation:
        answer=row["prediction"] if row["action"]=="answer" else ""
        em=int(bool(answer) and normalize_answer(answer)==normalize_answer(gold[(row["split"],row["case_id"])]))
        f1=f1_score(answer,gold[(row["split"],row["case_id"])]) if answer else 0.0
        if em!=row["em"] or abs(f1-row["f1"])>1e-9:
            raise RuntimeError(f"training evaluation score mismatch {row['arm']}/{row['case_id']}")
        by.setdefault((row["arm"],row["split"]),[]).append(row)
    frozen=load(TRAIN/"evaluation-summary.json")["sets"]
    for (arm,split),part in by.items():
        if len(part)!=16 or frozen[arm][split]["em"]!=sum(x["em"] for x in part):
            raise RuntimeError("evaluation summary mismatch")
    paired={arm:{split:delta(by[("goal_J_sft",split)],by[(arm,split)])
                 for split in ("train","new_eval","goal_G_regression")}
            for arm in ("sft","rl","joint")}
    pair_vs_sft={split:delta(by[("sft",split)],by[("joint",split)])
                 for split in ("train","new_eval","goal_G_regression")}
    difficult=[x["case_id"] for x in load(OUT/"p3-f0-signal.json")["never_strict_correct_cases"]]
    hard={arm:[key for key in difficult if any(x["case_id"]==key and x["em"] for x in by[(arm,"train")])]
          for arm in ("v118","goal_J_sft","sft","rl","joint")}
    curves={}
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for arm in ("sft","rl","joint"):
        groups=lines(TRAIN/arm/"groups.jsonl")
        if len(groups)!=64: raise RuntimeError("training groups incomplete")
        engineering=load(TRAIN/arm/"summary.json")
        rollouts=lines(TRAIN/arm/"rollouts.jsonl") if arm!="sft" else []
        if (engineering["status"]!="PASSED" or engineering["updates"]!=sum(x["updated"] for x in groups)
                or engineering["updates"]>64 or len(rollouts)!=(0 if arm=="sft" else 256)
                or not engineering["base_unchanged"] or not engineering["reference_unchanged"]
                or engineering["reload_max_abs"]>0.001):
            raise RuntimeError(f"{arm} engineering gate mismatch")
        checkpoint=TRAIN/arm/"adapter-final/adapter_model.safetensors"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=engineering["checkpoint_sha256"]:
            raise RuntimeError(f"{arm} checkpoint hash mismatch")
        if arm=="rl" and any(x["updated"]!=x["reward_variable"] for x in groups):
            raise RuntimeError("RL zero-variance skip mismatch")
        if arm=="joint" and any(not x["updated"] for x in groups):
            raise RuntimeError("joint zero-variance SFT update missing")
        curve=[]
        for epoch in range(1,5):
            chunk=[g for g in groups if g["pass"]==epoch]
            if len(chunk)!=16: raise RuntimeError("pass incomplete")
            curve.append({"pass":epoch,"updates":sum(x["updated"] for x in chunk),
                "rl_updates":sum(x["rl_updated"] for x in chunk),
                "mean_sft_loss":statistics.mean(x["sft_loss"] for x in chunk if x["sft_loss"] is not None) if arm!="rl" else None,
                "mean_sampled_reward":statistics.mean(x["mean_reward"] for x in chunk) if arm!="sft" else None,
                "mean_kl_updated":statistics.mean(x["kl"] for x in chunk if x["rl_updated"]) if any(x["rl_updated"] for x in chunk) else None,
                "max_reserved_gib":max(x["gpu_memory"]["peak_reserved_gb"] for x in chunk if x["updated"])})
        curves[arm]=curve
        for ax,key in zip(axes,("updates","mean_sampled_reward","mean_sft_loss")):
            if curve[0][key] is not None:
                ax.plot(range(1,5),[x[key] for x in curve],marker="o",label=arm)
    for ax,title in zip(axes,("Updates","Mean sampled reward","Mean SFT loss")):
        ax.set_title(title);ax.set_xticks(range(1,5));ax.set_xlabel("Training pass");ax.grid(alpha=.25);ax.legend()
    fig.savefig(OUT/"p3-training-curves.png",dpi=160);plt.close(fig)
    new=frozen["joint"];sft=frozen["sft"];j=frozen["goal_J_sft"]
    gates={"joint_main_at_least_sft":new["new_eval"]["em"]>=sft["new_eval"]["em"],
           "joint_regression_at_least_sft":new["goal_G_regression"]["em"]>=sft["goal_G_regression"]["em"],
           "joint_strictly_above_sft_one_dev":any(new[k]["em"]>sft[k]["em"] for k in ("new_eval","goal_G_regression")),
           "joint_main_at_least_goal_j":new["new_eval"]["em"]>=j["new_eval"]["em"],
           "joint_regression_at_least_goal_j":new["goal_G_regression"]["em"]>=j["goal_G_regression"]["em"]}
    result={"status":"AUDITED","system_rows_rescored":64,"training_rows_rescored":240,
            "system":p2,"training":{"summary":frozen,"paired_vs_goal_j":paired,
                "joint_vs_sft":pair_vs_sft,"original_six_difficult_correct":hard,
                "curves":curves,"engineering":{arm:load(TRAIN/arm/"summary.json") for arm in ("sft","rl","joint")},
                "gates":gates,"recommend_further_reproduction":all(gates.values())}}
    (OUT/"offline-summary.json").write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"AUDITED","system":p2,"six_difficult_correct":hard,
                      "training_gates":gates,"recommend_further_reproduction":all(gates.values())}))


if __name__=="__main__":main()
