"""Add independently reviewed validation hard negatives until the v107 gate is met."""
from __future__ import annotations

import argparse, json
from pathlib import Path

import httpx

from build_answer_verifier_v107 import BUILD_SYSTEM, REVIEW_SYSTEM, call, sft_row, split, table
from mini_agent_rl.answer_verifier import parse_and_apply, visible_documents
from mini_agent_rl.training.thinking_sft import _env


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--db",type=Path,required=True); parser.add_argument("--env-file",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path,required=True); parser.add_argument("--needed",type=int,default=2); args=parser.parse_args()
    rollouts={row["task_id"]:row for row in table(args.db,"rollouts")}; tasks={row["id"]:row for row in table(args.db,"tasks")}
    transitions={}
    for row in table(args.db,"transitions"): transitions.setdefault(row["rollout_id"],[]).append(row)
    existing=[json.loads(line) for line in (args.data_dir/"validation.jsonl").read_text(encoding="utf-8").splitlines() if line]
    existing_candidates={row["messages"][-2]["content"] for row in existing}
    key=_env(args.env_file).get("DEEPSEEK_API_KEY"); added=[]; audits=[]; usage_total=0
    with httpx.Client(timeout=90) as client:
        for task_id in sorted(rollouts):
            if split(task_id)!="validation" or len(added)>=args.needed: continue
            rollout=rollouts[task_id]; steps=sorted(transitions.get(rollout["id"],[]),key=lambda row:row["step"])
            if not steps or rollout.get("termination_reason")!="answer": continue
            docs=visible_documents(steps[-1].get("messages",[])); question=tasks[task_id]["question"]
            prompt={"question":question,"evidence":docs,"candidate_answer":rollout.get("final_answer"),
                    "requirement":"Generate a different hard negative from any obvious original candidate; focus on entity, relation, date, or boundary confusion."}
            teacher,usage=call(client,key,BUILD_SYSTEM,prompt); usage_total+=int(usage.get("total_tokens",0)); hard=teacher.get("hard_negative") or {}
            candidate=str(hard.get("candidate", "")); action=hard.get("action")
            if not candidate or not isinstance(action,dict) or candidate in existing_candidates: continue
            applied=parse_and_apply(json.dumps(action,ensure_ascii=False),candidate,docs)
            if applied.status!="replace": continue
            review,review_usage=call(client,key,REVIEW_SYSTEM,{"question":question,"evidence":docs,"candidate_answer":candidate,"proposed_action":action})
            usage_total+=int(review_usage.get("total_tokens",0)); audits.append({"task_id":task_id,"review":review})
            if not review.get("accepted"): continue
            added.append(sft_row(task_id,question,docs,candidate,action,"hard_negative_supplement")); existing_candidates.add(candidate)
    if len(added)<args.needed: raise RuntimeError(f"only {len(added)} supplemental records accepted")
    with (args.data_dir/"validation.jsonl").open("a",encoding="utf-8") as stream:
        for row in added: stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    with (args.data_dir/"teacher-audit.jsonl").open("a",encoding="utf-8") as stream:
        for row in audits: stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    (args.data_dir/"supplement.json").write_text(json.dumps({"added":len(added),"usage_total_tokens":usage_total,"records":added},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"added":len(added),"usage_total_tokens":usage_total},ensure_ascii=False))


if __name__=="__main__": main()
