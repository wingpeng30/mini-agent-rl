"""Reconstruct exact E0 prompts and freeze E0/E1 evidence for v109."""
from __future__ import annotations

import argparse, hashlib, json, sqlite3
from pathlib import Path
from typing import Any

from transformers import AutoProcessor

from mini_agent_rl.answer_verifier import visible_documents
from mini_agent_rl.domain import Message
from mini_agent_rl.hybrid_answer import answer_metrics, evidence_hash
from mini_agent_rl.model.context import budget_messages


def table(path: Path, name: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try: return [json.loads(row[0]) for row in connection.execute(f"SELECT data FROM {name}")]
    finally: connection.close()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--db",type=Path,required=True); p.add_argument("--model-path",required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--report",type=Path,required=True); args=p.parse_args()
    source_hash=hashlib.sha256(args.db.read_bytes()).hexdigest(); processor=AutoProcessor.from_pretrained(args.model_path,local_files_only=True)
    tasks={r["id"]:r for r in table(args.db,"tasks")}; rollouts={r["task_id"]:r for r in table(args.db,"rollouts")}; grouped={}
    for r in table(args.db,"transitions"): grouped.setdefault(r["rollout_id"],[]).append(r)
    def render(messages):
        chat=[{"role":m.role,"content":m.content} for m in messages]
        value=processor.apply_chat_template(chat,add_generation_prompt=True,tokenize=True,return_dict=True,enable_thinking=False)
        ids=value["input_ids"]
        if ids and isinstance(ids[0],list): ids=ids[0]
        return list(ids)
    def length(messages): return len(render(messages))
    tok=getattr(processor,"tokenizer",processor)
    records=[]; replay_diagnostics=[]
    for task_id in sorted(rollouts):
        rollout=rollouts[task_id]; steps=sorted(grouped.get(rollout["id"],[]),key=lambda r:r["step"]); final=steps[-1]
        original=[Message.model_validate(m) for m in final["messages"]]
        budgeted=budget_messages(original,2048,length,lambda text:len(tok.encode(text,add_special_tokens=False)),"compact-v3")
        ids=render(budgeted.messages); response=final.get("response") or {}; saved=response.get("prompt_token_ids") or []
        mask=response.get("prompt_attention_mask") or []; replay=ids==saved and len(mask)==len(saved) and all(int(x)==1 for x in mask)
        e0=visible_documents([m.model_dump(mode="json") for m in budgeted.messages]); e1=visible_documents(final["messages"])
        prediction=str(rollout.get("final_answer") or ""); gold=str(tasks[task_id]["answer"])
        if not replay and len(replay_diagnostics) < 5:
            first = next((index for index,(left,right) in enumerate(zip(ids,saved)) if left != right), min(len(ids),len(saved)))
            replay_diagnostics.append({"task_id":task_id,"rendered_length":len(ids),"saved_length":len(saved),
                "first_difference":first,"rendered_window":ids[max(0,first-5):first+10],"saved_window":saved[max(0,first-5):first+10],
                "message_roles":[m.role for m in original],"budget_usage":budgeted.usage})
        records.append({"task_id":task_id,"question":tasks[task_id]["question"],"gold":gold,"original_answer":prediction,
            "metrics":answer_metrics(prediction,gold),"e0":e0,"e1":e1,"e0_sha256":evidence_hash(e0),"e1_sha256":evidence_hash(e1),
            "prompt_replay_exact":replay,"saved_prompt_sha256":response.get("usage",{}).get("prompt_sha256"),
            "replayed_prompt_sha256":hashlib.sha256(bytes(str(ids),"utf-8")).hexdigest(),"budget_usage":budgeted.usage})
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    report={"version":"v109","source_db_sha256":source_hash,"source_unchanged":source_hash==hashlib.sha256(args.db.read_bytes()).hexdigest(),
        "tasks":len(records),"exact_prompt_replay":sum(r["prompt_replay_exact"] for r in records),
        "strict_em":sum(r["metrics"]["strict_em"] for r in records)/len(records),"normalized_em":sum(r["metrics"]["normalized_em"] for r in records)/len(records),
        "mean_f1":sum(r["metrics"]["f1"] for r in records)/len(records),"e0_e1_different":sum(r["e0_sha256"]!=r["e1_sha256"] for r in records),
        "replay_diagnostics":replay_diagnostics}
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
