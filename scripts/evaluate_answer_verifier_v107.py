"""Greedy evaluation of an answer-verifier adapter on frozen JSONL records."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

from mini_agent_rl.answer_scoring import normalize_answer
from mini_agent_rl.answer_verifier import parse_and_apply
from mini_agent_rl.domain import Message
from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient, generation_terminator_ids, trim_generation_terminators


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--model-path",required=True); parser.add_argument("--adapter",required=True)
    parser.add_argument("--data",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    import torch
    client=TransformersModelClient(LocalModelConfig(model_id=args.model_path,adapter_path=args.adapter,max_input_tokens=2048,
        max_new_tokens=128,observation_encoding="compact-v3",enable_thinking=False,gradient_checkpointing=False,recompute_old_logprobs=False))
    client._load(); tokenizer=getattr(client._processor,"tokenizer",client._processor); terminators=generation_terminator_ids(tokenizer)
    records=[json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]; rows=[]; started=time.perf_counter()
    for index,record in enumerate(records,1):
        messages=[Message.model_validate(item) for item in record["messages"][:-1]]; payload=json.loads(messages[-1].content)
        candidate=payload["candidate_answer"]; docs=payload["evidence"]; expected=json.loads(record["messages"][-1]["content"])
        row={"index":index,"task_id":record["metadata"]["task_id"],"origin":record["metadata"]["origin"],"expected":expected,"candidate":candidate}
        try:
            inputs=client._inputs(messages,"verifier_eval"); client._model.config.use_cache=True
            with torch.inference_mode(): output=client._model.generate(**inputs,max_new_tokens=128,do_sample=False,eos_token_id=terminators or None)
            action_ids=output[0,inputs["input_ids"].shape[1]:].tolist(); parse_ids,terminal=trim_generation_terminators(action_ids,tokenizer)
            raw=client._processor.decode(parse_ids,skip_special_tokens=False); applied=parse_and_apply(raw,candidate,docs)
            valid=not applied.status.startswith("fallback_"); preserve=applied.answer==candidate
            repair=normalize_answer(applied.answer)==normalize_answer(str(expected.get("answer","")))
            row.update({"raw":raw,"answer":applied.answer,"status":applied.status,"valid":valid,"preserve":preserve,"repair":repair,
                        "completion_tokens":len(action_ids),"terminal":terminal})
        except Exception as exc:
            row.update({"raw":"","answer":candidate,"status":"infrastructure_error","valid":False,"preserve":True,"repair":False,"error":f"{type(exc).__name__}: {exc}"})
        rows.append(row); print(f"{index}/{len(records)} {row['status']}",flush=True)
        if row["status"]=="infrastructure_error": break
    keep=[row for row in rows if row["expected"].get("decision")=="keep"]
    replace=[row for row in rows if row["expected"].get("decision")=="replace"]
    report={"version":"v107","adapter":args.adapter,"expected":len(records),"completed":len(rows),
        "protocol_legal_rate":sum(row["valid"] for row in rows)/len(records),
        "correct_candidate_preserve_rate":sum(row["preserve"] for row in keep)/len(keep) if keep else None,
        "wrong_candidate_repair_rate":sum(row["repair"] for row in replace)/len(replace) if replace else None,
        "keep_count":len(keep),"replace_count":len(replace),"infrastructure_errors":sum(row["status"]=="infrastructure_error" for row in rows),
        "elapsed_seconds":time.perf_counter()-started,"memory":client.gpu_memory_stats(),"rows":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
