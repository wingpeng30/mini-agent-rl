"""Build leak-free final-answer-only SFT rows from the existing training split."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

from mini_agent_rl.prompts import DIRECT_SYSTEM


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--source",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    rows=[json.loads(x) for x in args.source.read_text(encoding="utf-8").splitlines() if x.strip()]
    args.output.mkdir(parents=True)
    handles={name:(args.output/f"{name}.jsonl").open("w",encoding="utf-8") for name in ("train","validation")}
    counts={"train":0,"validation":0}; task_ids={"train":[],"validation":[]}
    try:
        for row in rows:
            task_id=row["metadata"]["task_id"]
            split="validation" if int(hashlib.sha256(task_id.encode()).hexdigest(),16)%10==0 else "train"
            messages=[dict(message) for message in row["messages"]]
            messages[0]={"role":"system","content":DIRECT_SYSTEM}
            assistant_indexes=[i for i,m in enumerate(messages) if m["role"]=="assistant"]
            if not assistant_indexes or json.loads(messages[assistant_indexes[-1]]["content"]).get("action")!="answer":
                raise ValueError(f"missing final answer action: {task_id}")
            item={"messages":messages,"target_assistant_index":assistant_indexes[-1],"thinking_enabled":False,
                  "metadata":{"task_id":task_id,"source":"hotpot-agent-v041/train","target":"final_answer_only"}}
            handles[split].write(json.dumps(item,ensure_ascii=False)+"\n"); counts[split]+=1; task_ids[split].append(task_id)
    finally:
        for handle in handles.values(): handle.close()
    overlap=set(task_ids["train"])&set(task_ids["validation"])
    manifest={"version":"v105-answer-repair-v1","source":str(args.source),"source_sha256":sha(args.source),"prompt":DIRECT_SYSTEM,
              "split":"sha256(task_id) % 10 == 0 is validation","counts":counts,"overlap":sorted(overlap),
              "files":{name:sha(args.output/f"{name}.jsonl") for name in counts}}
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
