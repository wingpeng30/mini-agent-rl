"""Interactive live search with Goal Q's selected adapter and shared protocol."""
from __future__ import annotations

import argparse
import asyncio
import json

import run_v133_goal_q_search as q


def present(result):
    trace = []
    for step in result["trace"]:
        item = {"step": step["step"], "raw": step["raw"], "action": step.get("action"),
                "query": step.get("query"), "answer": step.get("answer"),
                "parse_error": step["parse_error"], "usage": step["usage"]}
        if "observation" in step:
            item["documents"] = step["observation"].get("result", [])
        trace.append(item)
    return {"question": result["question"], "answer": result["answer"],
            "reason": result["reason"], "searches": result["searches"],
            "elapsed_seconds": result["elapsed_seconds"], "trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question")
    args = parser.parse_args()
    decision = q.read(q.OUT / "decision.json")
    q.load_config()
    client, wrapped, registry = q.make_client(decision["adapter"], deterministic=True)
    print(json.dumps({"selected": decision["selected"], "adapter": decision["adapter"]}, ensure_ascii=False))
    if args.question:
        print(json.dumps(present(asyncio.run(q.rollout(args.question, wrapped, registry, sample=False, seed=42))), ensure_ascii=False, indent=2))
        return
    while True:
        try: question = input("Question (empty to exit): ").strip()
        except EOFError: break
        if not question: break
        print(json.dumps(present(asyncio.run(q.rollout(question, wrapped, registry, sample=False, seed=42))), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
