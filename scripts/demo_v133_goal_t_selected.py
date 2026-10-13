"""Interactive real-search demo for the Goal T selected adapter."""
from __future__ import annotations

import argparse
import asyncio
import json

import run_v133_goal_q_search as q
import run_v133_goal_t_update_length as t


def show(value):
    return {"question": value["question"], "answer": value["answer"],
            "reason": value["reason"], "searches": value["searches"],
            "elapsed_seconds": value["elapsed_seconds"],
            "trace": [{"step": row["step"], "raw": row["raw"],
                       "query": row.get("query"), "answer": row.get("answer"),
                       "documents": row.get("observation", {}).get("result", []),
                       "parse_error": row["parse_error"]} for row in value["trace"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question")
    args = parser.parse_args()
    t.config()
    if t.read(t.STATE)["status"] != "complete": raise RuntimeError("Goal T has no complete product decision")
    decision = t.read(t.OUT / "decision.json")
    client, wrapped, registry = q.make_client(decision["adapter"], deterministic=True)
    print(json.dumps({"selected": decision["selected"], "adapter": decision["adapter"]}, ensure_ascii=False))
    if args.question:
        result = asyncio.run(q.rollout(args.question, wrapped, registry, sample=False, seed=42))
        print(json.dumps(show(result), ensure_ascii=False, indent=2))
        return
    while True:
        try: question = input("Question (empty to exit): ").strip()
        except EOFError: break
        if not question: break
        result = asyncio.run(q.rollout(question, wrapped, registry, sample=False, seed=42))
        print(json.dumps(show(result), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
