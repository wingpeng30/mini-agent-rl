"""Interactive live-search demo using Goal S's selected product adapter."""
from __future__ import annotations

import argparse
import asyncio
import json

import run_v133_goal_q_search as q
import run_v133_goal_s_label_granularity as s


def show(value):
    return {"question": value["question"], "answer": value["answer"],
            "reason": value["reason"], "searches": value["searches"],
            "elapsed_seconds": value["elapsed_seconds"],
            "trace": [{"step": x["step"], "raw": x["raw"], "query": x.get("query"),
                       "answer": x.get("answer"), "documents": x.get("observation", {}).get("result", []),
                       "parse_error": x["parse_error"]} for x in value["trace"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question")
    args = parser.parse_args()
    s.config()
    decision = s.read(s.OUT / "decision.json")
    if s.read(s.STATE)["status"] != "complete":
        raise RuntimeError("Goal S has no completed product decision")
    client, wrapped, registry = q.make_client(decision["adapter"], deterministic=True)
    print(json.dumps({"selected": decision["selected"], "adapter": decision["adapter"]}, ensure_ascii=False))
    if args.question:
        print(json.dumps(show(asyncio.run(q.rollout(args.question, wrapped, registry, sample=False, seed=42))), ensure_ascii=False, indent=2))
        return
    while True:
        try: question = input("Question (empty to exit): ").strip()
        except EOFError: break
        if not question: break
        print(json.dumps(show(asyncio.run(q.rollout(question, wrapped, registry, sample=False, seed=42))), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
