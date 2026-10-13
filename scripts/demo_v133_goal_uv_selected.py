"""Interactive real-search entry for the selected Goal U/V adapter."""
from __future__ import annotations

import argparse
import asyncio
import json

import run_v133_goal_q_search as q
import run_v133_goal_u_v_evidence_preference as uv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question")
    args = parser.parse_args()
    if uv.read(uv.STATE)["status"] != "complete":
        raise RuntimeError("Goal U/V has not reached a final decision")
    selected = uv.read(uv.OUT / "decision.json")
    client, wrapped, registry = q.make_client(selected["adapter"], deterministic=True)
    print(json.dumps({"selected": selected["selected"], "adapter": selected["adapter"]}, ensure_ascii=False))
    questions = [args.question] if args.question else None
    while True:
        if questions is None:
            try: question = input("Question (empty to exit): ").strip()
            except EOFError: break
            if not question: break
        elif questions:
            question = questions.pop(0)
        else: break
        result = asyncio.run(q.rollout(question, wrapped, registry, sample=False, seed=42))
        steps = []
        for step in result["trace"]:
            entry = {"action": step.get("action")}
            if entry["action"] == "search":
                entry["query"] = step.get("query")
                entry["results"] = [{"id": doc["id"], "title": doc["title"], "text": doc["text"]}
                                    for doc in step.get("observation", {}).get("result", [])]
            elif entry["action"] == "answer":
                entry["answer"] = step.get("answer")
            steps.append(entry)
        print(json.dumps({"question": question, "answer": result.get("answer"),
                          "reason": result.get("reason"), "searches": result.get("searches"),
                          "elapsed_seconds": result.get("elapsed_seconds"), "steps": steps},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
