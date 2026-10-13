"""Use the Goal Z product decision through the existing real-search product path."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import hashlib

import run_v133_goal_q_search as q
import run_v133_goal_u_v_evidence_preference as uv
import run_v133_goal_aa_audit_confirmation as aa
import run_v133_goal_z_multihop_sft as z


def selected_adapter() -> tuple[str, str]:
    ab_dir = z.ROOT / "runs/v133-goal-ab-evidence-decision"
    ab_decision = ab_dir / "product-decision.json"
    if ab_decision.exists():
        decision = z.read(ab_decision)
        state = z.read(ab_dir / "state.json")
        if state["status"] != decision["status"]:
            raise RuntimeError("Goal AB decision/state mismatch")
        if decision["published"]:
            if decision["selected_seed"] != 42 or decision["selected_route"] not in ("A", "B"):
                raise RuntimeError("Invalid Goal AB product decision")
            return "Goal AB confirmed seed 42 " + decision["selected_route"], decision["adapter"]
    aa_decision_path = aa.OUT / "product-decision.json"
    if aa_decision_path.exists():
        aa_decision = z.read(aa_decision_path)
        aa_state = z.read(aa.STATE)
        if aa_state["status"] != aa_decision["status"]:
            raise RuntimeError("Goal AA decision/state mismatch")
        if aa_decision["published"]:
            if aa_decision["selected_route"] != "trajectory" or aa_decision["selected_seed"] != 42:
                raise RuntimeError("Invalid Goal AA product decision")
            return "Goal AA confirmed trajectory SFT seed 42", aa_decision["adapter"]
        return "Goal T seed 42 update 32", aa_decision["adapter"]
    decision = z.read(z.OUT / "development-decision.json")
    state = z.read(z.STATE)
    if state["status"] != decision["status"]:
        raise RuntimeError("Goal Z decision/state mismatch")
    if decision["status"] == "complete_development_gate_failed":
        previous = uv.read(uv.OUT / "decision.json")
        return "Goal T seed 42 update 32", previous["adapter"]
    if decision["status"] == "complete_confirmed_and_published":
        final = z.read(z.OUT / "product-decision.json")
        if not final["published"] or final["selected_seed"] != 42:
            raise RuntimeError("Invalid Goal Z product decision")
        return final["selected_route"], final["adapter"]
    raise RuntimeError("Goal Z has not reached a product decision")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question")
    parser.add_argument("--corpus", type=Path, help="JSON document library; defaults to the original 96-document product library")
    args = parser.parse_args()
    selected, adapter = selected_adapter()
    corpus = args.corpus or q.p.CORPUS
    if not corpus.is_file():
        raise FileNotFoundError(corpus)
    if args.corpus:
        import run_v133_goal_z_multihop_sft as z_runner
        client, wrapped, registry = z_runner.make_client(Path(adapter), corpus)
    else:
        client, wrapped, registry = q.make_client(adapter, deterministic=True)
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    print(json.dumps({"selected": selected, "adapter": adapter, "corpus": str(corpus),
                      "corpus_sha256": digest}, ensure_ascii=False))
    questions = [args.question] if args.question else None
    while True:
        if questions is None:
            try:
                question = input("Question (empty to exit): ").strip()
            except EOFError:
                break
            if not question:
                break
        elif questions:
            question = questions.pop(0)
        else:
            break
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
                          "elapsed_seconds": result.get("elapsed_seconds"),
                          "steps": steps}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
