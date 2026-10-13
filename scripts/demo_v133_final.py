"""Final interview demo: explicitly labelled replay or bounded live research preset."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v133_goal_bd as bd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--model", choices=["goal-z", "am-research"], default="goal-z")
    parser.add_argument("--case-id", default=bd.DEMOS[0], choices=bd.DEMOS)
    parser.add_argument("--question")
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    arm = "A" if args.model == "goal-z" else "B"
    if args.mode == "replay":
        if args.question or args.corpus:
            parser.error("custom question/corpus only supported with --mode live")
        row = next(r for r in bd.rows(bd.OUT / "demo-replays.jsonl") if r["case_id"] == args.case_id and r["arm"] == arm)
        m = bd.read(bd.OUT / "manifest.json")
        spec = next(s for s in m["specs"] if s["split"] == "goal_z_dev")
        corpus_identity = dict(path=spec["corpus"], sha256=spec["corpus_sha256"])
        mode = "archived replay; no model inference; elapsed_seconds is historical"
    else:
        if args.question and args.corpus is None:
            parser.error("custom --question requires explicit --corpus")
        bd.frozen()
        if args.question:
            question = args.question
        else:
            question = next(r["question"] for r in bd.rows(bd.OUT / "evaluation-cases.jsonl") if r["case_id"] == args.case_id)
        corpus = args.corpus or bd.ROOT / "runs/v133-goal-z-multihop-sft/corpus-dev.json"
        if not corpus.is_file():
            parser.error("corpus file does not exist")
        corpus_identity = dict(path=str(corpus), sha256=bd.sha(corpus))
        print(json.dumps(dict(mode="live", model=args.model, corpus=corpus_identity, profile="BD research preset"), ensure_ascii=False), flush=True)
        client, wrapped, registry = bd.q.make_client(bd.ADAPTERS[arm], deterministic=True,
                 product_config=bd.product(), model_path=bd.BASE, corpus=corpus)
        row = asyncio.run(bd.q.rollout(question, wrapped, registry, sample=False, seed=42, product_config=bd.product()))
        mode = "live inference; manually invoked outside frozen BD evaluation"
    steps = []
    for t in row["trace"]:
        steps.append(dict(action=t.get("action"), query=t.get("query"), answer=t.get("answer"),
                    documents=t.get("observation", {}).get("result", []),
                    input_tokens=len(t["prompt_token_ids"]), output_tokens=len(t["action_token_ids"])))
    print(json.dumps(dict(mode=mode, profile="BD wide-context research preset; original product defaults unchanged",
          model=args.model, model_status="published weights" if arm == "A" else "unpublished research candidate",
          adapter_sha256=bd.EXPECTED[arm], corpus=corpus_identity, question=row["question"], answer=row.get("answer"),
          reason=row["reason"], searches=row["searches"], elapsed_seconds=row["elapsed_seconds"], steps=steps), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
