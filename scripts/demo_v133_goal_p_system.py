"""Interactive Goal P live retrieval with Goal J and retained protocol recovery."""
import argparse
import asyncio
import json

import run_v133_goal_p_system as system
from run_v133_goal_p_variant import SearchTextAliasClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="One question; omit for an interactive session")
    args = parser.parse_args()
    cfg = system.config()
    model, registry = system.make_components(cfg)
    client = SearchTextAliasClient(model)
    def show(question):
        result = asyncio.run(system.ask(question, client, registry, cfg))
        result["scope"] = "live_retrieval_shared_96_document_demo_corpus"
        result["retrieved_evidence"] = [doc for step in result["trace"]
            for doc in (step.get("observation", {}).get("result") or [])]
        result["retrieved_evidence_note"] = "These documents were retrieved by the tool; the model did not cite them explicitly."
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.question:
        show(args.question)
        return
    print("Goal P live retrieval ready. Enter a question; empty line exits.", flush=True)
    while True:
        try:
            question = input("question> ").strip()
        except EOFError:
            break
        if not question:
            break
        show(question)


if __name__ == "__main__":
    main()
