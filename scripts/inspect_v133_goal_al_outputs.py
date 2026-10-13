"""Read-only views of actual AL input bodies for the primary one-pass review."""
import argparse
import json
from transformers import AutoProcessor
from prepare_v133_goal_ah import documents
import run_v133_goal_al_wide_context as al


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=29)
    parser.add_argument("--case")
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(str(al.assets.BASE), local_files_only=True)
    manifest = al.read(al.OUT / "manifest.json")
    data = {arm: {r["case_id"]: r for r in al.rows(al.OUT / f"trajectory-{arm}.jsonl")} for arm in ("A", "B") if (al.OUT / f"trajectory-{arm}.jsonl").exists()}
    edges = al.rows(al.assets.EDGES)
    order = [r["case_id"] for r in al.rows(al.assets.CASES) if r["case_id"] in manifest["admitted"]]
    selected = [args.case] if args.case else order[args.start:args.end]
    for case_id in selected:
        a = data["A"][case_id]
        b = data.get("B", {}).get(case_id)
        print(json.dumps(dict(case_id=case_id, question=a["question"], gold=a["official_answer"],
            edges=[dict(id=e["edge_id"], subject=e["subject"], relation=e["relation"], object=e["object"], doc_id=e.get("document_id")) for e in edges if e["case_id"]==case_id]), ensure_ascii=False))
        for arm, row in (("A", a), ("B", b)):
            if row is None: continue
            same = arm=="B" and row["trace"][-1]["prompt_token_ids"] == a["trace"][-1]["prompt_token_ids"] and row["answer"]==a["answer"]
            final = row["trace"][-1]
            print(json.dumps(dict(arm=arm, answer=row["answer"], reason=row["reason"], same_input_and_answer_as_A=same,
                input_tokens=len(final["prompt_token_ids"]), queries=[t.get("query") for t in row["trace"] if t.get("query")],
                documents=[] if same else documents(processor.decode(final["prompt_token_ids"], skip_special_tokens=False))), ensure_ascii=False))


if __name__ == "__main__": main()
