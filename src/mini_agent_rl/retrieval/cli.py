"""Local retrieval CLI. Run via mini-agent-search or python -m ...retrieval.cli."""
import argparse
import json
from pathlib import Path
import time
from uuid import uuid4
from .local import MODES, ROOT, LocalRetrieval, build_index, config, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index")
    query = sub.add_parser("query")
    query.add_argument("query")
    query.add_argument("--retriever", choices=MODES, default="hybrid-rrf-v1")
    sub.add_parser("evaluate")
    args = parser.parse_args()
    cfg = config(args.config)
    if args.command == "index":
        result = build_index(cfg)
    else:
        engine = LocalRetrieval(cfg)
        if args.command == "query":
            started = time.perf_counter()
            documents = engine.search(args.query, args.retriever)
            result = dict(query=args.query, documents=documents, retrieval=engine.metadata(args.retriever), elapsed_seconds=time.perf_counter() - started)
        else:
            cases_path = ROOT / "runs/v133-goal-z-multihop-sft/selected-dev.jsonl"
            cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(cases) != 64:
                raise ValueError("Expected 64 development queries")
            # Warm up the encoder once, outside measured per-query latency.
            engine.search("retrieval warmup", "dense-v1")
            ledger, scores = [], {}
            for mode in MODES:
                records = []
                for case in cases:
                    started = time.perf_counter()
                    docs = engine.search(case["question"], mode, 10)
                    expected = set(case["support_titles"])
                    row = dict(case_id=case["id"], retriever=mode, question=case["question"],
                        support_titles=sorted(expected), retrieved_ids=[d["id"] for d in docs],
                        retrieved_titles=[d["title"] for d in docs], elapsed_seconds=time.perf_counter() - started)
                    for k in (3, 10):
                        found = {d["title"] for d in docs[:k]}
                        row[f"recall_at_{k}"] = len(expected & found) / len(expected)
                        row[f"both_at_{k}"] = int(expected.issubset(found))
                    records.append(row)
                ledger.extend(records)
                scores[mode] = {key: sum(r[key] for r in records) / len(records) for key in ("recall_at_3", "recall_at_10", "both_at_3", "both_at_10", "elapsed_seconds")}
            result = dict(kind="64 exposed development question queries; retrieval diagnostic, not end-to-end QA", scores=scores,
                retrieval=engine.metadata("hybrid-rrf-v1"), questions=64)
            folder = Path(cfg["run_dir"]) / ("evaluate-" + uuid4().hex)
            folder.mkdir(parents=True)
            (folder / "ledger.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ledger) + "\n", encoding="utf-8")
            (folder / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            result["report_path"] = str(folder / "report.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
