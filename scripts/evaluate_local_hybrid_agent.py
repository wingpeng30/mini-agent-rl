"""Paired fixed-weight local retrieval evaluation; no training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time
import urllib.request
from uuid import uuid4

from mini_agent_rl.evaluation import answer_metrics
from mini_agent_rl.retrieval.local import ROOT, LocalRetrieval, config, corpus, sha

MODES = ("bm25-v2", "hybrid-rrf-v1")
RUN_DIR = Path("E:/AgentLightningAssets/runs/retrieval-v1")
SUPPLEMENT = Path("E:/AgentLightningAssets/runs/retrieval-v1/supplement-reviewed-64-7011e1801e974f0eb055256abddbeab9")
PREVIOUS_LEDGERS = (
    RUN_DIR / "agent-paired-dev-e24475adc24549e1b19c7d12fce7380c" / "ledger.jsonl",
    RUN_DIR / "agent-paired-dev-29c9988ce7ca4dac882a3dbb73dd5c9f" / "ledger.jsonl",
)


def api(path, payload=None):
    request = urllib.request.Request("http://127.0.0.1:8765" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def freeze_combined_cases():
    """Freeze the already reviewed 64 dev + 40 natural + 24 diagnostic cases."""
    paths = {
        "development": ROOT / "runs/v133-goal-z-multihop-sft/selected-dev.jsonl",
        "natural_supplement": SUPPLEMENT / "natural.jsonl",
        "constructed_singlehop": SUPPLEMENT / "constructed.jsonl",
    }
    cases = []
    inputs = {}
    for stratum, path in paths.items():
        rows = read_rows(path)
        inputs[stratum] = {"path": str(path), "sha256": sha(path), "count": len(rows)}
        for row in rows:
            row = dict(row)
            row["stratum"] = stratum
            cases.append(row)
    ids = [case.get("id") for case in cases]
    questions = [case.get("question", "").strip().casefold() for case in cases]
    if len(cases) != 128 or len(set(ids)) != 128 or len(set(questions)) != 128:
        raise ValueError("Frozen set must contain 128 unique IDs and questions")
    docs = corpus(config())
    titles = {doc["title"] for doc in docs}
    for row in cases:
        if not all(row.get(key) for key in ("id", "question", "answer", "support_titles")):
            raise ValueError(f"Missing required field: {row.get('id')}")
        missing = set(row["support_titles"]) - titles
        if missing:
            raise ValueError(f"Support documents absent from corpus for {row['id']}: {sorted(missing)}")
    root = RUN_DIR / ("paired-128-frozen-" + uuid4().hex)
    root.mkdir(parents=True)
    # Deterministic shuffle; source files and answer labels remain untouched.
    random.Random(42).shuffle(cases)
    case_path = root / "cases.jsonl"
    case_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases), encoding="utf-8")
    manifest = {
        "kind": "exposed fixed-weight replication/diagnostic; not independent generalization",
        "seed": 42, "questions": len(cases), "strata": {k: v["count"] for k, v in inputs.items()},
        "inputs": inputs, "cases_sha256": sha(case_path), "corpus_sha256": config()["corpus_sha256"],
        "corpus_document_count": len(docs), "training": False,
        "selection_uses_model_or_retrieval_results": False,
    }
    write_json(root / "manifest.json", manifest)
    return case_path, root, manifest


def summarize(rows):
    if not rows:
        return {"questions": 0}
    metrics = ("strict_exact_match", "normalized_exact_match", "token_f1", "both_support_seen",
        "searches", "model_calls", "auxiliary_calls", "input_tokens", "output_tokens", "elapsed_seconds")
    scores, reasons = {}, {}
    for mode in MODES:
        group = [row for row in rows if row["retriever"] == mode]
        scores[mode] = {key: sum(row[key] for row in group) / len(group) for key in metrics}
        reasons[mode] = dict(Counter(row["reason"] for row in group))
        scores[mode].update({
            "correct_strict": sum(row["strict_exact_match"] for row in group),
            "correct_normalized": sum(row["normalized_exact_match"] for row in group),
            "repeated_actions": sum(row["repeated_actions"] for row in group),
            "invalid_actions": sum(row["invalid_actions"] for row in group),
        })
    pairs = {}
    for row in rows:
        pairs.setdefault(row["case_id"], {})[row["retriever"]] = row
    complete_pairs = [p for p in pairs.values() if all(mode in p for mode in MODES)]
    deltas = [p[MODES[1]]["normalized_exact_match"] - p[MODES[0]]["normalized_exact_match"] for p in complete_pairs]
    if not deltas:
        paired = {"complete_pairs": 0}
    else:
        rng = random.Random(42)
        draws = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
        paired = dict(complete_pairs=len(deltas), improved=sum(d > 0 for d in deltas),
            regressed=sum(d < 0 for d in deltas), unchanged=sum(d == 0 for d in deltas),
            mean_normalized_em_delta=sum(deltas) / len(deltas),
            bootstrap_95_percent_interval=[draws[50], draws[1949]],
            interval_metric="paired normalized exact match difference")
    return dict(questions=len({row["case_id"] for row in rows}), scores=scores,
        reasons=reasons, paired_normalized_em=paired)


def comparison(rows, old_rows):
    old = {(row["case_id"], row["retriever"]): row for row in old_rows}
    changed = []
    for row in rows:
        before = old.get((row["case_id"], row["retriever"]))
        if before is None:
            continue
        old_queries = [step.get("query") for step in before.get("steps", []) if step.get("action") == "search"]
        new_queries = [step.get("query") for step in row.get("steps", []) if step.get("action") == "search"]
        old_documents = [[doc.get("id") for doc in step.get("documents", [])] for step in before.get("steps", [])]
        new_documents = [[doc.get("id") for doc in step.get("documents", [])] for step in row.get("steps", [])]
        fields = ("answer", "reason", "searches")
        if any(before.get(key) != row.get(key) for key in fields) or old_queries != new_queries or old_documents != new_documents:
            changed.append({"case_id": row["case_id"], "stratum": row["stratum"], "retriever": row["retriever"],
                "old": {key: before.get(key) for key in fields}, "new": {key: row.get(key) for key in fields},
                "old_queries": old_queries, "new_queries": new_queries,
                "old_document_ids_by_step": old_documents, "new_document_ids_by_step": new_documents})
    return {"matched_runs": sum((r["case_id"], r["retriever"]) in old for r in rows),
        "changed_runs": len(changed), "changes": changed}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, help="Frozen JSONL question set; defaults to the 64-question dev set")
    parser.add_argument("--combine-reviewed-128", action="store_true", help="Freeze the existing reviewed 64+40+24 set and run it")
    parser.add_argument("--resume", type=Path, help="Resume an incomplete run folder without repeating completed case/mode pairs")
    args = parser.parse_args()
    if sum((bool(args.combine_reviewed_128), bool(args.cases), bool(args.resume))) > 1:
        parser.error("Use only one of --cases, --combine-reviewed-128, and --resume")
    frozen_root = None
    resume = bool(args.resume)
    if resume:
        folder = args.resume.resolve()
        if not (folder / "protocol.json").exists() or (folder / "report.json").exists():
            parser.error("--resume requires an incomplete run with protocol.json and no report.json")
        protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
        case_path = Path(protocol["cases_path"])
        frozen_root = Path(protocol.get("frozen_cases_path") or case_path).parent
        frozen_manifest_path = case_path.parent / "manifest.json"
        frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8")) if frozen_manifest_path.exists() else None
    elif args.combine_reviewed_128:
        case_path, frozen_root, frozen_manifest = freeze_combined_cases()
        print(json.dumps({"frozen_cases": str(case_path), "questions": 128}, ensure_ascii=False), flush=True)
    else:
        case_path = args.cases or (ROOT / "runs/v133-goal-z-multihop-sft/selected-dev.jsonl")
        frozen_manifest = None
    cases = read_rows(case_path)
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    dataset_manifest_path = Path(case_path).parent / "manifest.json"
    dataset_manifest = frozen_manifest
    if dataset_manifest is None and dataset_manifest_path.exists():
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if dataset_manifest.get("cases_sha256") != sha(case_path) or dataset_manifest.get("corpus_sha256") != config()["corpus_sha256"]:
            raise ValueError("Dataset manifest or corpus hash mismatch")
    corpus_rows = corpus(config())
    titles = {doc["title"] for doc in corpus_rows}
    for case in cases:
        if not all(case.get(key) for key in ("id", "question", "answer", "support_titles")):
            raise ValueError(f"Missing required field in {case.get('id')}")
        if set(case["support_titles"]) - titles:
            raise ValueError(f"Unknown support title in {case['id']}")
    # Check service before doing the CPU diagnostic or switching weights.
    initial_status = api("/api/status")
    if initial_status["busy"]:
        raise RuntimeError("Local service is busy; no model or experiment state was changed")
    if initial_status.get("state") != "ready" or not initial_status.get("model_id"):
        raise RuntimeError(f"Local service is not ready: {initial_status}")
    original_model = protocol.get("original_model", initial_status["model_id"]) if resume else initial_status["model_id"]
    if resume:
        if initial_status["model_id"] not in (original_model, "goal-z"):
            raise RuntimeError(f"Service model changed since interrupted run: {initial_status['model_id']}")
    else:
        folder = RUN_DIR / ("agent-paired-128-" + uuid4().hex)
        folder.mkdir(parents=True)
    old_rows = [row for path in PREVIOUS_LEDGERS if path.exists() for row in read_rows(path)]
    if not resume:
        protocol = dict(kind="exposed fixed-weight replication/diagnostic; not independent generalization",
            cases_path=str(case_path), cases_sha256=sha(case_path), frozen_cases_path=str(case_path) if frozen_root else None,
            questions=len(cases), strata={s: sum(row.get("stratum", "development") == s for row in cases)
                for s in sorted({row.get("stratum", "development") for row in cases})},
            corpus_sha256=config()["corpus_sha256"], corpus_document_count=len(corpus_rows),
            model_id="goal-z", expected_adapter_sha256="d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52",
            original_model=original_model, modes=MODES, reasoning_mode="off", seed=42,
            limits={"model_calls": 4, "searches": 3, "auxiliary_calls": 0}, training=False,
            output_dir=str(folder), started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if resume:
        with (folder / "recovery-attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "reason": "resume incomplete paired experiment; completed case/mode pairs are retained"}) + "\n")
    write_json(folder / "protocol.json", protocol)
    ledger_path = folder / "ledger.jsonl"
    rows = read_rows(ledger_path) if resume and ledger_path.exists() else []
    existing_pairs = {(row["case_id"], row["retriever"]) for row in rows}
    stage = "cpu_retrieval_diagnostic"
    try:
        engine = LocalRetrieval(config())
        engine.search("warmup", "dense-v1")
        retrieval_rows = []
        for case in cases:
            for mode in ("bm25-v2", "dense-v1", "hybrid-rrf-v1"):
                start = time.perf_counter()
                docs = engine.search(case["question"], mode, 10)
                required = set(case["support_titles"])
                found_titles = {doc["title"] for doc in docs}
                result = dict(case_id=case["id"], retriever=mode,
                    stratum=case.get("stratum", "development"),
                    retrieved_ids=[doc["id"] for doc in docs], elapsed_seconds=time.perf_counter()-start)
                for k in (3, 10):
                    selected = {doc["title"] for doc in docs[:k]}
                    result[f"recall_at_{k}"] = len(required & selected) / len(required)
                    result[f"all_support_at_{k}"] = int(required.issubset(selected))
                retrieval_rows.append(result)
        (folder / "retrieval-ledger.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in retrieval_rows), encoding="utf-8")
        retrieval_scores = {}
        for stratum in protocol["strata"]:
            retrieval_scores[stratum] = {}
            for mode in ("bm25-v2", "dense-v1", "hybrid-rrf-v1"):
                group = [r for r in retrieval_rows if r["stratum"] == stratum and r["retriever"] == mode]
                retrieval_scores[stratum][mode] = {key: sum(r[key] for r in group)/len(group)
                    for key in ("recall_at_3", "recall_at_10", "all_support_at_3", "all_support_at_10", "elapsed_seconds")}
        protocol["retrieval_identity"] = engine.metadata("hybrid-rrf-v1")
        protocol["retrieval_scores"] = retrieval_scores
        protocol["retrieval_diagnostic_rows"] = len(retrieval_rows)
        write_json(folder / "protocol.json", protocol)

        stage = "load_goal_z"
        if original_model != "goal-z":
            api("/api/model/load", dict(model_id="goal-z"))
        deadline = time.monotonic() + 180
        while True:
            status = api("/api/status")
            if status.get("state") == "ready" and status.get("model_id") == "goal-z":
                break
            if status.get("error") or time.monotonic() > deadline:
                raise RuntimeError(f"Goal Z failed to load: {status}")
            time.sleep(1)
        spec = next((m for m in status.get("models", []) if m.get("id") == "goal-z"), {})
        protocol["loaded_model_status"] = {k: status.get(k) for k in ("state", "model_id", "busy", "error")}
        protocol["loaded_adapter_sha256"] = spec.get("sha256")
        if spec.get("sha256") != protocol["expected_adapter_sha256"]:
            raise RuntimeError("Loaded Goal Z adapter hash differs from frozen identity")
        write_json(folder / "protocol.json", protocol)

        stage = "paired_agent_runs"
        for i, case in enumerate(cases):
            order = MODES if i % 2 == 0 else MODES[::-1]
            for mode in order:
                if (case["id"], mode) in existing_pairs:
                    continue
                job = api("/api/runs", dict(question=case["question"], model_id="goal-z",
                    retriever=mode, reasoning_mode="off"))
                deadline = time.monotonic() + 300
                while job.get("state") not in ("complete", "error", "failed"):
                    if time.monotonic() > deadline:
                        raise RuntimeError("Inference timeout: " + job["id"])
                    time.sleep(0.3)
                    job = api("/api/runs/" + job["id"])
                if job.get("state") != "complete":
                    raise RuntimeError(f"Inference failed for {case['id']} {mode}: {job}")
                result = job["result"]
                if result.get("reasoning_mode") != "off" or result.get("retrieval", {}).get("retriever") != mode:
                    raise RuntimeError("Actual reasoning/retrieval mode differs from frozen experiment")
                if result.get("model_id") != "goal-z" or result.get("adapter_sha256") != protocol["expected_adapter_sha256"]:
                    raise RuntimeError("Run model/adapter identity mismatch")
                if result.get("model_calls", 999) > 4 or result.get("searches", 999) > 3 or result.get("auxiliary_calls", 999) != 0:
                    raise RuntimeError("Per-case inference budget violated")
                references = [case["answer"]] + case.get("answer_aliases", [])
                scored = [answer_metrics(result.get("answer"), answer) for answer in references]
                metrics = {key: max(item[key] for item in scored) for key in scored[0]}
                found = {doc["title"] for step in result["steps"] for doc in step["documents"]}
                steps = result["steps"]
                row = dict(case_id=case["id"], retriever=mode, reference=case["answer"],
                    stratum=case.get("stratum", "development"), source=case.get("source"),
                    **metrics, both_support_seen=int(set(case["support_titles"]).issubset(found)),
                    repeated_actions=sum(bool(step.get("repeated")) for step in steps),
                    invalid_actions=sum(bool(step.get("parse_error")) for step in steps),
                    input_tokens=sum(step.get("input_tokens", 0) for step in steps),
                    output_tokens=sum(step.get("output_tokens", 0) for step in steps), **result)
                rows.append(row)
                with (folder / "ledger.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing_pairs.add((case["id"], mode))
            print(json.dumps(dict(completed_questions=i+1, total=len(cases), last_answers=[r["answer"] for r in rows[-2:]]), ensure_ascii=False), flush=True)

        stage = "report"
        if len(existing_pairs) != len(cases) * 2 or len(rows) != len(cases) * 2:
            raise RuntimeError("Expected two complete runs per question")
        old_pairs = {(row["case_id"], row["retriever"]) for row in old_rows}
        new_pairs = {(row["case_id"], row["retriever"]) for row in rows}
        if not new_pairs.issubset(old_pairs):
            raise RuntimeError("Historical paired ledgers do not cover every case and mode; cannot complete requested replication")
        strata_scores = {stratum: summarize([r for r in rows if r["stratum"] == stratum]) for stratum in protocol["strata"]}
        report = dict(**protocol, complete=True, strata_scores=strata_scores, all_questions=summarize(rows),
            comparison_to_previous=comparison(rows, old_rows),
            primary_104_natural_multihop=summarize([r for r in rows if r["stratum"] in ("development", "natural_supplement")]),
            historical_comparison_sources=[str(path) for path in PREVIOUS_LEDGERS if path.exists()],
            recovered_after_initial_failure=bool(resume),
            support_metric_note="both_support_seen means all required support documents appeared at some point; single-hop set has one required document",
            report_note="Exposed replication/diagnostic set; repeated measurements do not create independent examples")
        write_json(folder / "report.json", report)
        print(json.dumps(dict(report=str(folder / "report.json"), strata_scores=strata_scores,
            all_questions=report["all_questions"], comparison_to_previous=report["comparison_to_previous"]["changed_runs"]), ensure_ascii=False), flush=True)
    except Exception as exc:
        write_json(folder / "failure.json", {"stage": stage, "error": repr(exc), "completed_runs": len(rows),
            "ledger_lines": sum(1 for _ in (folder / "ledger.jsonl").open(encoding="utf-8")) if (folder / "ledger.jsonl").exists() else 0})
        raise
    finally:
        try:
            current = api("/api/status")
            if original_model and current.get("model_id") != original_model and not current.get("busy"):
                api("/api/model/load", dict(model_id=original_model))
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    current = api("/api/status")
                    if current.get("state") == "ready" and current.get("model_id") == original_model:
                        protocol["restored_model_status"] = {k: current.get(k) for k in ("state", "model_id", "busy", "error")}
                        write_json(folder / "protocol.json", protocol)
                        if (folder / "report.json").exists():
                            completed_report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
                            completed_report["restored_model_status"] = protocol["restored_model_status"]
                            write_json(folder / "report.json", completed_report)
                        break
                    if current.get("error"):
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError(f"Original model restoration timed out: {current}")
                if current.get("model_id") != original_model or current.get("state") != "ready":
                    raise RuntimeError(f"Original model restoration failed: {current}")
        except Exception as restore_error:
            write_json(folder / "restore-failure.json", {"error": repr(restore_error)})
            if (folder / "report.json").exists():
                completed_report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
                completed_report["complete"] = False
                completed_report["restore_error"] = repr(restore_error)
                write_json(folder / "report.json", completed_report)
            print(json.dumps({"restore_failure": repr(restore_error), "folder": str(folder)}), flush=True)


if __name__ == "__main__":
    main()
