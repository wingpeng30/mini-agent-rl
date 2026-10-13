"""Goal AE: compare frozen BM25-v2 with local natural-paragraph BM25-v3.

All source acquisition is from a pinned public static Parquet snapshot. This
runner makes no search-engine, page, or dataset API requests.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import run_v133_goal_p_system as p
import run_v133_goal_q_search as q
import run_v133_goal_y_three_class_transfer as y
import run_v133_goal_z_multihop_sft as z

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ae-corpus-repair"
RUN = OUT / "evaluation"
STATIC = OUT / "static-data"
CFG, STATE = RUN / "config.json", RUN / "state.json"
SOURCE_ADAPTER = z.SOURCE
SPLITS = ("goal_z_dev", "old_regression", "goal_y_regression", "aa_regression")
LABELS = ("bm25_v2", "bm25_v3_paragraph")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_once(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def original_paths() -> dict[str, Path]:
    return {
        "goal_z_dev": z.OUT / "corpus-dev.json",
        "old_regression": p.CORPUS,
        "goal_y_regression": y.OUT / "dev-corpus.json",
        "aa_regression": z.OUT / "corpus-confirm.json",
    }


def paragraph_paths() -> dict[str, Path]:
    return {key: STATIC / f"paragraphs-{key}.json" for key in SPLITS}


def title_by_id(path: Path) -> dict[str, str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("id")): str(row.get("title", "")) for row in values}


def frozen_cases() -> dict[str, list[dict]]:
    old_title = title_by_id(p.CORPUS)
    y_title = title_by_id(y.OUT / "dev-corpus.json")
    old = [row for row in p.records(p.CASES) if row["split"] != "train"]
    if len(old) != 32:
        raise RuntimeError(f"expected 32 old regression cases, found {len(old)}")
    old_cases = [{"case_id": row["case_id"], "question": row["question"],
                  "answer": row["official_answer"], "support_titles": [old_title[x] for x in row["support_ids"] if x in old_title]}
                 for row in old]
    y_cases = [{"case_id": row["case_id"], "question": row["question"],
                "answer": row["official_answer"], "support_titles": [y_title[x] for x in row["support_ids"] if x in y_title]}
               for row in y.dev_cases()]
    return {
        "goal_z_dev": [{"case_id": row["id"], "question": row["question"], "answer": row["answer"],
                        "support_titles": row["support_titles"]} for row in z.cases("dev")],
        "old_regression": old_cases,
        "goal_y_regression": y_cases,
        "aa_regression": [{"case_id": row["id"], "question": row["question"], "answer": row["answer"],
                           "support_titles": row["support_titles"]} for row in z.cases("confirm")],
    }


def product_config() -> dict:
    # Goal Q's config is read as frozen data. Its loader still verifies inputs
    # used by the common rollout loop; bm25-v2 itself remains unchanged.
    return q.load_config()["product"]


def inputs() -> dict[str, Path]:
    return {"adapter": SOURCE_ADAPTER / "adapter_model.safetensors",
            "tokenizer": q.p.BASE / "tokenizer.json",
            "search_code": ROOT / "src/mini_agent_rl/tools/search.py",
            "q_runner": Path(q.__file__), "z_runner": Path(z.__file__),
            "ae_runner": Path(__file__), "snapshot_manifest": STATIC / "manifest.json",
            **{f"original_{k}": v for k, v in original_paths().items()},
            **{f"paragraph_{k}": v for k, v in paragraph_paths().items()},
            **{f"base_weight_{i}": path for i, path in enumerate(sorted(q.p.BASE.glob("*.safetensors")))}}


def prepare() -> None:
    if RUN.exists():
        raise FileExistsError(f"refusing to overwrite Goal AE run: {RUN}")
    manifest = read(STATIC / "manifest.json")
    if manifest.get("fresh_goal_ab_confirmation_accessed") is not False:
        raise RuntimeError("fresh Goal AB confirmation must stay sealed before the development gate")
    corpora, paragraphs = original_paths(), paragraph_paths()
    if set(corpora) != set(SPLITS) or set(paragraphs) != set(SPLITS):
        raise RuntimeError("corpus inventory mismatch")
    for split, path in paragraphs.items():
        docs = json.loads(path.read_text(encoding="utf-8"))
        if not docs or any(not d.get("text") or not d.get("title") for d in docs):
            raise RuntimeError(f"empty or invalid paragraph corpus: {split}")
    cases = frozen_cases()
    counts = {split: len(rows) for split, rows in cases.items()}
    if counts != {"goal_z_dev": 64, "old_regression": 32, "goal_y_regression": 48, "aa_regression": 64}:
        raise RuntimeError(f"evaluation case counts changed: {counts}")
    for key, path in corpora.items():
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(sorted(q.p.BASE.glob("*.safetensors"))) == 0:
        raise RuntimeError("actual base model weight files not found")
    cfg = {"version": "goal-ae-corpus-repair-v1", "inputs": {k: sha(v) for k, v in inputs().items()},
           "snapshot": manifest, "case_counts": counts,
           "question_ids": {split: [row["case_id"] for row in items] for split, items in cases.items()},
           "product": product_config(), "models": ["Goal Z trajectory seed 42"],
           "arms": ["bm25-v2 original corpus", "bm25-v3-paragraph pinned natural text"],
           "evaluation_trajectories": 416,
           "fresh_goal_ab_confirmation_accessed": False}
    RUN.mkdir(parents=True)
    write_once(CFG, cfg)
    write_once(STATE, {"status": "prepared", "completed": 0, "fresh_goal_ab_confirmation_accessed": False})
    write_once(RUN / "frozen-cases.json", cases)
    print(json.dumps({"prepared": str(RUN), "counts": counts, "hashes": cfg["inputs"]}, ensure_ascii=True))


def config() -> dict:
    cfg = read(CFG)
    current = {k: sha(v) for k, v in inputs().items()}
    if current != cfg["inputs"]:
        changed = sorted(k for k in current if current.get(k) != cfg["inputs"].get(k))
        raise RuntimeError(f"frozen inputs changed: {changed}")
    return cfg


def evaluate_arm(retriever: str, label: str) -> None:
    cfg = config()
    state = read(STATE)
    if state["status"] not in {"prepared", "evaluating"}:
        raise RuntimeError(f"cannot evaluate from state {state['status']}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    from run_v133_goal_p_variant import SearchTextAliasClient

    product = cfg["product"]
    client = TransformersModelClient(LocalModelConfig(
        model_id=str(q.p.BASE), adapter_path=str(SOURCE_ADAPTER),
        max_input_tokens=product["max_input_tokens"], max_new_tokens=product["max_new_tokens"],
        enable_thinking=False, deterministic=True, observation_encoding=product["observation_encoding"],
        recompute_old_logprobs=False))
    wrapped = SearchTextAliasClient(client)
    cases = read(RUN / "frozen-cases.json")
    outputs = RUN / f"eval-{label}.jsonl"
    if outputs.exists():
        raise FileExistsError(outputs)
    started = time.monotonic()
    rows_written = 0
    for split in SPLITS:
        corpus_path = original_paths()[split] if retriever == "bm25-v2" else paragraph_paths()[split]
        docs = json.loads(corpus_path.read_text(encoding="utf-8"))
        registry = ToolRegistry()
        registry.register(SearchTool(docs, retriever))
        for row in cases[split]:
            result = asyncio.run(q.rollout(row["question"], wrapped, registry, sample=False, seed=42))
            retrieved_titles = {str(doc.get("title", "")) for step in result["trace"]
                                for doc in step.get("observation", {}).get("result", [])}
            result.update(case_id=row["case_id"], split=split, model=label,
                          support_titles=row["support_titles"],
                          support_retrieved=bool(row["support_titles"]) and set(row["support_titles"]).issubset(retrieved_titles),
                          **q.score(result["answer"], row["answer"]))
            with outputs.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            rows_written += 1
            if rows_written % 8 == 0:
                print(json.dumps({"model": label, "completed": rows_written, "total": 208,
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}, ensure_ascii=True), flush=True)
    del client, wrapped
    gc.collect(); torch.cuda.empty_cache()
    if rows_written != 208:
        raise RuntimeError(f"incomplete evaluation for {label}: {rows_written}/208")
    state = read(STATE)
    state.setdefault("arms", {})[label] = {"status": "completed", "output_sha256": sha(outputs),
                                           "rows": rows_written, "seconds": time.monotonic() - started,
                                           "retriever": retriever}
    state["completed"] = sum(x.get("rows", 0) for x in state["arms"].values())
    state["status"] = "evaluation_complete" if len(state["arms"]) == 2 else "evaluating"
    (RUN / "state.json").write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def summarize() -> dict:
    config()
    report = {"version": "goal-ae-corpus-repair-v1", "fresh_goal_ab_confirmation_accessed": False,
              "arms": {}, "note": "EM/F1 and retrieval diagnostics are mechanical; semantic evidence review is separate."}
    for label in LABELS:
        path = RUN / f"eval-{label}.jsonl"
        if not path.exists():
            report["arms"][label] = {"status": "not_run"}
            continue
        values = read_rows(path)
        by_split = {}
        for split in SPLITS:
            rows = [x for x in values if x["split"] == split]
            by_split[split] = {"n": len(rows), "em": sum(x["em"] for x in rows),
                "mean_f1": statistics.mean([x["f1"] for x in rows]) if rows else 0.,
                "support_retrieved": sum(x["support_retrieved"] for x in rows),
                "invalid_actions": sum(x["reason"] == "invalid_action" for x in rows),
                "unanswered": sum(not x.get("answer") for x in rows),
                "searches": sum(x["searches"] for x in rows),
                "mean_latency_seconds": statistics.mean([x["elapsed_seconds"] for x in rows]) if rows else 0.}
        report["arms"][label] = {"n": len(values), "by_split": by_split,
                                   "em_total": sum(x["em"] for x in values), "output_sha256": sha(path)}
    out = RUN / "summary.json"
    if out.exists():
        out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    else:
        write_once(out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "summarize":
        summarize()
    else:
        if not CFG.exists():
            prepare()
        if not (RUN / "eval-bm25_v2.jsonl").exists():
            evaluate_arm("bm25-v2", "bm25_v2")
        if not (RUN / "eval-bm25_v3_paragraph.jsonl").exists():
            evaluate_arm("bm25-v3-paragraph", "bm25_v3_paragraph")
        summarize()


if __name__ == "__main__":
    main()
