"""Complete Goal R order diagnostic after correcting JSON key-order replay."""
from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import time
from pathlib import Path

import run_v133_goal_r_evidence_order as r
import run_v133_goal_q_search as q
from evaluate_v133_goal_c_sft import normalize_answer, f1_score


def runtime_order(trace):
    """Restore SearchTool's id/title/text/score insertion order after sorted JSON persistence."""
    result = json.loads(json.dumps(trace, ensure_ascii=False))
    for step in result:
        if "observation" not in step: continue
        docs = []
        for value in step["observation"].get("result", []):
            if set(value) != {"id", "title", "text", "score"}:
                raise RuntimeError("unexpected search document schema")
            docs.append({key: value[key] for key in ("id", "title", "text", "score")})
        step["observation"]["result"] = docs
    return result


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    started = time.monotonic()
    r.config()
    state = r.read(r.STATE)
    if state["status"] != "stopped" or state["stage"] != "diagnostic":
        raise RuntimeError("repair only applies to the recorded diagnostic failure")
    if state["gpu_seconds"] + 900 + 300 > 7200:
        raise RuntimeError("insufficient remaining GPU budget")
    path = r.OUT / "order-diagnostic.jsonl"
    if path.exists(): raise FileExistsError("diagnostic output already started")
    baseline = r.rows(r.OUT / "eval-incumbent.jsonl")
    gold = {x["case_id"]: x for x in q.dev_cases()}
    if len(baseline) != 32: raise RuntimeError("baseline incomplete")
    for label, adapter in (("incumbent", r.START), ("a", r.OUT / "a/adapter-final"),
                           ("b", r.OUT / "b/adapter-final")):
        client, wrapped, registry = q.make_client(adapter, deterministic=True)
        client._load()
        for row in baseline:
            if time.monotonic() - started > 900: raise TimeoutError("diagnostic cap")
            trace = runtime_order(row["trace"])
            if not trace or trace[-1].get("action") != "answer":
                raise RuntimeError("baseline lacks an answer state")
            for mode in ("original", "reversed"):
                messages = r.rebuild(row["question"], trace, reverse=mode == "reversed")
                ids, mask = r.prompt_ids(client, messages)
                if mode == "original" and ids != row["trace"][-1]["prompt_token_ids"]:
                    raise RuntimeError(f"product prompt mismatch {row['case_id']}")
                response = asyncio.run(wrapped.generate(messages, registry.definitions(), temperature=0., seed=42))
                answer = response.content if not response.parse_error and not response.tool_call else None
                actual = gold[row["case_id"]]["official_answer"]
                r.append(path, {"model": label, "case_id": row["case_id"], "mode": mode,
                                "prompt_sha256": hashlib.sha256(str(ids).encode()).hexdigest(),
                                "raw": response.raw_text, "answer": answer,
                                "em": int(bool(answer) and normalize_answer(answer) == normalize_answer(actual)),
                                "f1": f1_score(answer, actual) if answer else 0.})
        del client, wrapped, registry
        gc.collect(); torch.cuda.empty_cache()
    values = r.rows(path)
    if len(values) != 192: raise RuntimeError("diagnostic incomplete")
    summary = {"status": "PASSED", "rows": 192, "models": {},
               "repair": "restore SearchTool document key insertion order for exact prompt replay",
               "script_sha256": sha(__file__), "elapsed_seconds": time.monotonic() - started}
    for label in ("incumbent", "a", "b"):
        original = {x["case_id"]: x for x in values if x["model"] == label and x["mode"] == "original"}
        reversed_ = {x["case_id"]: x for x in values if x["model"] == label and x["mode"] == "reversed"}
        if set(original) != set(gold) or set(reversed_) != set(gold): raise RuntimeError("diagnostic cases missing")
        summary["models"][label] = {"original_em": sum(x["em"] for x in original.values()),
            "reversed_em": sum(x["em"] for x in reversed_.values()),
            "changed_answer": sum(normalize_answer(original[k]["answer"] or "") !=
                                  normalize_answer(reversed_[k]["answer"] or "") for k in gold)}
    r.write(r.OUT / "order-diagnostic-summary.json", summary)
    before_final = time.monotonic()
    r.stage("final")
    state["gpu_seconds"] += time.monotonic() - started
    state["stages"].append({"name": "diagnostic_repair", "seconds": before_final - started,
                             "reason": "JSON key-order replay correction", "script_sha256": sha(__file__)})
    state["stages"].append({"name": "final", "seconds": time.monotonic() - before_final})
    state.update(status="complete", stage="complete")
    state.pop("error", None)
    r.write(r.STATE, state)
    r.summarize()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__": main()
