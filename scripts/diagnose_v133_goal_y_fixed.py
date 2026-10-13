"""Replay Goal Y's exact incumbent answer prompts across frozen checkpoints."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import run_v133_goal_q_search as q
import run_v133_goal_s_label_granularity as s
import run_v133_goal_t_update_length as t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-y-three-class-transfer"


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run(label):
    if label not in ("incumbent", "42-a", "42-b", "43-a", "43-b"):
        raise ValueError(label)
    baseline = rows(OUT / "dev-incumbent.jsonl")
    cases = {item["case_id"]: item for item in rows(OUT / "dev-cases.jsonl")}
    if len(baseline) != 48 or len(cases) != 48:
        raise RuntimeError("Frozen baseline incomplete")
    adapter = t.checkpoint_dir(42, 32) / "adapter-final" if label == "incumbent" else OUT / f"seed{label}/adapter-final"
    target = OUT / f"fixed-{label}.jsonl"
    if target.exists(): raise FileExistsError(target)
    client, wrapped, registry = q.make_client(adapter, deterministic=True)
    client._load()
    try:
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            for item in baseline:
                final = item["trace"][-1]
                if final["action"] != "answer": raise RuntimeError("No baseline answer prompt")
                expected = final["prompt_token_ids"]
                result = s.fixed_answer(client, wrapped, registry, item["question"], item["trace"], expected)
                score = q.score(result["answer"], cases[item["case_id"]]["official_answer"])
                value = {"case_id": item["case_id"], "model": label,
                         "answer": result["answer"], "raw": result["raw"],
                         "invalid": result["invalid"], "input_tokens": result["input_tokens"],
                         "prompt_token_sha256": hashlib.sha256(json.dumps(expected).encode()).hexdigest(), **score}
                stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                print(json.dumps({"fixed_model": label, "case": item["case_id"]}), flush=True)
    finally:
        del client, wrapped, registry
        gc.collect()
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=("incumbent", "42-a", "42-b", "43-a", "43-b"))
    run(parser.parse_args().label)
