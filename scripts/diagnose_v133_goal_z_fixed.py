"""Replay incumbent's exact Goal Z development answer contexts after training."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json

import run_v133_goal_s_label_granularity as s
import run_v133_goal_z_multihop_sft as z


def run(label: str) -> None:
    if label not in ("incumbent", "42-answer", "42-trajectory", "43-answer", "43-trajectory"):
        raise ValueError(label)
    baseline = z.rows(z.OUT / "eval-dev-incumbent.jsonl")
    cases = {v["id"]: v for v in z.cases("dev")}
    if len(baseline) != 64 or len(cases) != 64:
        raise RuntimeError("Development baseline incomplete")
    adapter = z.SOURCE if label == "incumbent" else z.OUT / f"seed{label}/adapter-final"
    target = z.OUT / f"fixed-dev-{label}.jsonl"
    if target.exists():
        raise FileExistsError(target)
    client, wrapped, registry = z.make_client(adapter, z.OUT / "corpus-dev.json")
    client._load()
    try:
        for item in baseline:
            final = item["trace"][-1]
            if final.get("action") != "answer":
                z.append(target, {"case_id": item["case_id"], "model": label,
                                  "answer": None, "reason": "incumbent_no_final_answer"})
                continue
            expected = final["prompt_token_ids"]
            result = s.fixed_answer(client, wrapped, registry, item["question"], item["trace"], expected)
            z.append(target, {"case_id": item["case_id"], "model": label,
                              "answer": result["answer"], "raw": result["raw"],
                              "invalid": result["invalid"], "input_tokens": result["input_tokens"],
                              "prompt_sha256": hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
                              **z.q.score(result["answer"], cases[item["case_id"]]["answer"])})
            print(json.dumps({"fixed": label, "case": item["case_id"]}), flush=True)
    finally:
        del client, wrapped, registry
        gc.collect()
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", choices=("incumbent", "42-answer", "42-trajectory",
                                          "43-answer", "43-trajectory"))
    run(parser.parse_args().label)
