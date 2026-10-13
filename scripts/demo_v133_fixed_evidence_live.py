"""Generate one fixed-evidence case with the three frozen adapters on AutoDL."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import set_seed

from run_v133_goal_n_v118_grpo import (J_ADAPTER, OUT, PATHS, STATUS, V118,
                                       generate, load_client, load_config,
                                       read_rows, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "new_eval", "goal_G_regression"), default="new_eval")
    parser.add_argument("--case-id", default="5ac00c72554299012d1db589")
    args = parser.parse_args()
    cfg = load_config()
    state = json.loads(STATUS.read_text(encoding="utf-8"))
    if state["status"] not in {"EXPERIMENT_COMPLETE", "COMPLETED_EXPERIMENT"} or state["gpu_seconds_used"] >= cfg["total_gpu_cap_seconds"]:
        raise RuntimeError("Goal N checkpoint or GPU budget unavailable")
    row = next((item for item in read_rows(PATHS[args.split]) if item["case_id"] == args.case_id), None)
    if row is None:
        parser.error("case ID is absent from the named frozen split")
    set_seed(cfg["seed"])
    started = time.monotonic()
    outputs = []
    arms = (("v118", V118), ("goal_J_sft", J_ADAPTER), ("goal_N_grpo", OUT / "adapter-final"))
    try:
        for arm, adapter in arms:
            if state["gpu_seconds_used"] + time.monotonic()-started >= cfg["total_gpu_cap_seconds"]:
                raise TimeoutError("Goal N total GPU budget")
            client = load_client(adapter, cfg)
            item = generate(client, row, cfg, sample=False)
            outputs.append({"adapter": arm, "action": item["raw"], "kind": item["action"],
                            "strict_em": item["em"], "f1": item["f1"],
                            "prompt_tokens": item["prompt_tokens"], "generation_seconds": item["generation_seconds"]})
            del client
            gc.collect(); torch.cuda.empty_cache()
        result = {"mode": "live_generation", "scope": "fixed evidence and synthetic two-search history",
                  "split": args.split, "case_id": row["case_id"], "question": row["messages"][1]["content"],
                  "evidence": row["evidence_documents"], "gold": row["official_answer"], "outputs": outputs}
        write_json(OUT / "demo-live-example.json", result)
        print(json.dumps({"status": "pass", "case_id": row["case_id"], "outputs": len(outputs)}))
    finally:
        state["gpu_seconds_used"] = round(state["gpu_seconds_used"] + time.monotonic()-started, 3)
        write_json(STATUS, state)


if __name__ == "__main__":
    main()
