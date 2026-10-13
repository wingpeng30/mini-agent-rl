"""Run one greedy search trace for each frozen Goal AG supplement question."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import run_v133_goal_af_retrieval_validation as af
import run_v133_goal_q_search as q
import run_v133_goal_z_multihop_sft as z
from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
from mini_agent_rl.tools import SearchTool, ToolRegistry


ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENTAL = ROOT / "runs/v133-goal-ag-failure-repair/supplemental"
OUT = SUPPLEMENTAL / "attempt-02"
CASES = SUPPLEMENTAL / "candidate-pool.jsonl"
CORPUS = SUPPLEMENTAL / "corpus.json"
OUTPUT = OUT / "start-model-traces.jsonl"
RUN_MANIFEST = OUT / "sample-config.json"
# z.SOURCE is Goal T's start point. The published Goal Z model is resolved
# through AF's AA-verified release path, then checked against its saved hash.
ADAPTER = af.PUBLISHED_ADAPTER


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not CASES.is_file() or not CORPUS.is_file():
        raise FileNotFoundError("Freeze the bounded supplement pool first")
    if OUTPUT.exists() or RUN_MANIFEST.exists():
        raise FileExistsError("Supplement sampling has already started; preserve its output")
    identity = af.validate_release_identity()
    if identity["adapter_sha256"] != "d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52":
        raise RuntimeError("Published Goal Z adapter identity mismatch")
    if Path(identity["release_adapter"]).resolve() != ADAPTER.resolve():
        raise RuntimeError("Published Goal Z adapter path does not match the AA release record")
    cases = rows(CASES)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if len(cases) != 128 or len(corpus) != 2048:
        raise RuntimeError(f"Frozen pool/corpus dimensions invalid: {len(cases)}/{len(corpus)}")

    # The frozen Goal Z/Goal Q manifests predate two narrowly scoped code
    # changes. Confirm that these are the only changed files and record their
    # current hashes. local.py only added nullable call timeout support;
    # search.py only added retrievers, leaving the BM25-v2 implementation intact.
    zcfg = json.loads(z.CFG.read_text(encoding="utf-8"))
    z_diffs = {}
    for name, path in z.frozen_files().items():
        actual = z.sha(path)
        if zcfg["files"].get(name) != actual:
            z_diffs[name] = {"frozen": zcfg["files"].get(name), "current": actual}
    if set(z_diffs) != {"model_code", "search_code"}:
        raise RuntimeError(f"Unexpected Goal Z frozen file differences: {sorted(z_diffs)}")
    qcfg = json.loads(q.CFG.read_text(encoding="utf-8"))
    if qcfg["hashes"]["local"] != zcfg["files"]["model_code"] or \
       qcfg["hashes"]["search"] != zcfg["files"]["search_code"]:
        raise RuntimeError("Goal Q and Goal Z do not share the same frozen pre-change code")
    if qcfg["product"] != zcfg["product"]:
        raise RuntimeError("Goal Q and Goal Z product settings differ")
    q_frozen = {
        "corpus": q.p.CORPUS, "cases": q.p.CASES,
        "j": q.J / "adapter_model.safetensors",
        "old_rl": q.OLD_RL / "adapter_model.safetensors",
        "train": q.p.J / "train-frozen.jsonl", "runner": Path(q.__file__),
    }
    for name, path in q_frozen.items():
        if q.sha(path) != qcfg["hashes"][name]:
            raise RuntimeError(f"Unexpected Goal Q frozen input difference: {name}")
    pc = zcfg["product"]
    expected_product = {"retriever": "bm25-v2", "system_prompt": "direct-v090",
                        "thinking": False, "observation_encoding": "legacy",
                        "max_input_tokens": 2048, "max_new_tokens": 128,
                        "max_model_calls_per_case": 4, "max_searches_per_case": 3,
                        "top_k": 3}
    if any(pc.get(key) != value for key, value in expected_product.items()):
        raise RuntimeError("Goal Z frozen product settings differ from the agreed Goal AG settings")
    sample_manifest = {
        "model_identity": "Goal Z full-trajectory seed 42",
        "adapter_path": str(ADAPTER),
        "adapter_sha256": identity["adapter_sha256"],
        "base_weight_sha256": identity["base_weight_sha256"],
        "candidate_pool_sha256": af.sha(CASES), "corpus_sha256": af.sha(CORPUS),
        "candidate_count": len(cases), "corpus_documents": len(corpus),
        "product": expected_product | {"seed": 42, "sampling": False, "temperature": 0,
                                      "timeout": None, "dtype": "bfloat16", "quantization": False},
        "known_frozen_code_differences": z_diffs,
        "code_behavior_review": {
            "model_local_py": "only adds timeout=None support; generation logic unchanged",
            "search_py": "adds BM25 paragraph retrievers; BM25-v2 scoring implementation unchanged",
        },
        "confirmation_questions_opened": False,
    }
    if args.prepare_only:
        print(json.dumps({"cases": len(cases), "corpus": len(corpus),
                          "adapter_sha256": identity["adapter_sha256"],
                          "known_code_differences": z_diffs}))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    RUN_MANIFEST.write_text(json.dumps(sample_manifest, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    client = TransformersModelClient(LocalModelConfig(
        model_id=str(q.p.BASE), adapter_path=str(ADAPTER),
        max_input_tokens=pc["max_input_tokens"], max_new_tokens=pc["max_new_tokens"],
        enable_thinking=False, deterministic=True, observation_encoding=pc["observation_encoding"],
        recompute_old_logprobs=False, timeout=None))
    registry = ToolRegistry()
    registry.register(SearchTool(corpus, pc["retriever"]))
    wrapped = q.SearchTextAliasClient(client)
    # rollout reads only its immutable product config; validation above checked
    # every Goal Q data input and runner hash except the two reviewed code files.
    q.load_config = lambda: qcfg
    loaded = af.load_and_verify_adapter(client)
    if loaded["adapter_sha256"] != identity["adapter_sha256"]:
        raise RuntimeError("Adapter on first model load is not the published Goal Z adapter")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
            for index, case in enumerate(cases):
                result = asyncio.run(q.rollout(case["question"], wrapped, registry,
                                               sample=False, seed=42))
                result.update(case_id=case["id"], source=case["source"],
                              supplement_index=index,
                              model_identity="goal-z-trajectory-seed42",
                              adapter_sha256=loaded["adapter_sha256"],
                              corpus_sha256=af.sha(CORPUS),
                              sample_config_sha256=af.sha(RUN_MANIFEST))
                stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                print(json.dumps({"sampled": index + 1, "total": len(cases),
                                  "seconds": round(time.monotonic() - started, 2)}), flush=True)
    except BaseException:
        # Keep partial output and require a human review before any continuation.
        raise
    print(json.dumps({"output": str(OUTPUT), "rows": len(rows(OUTPUT)),
                      "elapsed_seconds": round(time.monotonic() - started, 2),
                      "adapter_sha256": loaded["adapter_sha256"]}))


if __name__ == "__main__":
    main()
