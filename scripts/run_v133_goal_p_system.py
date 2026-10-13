"""Goal P: shared-corpus live retrieval with a persistent Goal J model."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-p-system"
CORPUS = OUT / "corpus.json"
CASES = OUT / "eval-only-cases.jsonl"
CONFIG = OUT / "system-config.json"
J = ROOT / "runs/v133-goal-j-transfer"
G = ROOT / "runs/v133-goal-g-cpu/evaluation-frozen.jsonl"
BASE = Path("/root/autodl-tmp/qwen_08b")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def prepare():
    if CORPUS.exists() or CASES.exists() or CONFIG.exists():
        raise FileExistsError("P1 corpus/config already frozen")
    inputs = {"train": J / "train-frozen.jsonl", "new_eval": J / "evaluation-frozen.jsonl",
              "goal_G_regression": G}
    expected = {"train": "d4b41cb3413c9fca736d26b30e6d5e5292c1b47308d21c01f174f8957e0b263b",
                "new_eval": "42531fc1dbd2c3cb1fbec12e85523483c1c0b40546c55a0a6d622474d9648d29",
                "goal_G_regression": "f689f94b598a1c0d89f1b87f0b008c8a03158a7ab479028368285564244aad69"}
    for name, path in inputs.items():
        if sha(path) != expected[name]:
            raise RuntimeError(f"frozen input changed: {name}")
    document_map = {}
    cases = []
    for split, path in inputs.items():
        source = records(path)
        if len(source) != 16:
            raise RuntimeError(f"unexpected {split} length")
        for row in source:
            docs = row["evidence_documents"]
            if len(docs) != 2:
                raise RuntimeError("expected two support documents")
            ids = []
            for doc in docs:
                small = {key: str(doc[key]) for key in ("id", "title", "text")}
                old = document_map.setdefault(small["id"], small)
                if old != small:
                    raise RuntimeError("same document ID has conflicting content")
                ids.append(small["id"])
            cases.append({"split": split, "case_id": row["case_id"],
                          "question": row["messages"][1]["content"],
                          "official_answer": row["official_answer"], "support_ids": ids})
    docs = [document_map[key] for key in sorted(document_map)]
    OUT.mkdir(parents=True, exist_ok=True)
    write(CORPUS, docs)
    with CASES.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    config = {"version": "goal-p-system-v1", "corpus_scope": "shared_corpus_from_48_frozen_evidence_pairs",
              "document_count": len(docs), "case_count": len(cases), "inputs_sha256": expected,
              "corpus_sha256": sha(CORPUS), "eval_only_cases_sha256": sha(CASES),
              "goal_j_adapter_sha256": sha(J / "adapter-final/adapter_model.safetensors"),
              "runner_sha256": sha(Path(__file__)),
              "search_sha256": sha(ROOT / "src/mini_agent_rl/tools/search.py"),
              "local_client_sha256": sha(ROOT / "src/mini_agent_rl/model/local.py"),
              "base_config_sha256": sha(BASE / "config.json") if (BASE / "config.json").exists() else None,
              "tokenizer_sha256": sha(BASE / "tokenizer.json") if (BASE / "tokenizer.json").exists() else None,
              "system_prompt": "direct-v090", "model": "goal_J_sft", "retriever": "bm25-v2",
              "max_model_calls_per_case": 4, "max_searches_per_case": 3, "max_input_tokens": 2048,
              "max_new_tokens": 128, "seed": 42, "greedy": True, "thinking": False,
              "observation_encoding": "legacy", "top_k": 3, "candidate": "baseline"}
    write(CONFIG, config)
    print(json.dumps({"status": "P0_READY", "documents": len(docs), "cases": len(cases),
                      "config_sha256": sha(CONFIG)}))


def config():
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    checks = {"corpus_sha256": CORPUS, "eval_only_cases_sha256": CASES,
              "goal_j_adapter_sha256": J / "adapter-final/adapter_model.safetensors",
              "runner_sha256": Path(__file__), "search_sha256": ROOT / "src/mini_agent_rl/tools/search.py",
              "local_client_sha256": ROOT / "src/mini_agent_rl/model/local.py"}
    for key, path in checks.items():
        if sha(path) != c[key]:
            raise RuntimeError(f"frozen asset changed: {key}")
    return c


def make_components(c):
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    client = TransformersModelClient(LocalModelConfig(model_id=str(BASE), adapter_path=str(J / "adapter-final"),
        max_input_tokens=c["max_input_tokens"], max_new_tokens=c["max_new_tokens"],
        enable_thinking=False, deterministic=True, observation_encoding=c["observation_encoding"],
        recompute_old_logprobs=False))
    registry = ToolRegistry()
    registry.register(SearchTool(json.loads(CORPUS.read_text(encoding="utf-8")), c["retriever"]))
    return client, registry


async def ask(question, client, registry, c):
    from mini_agent_rl.domain import Message
    from mini_agent_rl.prompts import system_prompt
    messages = [Message(role="system", content=system_prompt(False, c["system_prompt"])),
                Message(role="user", content=question)]
    trace = []; seen = set(); started = time.monotonic(); answer = None; reason = "max_steps"
    for step in range(c["max_model_calls_per_case"]):
        before = time.monotonic()
        response = await client.generate(messages, registry.definitions(), temperature=0.0, seed=c["seed"] + step)
        item = {"step": step, "raw": response.raw_text, "parse_error": response.parse_error,
                "prompt_tokens": response.usage.get("prompt_tokens"),
                "completion_tokens": response.usage.get("completion_tokens"),
                "context_usage": response.usage, "model_seconds": round(time.monotonic()-before, 3),
                "prompt_token_sha256": hashlib.sha256(str(response.prompt_token_ids).encode()).hexdigest()}
        if response.parse_error:
            reason = "invalid_action"; trace.append(item); break
        if response.tool_call:
            query = str(response.tool_call.arguments.get("query", ""))
            key = " ".join(query.casefold().split())
            item.update(action="search", query=query, repeated=key in seen)
            if step >= c["max_searches_per_case"]:
                reason = "max_steps"; trace.append(item); break
            if key in seen:
                reason = "repeated_action"; trace.append(item); break
            seen.add(key)
            observation = await registry.execute("search", {"query": query})
            observation["searched_queries"] = sorted(seen)
            item["observation"] = observation
            trace.append(item)
            if not observation.get("ok"):
                reason = "tool_error"; break
            action_text = json.dumps({"action": "search", "query": query}, ensure_ascii=False)
            model_observation = {key: value for key, value in observation.items() if key != "latency_ms"}
            messages.extend([Message(role="assistant", content=action_text, tool_calls=[response.tool_call]),
                             Message(role="tool", content=json.dumps(model_observation, ensure_ascii=False),
                                     tool_call_id=response.tool_call.id)])
        else:
            answer = response.content
            item.update(action="answer", answer=answer)
            trace.append(item); reason = "answer"; break
    return {"question": question, "answer": answer, "reason": reason, "trace": trace,
            "model_calls": len(trace), "searches": sum("observation" in x for x in trace),
            "elapsed_seconds": round(time.monotonic()-started, 3)}


def run_eval():
    c = config()
    path = OUT / "p1-baseline.jsonl"
    if path.exists():
        raise FileExistsError("P1 baseline already started")
    cases = [row for row in records(CASES) if row["split"] != "train"]
    if len(cases) != 32:
        raise RuntimeError("expected 32 dev cases")
    from evaluate_v133_goal_c_sft import normalize_answer, f1_score
    client, registry = make_components(c)
    started = time.monotonic()
    for row in cases:
        if time.monotonic()-started >= 1200:
            raise TimeoutError("P1 baseline cap")
        try:
            item = asyncio.run(ask(row["question"], client, registry, c))
        except Exception as exc:
            item = {"question": row["question"], "answer": None, "reason": "model_error",
                    "error": str(exc), "trace": [], "model_calls": 0, "searches": 0, "elapsed_seconds": 0}
        answer = item["answer"] or ""
        item.update(case_id=row["case_id"], split=row["split"],
                    em=int(bool(answer) and normalize_answer(answer)==normalize_answer(row["official_answer"])),
                    f1=f1_score(answer, row["official_answer"]) if answer else 0.0)
        observed = {doc["id"] for step in item["trace"] for doc in (step.get("observation",{}).get("result") or [])}
        item["retrieved_support_count"] = len(observed.intersection(row["support_ids"]))
        item["both_support_retrieved"] = item["retrieved_support_count"] == 2
        append(path, item)
        print(json.dumps({"case_id":row["case_id"],"reason":item["reason"],"em":item["em"],
                          "searches":item["searches"]}), flush=True)
    output = records(path)
    summary = {"status":"PASSED","n":len(output),"em":sum(x["em"] for x in output),
               "f1":statistics.mean(x["f1"] for x in output),
               "both_support_retrieved":sum(x["both_support_retrieved"] for x in output),
               "reasons":{kind:sum(x["reason"]==kind for x in output) for kind in sorted({x["reason"] for x in output})},
               "model_calls":sum(x["model_calls"] for x in output),
               "searches":sum(x["searches"] for x in output),
               "elapsed_seconds":round(time.monotonic()-started,3),
               "gpu_peak_allocated_gib":client.gpu_memory_peak_gb()}
    write(OUT/"p1-summary.json", summary)
    print(json.dumps(summary))


def demo(question):
    c = config(); client, registry = make_components(c)
    result = asyncio.run(ask(question, client, registry, c))
    result["scope"] = "live_retrieval_from_shared_48_case_evidence_corpus"
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "eval", "demo"])
    parser.add_argument("--question")
    args=parser.parse_args()
    if args.command=="prepare": prepare()
    elif args.command=="eval": run_eval()
    elif args.question: demo(args.question)
    else: parser.error("--question is required for demo")


if __name__=="__main__":
    main()
