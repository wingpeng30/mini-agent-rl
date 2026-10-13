"""Evaluate the untouched base 0.8B after the original experiment exits."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import run_final_online_grpo as run

LABEL = "base-untrained"
TERMINAL = {"COMPLETED", "CUTOFF", "FAILED"}


def pid_alive(pid):
    # A Linux zombie has exited even while its parent has not yet reaped it.
    stat = Path(f"/proc/{pid}/stat")
    if sys.platform.startswith("linux") and stat.exists():
        try:
            if stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return False
        except (FileNotFoundError, IndexError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_original(args, budget, status):
    path = args.run_dir / "status.json"
    while True:
        budget.check()
        original = run.read(path) if path.exists() else {}
        alive = pid_alive(args.wait_pid)
        if original.get("status") in TERMINAL and not alive:
            status.update(original_status=original["status"], original_stopped_at=original.get("stopped_at", original.get("completed_at")),
                          original_pid_exited=True, wait_completed_at=time.time())
            return
        time.sleep(min(5., max(.01, args.deadline_epoch - time.time())))


def base_identity(base):
    files = sorted(base.glob("*.safetensors")) or sorted(base.glob("pytorch_model*.bin"))
    if not files:
        raise FileNotFoundError(f"no base model weights found in {base}")
    hashes = {p.name: run.sha(p) for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(base_model_path=str(base.resolve()), base_weights_sha256=hashes, base_model_sha256=digest,
                parent_adapter=None, adapter_path=None, adapter_sha256=None)


def make_client(args):
    import torch
    from transformers import set_seed
    from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
    set_seed(42)
    product = run.PRODUCT
    client = TransformersModelClient(LocalModelConfig(model_id=str(args.base_model), adapter_path=None,
        max_input_tokens=product["max_input_tokens"], max_new_tokens=product["max_new_tokens"],
        enable_thinking=False, device="cuda", dtype="bfloat16", load_in_4bit=False,
        gradient_checkpointing=True, deterministic=True, timeout=None, recompute_old_logprobs=False,
        action_completion="first-json-v1", observation_encoding="legacy", observation_presentation="legacy-json"))
    started = time.monotonic()
    client._load()
    torch.cuda.synchronize()
    generation = client._model.generation_config
    for key, value in dict(top_p=1., top_k=0, typical_p=1., repetition_penalty=1.,
                           no_repeat_ngram_size=0, min_new_tokens=0, min_length=0,
                           suppress_tokens=None, begin_suppress_tokens=None, bad_words_ids=None).items():
        setattr(generation, key, value)
    client._model.eval()
    return client, dict(model_loading_seconds=time.monotonic() - started,
                        generation_config=generation.to_dict(), gpu_memory=client.gpu_memory_stats(),
                        policy_fingerprint=client.policy_fingerprint())


def summary(values, identity, *, elapsed, current_count, resumed, memory, reconstructed=False):
    total_seconds = sum(v["elapsed_seconds"] for v in values)
    return dict(status="COMPLETED", branch=LABEL, case_count=len(values), current_process_cases=current_count,
        evaluation_wall_seconds=elapsed, total_question_seconds=total_seconds, batch_size=1,
        questions_per_second=current_count / max(elapsed, 1e-9) if current_count else len(values) / max(elapsed, 1e-9),
        model_calls=sum(v["model_calls"] for v in values), searches=sum(v["searches"] for v in values),
        input_tokens=sum(v["input_tokens"] for v in values), output_tokens=sum(v["output_tokens"] for v in values),
        exact_match=sum(v["exact_match"] for v in values),
        normalized_exact_match=sum(v["normalized_exact_match"] for v in values),
        f1=statistics.mean(v["f1"] for v in values), gpu_memory=memory, resumed=resumed,
        timing_reconstructed=reconstructed, **identity)


def evaluate(args, budget, status):
    folder = args.run_dir / "branches" / LABEL
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "evaluation.jsonl"
    cases, existing = run.rows(args.eval_cases), run.rows(path)
    case_ids = {c["id"] for c in cases}
    done = {v["case_id"] for v in existing}
    if len(cases) != 128 or len(case_ids) != 128 or len(done) != len(existing) or not done <= case_ids:
        raise ValueError("expected the same 128 unique evaluation cases and a valid resumable ledger")
    identity = base_identity(args.base_model)
    configuration = dict(**identity, product=run.PRODUCT, eval_cases=str(args.eval_cases.resolve()),
        eval_cases_sha256=run.sha(args.eval_cases), eval_retrieval_config=str(args.eval_retrieval_config.resolve()),
        eval_retrieval_config_sha256=run.sha(args.eval_retrieval_config), seed=42, sample=False, batch_size=1)
    config_path = folder / "configuration.json"
    if config_path.exists() and run.read(config_path) != configuration:
        raise ValueError("bare evaluation identity changed from the existing ledger")
    run.write(config_path, configuration)
    if done == case_ids:
        summary_path = folder / "evaluation-summary.json"
        if not summary_path.exists():
            elapsed = sum(v["elapsed_seconds"] for v in existing)
            run.write(summary_path, summary(existing, identity, elapsed=elapsed, current_count=0,
                resumed=True, memory=None, reconstructed=True))
        status["case_count"] = len(existing)
        return
    budget.check()
    import torch
    import run_v133_goal_q_search as q
    from mini_agent_rl.retrieval.local import LocalRetrieval
    from mini_agent_rl.tools import SearchTool, ToolRegistry
    retrieval_config = run.read(args.eval_retrieval_config)
    started = time.monotonic()
    engine = LocalRetrieval(retrieval_config)
    engine.dense_index()
    run.write(folder / "retrieval-loading.json", dict(loading_seconds=time.monotonic() - started,
        config=retrieval_config, metadata=engine.metadata("hybrid-rrf-v1")))
    budget.check()
    client, loading = make_client(args)
    run.write(folder / "evaluation-loading.json", loading)
    registry = ToolRegistry()
    registry.register(SearchTool(engine.documents, "hybrid-rrf-v1", retrieval=engine))
    wrapped = q.SearchTextAliasClient(client)
    todo = [c for c in cases if c["id"] not in done]
    wall_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        for case in todo:
            budget.check()
            result = asyncio.run(q.rollout(case["question"], wrapped, registry,
                sample=False, seed=42, product_config=run.PRODUCT))
            trace = result["trace"]
            value = dict(result, case_id=case["id"], id=case["id"], gold=case["answer"],
                stratum=case.get("stratum", case.get("source", "unspecified")),
                policy_version=LABEL, batch_size=1, retrieval=engine.metadata("hybrid-rrf-v1"),
                **identity, **run.metrics(result["answer"], case),
                repeated_actions=sum(bool(s.get("repeated")) for s in trace),
                invalid_actions=sum(bool(s.get("parse_error")) for s in trace),
                input_tokens=sum(s.get("usage", {}).get("prompt_tokens", len(s.get("prompt_token_ids") or [])) for s in trace),
                output_tokens=sum(s.get("usage", {}).get("completion_tokens", len(s.get("action_token_ids") or [])) for s in trace))
            run.append(path, value)
            done.add(case["id"])
            status.update(case_count=len(done), updated_at=time.time())
            run.write(args.run_dir / "bare-status.json", status)
            run.log("bare_evaluation", branch=LABEL, done=len(done), total=len(cases),
                    case_id=case["id"], normalized_exact_match=value["normalized_exact_match"])
            budget.check()
        elapsed = time.monotonic() - wall_started
        run.write(folder / "evaluation-summary.json", summary(run.rows(path), identity, elapsed=elapsed,
            current_count=len(todo), resumed=bool(existing), memory=client.gpu_memory_stats()))
    finally:
        status["current_process_evaluation_seconds"] = time.monotonic() - wall_started
        status["gpu_memory"] = client.gpu_memory_stats()
        del wrapped, registry
        run.release(client)
        del client


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "base-model", "eval-cases", "eval-retrieval-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--train-deadline-epoch", type=float, required=True)
    args = parser.parse_args(argv)
    if args.wait_pid <= 0 or args.wait_pid == os.getpid():
        parser.error("--wait-pid must identify the original experiment process")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    budget = run.Budget(args)
    status = dict(status="WAITING", started_at=time.time(), wait_pid=args.wait_pid,
        deadline_epoch=args.deadline_epoch, train_deadline_epoch=args.train_deadline_epoch,
        parent_adapter=None, branch=LABEL, runner_sha256=run.sha(__file__))
    run.write(args.run_dir / "bare-status.json", status)
    try:
        wait_original(args, budget, status)
        status.update(status="RUNNING", evaluation_started_at=time.time())
        run.write(args.run_dir / "bare-status.json", status)
        evaluate(args, budget, status)
        status.update(status="COMPLETED", completed_at=time.time())
    except BaseException as exc:
        status.update(status="CUTOFF" if isinstance(exc, run.DeadlineReached) else "FAILED",
            stopped_at=time.time(), error=repr(exc), error_type=type(exc).__name__)
        raise
    finally:
        run.write(args.run_dir / "bare-status.json", status)
        run.log("bare_stopped", **status)


if __name__ == "__main__":
    main()
