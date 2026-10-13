"""Goal BD: one bounded, fixed-weight final regression; never trains or publishes."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v133_goal_q_search as q

OUT = ROOT / "runs/v133-goal-bd-final-regression-v1"
BASE = Path("/root/autodl-tmp/qwen_08b")
SPLITS = [("goal_z_dev", 64), ("old_main", 16), ("old_regression", 16),
          ("goal_y_regression", 48), ("aa_regression", 64)]
ADAPTERS = {
    "A": ROOT / "runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final",
    "B": ROOT / "runs/v133-goal-am-action-sft-v2/seed42-A/adapter-final",
}
EXPECTED = {
    "A": "d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52",
    "B": "3bcd1f85382c1d7775fa674c1a6e1e68c80911a69b86de209d20f2ab6269280c",
}
DEMOS = ["musique:2hop__796221_18803", "2wiki:f299dd600bdd11eba7f7acde48001122",
         "musique:2hop__96194_78303"]
CATEGORIES = {"success", "factual_error", "unsupported_by_visible_evidence",
              "incomplete_required_qualifier", "invalid_action", "unconfirmed"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write(path, value, jsonl=False):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        if jsonl:
            for item in value:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()


def product():
    return read(OUT / "config.json")["product"]


def final_key(question, ids, answer):
    return hashlib.sha256(json.dumps([question, ids, answer], ensure_ascii=False).encode()).hexdigest()


def resolve_specs(freeze):
    by = {s["split"]: s for s in freeze["evaluation"]}
    selected = [dict(by[name]) for name, _ in SPLITS]
    if [(s["split"], s["count"]) for s in selected] != SPLITS:
        raise RuntimeError("BD fixed split sizes")
    if any("confirm" in s["cases"] or s["split"] == "ag_regression" for s in selected):
        raise RuntimeError("BD forbidden evaluation case source")
    return selected


def validate_reviews(packets, reviews):
    expected = {(p["arm"], p["case_id"]): p for p in packets}
    actual = {(r["arm"], r["case_id"]): r for r in reviews}
    if len(packets) != 58 or len(actual) != 58 or set(actual) != set(expected):
        raise RuntimeError("BD requires exactly 58 identity-linked verdicts")
    for key, p in expected.items():
        r = actual[key]
        if r["combination_sha256"] != p["combination_sha256"]:
            raise RuntimeError("BD verdict bound to different actual input")
        if r["category"] not in CATEGORIES or not r.get("reason"):
            raise RuntimeError("BD invalid/absent semantic verdict")
        if r["category"] == "success" and (not r.get("complete_chain_visible") or
                not r.get("edges") or not all(e.get("actual_visible") and e.get("citation") for e in r["edges"])):
            raise RuntimeError("BD success lacks per-edge visible citations")
    return actual


def prepare():
    if (OUT / "manifest.json").exists() or (OUT / "experiment-failure.json").exists():
        raise FileExistsError("BD already prepared/stopped")
    pc = product()
    required = dict(max_input_tokens=16384, max_new_tokens=2048, max_model_calls_per_case=4,
                    max_searches_per_case=3, system_prompt="direct-v090", thinking=False,
                    action_completion="first-json-v1", observation_encoding="legacy",
                    observation_presentation="legacy-json", retriever="bm25-v2", top_k=3,
                    dtype="bfloat16", load_in_4bit=False, device="cuda")
    if any(pc.get(k) != v for k, v in required.items()) or pc.get("context_aware_decoding"):
        raise RuntimeError("BD product configuration drift")
    aa = ROOT / "runs/v133-goal-aa-audit-confirmation/product-decision.json"
    decision = read(aa)
    if not decision["published"] or decision["selected_seed"] != 42 or decision["selected_route"] != "trajectory":
        raise RuntimeError("BD published model identity")
    specs = resolve_specs(read(ROOT / "runs/v133-goal-ah-entity-relation/execution-freeze.json"))
    all_cases = []
    paths = {"plan": OUT / "PLAN.md", "config": OUT / "config.json", "AA": aa,
             "AH": ROOT / "runs/v133-goal-ah-entity-relation/execution-freeze.json",
             "admission": ROOT / "runs/v133-goal-al-wide-context-v1/manifest.json"}
    for spec in specs:
        cases = rows(ROOT / spec["cases"])
        if len(cases) != spec["count"] or sha(ROOT / spec["corpus"]) != spec["corpus_sha256"]:
            raise RuntimeError("BD case/corpus archive identity")
        paths[spec["split"] + ":cases"] = ROOT / spec["cases"]
        paths[spec["split"] + ":corpus"] = ROOT / spec["corpus"]
        for case in cases:
            all_cases.append(dict(case, split=spec["split"]))
    if len(all_cases) != 208 or len({(r["split"], r["case_id"]) for r in all_cases}) != 208:
        raise RuntimeError("BD 208 fixed entries required")
    admission = read(paths["admission"])
    if len(admission["admitted"]) != 29 or len(admission["protected"]) != 7:
        raise RuntimeError("BD semantic/protection inventory")
    for arm, folder in ADAPTERS.items():
        if sha(folder / "adapter_model.safetensors") != EXPECTED[arm]:
            raise RuntimeError("BD adapter mismatch " + arm)
        for file in folder.glob("*.json"):
            paths[arm + ":" + file.name] = file
        paths[arm + ":weights"] = folder / "adapter_model.safetensors"
    for file in BASE.iterdir():
        if file.suffix in {".json", ".jinja", ".safetensors", ".txt"}:
            paths["base:" + file.name] = file
    for folder in (ROOT / "src", ROOT / "scripts"):
        for file in folder.rglob("*.py"):
            paths["code:" + file.relative_to(ROOT).as_posix()] = file
    paths["demo"] = ROOT / "scripts/demo_v133_final.py"
    paths["tests"] = ROOT / "tests/test_v133_goal_bd.py"
    for name in ("eval-42-A.jsonl", "semantic-review-seed42.jsonl", "review-packets-seed42-corrected.jsonl"):
        paths["AM:" + name] = ROOT / "runs/v133-goal-am-action-sft-v2" / name
    for name in ("trajectory-B.jsonl", "semantic-review.jsonl"):
        paths["AL:" + name] = ROOT / "runs/v133-goal-al-wide-context-v1" / name
    paths["probe_inputs"] = ROOT / "runs/v133-goal-bc-context-aware-decoding-v1/probe-inputs.jsonl"
    write(OUT / "evaluation-cases.jsonl", all_cases, True)
    paths["cases"] = OUT / "evaluation-cases.jsonl"
    manifest = dict(goal="BD", specs=specs, configuration=read(OUT / "config.json"),
                    paths={k: str(v) for k, v in paths.items()}, hashes={k: sha(v) for k, v in paths.items()},
                    adapters={a: str(p) for a, p in ADAPTERS.items()}, adapter_hashes=EXPECTED,
                    admitted=admission["admitted"], protected=admission["protected"], demo_ids=DEMOS,
                    product_model="Goal Z seed42", model_changed=False, training=0, teacher_calls=0,
                    new_confirmation=0, evaluation_use_for_training_forbidden=True)
    write(OUT / "manifest.json", manifest)
    write(OUT / "usage.jsonl", [], True)
    from transformers import AutoProcessor
    from mini_agent_rl.domain import Message
    from run_v133_goal_ap import parse_archived_messages, restore_tool_messages
    processor = AutoProcessor.from_pretrained(str(BASE), local_files_only=True)
    tok = getattr(processor, "tokenizer", processor)
    client, _, _ = q.make_client(ADAPTERS["A"], deterministic=True, product_config=pc,
                                 model_path=BASE, corpus=ROOT / specs[0]["corpus"])
    client._processor = processor
    client._model = SimpleNamespace(device="cpu")
    inputs = rows(paths["probe_inputs"])
    for item in inputs:
        messages = [Message(**m) for m in item["messages"]]
        encoded = client._inputs(messages, "bd_cpu")
        if encoded["input_ids"][0].tolist() != item["prompt_token_ids"]:
            raise RuntimeError("BD historical input reconstruction mismatch")
        restored = restore_tool_messages(parse_archived_messages(tok, item["prompt_token_ids"]))
        if client._inputs([Message(**m) for m in restored], "bd_roundtrip")["input_ids"][0].tolist() != item["prompt_token_ids"]:
            raise RuntimeError("BD archived-message roundtrip mismatch")
    write(OUT / "cpu-preflight.json", dict(passed=True, reconstructed_inputs=len(inputs),
          cases=208, admitted=29, protected=7, weights_loaded=False,
          forbidden_confirmation_accessed=False, manifest_sha256=sha(OUT / "manifest.json")))
    import torch, transformers, peft
    write(OUT / "environment.json", dict(python=platform.python_version(), torch=torch.__version__,
          transformers=transformers.__version__, peft=peft.__version__, cuda=torch.version.cuda,
          executable=sys.executable))
    print("BD prepared: 208 fixed cases, 32 exact archived inputs, zero model generation", flush=True)


def frozen():
    if (OUT / "experiment-failure.json").exists():
        raise RuntimeError("BD stopped; no automatic rerun")
    m = read(OUT / "manifest.json")
    for key, path in m["paths"].items():
        if sha(path) != m["hashes"][key]:
            raise RuntimeError("BD frozen asset drift: " + key)
    if not read(OUT / "cpu-preflight.json")["passed"]:
        raise RuntimeError("BD CPU preflight missing")
    return m


def attempt(kind, arm, split, cid):
    cfg = read(OUT / "config.json")
    used = rows(OUT / "usage.jsonl")
    if sum(r["kind"] == kind for r in used) >= cfg["caps"][kind]:
        raise RuntimeError("BD total quota exhausted")
    if sum(r["kind"] == kind and r["arm"] == arm for r in used) >= cfg["arm_caps"][kind]:
        raise RuntimeError("BD arm quota exhausted")
    append(OUT / "usage.jsonl", dict(kind=kind, arm=arm, split=split, case_id=cid,
           status="attempt_started", time=time.time()))


def evaluate(arm):
    m = frozen()
    target = OUT / f"evaluation-{arm}.jsonl"
    if target.exists() or (OUT / f"evaluation-{arm}-started.json").exists():
        raise FileExistsError("BD refuses to rerun " + arm)
    if arm == "B" and not (OUT / "evaluation-A-complete.json").exists():
        raise RuntimeError("BD must finish A before B")
    import torch
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BD single BF16 GPU required")
    write(OUT / f"evaluation-{arm}-started.json", dict(time=time.time(), arm=arm))
    write(target, [], True)
    loading = []
    for spec in m["specs"]:
        split = spec["split"]
        started = time.monotonic()
        client, wrapped, registry = q.make_client(Path(m["adapters"][arm]), deterministic=True,
              product_config=product(), model_path=BASE, corpus=ROOT / spec["corpus"])
        index_seconds = time.monotonic() - started
        started = time.monotonic()
        client._load()
        model_seconds = time.monotonic() - started
        torch.cuda.reset_peak_memory_stats()
        active = {"id": None}
        original_generate, original_execute = wrapped.generate, registry.execute
        original_inputs = client._inputs
        def captured_inputs(messages, *args, **kwargs):
            encoded = original_inputs(messages, *args, **kwargs)
            client._bd_messages = [x.model_dump(mode="json") for x in encoded._budgeted_messages]
            return encoded
        client._inputs = captured_inputs
        def finite_logits(module, args, output):
            if hasattr(output, "logits") and not torch.isfinite(output.logits[:, -1, :]).all().item():
                raise RuntimeError("BD nonfinite model logits")
        handle = client._model.register_forward_hook(finite_logits)
        async def counted_generate(messages, tools, **kwargs):
            attempt("generation", arm, split, active["id"])
            response = await original_generate(messages, tools, **kwargs)
            if len(response.prompt_token_ids) > product()["max_input_tokens"] or len(response.action_token_ids) > product()["max_new_tokens"]:
                raise RuntimeError("BD actual token limits exceeded")
            if not all(math.isfinite(x) for x in response.old_logprobs):
                raise RuntimeError("BD nonfinite generated probabilities")
            append(OUT / f"actual-inputs-{arm}.jsonl", dict(split=split, case_id=active["id"],
              prompt_token_ids=response.prompt_token_ids, messages=client._bd_messages,
              raw=response.raw_text, parsed_answer=response.content,
              parsed_search=response.tool_call.model_dump(mode="json") if response.tool_call else None))
            return response
        async def counted_execute(name, args):
            if name == "search":
                attempt("search", arm, split, active["id"])
            return await original_execute(name, args)
        wrapped.generate, registry.execute = counted_generate, counted_execute
        cases = [r for r in rows(OUT / "evaluation-cases.jsonl") if r["split"] == split]
        for case in cases:
            active["id"] = case["case_id"]
            result = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False,
                                          seed=42, product_config=product()))
            if len(result["trace"]) > 4 or result["searches"] > 3:
                raise RuntimeError("BD per-case action budget exceeded")
            result.update(case_id=case["case_id"], split=split, arm=arm,
                          official_answer=case["answer"], **q.score(result.get("answer"), case["answer"]))
            append(target, result)
            print(json.dumps(dict(arm=arm, split=split, case_id=case["case_id"],
                 done=len(rows(target)), em=result["em"], searches=result["searches"])), flush=True)
        loading.append(dict(split=split, model_loading_seconds=model_seconds, index_seconds=index_seconds,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        handle.remove()
        del client, wrapped, registry, handle, original_generate, original_execute, original_inputs
        gc.collect()
        torch.cuda.empty_cache()
    items = rows(target)
    if [(r["split"], r["case_id"]) for r in items] != [(r["split"], r["case_id"]) for r in rows(OUT / "evaluation-cases.jsonl")]:
        raise RuntimeError("BD result order/count drift")
    write(OUT / f"loading-{arm}.json", loading)
    write(OUT / f"evaluation-{arm}-complete.json", dict(cases=208, sha256=sha(target), arm=arm))


def metrics(items):
    calls = [t for r in items for t in r["trace"]]
    parse = sum(bool(t.get("parse_error")) for t in calls)
    repeats = sum(bool(t.get("repeated")) for t in calls)
    outside = sum(t.get("action") == "search" and t["step"] >= 3 for t in calls)
    return dict(cases=len(items), EM=sum(r["em"] for r in items), F1=statistics.mean(r["f1"] for r in items),
        action_problem_cases=sum(r["reason"] in {"invalid_action", "repeated_action"} or any(
            t.get("action") == "search" and t["step"] >= 3 for t in r["trace"]) for r in items),
        invalid_actions=parse, repeated_actions=repeats, overbudget_actions=outside,
        invalid_cases=sum(r["reason"] == "invalid_action" for r in items),
        repeated_cases=sum(r["reason"] == "repeated_action" for r in items),
        overbudget_cases=sum(any(t.get("action") == "search" and t["step"] >= 3 for t in r["trace"]) for r in items),
        calls=len(calls), searches=sum(r["searches"] for r in items),
        alias_recoveries=sum(bool(t.get("usage", {}).get("protocol_recovery")) for t in calls),
        max_input_tokens=max(len(t["prompt_token_ids"]) for t in calls),
        max_output_tokens=max(len(t["action_token_ids"]) for t in calls),
        length_stops=sum(t.get("usage", {}).get("generation_stop_reason") == "length" for t in calls),
        clipped_calls=sum(any(t.get("context_usage", t.get("usage", {})).get(k) for k in
                 ("lossy_body_trimmed", "documents_deleted", "history_dropped", "prompt_truncated")) for t in calls),
        mean_question_seconds=statistics.mean(r["elapsed_seconds"] for r in items),
        mean_generation_seconds=statistics.mean(t["model_seconds"] for t in calls))


def review_packets():
    m = frozen()
    for arm in ADAPTERS:
        if not (OUT / f"evaluation-{arm}-complete.json").exists():
            raise RuntimeError("BD both arms required before review")
    if (OUT / "review-packets.jsonl").exists():
        raise FileExistsError("BD one review pass only")
    from run_v133_goal_am import tokenizer_from_base
    from finalize_v133_goal_al_report import cite
    from prepare_v133_goal_ah import documents
    tok = tokenizer_from_base()
    am = ROOT / "runs/v133-goal-am-action-sft-v2"
    al = ROOT / "runs/v133-goal-al-wide-context-v1"
    am_eval = {r["case_id"]: r for r in rows(am / "eval-42-A.jsonl")}
    am_sem = {r["case_id"]: r for r in rows(am / "semantic-review-seed42.jsonl") if r["model"] == "42-A"}
    am_sources = {r["case_id"]: r for r in rows(am / "review-packets-seed42-corrected.jsonl") if r["model"] == "42-A"}
    al_eval = {r["case_id"]: r for r in rows(al / "trajectory-B.jsonl")}
    al_sem = {r["case_id"]: r["B"] for r in rows(al / "semantic-review.jsonl")}
    cache = {}
    for ev, sem in ((al_eval, al_sem), (am_eval, am_sem)):
        for cid in m["admitted"]:
            r = ev[cid]
            key = final_key(r["question"], r["trace"][-1]["prompt_token_ids"], r.get("answer"))
            cache[key] = sem[cid]
    packets = []
    reused = []
    for arm in ADAPTERS:
        for r in rows(OUT / f"evaluation-{arm}.jsonl"):
            if r["split"] != "goal_z_dev" or r["case_id"] not in m["admitted"]:
                continue
            cid = r["case_id"]
            ids = r["trace"][-1]["prompt_token_ids"]
            edges = []
            for edge in am_sources[cid]["edges"]:
                candidates = []
                for sem in (am_sem[cid], al_sem[cid]):
                    for e in sem["edges"]:
                        if e["edge_id"] == edge["edge_id"] and e.get("citation"):
                            candidates.append(e["citation"])
                citation = next((found for c in candidates if
                    (found := cite(ids, tok, c["document_id"], c["quote"])) is not None), None)
                first = next((t["step"] for t in r["trace"] if any(c["document_id"] == d.get("id") and
                    c["quote"] in d.get("text", "") for c in candidates for d in t.get("observation", {}).get("result", []))), None)
                edges.append(dict(edge_id=edge["edge_id"], subject=edge["subject"], relation=edge["relation"],
                    object=edge["object"], actual_visible=citation is not None, citation=citation,
                    first_return_search=first))
            key = final_key(r["question"], ids, r.get("answer"))
            p = dict(arm=arm, case_id=cid, question=r["question"], answer=r.get("answer"), reason=r["reason"],
                combination_sha256=key, complete_chain_visible=all(e["actual_visible"] for e in edges),
                edges=edges, documents=documents(tok.decode(ids, skip_special_tokens=False)),
                prompt_token_ids=ids, queries=[t["query"] for t in r["trace"] if t.get("query")],
                exact_cached_verdict=cache.get(key), training_use_forbidden=True)
            packets.append(p)
            if key in cache:
                old = cache[key]
                if old["category"] == "success" and not p["complete_chain_visible"]:
                    raise RuntimeError("BD inherited success citation mapping incomplete")
                reused.append(dict(arm=arm, case_id=cid, combination_sha256=key, category=old["category"],
                    reason=old["reason"], entity_relation_mismatch=old.get("entity_relation_mismatch", False),
                    complete_chain_visible=p["complete_chain_visible"], edges=edges, exact_cache_reuse=True))
    write(OUT / "review-packets.jsonl", packets, True)
    write(OUT / "exact-reused-verdicts.jsonl", reused, True)
    print(json.dumps(dict(packets=len(packets), exact_reused=len(reused), new_combinations=len(packets)-len(reused))), flush=True)


def report():
    m = frozen()
    packets = rows(OUT / "review-packets.jsonl")
    verdicts = validate_reviews(packets, rows(OUT / "semantic-adjudications.jsonl"))
    data = {a: rows(OUT / f"evaluation-{a}.jsonl") for a in ADAPTERS}
    stats = {a: {s: metrics([r for r in data[a] if r["split"] == s]) for s, _ in SPLITS} for a in ADAPTERS}
    for a in ADAPTERS:
        stats[a]["all208"] = metrics(data[a])
    sem = {}
    for a in ADAPTERS:
        v = [verdicts[(a, c)] for c in m["admitted"]]
        sem[a] = dict(supported_success=sum(r["category"] == "success" for r in v),
            error_total=sum(r["category"] in {"factual_error", "unsupported_by_visible_evidence"} for r in v),
            complete_chain_visible=sum(r["complete_chain_visible"] for r in v),
            unconfirmed=sum(r["category"] == "unconfirmed" for r in v),
            entity_relation_mismatch=sum(bool(r.get("entity_relation_mismatch")) for r in v))
    gained = [c for c in m["admitted"] if verdicts[("B", c)]["category"] == "success" and verdicts[("A", c)]["category"] != "success"]
    lost = [c for c in m["admitted"] if verdicts[("A", c)]["category"] == "success" and verdicts[("B", c)]["category"] != "success"]
    checks = dict(supported_net_plus_two=sem["B"]["supported_success"] >= sem["A"]["supported_success"]+2,
        errors_not_increased=sem["B"]["error_total"] <= sem["A"]["error_total"],
        retain_rapana_and_duke=all(verdicts[("B", c)]["category"] == "success" for c in DEMOS[:2]),
        protected_max_one_loss=sum(verdicts[("B", c)]["category"] != "success" for c in m["protected"]) <= 1,
        all_verdicts_confirmed=all(v["category"] != "unconfirmed" for v in verdicts.values()))
    for s, _ in SPLITS[1:]:
        checks[s+"_EM_max_one_loss"] = stats["B"][s]["EM"] >= stats["A"][s]["EM"]-1
    for k in ("action_problem_cases", "invalid_cases", "repeated_cases", "overbudget_cases",
              "invalid_actions", "repeated_actions", "overbudget_actions"):
        checks[k+"_not_increased"] = stats["B"]["all208"][k] <= stats["A"]["all208"][k]
    paired = []
    for x, y in zip(data["A"], data["B"]):
        if (x["split"], x["case_id"]) != (y["split"], y["case_id"]):
            raise RuntimeError("BD paired IDs/order")
        divergence = None
        for i in range(max(len(x["trace"]), len(y["trace"]))):
            tx = x["trace"][i] if i < len(x["trace"]) else None
            ty = y["trace"][i] if i < len(y["trace"]) else None
            if tx is None or ty is None or tx["raw"] != ty["raw"] or tx["prompt_token_ids"] != ty["prompt_token_ids"]:
                divergence = dict(call=i, same_input=bool(tx and ty and tx["prompt_token_ids"] == ty["prompt_token_ids"]),
                    A_raw=tx["raw"] if tx else None, B_raw=ty["raw"] if ty else None)
                break
        cid = x["case_id"]
        entry = dict(split=x["split"], case_id=cid, question=x["question"], A_answer=x["answer"], B_answer=y["answer"],
                     A_EM=x["em"], B_EM=y["em"], first_divergence=divergence)
        if x["split"] == "goal_z_dev" and cid in m["admitted"]:
            av, bv = verdicts[("A", cid)], verdicts[("B", cid)]
            same = x["trace"][-1]["prompt_token_ids"] == y["trace"][-1]["prompt_token_ids"]
            mechanism = "same_evidence_answer_selection" if same else "different_visible_evidence_or_search_path"
            if not av["complete_chain_visible"] and bv["complete_chain_visible"]:
                mechanism = "necessary_evidence_acquired_or_retained"
            if x["reason"] == "repeated_action" and bv["category"] == "success" and av["complete_chain_visible"]:
                mechanism = "answer_after_sufficient_evidence_instead_of_repeat"
            entry.update(A_category=av["category"], B_category=bv["category"], gained=cid in gained, lost=cid in lost,
                         mechanism=mechanism, A_edges=av["edges"], B_edges=bv["edges"])
        paired.append(entry)
    write(OUT / "paired-ledger.jsonl", paired, True)
    write(OUT / "metrics.json", stats)
    write(OUT / "decision.json", dict(checks=checks, local_gain_mechanical_protection=all(checks.values()),
        semantic_dev29=sem, gained=gained, lost=lost, model_published=False, product_model="Goal Z seed42",
        full_semantic_regression_protection="not_evaluated", historical_AM_stop_unchanged=True))
    write(OUT / "demo-replays.jsonl", [r for a in ADAPTERS for r in data[a]
          if r["split"] == "goal_z_dev" and r["case_id"] in DEMOS], True)
    count = {k: sum(r["kind"] == k for r in rows(OUT / "usage.jsonl")) for k in ("generation", "search")}
    table = "\n".join(f"| {s} | {n} | {stats['A'][s]['EM']} | {stats['B'][s]['EM']} | {stats['A'][s]['F1']:.4f} | {stats['B'][s]['F1']:.4f} |" for s, n in SPLITS)
    text = f"""# Goal BD：最终统一回归与面试版\n\n固定Goal Z与AM42-A，无训练、教师或新确认。所有208题已暴露；统一宽窗/first-json是研究配置，不改正式入口。\n\n| 集合 | 题数 | Goal Z EM | AM EM | Goal Z F1 | AM F1 |\n|---|---:|---:|---:|---:|---:|\n{table}\n\n开发29题证据支持：{sem['A']['supported_success']}→{sem['B']['supported_success']}；错误合计：{sem['A']['error_total']}→{sem['B']['error_total']}。支持新增{len(gained)}、丢失{len(lost)}。\n全208题动作问题：{stats['A']['all208']['action_problem_cases']}→{stats['B']['all208']['action_problem_cases']}。\n\n局部收益并通过机械回归保护：{all(checks.values())}。具体门槛见decision.json；旧144题只计EM/F1，不能称全面证据保护。开发其余35题来源未确认。原AM探针停止不撤销，正式模型始终Goal Z；不发布AM。\n\n本轮416轨迹，{count['generation']}次生成、{count['search']}次真实搜索、0更新、0教师、0新确认。加载/索引时间与显存见loading-A/B.json；延迟排除加载，不据小差异推断加速。\n\nAA历史确认12/64→35/64、36/64保持独立口径：当时冻结新题、有限语料、代理辅助审核。它与本轮EM及29题支持分数不能混算。\n\n## 复现与演示\n\n远程仓库使用.venv/bin/python，PYTHONPATH=src:scripts。CPU核验：scripts/run_v133_goal_bd.py verify。默认离线回放：scripts/demo_v133_final.py；--model am-research选择候选。--mode live复用生产循环；自定义--question必须提供--corpus。回放显示真实档案、研究配置、语料身份及历史延迟，不冒充实时推理。\n\n已有GPU结果拒绝重跑。原始轨迹、实际消息、逐token输出与usage.jsonl保留。评分只有实际问题—输入—答案的58组合，等价表达允许、标题和gold命中不代替关系证明。\n"""
    write(OUT / "REPORT.md", text)
    write(OUT / "report-complete.json", dict(generations=count["generation"], searches=count["search"],
          trajectories=416, report_sha256=sha(OUT / "REPORT.md"), decision_sha256=sha(OUT / "decision.json")))
    print(json.dumps(dict(semantic=sem, checks=checks, count=count), ensure_ascii=False), flush=True)


def verify():
    m = frozen()
    cases = rows(OUT / "evaluation-cases.jsonl")
    for a in ADAPTERS:
        items = rows(OUT / f"evaluation-{a}.jsonl")
        done = read(OUT / f"evaluation-{a}-complete.json")
        if len(items) != 208 or sha(OUT / f"evaluation-{a}.jsonl") != done["sha256"]:
            raise RuntimeError("BD incomplete/tampered trajectories")
        if [(r["split"], r["case_id"]) for r in items] != [(r["split"], r["case_id"]) for r in cases]:
            raise RuntimeError("BD case identity")
        inputs = rows(OUT / f"actual-inputs-{a}.jsonl")
        if len(inputs) != sum(len(r["trace"]) for r in items):
            raise RuntimeError("BD actual input coverage")
        for r in items:
            if len(r["trace"]) > 4 or r["searches"] > 3 or any(len(t["prompt_token_ids"]) > 16384 or
                    len(t["action_token_ids"]) > 2048 for t in r["trace"]):
                raise RuntimeError("BD actual budget invalid")
    validate_reviews(rows(OUT / "review-packets.jsonl"), rows(OUT / "semantic-adjudications.jsonl"))
    count = {k: sum(r["kind"] == k for r in rows(OUT / "usage.jsonl")) for k in ("generation", "search")}
    if count["generation"] > 1664 or count["search"] > 1248:
        raise RuntimeError("BD total cap")
    if count["generation"] != sum(len(r["trace"]) for a in ADAPTERS for r in rows(OUT / f"evaluation-{a}.jsonl")):
        raise RuntimeError("BD generated-call accounting")
    if count["search"] != sum(r["searches"] for a in ADAPTERS for r in rows(OUT / f"evaluation-{a}.jsonl")):
        raise RuntimeError("BD search accounting")
    result = dict(passed=True, trajectories=416, counts=count, semantic_combinations=58,
                  product_unchanged=True, training=0, teacher=0, new_confirmation=0)
    print(json.dumps(result), flush=True)
    return result


def package():
    result = verify()
    if (OUT / "freeze-manifest.json").exists():
        raise FileExistsError("BD package already frozen")
    names = {ROOT / "pyproject.toml", ROOT / "README.md", ROOT / ".gitignore"}
    for directory, suffixes in (("src", {".py"}), ("scripts", {".py"}), ("tests", {".py"}),
                                 ("configs", {".json", ".toml"}), ("docs", {".md"})):
        names.update(p for p in (ROOT / directory).rglob("*") if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts)
    for p in OUT.iterdir():
        if p.is_file() and p.name not in {"actual-inputs-A.jsonl", "actual-inputs-B.jsonl", "evaluation-A.jsonl", "evaluation-B.jsonl", "review-packets.jsonl"} and p.suffix in {".md", ".json", ".jsonl"}:
            names.add(p)
    files = []
    for p in sorted(names):
        relative = p.relative_to(ROOT)
        if any(part.startswith(".env") or part == "__pycache__" for part in relative.parts):
            raise RuntimeError("BD forbidden snapshot file")
        files.append(dict(path=relative.as_posix(), sha256=sha(p), size=p.stat().st_size))
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    git_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", "src", "scripts", "tests", "configs", "docs", "README.md", "pyproject.toml", ".gitignore"], cwd=ROOT)
    write(OUT / "workspace-state.json", dict(git_head=git_head, status=git_status,
          tracked_diff_sha256=hashlib.sha256(git_diff).hexdigest(), unrelated_work_preserved=True,
          auto_commit=False, auto_cleanup=False))
    (OUT / "tracked-code.patch").write_bytes(git_diff)
    assets = {key: dict(path=path, sha256=read(OUT / "manifest.json")["hashes"][key]) for key, path in read(OUT / "manifest.json")["paths"].items()
              if key.startswith(("base:", "A:", "B:")) or key.endswith((":corpus", ":cases"))}
    for a in ADAPTERS:
        for name in (f"evaluation-{a}.jsonl", f"actual-inputs-{a}.jsonl"):
            assets[name] = dict(path=str(OUT / name), sha256=sha(OUT / name))
    write(OUT / "remote-assets.json", assets)
    archive = OUT / "interview-code-snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for item in files:
            tar.add(ROOT / item["path"], arcname=item["path"], recursive=False)
        for name in ("workspace-state.json", "tracked-code.patch", "remote-assets.json"):
            tar.add(OUT / name, arcname=(OUT / name).relative_to(ROOT).as_posix(), recursive=False)
    write(OUT / "freeze-manifest.json", dict(files=files, snapshot_sha256=sha(archive), git_head=git_head,
         validation=result, environment=read(OUT / "environment.json"), product="Goal Z seed42",
         raw_assets_remote_only=True, excluded=[".env", "keys", "private_request_logs", "caches", "model_weights"],
         no_commit=True, no_release=True, no_more_experiments=True))
    print(json.dumps(dict(frozen=True, files=len(files), archive_bytes=archive.stat().st_size)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "evaluate-a", "evaluate-b", "review", "report", "verify", "package"])
    args = parser.parse_args()
    try:
        if args.stage == "prepare": prepare()
        elif args.stage.startswith("evaluate-"): evaluate(args.stage[-1].upper())
        elif args.stage == "review": review_packets()
        elif args.stage == "report": report()
        elif args.stage == "verify": verify()
        elif args.stage == "package": package()
    except Exception as exc:
        if OUT.exists() and not (OUT / "experiment-failure.json").exists():
            write(OUT / "experiment-failure.json", dict(stage=args.stage, error_type=type(exc).__name__,
                  error=str(exc), time=time.time(), automatic_retry=False, training=0))
        raise


if __name__ == "__main__":
    main()
