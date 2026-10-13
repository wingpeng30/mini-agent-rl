"""Goal AC: train repairs from real policy failure states and gated evaluation."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import run_v133_goal_ab_evidence_decision as ab
import run_v133_goal_aa_audit_confirmation as aa
import run_v133_goal_q_search as q
import run_v133_goal_z_multihop_sft as z
import prepare_v133_goal_z_multihop_data as source
import curate_v133_goal_z_multihop as curator
import run_v133_goal_s_label_granularity as s

ROOT = z.ROOT
AB = ab.OUT
OUT = ROOT / "runs/v133-goal-ac-failure-state-repair"
CFG, STATE = OUT / "config.json", OUT / "state.json"
SOURCE = z.OUT / "seed42-trajectory/adapter-final"
SPLITS = ("dev", "old", "y", "aa_regression")
LABELS = ("baseline", "42-A", "42-B", "43-A", "43-B", "42-C", "43-C")
EVENTS = set(ab.EVENTS)
REVIEW_CATEGORIES = set(aa.CATEGORIES)
MIN_PAIRS, MAX_PAIRS, MIN_STOP_REPAIRS = 32, 64, 16
MIN_DEV_GAIN = 4


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    content = Path(path).read_text(encoding="utf-8")
    if content.lstrip().startswith("["):
        value = json.loads(content)
        if not isinstance(value, list):
            raise ValueError(f"Expected JSON array: {path}")
        return value
    return [json.loads(x) for x in content.splitlines() if x.strip()]


def write(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def append(path: Path, value) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def write_jsonl(path: Path, values: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def set_state(status: str, **updates) -> dict:
    value = z.read(STATE)
    value.update(status=status, **updates)
    write(STATE, value)
    return value


def required_assets() -> dict[str, Path]:
    values = {"source_adapter": SOURCE / "adapter_model.safetensors",
              "ac_runner": Path(__file__),
              "ab_runner": Path(ab.__file__), "goal_z_runner": Path(z.__file__),
              "search_runner": Path(q.__file__), "local_model": ROOT / "src/mini_agent_rl/model/local.py",
              "tokenizer": q.p.BASE / "tokenizer.json",
              "train_cases": z.OUT / "selected-train.jsonl",
              "train_inputs": z.OUT / "training-inputs.jsonl",
              "train_corpus": z.OUT / "corpus-train.json",
              "dev_cases": z.OUT / "selected-dev.jsonl", "dev_corpus": z.OUT / "corpus-dev.json",
              "aa_cases": z.OUT / "selected-confirm.jsonl", "aa_corpus": z.OUT / "corpus-confirm.json",
              "ab_state": AB / "state.json", "ab_final_review": AB / "development-review-final.jsonl",
              "ab_review_map": AB / "development-review-map.json",
              "ab_pair_decision": AB / "pair-decision.json",
              "ab_decision_pairs": AB / "decision-pairs.jsonl",
              "ab_train_packet": AB / "train-decision-packet.jsonl",
              "ab_train_reviews": AB / "train-decision-review.jsonl",
              "ab_onpolicy": AB / "train-on-policy.jsonl",
              "ab_confirm_cases": AB / "fresh-confirm-cases.jsonl",
              "ab_confirm_corpus": AB / "fresh-confirm-corpus.json",
              "ab_confirm_manifest": AB / "fresh-confirm-manifest.json",
              "musique_source": source.MUSIQUE, "2wiki_source": source.WIKI}
    values.update({f"base_weights_{i}": p for i, p in enumerate(sorted(q.p.BASE.glob("*.safetensors")))})
    for key, path in list(values.items()):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required frozen asset absent: {key}: {path}")
    return values


def frozen_ab_scores() -> tuple[dict, dict]:
    if z.read(AB / "state.json")["status"] != "complete_development_gate_failed":
        raise RuntimeError("Goal AB is not in the expected terminal development state")
    if z.read(AB / "fresh-confirm-manifest.json").get("confirmation_accessed"):
        raise RuntimeError("Goal AB fresh confirmation has already been accessed")
    counts, by_case = ab.scored("development")
    baseline = {key: value for key, value in counts.items() if key.endswith("-baseline")}
    expected = {"dev-baseline": (36, 19, 0), "old-baseline": (27, 5, 0),
                "y-baseline": (40, 8, 0), "aa_regression-baseline": (32, 23, 0)}
    for key, (success, bad, invalid) in expected.items():
        value = baseline.get(key)
        if not value or (value.get("correct_sufficient", 0), value["bad_total"], value["invalid"]) != (success, bad, invalid):
            raise RuntimeError(f"AB frozen baseline mismatch: {key}: {value}")
    return counts, by_case


def prepare_extra_candidates(exclusion_path: Path) -> dict:
    """Freeze at most 64 eligible official train candidates per source once."""
    ids_hist, questions_hist, titles_hist, entities_hist = source.w.history_exposure()
    ids_hist, questions_hist, titles_hist = set(ids_hist), set(questions_hist), set(titles_hist)
    # The fresh confirmation is opened only by this isolated CPU data-audit step;
    # downstream training sees this hash-only index and its selected train cases.
    fresh = rows(AB / "fresh-confirm-cases.jsonl")
    if len(fresh) != 64:
        raise RuntimeError("Frozen fresh confirmation has wrong size")
    def h(value: str) -> str:
        return hashlib.sha256(source.norm(value).encode()).hexdigest()
    exclusion = {"source_ids": set(), "questions": set(), "entities": set(), "titles": set()}
    for case in fresh:
        exclusion["source_ids"].update(h(str(x)) for x in case["source_ids"])
        exclusion["questions"].add(h(case["question"]))
        exclusion["titles"].update(h(str(x)) for x in case["support_titles"])
        exclusion["entities"].update(h(str(x)) for x in [case["question"], *case["support_titles"]])
    for item in sorted(ids_hist): exclusion["source_ids"].add(h(str(item)))
    for item in sorted(questions_hist): exclusion["questions"].add(h(str(item)))
    for item in sorted(titles_hist): exclusion["titles"].add(h(str(item)))
    for item in sorted(entities_hist): exclusion["entities"].add(h(str(item)))
    frozen_index = {k: sorted(v) for k, v in exclusion.items()}
    write(exclusion_path, frozen_index)

    accepted_by_source: dict[str, list[dict]] = {}
    excluded = Counter()
    for name, loader in (("musique", source.load_musique), ("2wiki", source.load_wiki)):
        pool = [source.canonical(name, row) for row in loader("train") if source.source_eligible(name, row)]
        pool.sort(key=lambda row: source.stable("goal-ac-extra:" + row["id"]))
        picked = []
        local_titles: set[str] = set()
        local_entities: set[str] = set()
        for case in pool:
            if len(picked) >= 64:
                break
            if h(case["id"]) in exclusion["source_ids"] or h(case["question"]) in exclusion["questions"]:
                excluded["history_id_or_question"] += 1
                continue
            supports = [d for d in case["docs"] if d["support"]]
            stitles = {source.norm(d["title"]) for d in supports}
            if len(supports) != 2 or len(stitles) != 2 or stitles & local_titles:
                excluded["invalid_or_repeated_support_title"] += 1
                continue
            if any(h(t) in exclusion["titles"] for t in stitles):
                excluded["historical_support_title"] += 1
                continue
            support_titles = [d["title"] for d in supports]
            ents = {source.norm(x) for x in [case["question"], *support_titles] if source.norm(x)}
            if ents & local_entities or any(h(x) in exclusion["entities"] for x in ents):
                excluded["historical_entity_or_local_overlap"] += 1
                continue
            picked.append(case)
            local_titles.update(stitles)
            local_entities.update(ents)
        if len(picked) != 64:
            raise RuntimeError(f"Fixed extra candidate pool shortfall for {name}: {len(picked)}/64")
        accepted_by_source[name] = picked
    flat = accepted_by_source["musique"] + accepted_by_source["2wiki"]
    write(OUT / "extra-candidate-pool.json", {"counts": {k: len(v) for k, v in accepted_by_source.items()},
          "case_sha256": None, "excluded": dict(excluded),
          "cases": [{k: v for k, v in item.items() if k != "docs"} |
                    {"support_docs": [d for d in item["docs"] if d["support"]],
                     "distractor_docs": [d for d in item["docs"] if not d["support"]]}
                    for item in flat]})
    pool_path = OUT / "extra-candidate-pool.json"
    result = {"counts": {k: len(v) for k, v in accepted_by_source.items()},
              "pool_sha256": digest(pool_path), "exclusion_sha256": digest(exclusion_path),
              "excluded": dict(excluded)}
    value = z.read(pool_path); value["case_sha256"] = result["pool_sha256"]; write(pool_path, value)
    result["pool_sha256"] = digest(pool_path)
    return result


def prepare() -> None:
    # Reviewer inputs were collected before prepare. Preserve those immutable
    # audit files, while refusing to reuse a directory once AC has started.
    if OUT.exists() and any((OUT / name).exists() for name in
            ("config.json", "state.json", "failure-audit-packet.jsonl")):
        raise FileExistsError(f"Goal AC already started: {OUT}")
    baseline, _ = frozen_ab_scores()
    assets = required_assets()
    manifest = z.read(AB / "fresh-confirm-manifest.json")
    if manifest.get("confirmation_accessed") or digest(AB / "fresh-confirm-cases.jsonl") != manifest["case_sha256"] or \
       digest(AB / "fresh-confirm-corpus.json") != manifest["corpus_sha256"]:
        raise RuntimeError("Fresh confirmation is not sealed with its frozen hashes")
    OUT.mkdir(parents=True, exist_ok=True)
    write(CFG, {"version": "goal-ac-v1", "source_adapter": str(SOURCE),
        "asset_hashes": {k: digest(v) for k, v in assets.items()},
        "software": z.software(), "product": q.load_config()["product"],
        "baseline_counts": baseline, "confirmation_case_sha256": manifest["case_sha256"],
        "confirmation_corpus_sha256": manifest["corpus_sha256"],
        "confirmation_was_sealed": True, "seeds": [42, 43],
        "max_updates_per_seed": 128, "max_extra_rollouts": 128,
        "min_pairs": MIN_PAIRS, "max_pairs": MAX_PAIRS,
        "min_stop_error_repairs": MIN_STOP_REPAIRS,
        "max_confirmation_outputs": 192,
        "optimization": {"learning_rate": 5e-6, "betas": [0.9, 0.999],
            "eps": 1e-8, "weight_decay": 0., "gradient_clip": 1.},
        "order": z.read(AB / "config.json")["order"], "created_at": time.time()})
    write(OUT / "baseline-counts.json", baseline)
    write(OUT / "ab-frozen-hashes.json", {k: digest(v) for k, v in assets.items()})
    audit_rows = rows(AB / "train-decision-packet.jsonl")
    old_votes = {x["case_id"]: x for x in rows(AB / "train-decision-review.jsonl")}
    if len(audit_rows) != 128 or len(old_votes) != 128:
        raise RuntimeError("Frozen AB training audit incomplete")
    # Reviewer packets never contain the hidden training answer or support mapping.
    for item in audit_rows:
        steps = [{k: step[k] for k in ("step", "action", "query", "answer", "actual_prompt", "visible_document_ids")}
                 for step in item["steps"]]
        append(OUT / "failure-audit-packet.jsonl", {"case_id": item["case_id"],
            "question": item["question"], "reason": item["on_policy_reason"], "steps": steps})
    extra = prepare_extra_candidates(OUT / "sealed-exclusion-index.json")
    write(OUT / "extra-pool-summary.json", extra)
    write(STATE, {"status": "awaiting_failure_audit", "confirmation_accessed": False,
        "stages": [{"name": "AC0_prepare", "seconds": time.time() - z.read(CFG)["created_at"]}],
        "extra_pool_frozen_before_model_rollouts": True})


def verify() -> dict:
    cfg = z.read(CFG)
    current = {k: digest(v) for k, v in required_assets().items()}
    if current != cfg["asset_hashes"]:
        raise RuntimeError("Goal AC frozen input asset hash changed")
    if cfg["software"] != z.software():
        raise RuntimeError("Goal AC software/GPU environment changed")
    manifest = z.read(AB / "fresh-confirm-manifest.json")
    if manifest.get("confirmation_accessed") or digest(AB / "fresh-confirm-cases.jsonl") != cfg["confirmation_case_sha256"] or \
       digest(AB / "fresh-confirm-corpus.json") != cfg["confirmation_corpus_sha256"]:
        raise RuntimeError("Goal AC confirmation set is no longer sealed")
    return cfg


def audited_inputs() -> dict[str, dict]:
    packet = rows(OUT / "failure-audit-packet.jsonl")
    votes = rows(OUT / "failure-audit-final.jsonl")
    if len(packet) != 128 or len(votes) != 128 or len({x["case_id"] for x in votes}) != 128:
        raise RuntimeError("All 128 real failure trajectories require a main-adjudicated audit")
    by_id = {x["case_id"]: x for x in votes}
    if set(by_id) != {x["case_id"] for x in packet}:
        raise RuntimeError("Failure audit IDs mismatch")
    for item in packet:
        vote = by_id[item["case_id"]]
        if vote.get("event") not in EVENTS | {"none", "ambiguous", "no_recoverable_evidence"} or \
           not isinstance(vote.get("repairable_within_budget"), bool) or not vote.get("reason"):
            raise RuntimeError("Failure audit has invalid classification")
        index = {x["case_id"]: x for x in packet}[item["case_id"]]
        stepmap = {s["step"]: s for s in index["steps"]}
        if vote["event"] in {"premature_answer", "wrong_bridge_query", "unnecessary_search"}:
            if vote.get("repairable_within_budget"):
                step = stepmap.get(vote.get("error_step"))
                queries = vote.get("repair_queries") or ([{"query": vote.get("repair_query"),
                    "bridge_source": vote.get("bridge_source")}] if vote.get("repair_query") else [])
                if step is None or not queries or (vote["event"] != "unnecessary_search" and
                                                   not vote.get("missing_relation")):
                    raise RuntimeError("Repairable stop error lacks an exact-state correction proposal")
                if vote.get("bridge_entity") and not vote.get("bridge_source"):
                    raise RuntimeError("Bridge entity has no source citation")
    return by_id


def combine_failure_reviews() -> dict:
    verify()
    if z.read(STATE)["status"] != "awaiting_failure_audit":
        raise RuntimeError("Failure review is not the active stage")
    packet = {x["case_id"]: x for x in rows(OUT / "failure-audit-packet.jsonl")}
    a, b = rows(OUT / "audit-a.jsonl"), rows(OUT / "audit-b.jsonl")
    original = {x["case_id"]: x for x in rows(AB / "train-decision-review.jsonl")}
    if set(x["case_id"] for x in a) & set(x["case_id"] for x in b):
        raise RuntimeError("Failure reviewers must be disjoint")
    audited = {x["case_id"]: x for x in a + b}
    expected = {key for key, value in original.items() if value["on_policy_stop_error"]}
    if set(audited) != expected:
        raise RuntimeError("Both reviewers must cover all and only AB-marked failure states")
    votes = {}
    # Main agent adjudicates the label conflict using the actual-prompt rubric.
    for case_id, old in original.items():
        vote = audited.get(case_id, {"case_id": case_id, "event": "none",
            "error_supported": False, "repairable_within_budget": False,
            "reason": "Inherited frozen AB review: no stop-decision error."})
        if vote.get("event") == "premature_answer" and not vote.get("missing_relation"):
            # Reviewer identified a missing answer detail but left the typed
            # field blank. Preserve their verbatim explanation as the gap.
            vote["missing_relation"] = vote.get("reason", "")
            vote["main_adjudication_note"] = "missing_relation normalized from reviewer explanation"
        if vote.get("event") not in EVENTS | {"none", "ambiguous", "no_recoverable_evidence"}:
            raise RuntimeError(f"Invalid error event for {case_id}")
        if vote.get("error_supported") is False and old["on_policy_stop_error"]:
            vote["main_adjudication"] = "corrected_to_nonstop_or_unsupported"
        else:
            vote["main_adjudication"] = "reviewer_event_retained"
        vote["ab_original_stop_error"] = old["on_policy_stop_error"]
        vote["ab_original_reason"] = old["reason"]
        if vote.get("event") == "premature_answer":
            trace = rows(AB / "train-on-policy.jsonl")
            actual = next(x for x in trace if x["case_id"] == case_id)
            vote["error_step"] = len(actual["trace"]) - 1
        elif vote.get("event") == "wrong_bridge_query":
            trace = rows(AB / "train-on-policy.jsonl")
            actual = next(x for x in trace if x["case_id"] == case_id)
            searches = [i for i, step in enumerate(actual["trace"]) if step["action"] == "search"]
            vote["error_step"] = searches[-1] if searches else None
        elif vote.get("event") == "unnecessary_search":
            trace = rows(AB / "train-on-policy.jsonl")
            actual = next(x for x in trace if x["case_id"] == case_id)
            searches = [i for i, step in enumerate(actual["trace"]) if step["action"] == "search"]
            vote["error_step"] = searches[-1] if searches else None
        vote["audit_packet_sha256"] = hashlib.sha256(json.dumps(packet[case_id], ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        votes[case_id] = vote
    for case_id, vote in votes.items():
        if not isinstance(vote.get("repairable_within_budget"), bool) or not vote.get("reason"):
            raise RuntimeError(f"Incomplete error review: {case_id}")
    for vote in votes.values():
        if vote.get("repairable_within_budget") and vote.get("event") in ("premature_answer", "wrong_bridge_query"):
            if not (vote.get("repair_query") or vote.get("repair_queries")) or not vote.get("missing_relation"):
                raise RuntimeError(f"Repair query missing for {vote['case_id']}")
    for case_id in packet:
        append(OUT / "failure-audit-final.jsonl", votes[case_id])
    eligible = [x for x in votes.values() if x.get("repairable_within_budget")]
    stop_eligible = [x for x in eligible if x["event"] in ("premature_answer", "unnecessary_search")]
    summary = {"audited_cases": 128, "ab_stop_errors": sum(x["on_policy_stop_error"] for x in original.values()),
        "reviewed_stop_errors": sum(x.get("error_supported") is True for x in audited.values()),
        "repairable_paths": len(eligible), "repairable_stop_paths": len(stop_eligible),
        "events": dict(Counter(x["event"] for x in votes.values())),
        "passed_source_gate": len(eligible) >= MIN_PAIRS and len(stop_eligible) >= MIN_STOP_REPAIRS,
        "extra_candidates_available": 128}
    write(OUT / "failure-audit-summary.json", summary)
    # Material sufficiency is decided only after the proposed repair path has
    # actually run through the frozen retriever and passed evidence review.
    set_state("repair_path_construction", failure_audit=summary)
    return summary


def messages_for(case: dict, until_step: int, tok) -> list:
    from mini_agent_rl.domain import Message, ToolCall
    from mini_agent_rl.prompts import system_prompt
    pc = q.load_config()["product"]
    messages = [Message(role="system", content=system_prompt(False, pc["system_prompt"])),
                Message(role="user", content=case["question"])]
    for item in case["trace"][:until_step]:
        if item.get("action") != "search":
            raise RuntimeError("A repair prefix cannot continue after an answer")
        raw = json.dumps({"action": "search", "query": item["query"]}, ensure_ascii=False)
        observation = {k: v for k, v in item["observation"].items() if k != "latency_ms"}
        call = ToolCall(name="search", arguments={"query": item["query"]})
        messages.extend([Message(role="assistant", content=raw, tool_calls=[call]),
                         Message(role="tool", content=json.dumps(observation, ensure_ascii=False),
                                 tool_call_id=call.id)])
    return messages


def exact_prompt(client, messages: list, expected: dict, phase: str) -> tuple[list[int], list[int]]:
    encoded = client._inputs(messages, phase)
    ids = encoded["input_ids"][0].tolist()
    mask = encoded["attention_mask"][0].tolist()
    if ids != expected["prompt_token_ids"] or mask != expected["prompt_attention_mask"]:
        raise RuntimeError(f"Actual context reconstruction mismatch at {expected.get('step')}")
    return ids, mask


def action_ids(tokenizer, action: str, query: str | None = None, answer: str | None = None) -> tuple[str, list[int]]:
    if action == "search":
        raw = json.dumps({"action": "search", "query": query}, ensure_ascii=False)
        return raw, tokenizer.encode(raw, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
    if action == "answer":
        raw = json.dumps({"action": "answer", "answer": answer}, ensure_ascii=False)
        return raw, s.target_ids(tokenizer, answer)
    raise ValueError(action)


def execute_search_path(client, registry, question: str, messages: list, query: str,
                        prior_trace: list[dict], query_source: dict) -> dict:
    """Execute one reviewed correction query from an exact saved model state."""
    from mini_agent_rl.domain import Message, ToolCall
    from mini_agent_rl.model.local import generation_terminator_ids
    import torch
    tok = client._processor.tokenizer
    before = client._inputs(messages, "goal_ac_repair_prefix")
    expected_ids = before["input_ids"][0].tolist()
    # The correction state is always the saved policy input, not a rewritten prompt.
    item = next((x for x in prior_trace if x.get("prompt_token_ids") == expected_ids), None)
    if item is None:
        raise RuntimeError("Correction prefix is not an actual policy input")
    raw, ids = action_ids(tok, "search", query=query)
    if len(ids) > q.load_config()["product"]["max_new_tokens"]:
        raise RuntimeError("Correction query action exceeds output-token limit")
    seen = {" ".join(str(x.get("query", "")).casefold().split()) for x in prior_trace if x.get("action") == "search"}
    normalized = " ".join(query.casefold().split())
    if not query.strip() or normalized in seen:
        raise RuntimeError("Correction query is empty or repeated")
    obs = asyncio.run(registry.execute("search", {"query": query}))
    if not obs.get("ok"):
        raise RuntimeError("Real search tool failed during correction")
    seen.add(normalized)
    obs["searched_queries"] = sorted(seen)
    visible = {k: v for k, v in obs.items() if k != "latency_ms"}
    call = ToolCall(name="search", arguments={"query": query})
    next_messages = [*messages, Message(role="assistant", content=raw, tool_calls=[call]),
                     Message(role="tool", content=json.dumps(visible, ensure_ascii=False), tool_call_id=call.id)]
    encoded = client._inputs(next_messages, "goal_ac_repair_answer")
    return {"query": query, "query_source": query_source, "action": "search", "raw": raw,
        "prompt_token_ids": expected_ids, "prompt_attention_mask": before["attention_mask"][0].tolist(),
        "action_token_ids": ids, "observation": obs,
        "next_prompt_token_ids": encoded["input_ids"][0].tolist(),
        "next_prompt_attention_mask": encoded["attention_mask"][0].tolist(),
        "next_prompt_actual": tok.decode(encoded["input_ids"][0].tolist(), skip_special_tokens=False)}


def collect_extra_rollouts() -> None:
    import torch
    verify()
    if z.read(STATE)["status"] != "collecting_extra_rollouts":
        raise RuntimeError("Additional candidates are not enabled")
    pool = z.read(OUT / "extra-candidate-pool.json")["cases"]
    if len(pool) != 128 or (OUT / "extra-on-policy.jsonl").exists():
        raise RuntimeError("Frozen 128-case candidate pool absent or already started")
    pool_cases = []
    for item in pool:
        pool_cases.append({k: v for k, v in item.items() if k not in ("support_docs", "distractor_docs")} |
            {"docs": [{**d, "support": True} for d in item["support_docs"]] +
                     [{**d, "support": False} for d in item["distractor_docs"]]})
    history = source.w.history_exposure()
    reserved = {source.norm(d["title"]) for d in rows(z.OUT / "corpus-train.json")}
    for corpus_name in ("corpus-dev.json", "corpus-confirm.json"):
        reserved.update(source.norm(d["title"]) for d in rows(z.OUT / corpus_name))
    for corpus_path in (q.p.CORPUS, AB / "fresh-confirm-corpus.json"):
        if Path(corpus_path).is_file():
            reserved.update(source.norm(d["title"]) for d in rows(corpus_path))
    extra_docs, _ = curator.build_corpus("train", pool_cases, reserved, set(history[2]))
    extra_corpus_path = OUT / "corpus-extra-train.json"
    write(extra_corpus_path, extra_docs)
    write(OUT / "extra-corpus-manifest.json", {"corpus_sha256": digest(extra_corpus_path),
        "document_count": len(extra_docs), "independent_of_frozen_corpora": True})
    client, wrapped, registry = z.make_client(SOURCE, extra_corpus_path)
    for case in pool:
        result = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False, seed=42))
        result.update(case_id=case["id"], source=case["source"])
        append(OUT / "extra-on-policy.jsonl", result)
    if len(rows(OUT / "extra-on-policy.jsonl")) != 128:
        raise RuntimeError("Additional candidate rollout count mismatch")
    del client, wrapped, registry
    gc.collect(); torch.cuda.empty_cache()
    set_state("awaiting_extra_audit", extra_rollouts=128)


def select_and_build_repairs() -> dict:
    """Build one bounded correction path from each reviewed real failure state."""
    import torch
    from transformers import AutoTokenizer
    verify()
    audits = audited_inputs()
    base = {x["case_id"]: x for x in rows(AB / "train-on-policy.jsonl")}
    all_cases = {x["id"]: x for x in rows(z.OUT / "selected-train.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(q.p.BASE, local_files_only=True)
    client, _, registry = z.make_client(SOURCE, z.OUT / "corpus-train.json")
    client._load()
    search_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(search_end, int) or search_end < 0:
        raise RuntimeError("Tokenizer has no message terminator")
    eligible = []
    for case_id, audit in audits.items():
        if audit.get("repairable_within_budget") and audit.get("error_supported") is True and \
           audit.get("event") in ("premature_answer", "wrong_bridge_query", "unnecessary_search"):
            eligible.append((case_id, audit))
    eligible.sort(key=lambda pair: hashlib.sha256(("goal-ac-repair-v1\0" + pair[0]).encode()).hexdigest())
    output_path = OUT / "repair-candidates.jsonl"
    if output_path.exists():
        raise FileExistsError(output_path)
    failures = Counter()
    for case_id, audit in eligible:
        case = base[case_id]
        problem = all_cases[case_id]
        trace = case["trace"]
        try:
            target_step = int(audit.get("error_step", -1))
            if target_step < 0 or target_step >= len(trace):
                raise ValueError("error_step_absent")
            event = audit["event"]
            target = trace[target_step]
            if target.get("action") not in ("answer", "search"):
                raise ValueError("unsupported_error_state_action")
            # Rebuild precisely the real prefix before the bad decision and
            # verify that its token IDs match the stored policy input.
            messages = messages_for({"question": case["question"], "trace": trace}, target_step, tokenizer)
            prompt_ids, prompt_mask = exact_prompt(client, messages, target, "goal_ac_failure_state")
            repair_actions = []
            prefix_end = target_step
            if event == "unnecessary_search":
                # Turn the actual over-search state into an answer state. The
                # earlier genuine search steps remain in the prefix.
                insufficient_i = next((i for i, step in enumerate(trace[:target_step])
                                       if step.get("action") == "search"), None)
                if insufficient_i is None:
                    raise ValueError("no_real_search_state_for_pair")
                prefix_end = insufficient_i
                first = trace[insufficient_i]
                insufficient_action = {"action": "search", "query": first["query"],
                    "prompt_token_ids": first["prompt_token_ids"],
                    "prompt_attention_mask": first["prompt_attention_mask"],
                    "action_token_ids": first["action_token_ids"], "evidence_source": "archived_real_state"}
                repair_actions.append({k: first[k] for k in ("action", "query", "answer",
                    "prompt_token_ids", "prompt_attention_mask", "action_token_ids") if k in first})
                for step in trace[insufficient_i + 1:target_step]:
                    repair_actions.append({k: step[k] for k in ("action", "query", "answer",
                        "prompt_token_ids", "prompt_attention_mask", "action_token_ids") if k in step})
                final_ids, final_mask = prompt_ids, prompt_mask
            else:
                query_specs = audit.get("repair_queries") or ([{"query": audit.get("repair_query"),
                    "bridge_source": audit.get("bridge_source")} ] if audit.get("repair_query") else [])
                if not query_specs or len(query_specs) > 2:
                    raise ValueError("reviewed_correction_query_absent")
                answer_norm = source.norm(problem["answer"])
                from mini_agent_rl.domain import Message, ToolCall
                current_messages = list(messages)
                seen_queries = {" ".join(str(x.get("query", "")).casefold().split())
                                for x in trace[:target_step] if x.get("action") == "search"}
                first_repair = None
                for query_index, spec in enumerate(query_specs):
                    query = spec.get("query") if isinstance(spec, dict) else str(spec)
                    if not isinstance(query, str) or not query.strip():
                        raise ValueError("reviewed_correction_query_absent")
                    visible_text = tokenizer.decode(client._inputs(current_messages,
                        "goal_ac_repair_query")["input_ids"][0].tolist(), skip_special_tokens=False)
                    visible_norm = source.norm(case["question"] + " " + visible_text)
                    if len(answer_norm) >= 5 and answer_norm in source.norm(query) and answer_norm not in visible_norm:
                        raise ValueError("query_contains_hidden_answer")
                    bridge = str(audit.get("bridge_entity") or "") if query_index == 0 else ""
                    bridge_source = str(spec.get("bridge_source", "")) if isinstance(spec, dict) else ""
                    for entity in re.findall(r"[A-Z][\w’'-]+(?:\s+[A-Z][\w’'-]+)*", bridge_source):
                        if source.norm(entity) not in visible_norm:
                            raise ValueError("query_bridge_entity_not_visible_at_query_time")
                    if bridge:
                        parts = [source.norm(x) for x in re.split(r"\s*(?:/|;|\bor\b)\s*", bridge) if source.norm(x)]
                        if any(part not in visible_norm for part in parts):
                            raise ValueError("bridge_entity_not_visible_at_query_time")
                    norm_query = " ".join(query.casefold().split())
                    if norm_query in seen_queries:
                        raise ValueError("repair_query_repeated")
                    encoded_now = client._inputs(current_messages, "goal_ac_repair_query")
                    current_prompt_ids = encoded_now["input_ids"][0].tolist()
                    current_mask = encoded_now["attention_mask"][0].tolist()
                    raw_search, search_ids = action_ids(tokenizer, "search", query=query)
                    if query_index == 0:
                        executed = execute_search_path(client, registry, case["question"], current_messages, query,
                            trace, {"type": "reviewed_from_visible_state", "bridge_entity": bridge,
                            "bridge_source": audit.get("bridge_source"), "missing_relation": audit.get("missing_relation")})
                        observation = executed["observation"]
                        search_ids = executed["action_token_ids"]
                        seen_queries.add(norm_query)
                    else:
                        observation = asyncio.run(registry.execute("search", {"query": query}))
                        if not observation.get("ok"):
                            raise RuntimeError("supplemental repair search failed")
                        seen_queries.add(norm_query)
                        observation["searched_queries"] = sorted(seen_queries)
                    visible_observation = {k: v for k, v in observation.items() if k != "latency_ms"}
                    call = ToolCall(name="search", arguments={"query": query})
                    current_messages.extend([Message(role="assistant", content=raw_search, tool_calls=[call]),
                        Message(role="tool", content=json.dumps(visible_observation, ensure_ascii=False),
                            tool_call_id=call.id)])
                    encoded_next = client._inputs(current_messages, "goal_ac_repair_answer")
                    action = {"action": "search", "query": query,
                        "prompt_token_ids": current_prompt_ids, "prompt_attention_mask": current_mask,
                        "action_token_ids": search_ids, "observation": observation,
                        "query_source": {"bridge_source": bridge_source}}
                    repair_actions.append(action)
                    if first_repair is None:
                        first_repair = action
                    final_ids = encoded_next["input_ids"][0].tolist()
                    final_mask = encoded_next["attention_mask"][0].tolist()
                insufficient_action = {"action": "search", "query": first_repair["query"],
                    "prompt_token_ids": first_repair["prompt_token_ids"],
                    "prompt_attention_mask": first_repair["prompt_attention_mask"],
                    "action_token_ids": first_repair["action_token_ids"],
                    "evidence_source": audit.get("bridge_source")}
            raw_answer, answer_ids = action_ids(tokenizer, "answer", answer=problem["answer"])
            repair_actions.append({"action": "answer", "answer": problem["answer"], "raw": raw_answer,
                "prompt_token_ids": final_ids, "prompt_attention_mask": final_mask,
                "action_token_ids": answer_ids})
            searches = sum(x.get("action") == "search" for x in trace[:prefix_end]) + \
                       sum(x["action"] == "search" for x in repair_actions)
            calls = prefix_end + len(repair_actions)
            if searches > 3 or calls > 4 or any(len(x["prompt_token_ids"]) > 2048 or
                len(x["action_token_ids"]) > 128 for x in repair_actions):
                raise ValueError("repair_path_exceeds_product_budget")
            final_text = source.norm(tokenizer.decode(final_ids, skip_special_tokens=False))
            visible_docs = []
            for step in trace[:target_step]:
                if step.get("action") == "search":
                    visible_docs.append({"step": step.get("step"), "query": step.get("query"),
                                         "documents": step.get("observation", {}).get("result", [])})
            for action in repair_actions:
                if action.get("action") == "search":
                    visible_docs.append({"step": "repair", "query": action.get("query"),
                                         "documents": action.get("observation", {}).get("result", [])})
            for group in visible_docs:
                group["documents"] = [d for d in group["documents"] if
                    source.norm(d.get("id", "")) in final_text and
                    source.norm(d.get("text", "")) and source.norm(d["text"]) in final_text]
            pair = {"case_id": case_id, "event": audit["event"],
                "error_supported": audit.get("error_supported", True),
                "targeted_stop_error": audit["event"] in ("premature_answer", "unnecessary_search"),
                "original_position_case_id": case_id, "source": problem.get("source", "original"),
                "question": problem["question"], "training_answer": problem["answer"],
                "support_titles": problem.get("teacher", {}).get("support_titles", []),
                "insufficient_state": insufficient_action,
                "sufficient_state": repair_actions[-1], "repair_actions": repair_actions,
                "repair_searches": searches, "repair_calls": calls,
                "visible_documents": visible_docs,
                "actual_final_prompt": tokenizer.decode(final_ids, skip_special_tokens=False),
                "repair_path_sha256": hashlib.sha256(json.dumps(repair_actions, sort_keys=True,
                    ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                "evidence_audit_required": True, "review_reason": audit["reason"]}
            append(output_path, pair)
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            failures[str(exc)] += 1
        if sum(failures.values()) + (len(rows(output_path)) if output_path.exists() else 0) % 16 == 0:
            pass
    del client, registry
    gc.collect(); torch.cuda.empty_cache()
    built = rows(output_path) if output_path.exists() else []
    picked = []
    stop = [x for x in built if x["targeted_stop_error"]]
    stop.sort(key=lambda x: hashlib.sha256(("goal-ac-stop:" + x["case_id"]).encode()).hexdigest())
    picked.extend(stop[:MAX_PAIRS])
    if len(picked) < MAX_PAIRS:
        remaining = [x for x in built if x not in picked]
        remaining.sort(key=lambda x: hashlib.sha256(("goal-ac-fill:" + x["case_id"]).encode()).hexdigest())
        picked.extend(remaining[:MAX_PAIRS - len(picked)])
    write_jsonl(OUT / "selected-repair-paths.jsonl", picked)
    stop_count = sum(x["targeted_stop_error"] for x in picked)
    result = {"built_paths": len(built), "selected_paths": len(picked),
        "selected_stop_error_repairs": stop_count, "failure_reasons": dict(failures),
        "passed": len(picked) >= MIN_PAIRS and stop_count >= MIN_STOP_REPAIRS,
        "path_sha256": digest(OUT / "selected-repair-paths.jsonl")}
    write(OUT / "repair-material-decision.json", result)
    if result["passed"]:
        build_repair_evidence_packet(picked)
        set_state("awaiting_repair_evidence_review", repair_material=result)
    else:
        set_state("collecting_extra_rollouts", repair_material=result)
    return result


def build_repair_evidence_packet(pairs: list[dict] | None = None) -> dict:
    pairs = pairs if pairs is not None else rows(OUT / "selected-repair-paths.jsonl")
    path = OUT / "repair-evidence-packet.jsonl"
    if path.exists():
        raise FileExistsError(path)
    packet = []
    for pair in pairs:
        packet.append({"case_id": pair["case_id"], "question": pair["question"],
            "missing_relation": pair["review_reason"], "answer_target": pair["training_answer"],
            "insufficient_state": pair["insufficient_state"],
            "repair_actions": pair["repair_actions"],
            "actual_final_prompt": pair["actual_final_prompt"],
            "visible_documents": pair["visible_documents"],
            "instructions": "Judge whether the final answer is fully supported by the actual visible passages and answers the requested relation. Cite document IDs and exact supporting snippets. Also verify every bridge entity was present in the question or visible evidence before its query."})
        append(path, packet[-1])
    return {"items": len(packet), "sha256": digest(path)}


def finalize_repair_evidence() -> dict:
    """Main-adjudicate two independent blind evidence reviews."""
    packet = rows(OUT / "repair-evidence-packet.jsonl")
    a, b = rows(OUT / "repair-evidence-a.jsonl"), rows(OUT / "repair-evidence-b.jsonl")
    if len(a) != len(packet) or len(b) != len(packet):
        raise RuntimeError("Both evidence reviewers must cover every repair path")
    by_a, by_b = ({x["case_id"]: x for x in xs} for xs in (a, b))
    if set(by_a) != {x["case_id"] for x in packet} or set(by_b) != set(by_a):
        raise RuntimeError("Repair evidence review ID mismatch")
    approved = []
    decisions = []
    for item in packet:
        ra, rb = by_a[item["case_id"]], by_b[item["case_id"]]
        docs = {d["id"]: d for step in item["visible_documents"] for d in step["documents"]}
        for vote in (ra, rb):
            ids = set(vote.get("evidence_ids", []))
            snippets = vote.get("supporting_snippets", [])
            if not ids or not ids.issubset(docs) or not snippets or \
               any(not str(snip.get("text", "")).strip() or
                   source.norm(snip["text"]) not in source.norm(docs[snip["document_id"]]["text"])
                   for snip in snippets if snip.get("document_id") in docs):
                vote["_valid_citations"] = False
            else:
                vote["_valid_citations"] = all(x.get("document_id") in ids for x in snippets)
        same = (ra.get("supported") is True and rb.get("supported") is True and
                ra.get("answers_relation") is True and rb.get("answers_relation") is True and
                ra.get("bridge_sources_valid") is True and rb.get("bridge_sources_valid") is True and
                ra["_valid_citations"] and rb["_valid_citations"])
        reason = "both reviewers cited visible passages supporting the requested relation" if same else \
            "reviewers disagreed, relation unsupported, bridge unverified, or citations invalid"
        decisions.append({"case_id": item["case_id"], "approved": same,
            "reviewer_a": ra, "reviewer_b": rb, "adjudication": reason})
        if same:
            original = next(x for x in rows(OUT / "selected-repair-paths.jsonl") if x["case_id"] == item["case_id"])
            approved.append(original)
    write(OUT / "repair-evidence-final.json", decisions)
    approved.sort(key=lambda x: hashlib.sha256(("goal-ac-final:" + x["case_id"]).encode()).hexdigest())
    # Select stopping-decision repairs first, then fill to 32-64 from other
    # qualified real failure corrections. The exact list is frozen here.
    stops = [x for x in approved if x["targeted_stop_error"]]
    rest = [x for x in approved if not x["targeted_stop_error"]]
    selected = stops[:MAX_PAIRS] + rest[:max(0, MAX_PAIRS - len(stops))]
    stop_count = sum(x["targeted_stop_error"] for x in selected)
    states = {"search": len(selected), "answer": len(selected)}
    passed = len(selected) >= MIN_PAIRS and stop_count >= MIN_STOP_REPAIRS and \
             states["search"] >= 32 and states["answer"] >= 32
    write_jsonl(OUT / "repair-pairs.jsonl", selected)
    result = {"proposed_paths": len(packet), "evidence_approved": len(approved),
        "selected_pairs": len(selected), "selected_stop_error_repairs": stop_count,
        "decision_states": states, "passed": passed,
        "rejected": sum(not x["approved"] for x in decisions)}
    write(OUT / "repair-material-decision.json", result)
    if passed:
        freeze_repair_binding(selected)
        set_state("ready_to_train", repair_material=result)
    else:
        set_state("collecting_extra_rollouts", repair_material=result)
    return result


def freeze_repair_binding(selected: list[dict]) -> dict:
    cfg = verify()
    repair_by_case = {x["case_id"]: x for x in selected}
    original_ids = set(repair_by_case)
    extras = [x for x in selected if x["case_id"] not in set(cfg["order"]["42"])]
    if extras:
        raise RuntimeError("Extra repairs require explicit frozen slot binding")
    binding = {seed: [] for seed in ("42", "43")}
    for seed in ("42", "43"):
        for pos, case_id in enumerate(cfg["order"][seed], 1):
            if case_id in repair_by_case:
                binding[seed].append({"update": pos, "case_id": case_id,
                    "repair_case_id": case_id, "kind": "original"})
    case_bindings = [{x["case_id"] for x in binding[s]} for s in ("42", "43")]
    if case_bindings[0] != case_bindings[1]:
        raise RuntimeError("Repair case bindings differ across seeds")
    write(OUT / "repair-update-binding.json", binding)
    return binding


def train(seed: int) -> dict:
    """One matched 128-update C run, restarting from the same published adapter."""
    import torch
    from transformers import set_seed
    from mini_agent_rl.training.grpo import guarded_optimizer_step
    import run_v133_goal_o_budget as o
    cfg = verify()
    if seed not in (42, 43) or z.read(STATE)["status"] != "training":
        raise RuntimeError("AC training stage is not active")
    folder = OUT / f"seed{seed}-C"
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir()
    inputs = {x["id"]: x for x in rows(z.OUT / "training-inputs.jsonl")}
    repairs = {x["case_id"]: x for x in rows(OUT / "repair-pairs.jsonl")}
    bindings = z.read(OUT / "repair-update-binding.json")[str(seed)]
    repair_for = {x["case_id"]: repairs[x["repair_case_id"]] for x in bindings}
    set_seed(seed)
    client, _, _ = z.make_client(SOURCE, z.OUT / "corpus-train.json")
    params = [p for _, p in client.prepare_for_training()]
    base_before = o.parameter_hashes(client._model, lora=False)
    lora_before = o.parameter_hashes(client._model, lora=True)
    opt = cfg["optimization"]
    optimizer = torch.optim.AdamW(params, lr=opt["learning_rate"], betas=tuple(opt["betas"]),
        eps=opt["eps"], weight_decay=opt["weight_decay"])
    last = None
    try:
        order = cfg["order"][str(seed)]
        if len(order) != 128 or len(set(order)) != 128:
            raise RuntimeError("Frozen training order invalid")
        for index, case_id in enumerate(order, 1):
            if index > 128:
                raise RuntimeError("AC update limit exceeded")
            actions = inputs[case_id]["actions"]
            original_tokens = sum(len(x["action_token_ids"]) for x in actions)
            if original_tokens <= 0:
                raise RuntimeError("Empty original trajectory supervision")
            optimizer.zero_grad(set_to_none=True)
            client.set_training_forward_mode()
            scalar = 0.
            blend = case_id in repair_for
            original_weight = .5 if blend else 1.
            for action in actions:
                values = client.score_action_tensor([], action["action_token_ids"], True,
                    action["prompt_token_ids"], action["prompt_attention_mask"]).float()
                loss = -original_weight * values.sum() / original_tokens
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite original trajectory loss")
                loss.backward(); scalar += float(loss.detach())
            repair_tokens = 0
            if blend:
                repair = repair_for[case_id]
                search_actions = [x for x in repair["repair_actions"] if x["action"] == "search"]
                answer_actions = [x for x in repair["repair_actions"] if x["action"] == "answer"]
                if len(answer_actions) != 1 or not search_actions:
                    raise RuntimeError("Repair path must include search and final answer")
                search_logprob_sum = None
                search_token_count = 0
                for action in search_actions:
                    values = client.score_action_tensor([], action["action_token_ids"], True,
                        action["prompt_token_ids"], action["prompt_attention_mask"]).float()
                    search_logprob_sum = values.sum() if search_logprob_sum is None else search_logprob_sum + values.sum()
                    repair_tokens += len(action["action_token_ids"])
                    search_token_count += len(action["action_token_ids"])
                search_loss = -.25 * search_logprob_sum / search_token_count
                if not torch.isfinite(search_loss):
                    raise RuntimeError("Nonfinite correction search loss")
                search_loss.backward(); scalar += float(search_loss.detach())
                action = answer_actions[0]
                values = client.score_action_tensor([], action["action_token_ids"], True,
                    action["prompt_token_ids"], action["prompt_attention_mask"]).float()
                loss = -.25 * values.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite correction answer loss")
                loss.backward(); scalar += float(loss.detach())
                repair_tokens += len(action["action_token_ids"])
            grad = float(torch.nn.utils.clip_grad_norm_(params, opt["gradient_clip"]))
            if not math.isfinite(grad) or grad <= 0:
                raise RuntimeError("Invalid gradient")
            guarded_optimizer_step(optimizer, params, client.gpu_memory_stats(), 21.)
            z.append(folder / "updates.jsonl", {"step": index, "case_id": case_id,
                "loss": scalar, "gradient": grad, "original_tokens": original_tokens,
                "repair_tokens": repair_tokens, "repair_blended": blend,
                "gpu_memory": client.gpu_memory_stats()})
            last = actions[-1]
            if index % 8 == 0:
                print(json.dumps({"seed": seed, "updates": index, "repair": blend}), flush=True)
        if len(rows(folder / "updates.jsonl")) != 128:
            raise RuntimeError("AC update count mismatch")
        if o.parameter_hashes(client._model, lora=False) != base_before:
            raise RuntimeError("Base weights changed")
        lora_after = o.parameter_hashes(client._model, lora=True)
        changed = sum(lora_after[k] != v for k, v in lora_before.items())
        if changed <= 0:
            raise RuntimeError("LoRA did not change")
        adapter = folder / "adapter-final"
        client._model.save_pretrained(adapter)
        expected = client.score_action_tensor([], last["action_token_ids"], False,
            last["prompt_token_ids"], last["prompt_attention_mask"]).detach().float().cpu()
        del client, optimizer, params
        gc.collect(); torch.cuda.empty_cache()
        loaded, _, _ = z.make_client(adapter, z.OUT / "corpus-train.json")
        actual = loaded.score_action_tensor([], last["action_token_ids"], False,
            last["prompt_token_ids"], last["prompt_attention_mask"]).detach().float().cpu()
        difference = float((expected - actual).abs().max())
        if difference > .001:
            raise RuntimeError(f"Independent reload mismatch: {difference}")
        result = {"updates": 128, "base_unchanged": True, "changed_lora_tensors": changed,
            "reload_max_abs": difference, "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
            "repair_update_count": len(repair_for)}
        write(folder / "summary.json", result)
        del loaded
        gc.collect(); torch.cuda.empty_cache()
        return result
    except BaseException as exc:
        write(folder / "failure.json", {"error": str(exc), "completed_updates":
            len(rows(folder / "updates.jsonl")) if (folder / "updates.jsonl").exists() else 0})
        raise


def evaluation_specs() -> list[tuple[str, list[dict], Path]]:
    return ab.evaluation_specs()


def evaluate_one(label: str, adapter: Path, split: str, cases: list[dict], corpus: Path) -> None:
    import torch
    path = OUT / f"eval-{split}-{label}.jsonl"
    if path.exists():
        raise FileExistsError(path)
    client, wrapped, registry = z.make_client(adapter, corpus)
    for case in cases:
        answer = case.get("answer", case.get("official_answer"))
        result = asyncio.run(q.rollout(case["question"], wrapped, registry, sample=False, seed=42))
        result.update(case_id=case.get("id", case.get("case_id")), split=split, model=label,
            official_answer=answer, **q.score(result["answer"], answer))
        append(path, result)
        if len(rows(path)) % 8 == 0:
            print(json.dumps({"eval": split, "model": label, "completed": len(rows(path))}), flush=True)
    del client, wrapped, registry
    gc.collect(); torch.cuda.empty_cache()


def copy_ab_controls() -> None:
    """Copy frozen baseline/A/B outputs and their finalized reviews verbatim."""
    for split in SPLITS:
        for label in ("baseline", "42-A", "42-B", "43-A", "43-B"):
            src = AB / f"eval-{split}-{label}.jsonl"
            dst = OUT / src.name
            if dst.exists():
                raise FileExistsError(dst)
            dst.write_bytes(src.read_bytes())
    for name in ("development-review-packet.jsonl", "development-review-map.json",
                 "development-review-final.jsonl"):
        dst = OUT / name
        if dst.exists():
            raise FileExistsError(dst)
        dst.write_bytes((AB / name).read_bytes())


def review_packet_c(split: str, labels: tuple[str, ...], stage: str = "development") -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(q.p.BASE, local_files_only=True)
    packet, mapping = {}, []
    for label in labels:
        for row in rows(OUT / f"eval-{split}-{label}.jsonl"):
            last = row["trace"][-1]
            actual = tok.decode(last["prompt_token_ids"], skip_special_tokens=False)
            visible = sorted({d["id"] for st in row["trace"] for d in st.get("observation", {}).get("result", [])
                              if d["id"] in actual})
            steps = [{"action": st.get("action"), "query": st.get("query"), "answer": st.get("answer"),
                      "actual_prompt": tok.decode(st["prompt_token_ids"], skip_special_tokens=False)}
                     for st in row["trace"]]
            key = hashlib.sha256(json.dumps([row["question"], steps, row.get("answer"), row["reason"]],
                ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            packet[key] = {"review_id": key, "question": row["question"], "answer": row.get("answer"),
                "split": split,
                "reason": row["reason"], "actual_prompt": actual, "visible_document_ids": visible, "steps": steps,
                "prompt_token_sha256": hashlib.sha256(json.dumps(last["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest()}
            mapping.append({"split": split, "case_id": row["case_id"], "model": label, "review_id": key})
    packet_path = OUT / f"{stage}-review-packet.jsonl"
    map_path = OUT / f"{stage}-review-map.json"
    existing = {x["review_id"] for x in rows(packet_path)} if packet_path.exists() else set()
    unique_new = [x for key, x in packet.items() if key not in existing]
    with packet_path.open("a", encoding="utf-8") as f:
        for x in unique_new:
            f.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n")
    prior_map = rows(map_path) if map_path.exists() else []
    write(map_path, prior_map + mapping)
    new_path = OUT / f"{stage}-review-new-packet.jsonl"
    if unique_new:
        with new_path.open("a", encoding="utf-8") as f:
            for x in unique_new:
                f.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n")
    return {"candidate_outputs": len(mapping), "unique_new_review_items": len(unique_new)}


def evaluate_development() -> None:
    cfg = verify()
    if z.read(STATE)["status"] != "training":
        raise RuntimeError("Training has not completed")
    copy_ab_controls()
    for seed in (42, 43):
        label = f"{seed}-C"
        summary = z.read(OUT / f"seed{seed}-C/summary.json")
        adapter = OUT / f"seed{seed}-C/adapter-final"
        if summary["updates"] != 128 or digest(adapter / "adapter_model.safetensors") != summary["adapter_sha256"]:
            raise RuntimeError("C checkpoint incomplete")
        for split, cases, corpus in evaluation_specs():
            evaluate_one(label, adapter, split, cases, corpus)
    info = {}
    for split in SPLITS:
        info[split] = review_packet_c(split, ("42-C", "43-C"))
    set_state("awaiting_development_review", development_review=info)


def development_scores() -> tuple[dict, dict]:
    if (OUT / "development-review-additional-final.jsonl").exists():
        prior_path = OUT / "development-review-final.jsonl"
        prior = rows(prior_path) if prior_path.exists() else []
        additions = rows(OUT / "development-review-additional-final.jsonl")
        merged = {x["review_id"]: x for x in prior}
        for x in additions:
            if x["review_id"] in merged and merged[x["review_id"]] != x:
                raise RuntimeError("Conflicting review for an already frozen review ID")
            merged[x["review_id"]] = x
        write_jsonl(prior_path, list(merged.values()))
        (OUT / "development-review-additional-final.jsonl").rename(
            OUT / "development-review-additional-final.merged.jsonl")
    packet = rows(OUT / "development-review-packet.jsonl")
    votes = rows(OUT / "development-review-final.jsonl")
    mapping = rows(OUT / "development-review-map.json")
    if len({x["review_id"] for x in votes}) != len(votes):
        raise RuntimeError("Duplicate review result ID")
    by_vote = {x["review_id"]: x for x in votes}
    if {x["review_id"] for x in packet} != set(by_vote):
        raise RuntimeError("Development semantic audit incomplete")
    result, by_case = {}, {}
    for item in packet:
        v = by_vote[item["review_id"]]
        if v.get("category") not in REVIEW_CATEGORIES or not v.get("reason") or \
           not set(v.get("evidence_ids", [])).issubset(item.get("visible_document_ids", [])):
            raise RuntimeError("Invalid evidence review")
        if v["category"] == "correct_sufficient" and (item["reason"] != "answer" or not v.get("evidence_ids")):
            raise RuntimeError("Success must cite visible evidence at answer time")
    vote_by_id = {x["review_id"]: x for x in votes}
    for m in mapping:
        vote = vote_by_id[m["review_id"]]
        by_case[m["split"], m["model"], m["case_id"]] = vote
    for split in SPLITS:
        for label in LABELS:
            raw = rows(OUT / f"eval-{split}-{label}.jsonl")
            expected = 64 if split in ("dev", "aa_regression") else 32 if split == "old" else 48
            if len(raw) != expected:
                raise RuntimeError(f"Incomplete evaluation {split}/{label}")
            cats = Counter(by_case[split, label, x["case_id"]]["category"] for x in raw)
            cats["invalid"] = sum(x["reason"] == "invalid_action" for x in raw)
            cats["em"] = sum(x["em"] for x in raw)
            cats["f1_sum"] = sum(x["f1"] for x in raw)
            cats["searches_sum"] = sum(x["searches"] for x in raw)
            cats["latency_sum"] = sum(x["elapsed_seconds"] for x in raw)
            cats["bad_total"] = cats["factual_error"] + cats["unsupported_by_visible_evidence"]
            cats["stop_errors"] = sum(bool({"premature_answer", "unnecessary_search"} &
                set(by_case[split, label, x["case_id"]].get("stop_events", []))) for x in raw)
            if split == "old":
                groups = {x["case_id"]: x["split"] for x in q.dev_cases()}
                cats["main"] = sum(by_case[split, label, x["case_id"]]["category"] == "correct_sufficient"
                                    for x in raw if groups[x["case_id"]] == "new_eval")
                cats["regression"] = sum(by_case[split, label, x["case_id"]]["category"] == "correct_sufficient"
                                         for x in raw if groups[x["case_id"]] == "goal_G_regression")
            result[f"{split}-{label}"] = dict(cats)
    return result, by_case


def current_protected_cases() -> set[str]:
    """Only protect current same-rubric successes requiring later evidence."""
    aa_rows = rows(aa.OUT / "confirmation-revised-results.jsonl")
    baseline_raw = rows(AB / "eval-aa_regression-baseline.jsonl")
    baseline_votes = {x["review_id"]: x for x in rows(AB / "development-review-final.jsonl")}
    baseline_map = {(x["split"], x["model"], x["case_id"]): x["review_id"]
                    for x in rows(AB / "development-review-map.json")}
    current_success = {x["case_id"] for x in baseline_raw if
        baseline_votes[baseline_map["aa_regression", "baseline", x["case_id"]]]["category"] == "correct_sufficient"}
    historical_later = {x["case_id"] for x in aa_rows if x["model"] == "42-trajectory" and
        x["category"] == "correct_sufficient" and x.get("second_side_first_seen_later")}
    protected = current_success & historical_later
    write(OUT / "protection-set.json", {"case_ids": sorted(protected),
        "historical_count": len(historical_later), "current_same_rubric_count": len(current_success),
        "intersection_count": len(protected),
        "rule": "current AB baseline semantic success intersected with historically verified later-acquired second-side evidence"})
    return protected


def development_gate() -> dict:
    counts, by_case = development_scores()
    baseline = counts["dev-baseline"]
    protected = current_protected_cases()
    gates = {}
    for seed in (42, 43):
        label = f"{seed}-C"
        dev = counts[f"dev-{label}"]
        old, y, aa_reg = (counts[f"{split}-{label}"] for split in ("old", "y", "aa_regression"))
        lost = sum(by_case["aa_regression", label, case_id]["category"] != "correct_sufficient"
                   for case_id in protected)
        old_base = counts["old-baseline"]
        passed = (dev.get("correct_sufficient", 0) >= baseline.get("correct_sufficient", 0) + MIN_DEV_GAIN and
            dev["bad_total"] <= baseline["bad_total"] and dev["invalid"] <= baseline["invalid"] and
            old["main"] >= old_base["main"] - 1 and old["regression"] >= old_base["regression"] - 1 and
            y["correct_sufficient"] >= counts["y-baseline"]["correct_sufficient"] - 1 and
            aa_reg["correct_sufficient"] >= counts["aa_regression-baseline"]["correct_sufficient"] - 1 and
            lost <= 1 and z.read(OUT / f"seed{seed}-C/summary.json")["base_unchanged"] and
            z.read(OUT / f"seed{seed}-C/summary.json")["reload_max_abs"] <= .001)
        gates[str(seed)] = {"passed": passed, "dev_success": dev["correct_sufficient"],
            "dev_bad": dev["bad_total"], "invalid": dev["invalid"], "old_main": old["main"],
            "old_regression": old["regression"], "goal_y": y["correct_sufficient"],
            "aa_regression": aa_reg["correct_sufficient"], "protected_multihop_lost": lost,
            "C_stop_errors": dev["stop_errors"], "AB_B_stop_errors": counts[f"dev-{seed}-B"]["stop_errors"],
            "mechanism_seed_pass": dev["stop_errors"] < counts[f"dev-{seed}-B"]["stop_errors"] and
                dev["correct_sufficient"] >= counts[f"dev-{seed}-B"]["correct_sufficient"]}
    eligible = all(x["passed"] for x in gates.values())
    mechanism = all(x["mechanism_seed_pass"] for x in gates.values())
    result = {"status": "confirmation_authorized" if eligible else "complete_development_gate_failed",
        "gates": gates, "mechanism_supported": mechanism, "counts": counts,
        "protection_set": z.read(OUT / "protection-set.json")}
    write(OUT / "development-decision.json", result)
    set_state(result["status"], development_decision=result)
    return result


def evaluate_confirmation() -> dict:
    cfg = verify()
    if z.read(STATE)["status"] != "confirmation_authorized":
        raise RuntimeError("Confirmation gate has not passed")
    manifest = z.read(AB / "fresh-confirm-manifest.json")
    if manifest.get("confirmation_accessed") or digest(AB / "fresh-confirm-cases.jsonl") != cfg["confirmation_case_sha256"]:
        raise RuntimeError("Frozen confirmation is not sealed")
    cases = rows(AB / "fresh-confirm-cases.jsonl")
    if len(cases) != 64:
        raise RuntimeError("Confirmation size mismatch")
    for label, adapter in (("baseline", SOURCE), ("42-C", OUT / "seed42-C/adapter-final"),
                           ("43-C", OUT / "seed43-C/adapter-final")):
        evaluate_one(label, adapter, "confirmation", cases, AB / "fresh-confirm-corpus.json")
    info = review_packet_c("confirmation", ("baseline", "42-C", "43-C"), stage="confirmation")
    # Opening is now authorized and the exact evaluated outputs are frozen.
    manifest["confirmation_accessed"] = True
    manifest["accessed_by"] = "Goal AC after both seed development gates passed"
    write(OUT / "confirmation-access.json", {"case_sha256": cfg["confirmation_case_sha256"],
        "corpus_sha256": cfg["confirmation_corpus_sha256"], "outputs": 192})
    write(OUT / "fresh-confirm-manifest-copy.json", manifest)
    return {"outputs": 192, "review": info}


def confirmation_gate() -> dict:
    packet = rows(OUT / "confirmation-review-packet.jsonl")
    votes = rows(OUT / "confirmation-review-final.jsonl")
    by_vote = {x["review_id"]: x for x in votes}
    if len(packet) != len(by_vote) or {x["review_id"] for x in packet} != set(by_vote):
        raise RuntimeError("Confirmation semantic audit incomplete")
    mapping = rows(OUT / "confirmation-review-map.json")
    by_case = {}
    for m in mapping:
        by_case[m["model"], m["case_id"]] = by_vote[m["review_id"]]
    counts = {}
    for label in ("baseline", "42-C", "43-C"):
        raw = rows(OUT / f"eval-confirmation-{label}.jsonl")
        cats = Counter(by_case[label, r["case_id"]]["category"] for r in raw)
        cats["invalid"] = sum(r["reason"] == "invalid_action" for r in raw)
        cats["bad_total"] = cats["factual_error"] + cats["unsupported_by_visible_evidence"]
        counts[label] = dict(cats)
    base = counts["baseline"]
    gates = {seed: counts[f"{seed}-C"]["correct_sufficient"] >= base["correct_sufficient"] + 4 and
             counts[f"{seed}-C"]["bad_total"] <= base["bad_total"] and
             counts[f"{seed}-C"]["invalid"] <= base["invalid"] for seed in ("42", "43")}
    published = all(gates.values())
    result = {"counts": counts, "seed_gates": gates, "published": published,
        "product_adapter": str(OUT / "seed42-C/adapter-final") if published else str(SOURCE),
        "status": "published_C42" if published else "retain_goal_z_seed42"}
    write(OUT / "product-decision.json", result)
    set_state("complete", product_decision=result)
    return result


def run() -> dict:
    if not CFG.exists():
        prepare()
    while True:
        status = z.read(STATE)["status"]
        print(json.dumps({"goal_ac_status": status}), flush=True)
        if status == "awaiting_failure_audit":
            combine_failure_reviews()
        elif status == "repair_path_construction":
            select_and_build_repairs()
        elif status == "awaiting_repair_evidence_review":
            if not ((OUT / "repair-evidence-a.jsonl").exists() and
                    (OUT / "repair-evidence-b.jsonl").exists()):
                return {"status": status, "awaiting": ["repair-evidence-a.jsonl", "repair-evidence-b.jsonl"]}
            finalize_repair_evidence()
        elif status == "collecting_extra_rollouts":
            collect_extra_rollouts()
            return {"status": "awaiting_extra_audit", "awaiting": ["extra-failure-audit.jsonl"]}
        elif status == "awaiting_extra_audit":
            # Supplemental candidates are never silently converted into examples.
            # They must have the same blind failure and evidence review before use.
            if not ((OUT / "extra-repair-pairs.jsonl").exists() and
                    (OUT / "extra-repair-evidence-a.jsonl").exists() and
                    (OUT / "extra-repair-evidence-b.jsonl").exists()):
                return {"status": status, "awaiting": ["extra-repair-pairs.jsonl",
                    "extra-repair-evidence-a.jsonl", "extra-repair-evidence-b.jsonl"]}
            raise RuntimeError("Supplemental material adapter is not yet validated; stop rather than lower gates")
        elif status == "ready_to_train":
            set_state("training")
        elif status == "training":
            for seed in (42, 43):
                folder = OUT / f"seed{seed}-C"
                if not (folder / "summary.json").exists():
                    train(seed)
            set_state("evaluating_development")
        elif status == "evaluating_development":
            evaluate_development()
        elif status == "awaiting_development_review":
            if not (OUT / "development-review-final.jsonl").exists():
                return {"status": status, "awaiting": ["development-review-final.jsonl"]}
            decision = development_gate()
            if decision["status"] != "confirmation_authorized":
                write(OUT / "REPORT.md", render_report())
                return {"status": decision["status"], "decision": decision}
        elif status == "confirmation_authorized":
            evaluate_confirmation()
            set_state("awaiting_confirmation_review")
            return {"status": "awaiting_confirmation_review", "awaiting": ["confirmation-review-final.jsonl"]}
        elif status == "awaiting_confirmation_review":
            if not (OUT / "confirmation-review-final.jsonl").exists():
                return {"status": status, "awaiting": ["confirmation-review-final.jsonl"]}
            result = confirmation_gate()
            write(OUT / "REPORT.md", render_report())
            return result
        elif status in ("complete", "complete_development_gate_failed", "complete_repair_material_shortfall"):
            write(OUT / "REPORT.md", render_report())
            return z.read(STATE)
        else:
            raise RuntimeError(f"Unknown/non-resumable AC state: {status}")


def render_report() -> str:
    state = z.read(STATE) if STATE.exists() else {"status": "not_prepared"}
    lines = ["# Goal AC: failure state repair", "", f"Status: `{state['status']}`", ""]
    if (OUT / "failure-audit-summary.json").exists():
        x = z.read(OUT / "failure-audit-summary.json")
        lines += ["## Failure state audit", "", f"- Reviewed trajectories: {x['audited_cases']}",
            f"- AB-marked stop errors reviewed: {x['reviewed_stop_errors']}",
            f"- Pre-review repairable path proposals: {x['repairable_paths']}", ""]
    if (OUT / "repair-material-decision.json").exists():
        x = z.read(OUT / "repair-material-decision.json")
        lines += ["## Evidence approved material", "", f"- Evidence-approved pairs: {x.get('evidence_approved', 0)}",
            f"- Selected pairs: {x.get('selected_pairs', 0)}",
            f"- Selected stop-error repairs: {x.get('selected_stop_error_repairs', 0)}",
            f"- Material gate: {x.get('passed')}", ""]
    if (OUT / "development-decision.json").exists():
        x = z.read(OUT / "development-decision.json")
        lines += ["## Development results", "", "```json",
            json.dumps(x, ensure_ascii=False, indent=2), "```", ""]
    if (OUT / "product-decision.json").exists():
        x = z.read(OUT / "product-decision.json")
        lines += ["## Product decision", "", "```json",
            json.dumps(x, ensure_ascii=False, indent=2), "```", ""]
    lines += ["All semantic outcomes are limited to the frozen project corpora and exposed evaluation sets.", ""]
    return "\n".join(lines)


def summarize() -> dict:
    """Recompute existing artifacts only; this function never constructs a model."""
    if not CFG.exists():
        raise FileNotFoundError(CFG)
    cfg = z.read(CFG)
    for path_name, expected in cfg["asset_hashes"].items():
        path = required_assets()[path_name]
        if digest(path) != expected:
            raise RuntimeError(f"Frozen asset changed: {path_name}")
    report = {"state": z.read(STATE), "baseline": z.read(OUT / "baseline-counts.json"),
        "failure_audit": z.read(OUT / "failure-audit-summary.json") if
            (OUT / "failure-audit-summary.json").exists() else None,
        "repair_material": z.read(OUT / "repair-material-decision.json") if
            (OUT / "repair-material-decision.json").exists() else None,
        "development_decision": z.read(OUT / "development-decision.json") if
            (OUT / "development-decision.json").exists() else None,
        "product_decision": z.read(OUT / "product-decision.json") if
            (OUT / "product-decision.json").exists() else None}
    write(OUT / "summary.json", report)
    write(OUT / "REPORT.md", render_report())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else run() if args.command == "run" else summarize()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
