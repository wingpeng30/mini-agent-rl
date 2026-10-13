"""Goal X: auditable historical exposure and W candidate preflight.

This entry point initially implements the CPU asset and exposure gate. It must
not claim that semantic review or training has completed. Later stages are
unavailable until their data and executable checks are implemented.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "runs/v133-goal-w-transfer/data"
OUT = ROOT / "runs/v133-goal-x-audit-and-transfer"
G = ROOT / "runs/v133-goal-g-cpu"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def qhash(value: str) -> str:
    return hashlib.sha256(norm(value).encode()).hexdigest()


def source_paths():
    # Only named research artifacts; omit temporary and sealed directories.
    patterns = ("locator.jsonl", "screen-ledger.jsonl", "semantic-ledger.jsonl",
                "review-candidates.jsonl", "provisional.jsonl", "train-frozen.jsonl",
                "train-inputs.jsonl", "evaluation-frozen.jsonl", "eval-*.jsonl",
                "fresh-cases.jsonl", "fresh-incumbent.jsonl", "fresh-42-b.jsonl",
                "fresh-43-b.jsonl", "eval-only-cases.jsonl")
    found = set()
    for directory in (ROOT / "runs").glob("v133-goal-*"):
        if not directory.is_dir() or directory.name in {"v133-goal-w-transfer", "v133-goal-x-audit-and-transfer"} or "temp" in directory.name:
            continue
        for pattern in patterns:
            for path in directory.glob(pattern):
                if path.is_file():
                    found.add(path)
    return sorted(found)


def tier(path: Path):
    name = path.name
    if name == "locator.jsonl" or name == "provisional.jsonl":
        return "metadata_only"
    if name.startswith("screen-") or name in {"semantic-ledger.jsonl", "review-candidates.jsonl"}:
        return "evidence_opened"
    if name in {"train-frozen.jsonl", "train-inputs.jsonl"}:
        return "training_input"
    return "evaluation_input_or_output"


def exposed_row(row):
    source = str(row.get("source_id") or row.get("case_id") or row.get("id") or "")
    question = row.get("question")
    if not question and isinstance(row.get("messages"), list):
        question = next((m.get("content") for m in row["messages"] if m.get("role") == "user"), None)
    return source, qhash(question) if isinstance(question, str) and question else row.get("question_sha256")


def history_index():
    from run_v133_goal_g_cpu import exposure

    old_ids, old_questions, old_titles, old_files = exposure()
    records = collections.defaultdict(list)
    counts = {}
    for path in source_paths():
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        level = tier(path)
        used = 0
        for row in rows(path):
            source_id, question_hash = exposed_row(row)
            if not source_id and not question_hash:
                continue
            used += 1
            entry = {"path": rel, "tier": level}
            if source_id:
                records[("source_id", source_id)].append(entry)
            if question_hash:
                records[("question_sha256", question_hash)].append(entry)
        counts[rel] = {"sha256": sha(path), "tier": level, "indexed_rows": used}
    return old_ids, old_questions, old_titles, old_files, records, counts


def prepare():
    if OUT.exists():
        raise FileExistsError(f"Goal X output already exists: {OUT}")
    assets = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
              for path in [W / "manifest.json", *[W / f"candidate-{split}.jsonl" for split in ("train", "dev", "confirm")],
                           *[W / f"corpus-{split}.json" for split in ("train", "dev", "confirm")]]}
    corpus = json.loads((W / "corpus-train.json").read_text(encoding="utf-8"))["documents"]
    docs = {doc["id"]: doc for doc in corpus}
    old_ids, old_questions, old_titles, old_files, index, source_manifest = history_index()
    ledger, tier_counts = [], collections.Counter()
    for item in rows(W / "candidate-train.jsonl"):
        matches = []
        if item["source_id"] in old_ids:
            matches.append({"path": "Goal G pre-existing exposure aggregate", "tier": "pre_g_index"})
        if qhash(item["question"]) in old_questions:
            matches.append({"path": "Goal G pre-existing exposure aggregate", "tier": "pre_g_index"})
        matches.extend(index.get(("source_id", item["source_id"]), []))
        matches.extend(index.get(("question_sha256", qhash(item["question"])), []))
        matches = list({(v["path"], v["tier"]): v for v in matches}.values())
        support = [docs[doc_id] for doc_id in item["support_ids"]]
        title_matches = [doc["title"] for doc in support if norm(doc["title"]) in old_titles]
        levels = sorted({v["tier"] for v in matches})
        tier_counts.update(levels)
        ledger.append({"case_id": item["case_id"], "question": item["question"],
                       "source_id": item["source_id"], "official_answer": item["official_answer"],
                       "original_keyword_category": item["attribute_category"],
                       "support": support, "historical_matches": matches,
                       "historical_support_title_matches": title_matches,
                       "historical_tiers": levels,
                       "semantic_review_status": "pending_full_question_and_evidence_review"})
    if len(ledger) != 512:
        raise RuntimeError(f"Expected 512 frozen candidates, got {len(ledger)}")
    OUT.mkdir(parents=True)
    with (OUT / "candidate-review-input.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for item in ledger:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (OUT / "history-index-manifest.json").write_text(json.dumps({"normalization": "NFKD ASCII lowercase alphanumeric space; sha256 for normalized question",
        "pre_g_input_files": old_files, "post_g_sources": source_manifest,
        "tier_policy": {"metadata_only": "record separately; no model answer or full evidence implied",
                        "evidence_opened": "source content was inspected; exclude from new set",
                        "training_input": "exclude", "evaluation_input_or_output": "exclude",
                        "pre_g_index": "exclude per prior protocol"},
        "w_asset_sha256": assets, "candidate_count": len(ledger),
        "training_candidate_tier_counts": dict(tier_counts)}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "state.json").write_text(json.dumps({"status": "x0_preflight_prepared",
        "training_started": False, "semantic_review_complete": False,
        "history_manifest_sha256": sha(OUT / "history-index-manifest.json"),
        "candidate_packet_sha256": sha(OUT / "candidate-review-input.jsonl")}, indent=2) + "\n", encoding="utf-8")
    return {"candidates": len(ledger), "historical_tier_counts": dict(tier_counts),
            "status": "x0_preflight_prepared"}


def summarize():
    state = json.loads((OUT / "state.json").read_text(encoding="utf-8"))
    if sha(OUT / "history-index-manifest.json") != state["history_manifest_sha256"]:
        raise RuntimeError("Historical index manifest changed")
    if sha(OUT / "candidate-review-input.jsonl") != state["candidate_packet_sha256"]:
        raise RuntimeError("Candidate packet changed")
    for key, name in (("baseline_sha256", "x0-old-baseline.jsonl"),
                      ("audit_sha256", "training-audit.jsonl"),
                      ("search_trace_sha256", "train-real-search.jsonl"),
                      ("decision_sha256", "decision.json")):
        if key in state and sha(OUT / name) != state[key]:
            raise RuntimeError(f"Goal X output changed: {name}")
    return state


def baseline():
    import run_v133_goal_q_search as q
    import run_v133_goal_t_update_length as t

    state_path = OUT / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["status"] != "x0_preflight_prepared":
        raise RuntimeError("Goal X baseline is not in a prepared state")
    output = OUT / "x0-old-baseline.jsonl"
    if output.exists():
        raise FileExistsError(output)
    q.eval_model("goal-x-incumbent", t.checkpoint_dir(42, 32) / "adapter-final",
                 q.dev_cases(), output)
    result = list(rows(output))
    counts = {"total": sum(x["em"] for x in result),
              "regression": sum(x["em"] for x in result if x["split"] == "goal_G_regression"),
              "invalid": sum(x["reason"] != "answer" for x in result)}
    state["baseline"] = counts
    state["baseline_sha256"] = sha(output)
    state["status"] = "x0_baseline_passed" if len(result) == 32 and counts == {
        "total": 20, "regression": 7, "invalid": 0} else "x0_baseline_failed"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": state["status"], "counts": counts, "rows": len(result)}


def review():
    state_path = OUT / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["status"] != "x0_baseline_passed":
        raise RuntimeError("X0 baseline gate has not passed")
    output = OUT / "training-audit.jsonl"
    summary_path = OUT / "training-audit-summary.json"
    if output.exists() or summary_path.exists():
        raise FileExistsError("Goal X question audit already finalized")
    packets = list(rows(OUT / "candidate-review-input.jsonl"))
    reviews = list(rows(OUT / "semantic-question-review.jsonl"))
    if len(packets) != 512 or len(reviews) != 512:
        raise RuntimeError(f"Full 512-case review required; found {len(reviews)}")
    if [v["case_id"] for v in packets] != [v["case_id"] for v in reviews]:
        raise RuntimeError("Semantic review order or case ids differ from frozen W pool")
    history = json.loads((OUT / "x0-history-exposure-index.json").read_text(encoding="utf-8"))
    past_titles = set(history["post_g_title_union"]["normalized_titles"])
    valid_categories = {"occupation", "entity_type", "affiliation", "origin"}
    audit = []
    for item, decision in zip(packets, reviews):
        if decision["revised_category"] not in valid_categories | {"other", "not_applicable", "uncertain"}:
            raise RuntimeError(f"Unknown category: {decision['revised_category']}")
        if decision["evidence_status"] not in {"supported", "partial", "unsupported", "uncertain", "conflict"}:
            raise RuntimeError(f"Unknown evidence status: {decision['evidence_status']}")
        historical_hard = [x for x in item["historical_matches"] if x["tier"] in {
            "evidence_opened", "training_input", "evaluation_input_or_output", "pre_g_index"}]
        title_overlap = sorted(set(item["historical_support_title_matches"]) |
                               {doc["title"] for doc in item["support"] if norm(doc["title"]) in past_titles})
        flags = []
        if historical_hard or title_overlap:
            flags.append("historical_overlap")
        if not decision["shared_attribute_question"] or decision["revised_category"] not in valid_categories:
            flags.append("question_inapplicable_or_outside_four_classes")
        if decision["evidence_status"] != "supported":
            flags.append("source_evidence_insufficient_or_uncertain")
        status = "eligible_for_real_search" if not flags else flags[0]
        audit.append({"case_id": item["case_id"], "source_id": item["source_id"],
                      "question": item["question"], "official_answer": item["official_answer"],
                      "original_keyword_category": item["original_keyword_category"],
                      "reviewed_category": decision["revised_category"],
                      "shared_attribute_question": decision["shared_attribute_question"],
                      "question_review_reason": decision["reason"],
                      "evidence_status": decision["evidence_status"],
                      "support": item["support"], "historical_matches": item["historical_matches"],
                      "historical_support_title_matches": title_overlap,
                      "metadata_only_overlap": any(x["tier"] == "metadata_only" for x in item["historical_matches"]),
                      "flags": flags, "primary_disposition": status})
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for item in audit:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    counts = collections.Counter(x["primary_disposition"] for x in audit)
    reviewed = collections.Counter(x["reviewed_category"] for x in audit if x["shared_attribute_question"])
    eligible = collections.Counter(x["reviewed_category"] for x in audit if x["primary_disposition"] == "eligible_for_real_search")
    summary = {"rows": len(audit), "question_review_sha256": sha(OUT / "semantic-question-review.jsonl"),
               "historical_index_sha256": sha(OUT / "x0-history-exposure-index.json"),
               "audit_sha256": sha(output), "primary_disposition_counts": dict(counts),
               "reviewed_shared_category_counts": dict(reviewed),
               "eligible_for_real_search_by_category": dict(eligible),
               "metadata_only_overlap_count": sum(x["metadata_only_overlap"] for x in audit),
               "label_change_count": sum(x["reviewed_category"] != x["original_keyword_category"] for x in audit),
               "provisional": "X1 reviewed source evidence; X2 actual search visibility and pair validity remain pending"}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    state.update(status="x1_audit_finalized", semantic_review_complete=True,
                 audit_sha256=summary["audit_sha256"], audit_summary_sha256=sha(summary_path))
    state_path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def collect():
    """One greedy real-search trace per X1-eligible train question."""
    import run_v133_goal_q_search as q
    import run_v133_goal_t_update_length as t
    from mini_agent_rl.tools import SearchTool, ToolRegistry

    state_path = OUT / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["status"] != "x1_audit_finalized":
        raise RuntimeError("X1 semantic and historical gate has not passed")
    if sha(OUT / "training-audit.jsonl") != state["audit_sha256"]:
        raise RuntimeError("Frozen training audit changed")
    output = OUT / "train-real-search.jsonl"
    if output.exists():
        raise FileExistsError(output)
    cases = [x for x in rows(OUT / "training-audit.jsonl")
             if x["primary_disposition"] == "eligible_for_real_search"]
    corpus = json.loads((W / "corpus-train.json").read_text(encoding="utf-8"))["documents"]
    client, wrapped, _ = q.make_client(t.checkpoint_dir(42, 32) / "adapter-final", deterministic=True)
    registry = ToolRegistry()
    registry.register(SearchTool(corpus, q.load_config()["product"]["retriever"]))
    state.update(status="x2_collecting", model_traces_expected=len(cases))
    state_path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for number, item in enumerate(cases, 1):
            value = asyncio.run(q.rollout(item["question"], wrapped, registry, sample=False, seed=42))
            observed = {doc["id"] for step in value["trace"]
                        for doc in step.get("observation", {}).get("result", [])}
            answer_step = next((step for step in reversed(value["trace"])
                                if step.get("action") == "answer"), None)
            prompt = (client._processor.decode(answer_step["prompt_token_ids"], skip_special_tokens=False)
                      if answer_step else "")
            visible_ids = [doc["id"] for doc in item["support"] if doc["id"] in prompt]
            visible_text = [doc["id"] for doc in item["support"]
                            if norm(doc["text"]) and norm(doc["text"]) in norm(prompt)]
            usage = answer_step["usage"] if answer_step else {}
            value.update(case_id=item["case_id"], source_id=item["source_id"],
                         reviewed_category=item["reviewed_category"],
                         official_answer=item["official_answer"],
                         support_ids=[doc["id"] for doc in item["support"]],
                         both_support_retrieved=all(doc["id"] in observed for doc in item["support"]),
                         both_support_ids_visible=all(doc["id"] in visible_ids for doc in item["support"]),
                         both_support_full_text_visible=all(doc["id"] in visible_text for doc in item["support"]),
                         answer_prompt_token_ids=answer_step["prompt_token_ids"] if answer_step else [],
                         answer_prompt_attention_mask=answer_step["prompt_attention_mask"] if answer_step else [],
                         answer_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                         answer_prompt_visibility_flags={
                             "prompt_truncated": usage.get("prompt_truncated"),
                             "documents_deleted": usage.get("documents_deleted"),
                             "lossy_body_trimmed": usage.get("lossy_body_trimmed")})
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            print(json.dumps({"candidate": number, "of": len(cases), "case_id": item["case_id"],
                              "reason": value["reason"], "both_support_ids_visible": value["both_support_ids_visible"]}), flush=True)
    state.update(status="x2_search_collected", search_trace_sha256=sha(output),
                 search_trace_count=len(cases))
    state_path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"status": state["status"], "traces": len(cases)}


def finalize():
    state_path = OUT / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["status"] != "x2_search_collected":
        raise RuntimeError("X2 real search is incomplete")
    output = OUT / "train-real-search.jsonl"
    if sha(output) != state["search_trace_sha256"]:
        raise RuntimeError("Search trace changed")
    audit = [x for x in rows(OUT / "training-audit.jsonl")
             if x["primary_disposition"] == "eligible_for_real_search"]
    traces = list(rows(output))
    if len(traces) != len(audit) or [x["case_id"] for x in traces] != [x["case_id"] for x in audit]:
        raise RuntimeError("Search trace incomplete or order changed")
    def usable(x):
        flags = x["answer_prompt_visibility_flags"]
        return (x["reason"] == "answer" and x["both_support_retrieved"]
                and x["both_support_ids_visible"] and x["both_support_full_text_visible"]
                and flags["prompt_truncated"] is False
                and flags["documents_deleted"] == 0
                and flags["lossy_body_trimmed"] is False
                and bool(x["answer_prompt_token_ids"]))
    usable_by_category = collections.Counter(x["reviewed_category"] for x in traces if usable(x))
    failed_reasons = collections.Counter(x["reason"] for x in traces if not usable(x))
    result = {"status": "data_shortfall" if usable_by_category["affiliation"] < 16 else "pair_review_required",
              "source_candidates": 512, "question_and_source_eligible": len(audit),
              "search_traces": len(traces), "usable_real_contexts": sum(map(usable, traces)),
              "usable_real_contexts_by_category": dict(usable_by_category),
              "not_usable_reason_counts": dict(failed_reasons),
              "affiliation_minimum": 16,
              "affiliation_maximum_possible_pairs": usable_by_category["affiliation"],
              "validated_chosen_rejected_pairs": 0,
              "pair_status": "not_constructed_because_affiliation_upper_bound_fails_predeclared_quota" if usable_by_category["affiliation"] < 16 else "requires_review",
              "training_started": False, "selected_product": "goal_t_seed42_update32",
              "gpu_trace_elapsed_seconds": sum(float(x["elapsed_seconds"]) for x in traces),
              "search_trace_sha256": sha(output)}
    decision = OUT / "decision.json"
    if decision.exists():
        raise FileExistsError(decision)
    decision.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    state.update(status="complete_data_shortfall" if result["status"] == "data_shortfall" else "x3_pair_review_required",
                 decision_sha256=sha(decision), training_started=False)
    state_path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def run():
    state = json.loads((OUT / "state.json").read_text(encoding="utf-8"))
    if state["status"] == "x0_preflight_prepared":
        baseline()
    state = json.loads((OUT / "state.json").read_text(encoding="utf-8"))
    if state["status"] == "x0_baseline_passed":
        review()
    state = json.loads((OUT / "state.json").read_text(encoding="utf-8"))
    if state["status"] == "x1_audit_finalized":
        collect()
    state = json.loads((OUT / "state.json").read_text(encoding="utf-8"))
    if state["status"] == "x2_search_collected":
        return finalize()
    return summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "baseline", "review", "collect", "finalize", "run", "summarize"))
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "baseline": baseline, "review": review, "collect": collect,
                      "finalize": finalize,
                      "run": run, "summarize": summarize}[args.command](), ensure_ascii=False))
