"""Enrich final experiment reports using saved ledgers only (stdlib, no model calls).

Answer aliases and source annotations are reused; missing annotations remain NA.
Run after the bare baseline and before archiving. Paired bootstrap is referenced
from summarize_final_online_grpo.py's summary.json and never computed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import string
from typing import Any


BRANCHES = ("am-serial", "am-batch", "sft42", "grpo42", "grpo43", "base-untrained")
SINGLE_STRATA = {"constructed_singlehop", "singlehop", "single24"}
NA = None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_rows(path: Path | None, warnings: list[str]) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                warnings.append(f"Ignored incomplete last record: {path}:{number}")
                continue
            raise ValueError(f"Malformed JSONL: {path}:{number}") from None
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {path}:{number}")
        result.append(value)
    return result


def identity(row: dict[str, Any]) -> str:
    return str(row.get("case_id", row.get("id", "")))


def index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        key = identity(row)
        if not key or key in result:
            raise ValueError(f"Missing/duplicate case ID in {label}: {key}")
        result[key] = row
    return result


def finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: list[Any]) -> dict[str, Any]:
    numbers = [number for value in values if (number := finite(value)) is not None]
    return dict(count=len(numbers), missing=len(values) - len(numbers),
                sum=sum(numbers) if numbers else None, mean=statistics.mean(numbers) if numbers else None,
                min=min(numbers) if numbers else None, max=max(numbers) if numbers else None,
                **{f"p{int(q * 100)}": percentile(numbers, q) for q in (.5, .9, .95, .99)})


def rate(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def normalize(text: Any) -> str:
    value = re.sub(r"\b(a|an|the)\b", " ", str(text or "").lower())
    return " ".join(value.translate(str.maketrans("", "", string.punctuation)).split())


def token_scores(prediction: Any, reference: Any) -> dict[str, float]:
    pred, gold = normalize(prediction).split(), normalize(reference).split()
    if not pred or not gold:
        match = float(not pred and not gold)
        return dict(precision=match, recall=match, f1=match)
    overlap = sum((Counter(pred) & Counter(gold)).values())
    precision, recall = overlap / len(pred), overlap / len(gold)
    return dict(precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if overlap else 0.)


def alias_scores(row: dict[str, Any], case: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    canonical = case.get("answer", row.get("gold", audit.get("gold_answer")))
    aliases = case.get("answer_aliases", case.get("aliases", audit.get("answer_aliases", []))) or []
    references = list(dict.fromkeys(str(v) for v in [canonical, *aliases] if v is not None))
    if not references:
        return dict(precision=None, recall=None, f1=None, normalized_exact_match=None,
                    reference_count=0, selected_reference=None, annotation_status="missing_gold")
    values = [(token_scores(row.get("answer"), ref), ref) for ref in references]
    # Select one reference by F1; precision/recall belong to that same reference.
    best, ref = max(values, key=lambda item: (item[0]["f1"],
                     normalize(row.get("answer")) == normalize(item[1])))
    return dict(best, normalized_exact_match=float(any(normalize(row.get("answer")) == normalize(v)
                                                       for v in references)),
                reference_count=len(references), selected_reference=ref, annotation_status="existing_labels")


def observations(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [step["observation"] for step in trace if isinstance(step.get("observation"), dict)]


def documents(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [doc for obs in observations(trace) for doc in obs.get("result", []) if isinstance(doc, dict)]


def context(step: dict[str, Any]) -> dict[str, Any]:
    usage = step.get("usage") or {}
    return step.get("context_usage") or usage.get("context_usage") or usage


def last_visible_documents(trace: list[dict[str, Any]]) -> tuple[dict[str, list[str]] | None, str]:
    """Reconstruct saved context metadata; never assume newly returned tools were seen."""
    if not trace:
        return None, "no_saved_model_input"
    metadata = context(trace[-1])
    presentation = metadata.get("presentation") or {}
    presentation_mode = presentation.get("observation_presentation", presentation.get("mode", presentation.get("encoding")))
    if presentation_mode is not None and presentation_mode != "legacy-json":
        return None, "nonlegacy_presentation_not_reconstructed"
    changes = metadata.get("document_changes")
    visible: dict[str, list[str]] = defaultdict(list)
    if changes:
        for change in changes:
            if not change.get("visible") or change.get("deleted"):
                continue
            key, text = change.get("document_id"), change.get("original_text")
            if key is None or not isinstance(text, str):
                return None, "visible_document_body_not_saved"
            for segment in change.get("visible_segments") or []:
                if len(segment) != 2 or not all(isinstance(i, int) for i in segment):
                    return None, "invalid_visible_segment_metadata"
                visible[str(key)].append(text[segment[0]:segment[1]])
        return dict(visible), "saved_document_changes_visible_segments"
    required = ("prompt_truncated", "lossy_body_trimmed", "history_dropped", "context_budget_error")
    if (metadata.get("observation_encoding") == "legacy" and metadata.get("truncation_stage") == "none"
            and all(metadata.get(key) is False for key in required)
            and metadata.get("documents_deleted") == 0 and "document_changes" in metadata):
        for doc in documents(trace[:-1]):
            if doc.get("id") is not None and isinstance(doc.get("text"), str):
                visible[str(doc["id"])].append(doc["text"])
        return dict(visible), "explicit_legacy_no_truncation_all_prior_tools_retained"
    return None, "last_input_visibility_not_certified_by_saved_metadata"


def source_coverage(row: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    status = audit.get("status", "missing")
    empty = dict(annotation_status=status, measurable=False, required_documents=None,
                 retrieved_document_recall=None, any_required_document_retrieved=None,
                 all_required_documents_retrieved=None, retrieved_quote_recall=None,
                 all_required_quotes_retrieved=None, last_visible_document_recall=None,
                 all_required_documents_last_visible=None, last_visible_quote_recall=None,
                 all_required_quotes_last_visible=None, last_visibility_status="NA")
    if status != "confirmed":
        return empty
    evidence = audit.get("support_documents") or [e.get("source_evidence") for e in audit.get("required_edges", [])]
    evidence = [e for e in evidence if isinstance(e, dict) and e.get("document_id") is not None]
    required = {str(e["document_id"]) for e in evidence}
    if not required:
        empty["annotation_status"] = "confirmed_but_no_document_annotation"
        return empty
    trace = row.get("trace") or []
    retrieved = documents(trace)
    found = {str(d["id"]) for d in retrieved if d.get("id") is not None}
    quotes = [(str(e["document_id"]), e["quote"]) for e in evidence if isinstance(e.get("quote"), str) and e["quote"]]
    quote_hits = sum(any(str(doc.get("id")) == key and quote in str(doc.get("text", ""))
                         for doc in retrieved) for key, quote in quotes)
    visible, visible_status = last_visible_documents(trace)
    visible_ids = set(visible) if visible is not None else None
    visible_quote_hits = (sum(any(quote in text for text in visible.get(key, [])) for key, quote in quotes)
                          if visible is not None else None)
    return dict(annotation_status=status, measurable=True, required_documents=len(required),
                retrieved_required_documents=len(required & found),
                retrieved_document_recall=len(required & found) / len(required),
                any_required_document_retrieved=bool(required & found),
                all_required_documents_retrieved=required <= found,
                required_quotes=len(quotes), retrieved_quote_recall=rate(quote_hits, len(quotes)),
                all_required_quotes_retrieved=quote_hits == len(quotes) if quotes else None,
                last_visible_document_recall=rate(len(required & visible_ids), len(required)) if visible_ids is not None else None,
                all_required_documents_last_visible=required <= visible_ids if visible_ids is not None else None,
                last_visible_quote_recall=rate(visible_quote_hits, len(quotes)) if visible_quote_hits is not None else None,
                all_required_quotes_last_visible=visible_quote_hits == len(quotes) if quotes and visible_quote_hits is not None else None,
                last_visibility_status=visible_status)


def step_metrics(step: dict[str, Any]) -> dict[str, Any]:
    usage, metadata = step.get("usage") or {}, context(step)
    input_tokens = finite(usage.get("prompt_tokens"))
    if input_tokens is None and isinstance(step.get("prompt_attention_mask"), list):
        input_tokens = sum(step["prompt_attention_mask"])
    if input_tokens is None and isinstance(step.get("prompt_token_ids"), list):
        input_tokens = len(step["prompt_token_ids"])
    output_tokens = finite(usage.get("completion_tokens"))
    if output_tokens is None and isinstance(step.get("action_token_ids"), list):
        output_tokens = len(step["action_token_ids"])
    obs = step.get("observation") or {}
    return dict(input_tokens=input_tokens, output_tokens=output_tokens,
                json_action_tokens=finite(usage.get("json_action_tokens")),
                raw_action_chars=len(step["raw"]) if isinstance(step.get("raw"), str) else None,
                model_seconds=finite(step.get("model_seconds")),
                batch_seconds=finite(step.get("batch_seconds")), batch_active_rows=finite(step.get("batch_active_rows")),
                tool_seconds=finite(obs.get("latency_ms")) / 1000 if finite(obs.get("latency_ms")) is not None else None,
                query_chars=len(step["query"]) if isinstance(step.get("query"), str) else None,
                left_padding_tokens=finite(metadata.get("left_padding_tokens")),
                original_prompt_tokens=finite(metadata.get("original_prompt_tokens")),
                dropped_observation_tokens=finite(metadata.get("dropped_observation_tokens")),
                generation_stop_reason=usage.get("generation_stop_reason", "unrecorded"),
                protocol_recovery=usage.get("protocol_recovery"),
                prompt_truncated=metadata.get("prompt_truncated"),
                output_truncated=usage.get("output_truncated"))


def case_metrics(row: dict[str, Any], case: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("trace") or []
    steps = [step_metrics(step) for step in trace]
    scores = alias_scores(row, case, audit)
    answered = bool(str(row.get("answer") or "").strip())
    exact = finite(row.get("normalized_exact_match"))
    if exact is None:
        exact = scores["normalized_exact_match"]
    queries = [" ".join(str(s["query"]).casefold().split()) for s in trace if "query" in s]
    docs = documents(trace)
    doc_ids = [str(d["id"]) for d in docs if d.get("id") is not None]
    successful = [obs.get("ok") is True for obs in observations(trace) if "ok" in obs]
    tool_times = [finite(obs.get("latency_ms")) for obs in observations(trace)]
    def total(metric: str, fallback: str | None = None):
        if fallback and finite(row.get(fallback)) is not None:
            return finite(row[fallback])
        values = [s[metric] for s in steps]
        return sum(values) if values and all(v is not None for v in values) else None
    return dict(case_id=identity(row), stratum=row.get("stratum", case.get("stratum", "unspecified")),
                reason=row.get("reason", "unrecorded"), answered=answered,
                answer_action=row.get("reason") == "answer", normalized_exact_match=exact,
                strict_exact_match=finite(row.get("strict_exact_match", row.get("exact_match"))),
                best_alias=scores, wrong_answer=bool(answered and exact == 0),
                model_calls=finite(row.get("model_calls", len(trace))),
                searches=finite(row.get("searches", len(observations(trace)))),
                input_tokens=total("input_tokens", "input_tokens"), output_tokens=total("output_tokens", "output_tokens"),
                model_seconds=total("model_seconds"), tool_seconds=sum(tool_times) / 1000 if all(v is not None for v in tool_times) else None,
                elapsed_seconds=finite(row.get("elapsed_seconds")),
                invalid_actions=sum(bool(s.get("parse_error")) for s in trace),
                repeated_actions=sum(bool(s.get("repeated")) for s in trace),
                query_count=len(queries), query_duplicates=len(queries) - len(set(queries)),
                document_results=len(docs), identified_document_results=len(doc_ids),
                unique_documents=len(set(doc_ids)), document_duplicates=len(doc_ids) - len(set(doc_ids)),
                search_ok_count=sum(successful), search_ok_measurable=len(successful),
                empty_search_results=sum(not obs.get("result") for obs in observations(trace)),
                source=source_coverage(row, audit), steps=steps)


def aggregate(rows: list[dict[str, Any]], expected: int | None = None) -> dict[str, Any]:
    n, answered = len(rows), sum(r["answered"] for r in rows)
    exact = [r["normalized_exact_match"] for r in rows]
    correct = sum(r["normalized_exact_match"] or 0 for r in rows)
    source = [r["source"] for r in rows]
    known = [s for s in source if s["measurable"]]
    steps = [s for r in rows for s in r["steps"]]
    result = dict(count=n, expected=expected, complete=n == expected if expected is not None else None,
        completion=dict(answered=answered, answer_action=sum(r["answer_action"] for r in rows), coverage=rate(answered, n),
            normalized_em_count=correct, normalized_em_rate=rate(correct, sum(v is not None for v in exact)),
            normalized_em_missing=sum(v is None for v in exact),
            accuracy_among_answered=rate(sum(r["normalized_exact_match"] or 0 for r in rows if r["answered"]),
                                        sum(r["answered"] and r["normalized_exact_match"] is not None for r in rows)),
            answered_accuracy_measurable=sum(r["answered"] and r["normalized_exact_match"] is not None for r in rows),
            wrong_answers=sum(r["wrong_answer"] for r in rows), wrong_answer_rate_all=rate(sum(r["wrong_answer"] for r in rows), n),
            reason_counts=dict(Counter(r["reason"] for r in rows)),
            invalid_action_cases=sum(r["invalid_actions"] > 0 for r in rows), invalid_action_count=sum(r["invalid_actions"] for r in rows),
            repeated_action_cases=sum(r["repeated_actions"] > 0 for r in rows), repeated_action_count=sum(r["repeated_actions"] for r in rows),
            budget_exhausted_cases=sum(r["reason"] in {"max_steps", "context_budget_error", "max_model_calls", "max_searches"} for r in rows)),
        best_alias={key: distribution([r["best_alias"][key] for r in rows]) for key in ("precision", "recall", "f1")},
        per_case={key: distribution([r[key] for r in rows]) for key in
            ("model_calls", "searches", "input_tokens", "output_tokens", "model_seconds", "tool_seconds", "elapsed_seconds", "unique_documents")},
        per_action={key: distribution([s[key] for s in steps]) for key in
            ("input_tokens", "output_tokens", "json_action_tokens", "raw_action_chars", "model_seconds", "batch_seconds", "batch_active_rows",
             "tool_seconds", "query_chars", "left_padding_tokens", "original_prompt_tokens", "dropped_observation_tokens")},
        trace=dict(actions=len(steps), generation_stop_counts=dict(Counter(s["generation_stop_reason"] for s in steps)),
            protocol_recovery_counts=dict(Counter(s["protocol_recovery"] for s in steps if s["protocol_recovery"])),
            prompt_truncated=sum(s["prompt_truncated"] is True for s in steps), prompt_truncation_metadata_missing=sum(s["prompt_truncated"] is None for s in steps),
            output_truncated=sum(s["output_truncated"] is True for s in steps), output_truncation_metadata_missing=sum(s["output_truncated"] is None for s in steps)),
        search=dict(query_count=sum(r["query_count"] for r in rows), duplicate_queries=sum(r["query_duplicates"] for r in rows),
            duplicate_query_fraction=rate(sum(r["query_duplicates"] for r in rows), sum(r["query_count"] for r in rows)),
            document_results=sum(r["document_results"] for r in rows), identified_document_results=sum(r["identified_document_results"] for r in rows),
            duplicate_document_results=sum(r["document_duplicates"] for r in rows),
            duplicate_document_fraction=rate(sum(r["document_duplicates"] for r in rows), sum(r["identified_document_results"] for r in rows)),
            successful_searches=sum(r["search_ok_count"] for r in rows), search_ok_measurable=sum(r["search_ok_measurable"] for r in rows),
            search_success_rate=rate(sum(r["search_ok_count"] for r in rows), sum(r["search_ok_measurable"] for r in rows)),
            empty_search_results=sum(r["empty_search_results"] for r in rows)),
        source=dict(annotation_counts=dict(Counter(s["annotation_status"] for s in source)), measurable_cases=len(known),
            unknown_cases=n - len(known), required_document_instances=sum(s["required_documents"] for s in known),
            retrieved_document_micro_recall=rate(sum(s["retrieved_required_documents"] for s in known), sum(s["required_documents"] for s in known)),
            last_visibility_status_counts=dict(Counter(s["last_visibility_status"] for s in known))))
    for key in ("retrieved_document_recall", "any_required_document_retrieved", "all_required_documents_retrieved",
                "retrieved_quote_recall", "all_required_quotes_retrieved", "last_visible_document_recall",
                "all_required_documents_last_visible", "last_visible_quote_recall", "all_required_quotes_last_visible"):
        result["source"][key] = distribution([s[key] for s in known])
    return result


def group_diversity(group: dict[str, Any], rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [r for r in rollouts if r.get("group") == group.get("group")
               and identity(r) == identity(group) and r.get("policy_version") == group.get("policy_version")]
    # Recovery can repeat a candidate; use the last saved complete attempt for each.
    candidates = list({r.get("candidate"): r for r in matched}.values())
    answers = [normalize(r.get("answer")) for r in candidates]
    queries, actions, action_types = [], [], []
    for row in candidates:
        trace = row.get("trace") or []
        queries.append(tuple(" ".join(str(s["query"]).casefold().split()) for s in trace if "query" in s))
        types = tuple(s.get("action", "invalid" if s.get("parse_error") else "unknown") for s in trace)
        action_types.append(types)
        actions.append(tuple((types[i], " ".join(str(s.get("query", "")).casefold().split()), normalize(s.get("answer")))
                             for i, s in enumerate(trace)))
    pairs = [(answers[i], answers[j]) for i in range(len(answers)) for j in range(i + 1, len(answers))]
    nonempty_pairs = [(a, b) for a, b in pairs if a and b]
    return dict(group=group.get("group"), case_id=identity(group), candidate_count=len(candidates),
        expected_candidates=len(group.get("rewards") or []) or 4,
        unique_normalized_answers_including_empty=len(set(answers)) if candidates else None,
        unique_nonempty_normalized_answers=len({a for a in answers if a}) if candidates else None,
        empty_candidate_count=sum(not a for a in answers) if candidates else None,
        unique_query_sequences=len(set(queries)) if candidates else None,
        unique_action_sequences=len(set(actions)) if candidates else None,
        unique_action_type_sequences=len(set(action_types)) if candidates else None,
        equal_normalized_answer_pair_fraction=rate(sum(a == b for a, b in pairs), len(pairs)),
        nonempty_answer_pair_token_f1=distribution([token_scores(a, b)["f1"] for a, b in nonempty_pairs]),
        pairs_involving_empty=len(pairs) - len(nonempty_pairs))


def training_metrics(folder: Path, warnings: list[str]) -> dict[str, Any]:
    summary = read_json(folder / "summary.json")
    updates = read_rows(folder / "updates.jsonl", warnings)
    groups = read_rows(folder / "group-metrics.jsonl", warnings)
    rollouts = read_rows(folder / "rollouts.jsonl", warnings)
    partial = read_rows(folder / "partial-rollouts.jsonl", warnings)
    attempts = read_rows(folder / "generation-attempts.jsonl", warnings)
    resumes = read_rows(folder / "resume-ledger.jsonl", warnings)
    if not (summary or updates or groups or rollouts or partial or attempts):
        return dict(status="no_training_ledger")
    # Physical ledger attempts stay charged; summary is the final logical state.
    by_group = {str(row.get("group")): row for row in groups}
    logical = list(by_group.values())
    all_correct = sum(bool(r.get("rewards")) and all(v == 1 for v in r["rewards"]) for r in logical)
    all_wrong = sum(bool(r.get("rewards")) and all(v == 0 for v in r["rewards"]) for r in logical)
    mixed = sum(len(set(r.get("rewards") or [])) > 1 for r in logical)
    rewards = [v for r in logical for v in r.get("rewards") or []]
    reward_entropy = []
    for row in logical:
        values = row.get("rewards") or []
        reward_entropy.append(-sum((count / len(values)) * math.log2(count / len(values))
                              for count in Counter(values).values()) if values else None)
    completed = [r for r in attempts if r.get("event") == "completed"]
    started = [r for r in attempts if r.get("event") == "started"]
    elapsed_values = [finite(r.get("elapsed_seconds")) for r in updates]
    increments, previous = [], 0.
    for elapsed in elapsed_values:
        if elapsed is None:
            increments.append(None)
        else:
            increments.append(elapsed - previous if elapsed >= previous else elapsed)
            previous = elapsed
    updated = sum(bool(r.get("updated")) for r in logical)
    diversity = [group_diversity(group, rollouts) for group in logical]
    sft_updates = [r for r in updates if finite(r.get("supervised_tokens")) is not None
                   and finite(r.get("supervised_tokens")) > 0 and finite(r.get("loss")) is not None]
    sft_tokens = sum(r["supervised_tokens"] for r in sft_updates)
    nll = sum(r["loss"] * r["supervised_tokens"] for r in sft_updates) / sft_tokens if sft_tokens else None
    result = dict(status=summary.get("status", "ledger_only"), final_summary=summary,
        physical_ledger=dict(update_records=len(updates), group_records=len(groups), rollout_records=len(rollouts),
            partial_trajectory_records=len(partial), resume_records=len(resumes),
            generation_batches_started=len(started), generation_batches_completed=len(completed),
            generation_row_attempts_started=sum(r.get("rows", 0) for r in started),
            generation_row_attempts_completed=sum(r.get("rows", 0) for r in completed),
            uncompleted_batch_attempts=max(0, len(started) - len(completed))),
        logical_groups=dict(count=len(logical), all_correct=all_correct, all_wrong=all_wrong, mixed=mixed,
            mixed_group_fraction=rate(mixed, len(logical)),
            effective_groups=updated, effective_group_fraction=rate(updated, len(logical)),
            zero_variance_groups=sum(finite(r.get("reward_std")) == 0 for r in logical),
            reward_histogram=dict(sorted(Counter(str(v) for v in rewards).items())),
            reward_mean=distribution([r.get("reward_mean") for r in logical]),
            reward_std=distribution([r.get("reward_std") for r in logical]),
            outcome_entropy_bits=distribution(reward_entropy),
            optimizer_steps=distribution([r.get("optimizer_steps") for r in logical])),
        optimizer=dict(unique_update_numbers=len({r.get("update") for r in updates}),
            final_checkpoint_updates=summary.get("updates"),
            gradient_clipping_updates=sum(finite(r.get("grad")) is not None and finite(r["grad"]) > 1 for r in updates),
            policy_epoch_counts=dict(Counter(str(r.get("policy_epoch", "SFT")) for r in updates)),
            metrics={key: distribution([r.get(key) for r in updates]) for key in ("loss", "kl", "clip_fraction", "grad", "supervised_tokens")},
            ratio_summary={key: distribution([(r.get("ratio") or {}).get(key) for r in updates]) for key in ("min", "max", "p95", "count")},
            elapsed_between_recorded_updates_seconds=distribution(increments)),
        cost=dict(training_wall_seconds=finite(summary.get("training_seconds")),
            completed_generation_batch_seconds=distribution([r.get("elapsed_seconds") for r in completed]),
            completed_rollout_metrics=aggregate([case_metrics(r, {}, {}) for r in rollouts]),
            peak_gpu_memory=summary.get("gpu_memory"),
            gpu_hours=finite(summary.get("training_seconds")) / 3600 if finite(summary.get("training_seconds")) is not None else None,
            gpu_utilization=None, monetary_cost=None, energy_kwh=None),
        policy_entropy=None, policy_entropy_reason="Full policy token probability distributions were not recorded; outcome entropy is separate.",
        group_diversity=dict(per_group=diversity,
            metrics={key: distribution([r[key] for r in diversity]) for key in
                ("candidate_count", "unique_normalized_answers_including_empty", "unique_nonempty_normalized_answers", "empty_candidate_count",
                 "unique_query_sequences", "unique_action_sequences", "unique_action_type_sequences", "equal_normalized_answer_pair_fraction")},
            note="Normalized empty answers are counted separately. Pairwise token F1 excludes pairs involving an empty answer; action sequences include parsed query/answer content."),
        sft_process_nll=dict(record_count=len(sft_updates), supervised_tokens=sft_tokens, token_weighted_nll=nll,
            exp_process_nll=math.exp(nll) if nll is not None and nll < 709 else None,
            interpretation="Token-weighted mean of recorded SFT forward NLL during training (physical records, including replay); exp is a process statistic, not final-checkpoint perplexity. RL surrogate loss is not NLL/PPL."),
        resumed=bool(resumes), deduplication="Last completed/cutoff record per group index; all physical records retain attempt cost.")
    return result


def provenance(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return dict(path=str(path) if path else None, available=False, sha256=None)
    return dict(path=str(path), available=True, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path: Path, value: Any):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fmt(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else str(round(value, digits)) if isinstance(value, float) else str(value)


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Comprehensive metrics from existing ledgers", "", "仅复用现有答案别名、trace 和既有来源审计；无模型调用、无新人工或语义审查。未知来源题的证据指标保留 NA。", "",
             "配对 bootstrap 复用 [summary.json](summary.json) 与 [REPORT.md](REPORT.md)，本脚本不重复抽样。完整逐题及分布见 [comprehensive-metrics.json](comprehensive-metrics.json)。", "",
             "| Branch | n / expected | answered coverage | accuracy among answered | normalized EM | best-alias P / R / F1 | wrong / invalid / repeated / budget |", "|---|---:|---:|---:|---:|---|---|"]
    for name, branch in report["evaluation"].items():
        group = branch["groups"]["all128"]
        c, aliases = group["completion"], group["best_alias"]
        lines.append(f"| {name} | {group['count']} / {fmt(group['expected'])} | {fmt(c['coverage'])} | {fmt(c['accuracy_among_answered'])} | {fmt(c['normalized_em_rate'])} | "
                     + " / ".join(fmt(aliases[k]["mean"]) for k in ("precision", "recall", "f1"))
                     + f" | {c['wrong_answers']} / {c['invalid_action_cases']} / {c['repeated_action_cases']} / {c['budget_exhausted_cases']} |")
    lines += ["", "## Sources and final input visibility", "", "Doc recall uses existing required document IDs. Quote recall is exact substring coverage of existing quoted spans, not a new semantic correctness judgment. Last input excludes tool results returned by the final model call. Macro recall denominators include only confirmed source annotations, and last-visible metrics additionally require saved input metadata.", "",
              "| Branch | confirmed / unknown | retrieved doc recall | all required docs | last-visible doc recall | all quotes last visible |", "|---|---:|---:|---:|---:|---:|"]
    for name, branch in report["evaluation"].items():
        s = branch["groups"]["all128"]["source"]
        lines.append(f"| {name} | {s['measurable_cases']} / {s['unknown_cases']} | {fmt(s['retrieved_document_recall']['mean'])} | {fmt(s['all_required_documents_retrieved']['mean'])} | {fmt(s['last_visible_document_recall']['mean'])} | {fmt(s['all_required_quotes_last_visible']['mean'])} |")
    for name, branch in report["evaluation"].items():
        lines += ["", f"## {name}: distributions and source strata", "", "| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        group = branch["groups"]["all128"]
        for scope in ("per_case", "per_action"):
            for metric, stats in group[scope].items():
                lines.append(f"| {scope}.{metric} | {stats['count']} / {stats['missing']} | "
                             + " | ".join(fmt(stats[k]) for k in ("mean", "min", "p50", "p90", "p95", "p99", "max")) + " |")
        lines += ["", "| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |", "|---|---:|---:|---:|---:|---:|"]
        for stratum, item in branch["source_strata"].items():
            lines.append(f"| {stratum} | {item['count']} | {fmt(item['completion']['normalized_em_rate'])} | {fmt(item['best_alias']['f1']['mean'])} | {item['source']['measurable_cases']} | {fmt(item['source']['all_required_documents_retrieved']['mean'])} |")
        lines += ["", "Termination reasons: `" + json.dumps(group["completion"]["reason_counts"], ensure_ascii=False) + "`.",
                  "Search/query diagnostics: `" + json.dumps(group["search"], ensure_ascii=False) + "`.",
                  "Measured cost/throughput: `" + json.dumps(branch["cost"], ensure_ascii=False) + "`."]
    lines += ["", "## Training", "", "| Branch | groups | mixed / all correct / all wrong | effective groups | unique updates | training seconds | completed generation seconds |", "|---|---:|---|---:|---:|---:|---:|"]
    recorded_training = {name: train for name, train in report["training"].items() if train["status"] != "no_training_ledger"}
    for name, train in recorded_training.items():
        group = train["logical_groups"]
        lines.append(f"| {name} | {group['count']} | {group['mixed']} / {group['all_correct']} / {group['all_wrong']} | {group['effective_groups']} | {train['optimizer']['unique_update_numbers']} | {fmt(train['cost']['training_wall_seconds'])} | {fmt(train['cost']['completed_generation_batch_seconds']['sum'])} |")
    for name, train in recorded_training.items():
        group = train["logical_groups"]
        lines += ["", f"{name} reward histogram: `{json.dumps(group['reward_histogram'])}`; outcome entropy (bits) mean `{fmt(group['outcome_entropy_bits']['mean'])}`; effective fraction `{fmt(group['effective_group_fraction'])}`.",
                  "Group diversity means: `" + json.dumps({k: v["mean"] for k, v in train["group_diversity"]["metrics"].items()}) + "`.",
                  "SFT process NLL (not final checkpoint PPL): `" + json.dumps(train["sft_process_nll"]) + "`.", "",
                  "| Update metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for metric, stats in {**train["optimizer"]["metrics"], **{"ratio." + k: v for k, v in train["optimizer"]["ratio_summary"].items()}, "group.reward_std": group["reward_std"]}.items():
            lines.append(f"| {metric} | {stats['count']} / {stats['missing']} | " + " | ".join(fmt(stats[k]) for k in ("mean", "min", "p50", "p90", "p95", "p99", "max")) + " |")
    lines += ["", "## Measurement boundaries", "", *[f"- {note}" for note in report["measurement_notes"]], "", "## Warnings", "", *[f"- {warning}" for warning in report["warnings"]]]
    return "\n".join(lines) + "\n"


def enrich(run_dir: Path, cases_path: Path | None = None, audit_path: Path | None = None) -> dict[str, Any]:
    warnings: list[str] = []
    manifest = read_json(run_dir / "manifest.json") or read_json(run_dir / "config.json")
    if cases_path is None and manifest.get("identity", {}).get("eval_cases"):
        cases_path = Path(manifest["identity"]["eval_cases"])
    if cases_path is not None and not cases_path.is_absolute():
        cases_path = run_dir / cases_path
    cases = index(read_rows(cases_path, warnings), "existing evaluation labels")
    audits = index(read_rows(audit_path, warnings), "existing source audit")
    if not cases:
        warnings.append("Existing case labels unavailable; aliases use saved row gold/source audit where available.")
    if not audits:
        warnings.append("Existing source audit unavailable; source coverage remains NA (no new review).")
    report = dict(schema_version=1, new_semantic_review=False, new_model_calls=0,
        inputs=dict(run_dir=str(run_dir), cases=provenance(cases_path), source_audit=provenance(audit_path)),
        paired_bootstrap=dict(source="summary.json", source_report="REPORT.md", recomputed=False,
            status="reference_to_existing_summarizer_output"),
        evaluation={}, training={}, warnings=warnings,
        measurement_notes=[
            "Best-alias precision/recall use one existing reference chosen by highest normalized token F1; existing normalization removes English articles, punctuation and extra spaces.",
            "Required source coverage is measured only on previously confirmed source annotations; unresolved/missing cases are NA, including the existing 35 unresolved cases.",
            "Document ID retrieval/visibility and exact quoted-span coverage do not certify answer entailment or relation correctness. No new semantic success rate is created.",
            "Last-visible evidence uses explicit document_changes segments or certified legacy untruncated saved-input metadata; the current call's returned tool observation is excluded.",
            "Per-row model_seconds in GPU batches are allocated batch time / active rows. batch_seconds repeats per row and its sum is not GPU wall time; generation-attempt ledger completed times are physical batch wall times.",
            "Per-case elapsed time overlaps across wavefront rows; use saved evaluation_wall_seconds for throughput. Resumed evaluation summaries describe the latest process and cannot certify all-process wall time.",
            "Training ledgers retain physical attempts, including recovery replay. Last group record per index summarizes logical groups; summary.json defines final checkpoint state. Update interval includes rollout/scoring overhead, not pure optimizer latency.",
            "Outcome entropy describes the binary group reward distribution; token policy entropy is NA because full probability distributions were not logged. Ratio percentiles summarize recorded per-update summaries, not a reconstructed global token distribution.",
            "Missing measurements are null/NA; monetary cost, energy, GPU utilization, and new semantic correctness are unavailable. No estimates from hardware pricing or extra detectors are used."])
    for name in BRANCHES:
        folder = run_dir / "branches" / name
        raw = read_rows(folder / "evaluation.jsonl", warnings)
        index(raw, name)
        evaluated = [case_metrics(row, cases.get(identity(row), {}), audits.get(identity(row), {})) for row in raw]
        summary = read_json(folder / "evaluation-summary.json")
        loading = read_json(folder / "evaluation-loading.json")
        expected = len(cases) if cases else 128
        selections = dict(all128=evaluated, natural104=[r for r in evaluated if r["stratum"] not in SINGLE_STRATA],
                          single24=[r for r in evaluated if r["stratum"] in SINGLE_STRATA])
        expecteds = dict(all128=expected, natural104=sum(c.get("stratum") not in SINGLE_STRATA for c in cases.values()) if cases else 104,
                         single24=sum(c.get("stratum") in SINGLE_STRATA for c in cases.values()) if cases else 24)
        expected_ids = dict(all128=set(cases), natural104={key for key, c in cases.items() if c.get("stratum") not in SINGLE_STRATA},
                            single24={key for key, c in cases.items() if c.get("stratum") in SINGLE_STRATA})
        aggregated_groups = {group: aggregate(rows, expecteds[group]) for group, rows in selections.items()}
        for group, item in aggregated_groups.items():
            item["case_set_matches_expected"] = {r["case_id"] for r in selections[group]} == expected_ids[group] if cases else None
            if cases:
                item["complete"] = item["complete"] and item["case_set_matches_expected"]
        strata = sorted({r["stratum"] for r in evaluated})
        wall = finite(summary.get("evaluation_wall_seconds"))
        resumed = bool(summary.get("resumed"))
        totals = aggregate(evaluated, expected)["per_case"]
        report["evaluation"][name] = dict(status=summary.get("status", "partial" if raw else "missing"),
            groups=aggregated_groups,
            source_strata={stratum: aggregate([r for r in evaluated if r["stratum"] == stratum]) for stratum in strata},
            cost=dict(evaluation_wall_seconds=wall, wall_scope="latest_process" if resumed else "complete_branch" if wall is not None else "unavailable",
                resumed=resumed, current_process_cases=summary.get("current_process_cases"),
                questions_per_second=finite(summary.get("questions_per_second")),
                output_tokens_per_second=rate(totals["output_tokens"]["sum"], wall) if not resumed and wall and totals["output_tokens"]["missing"] == 0 else None,
                model_calls_per_second=rate(totals["model_calls"]["sum"], wall) if not resumed and wall and totals["model_calls"]["missing"] == 0 else None,
                model_loading_seconds=finite(loading.get("model_loading_seconds")), gpu_memory=summary.get("gpu_memory"),
                gpu_hours=wall / 3600 if wall is not None else None, monetary_cost=None),
            per_case=evaluated)
        report["training"][name] = training_metrics(folder, warnings)
        training = report["training"][name]
        if training["status"] != "no_training_ledger":
            planned_updates = manifest.get("sft_updates", 128) if name == "sft42" else manifest.get("grpo_max_updates_per_seed", 256)
            final_updates = finite(training["optimizer"]["final_checkpoint_updates"])
            training["optimizer"].update(planned_update_budget=planned_updates,
                final_update_fraction_of_planned_budget=rate(final_updates, planned_updates) if final_updates is not None else None)
        if cases and set(cases) != {identity(row) for row in raw}:
            warnings.append(f"{name}: evaluation case set incomplete/different; {len(raw)} saved / {len(cases)} expected.")
    report["complete"] = all(report["evaluation"][name]["groups"]["all128"]["complete"] for name in BRANCHES)
    write(run_dir / "comprehensive-metrics.json", report)
    (run_dir / "comprehensive-metrics.md").write_text(markdown(report), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--cases", type=Path, help="Existing evaluation JSONL (default: manifest/config identity.eval_cases).")
    parser.add_argument("--source-audit", type=Path, help="Previously saved source-audit JSONL; absent annotations remain NA.")
    args = parser.parse_args(argv)
    report = enrich(args.run_dir, args.cases, args.source_audit)
    print(json.dumps(dict(complete=report["complete"], output=str(args.run_dir / "comprehensive-metrics.json"),
                          warnings=len(report["warnings"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
