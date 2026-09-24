"""Audit historical overlap, freeze a training-only internal set, and build blinded SFT candidates."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/hotpot-full-train-v114"
OUT = ROOT / "data/evidence-sft-v114-clean"
TRAIN_FILES = sorted(SOURCE.glob("train-*-of-*.parquet"))
SEED_INTERNAL = "v114-internal-v1:"
SEED_TRAIN = "v114-training-v1:"
RISK_PATTERNS = {
    "relation": re.compile(r"\b(whose|who|which|that|same|directed|written|founded|member|part|author|created|played|acted|associated|compared|between|both)\b", re.I),
    "time": re.compile(r"\b(year|when|born|older|earlier|first|century|date|founded|started|established|before|after|during|later|earliest|latest)\b", re.I),
    "measurement": re.compile(r"\b(long|length|height|tall|mile|km|kilometer|metre|meter|population|area|number|how many|how much|larger|greater|duration|weight|percent|percentage)\b", re.I),
    "scope": re.compile(r"\b(both|same|common|which|what|known|type|kind|only|among|between|all|most|least|first|largest|smallest|main|primary)\b", re.I),
}


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def sha(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def record_questions(value, found: set[str]):
    if isinstance(value, dict):
        question = value.get("question")
        if isinstance(question, str) and norm(question):
            found.add(norm(question))
        for key, child in value.items():
            if key in {"task_id", "id"} and isinstance(child, str):
                match = re.fullmatch(r"hotpot[-:]?([0-9a-f]{24})", child, re.I)
                if match:
                    found.add("__id__" + match.group(1).casefold())
            record_questions(child, found)
    elif isinstance(value, list):
        for child in value:
            record_questions(child, found)


def collect_history():
    questions: set[str] = set()
    files = []
    roots = [ROOT / "data", ROOT / "reports", ROOT / "backups", ROOT / "runs"]
    excluded_fragments = ("final-test", "final_test", "holdout", "independent100", "evidence-sft-v114")
    for base in roots:
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            relative_parts = [part.casefold() for part in path.relative_to(ROOT).parts]
            if any(fragment in part for part in relative_parts for fragment in excluded_fragments):
                continue
            try:
                if path.stat().st_size > 32 * 1024 * 1024:
                    continue
                found_before = len(questions)
                if path.suffix.lower() == ".jsonl":
                    with path.open("r", encoding="utf-8") as stream:
                        for line in stream:
                            try:
                                record_questions(json.loads(line), questions)
                            except json.JSONDecodeError:
                                continue
                else:
                    record_questions(read_json(path), questions)
                if len(questions) != found_before:
                    files.append(str(path.relative_to(ROOT)).replace("\\", "/"))
            except (OSError, PermissionError):
                continue
    return questions, sorted(set(files))


def row_meta_batches():
    for path in TRAIN_FILES:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["id", "question", "type"], batch_size=4096):
            for row in batch.to_pylist():
                yield {"id": str(row["id"]), "question": row["question"], "type": row.get("type")}


def risk_tags(question: str):
    result = [name for name, pattern in RISK_PATTERNS.items() if pattern.search(question)]
    if "relation" not in result:
        result.append("relation")
    return result


def ordered(rows, prefix):
    return sorted(rows, key=lambda row: hashlib.sha256((prefix + row.get("id", row.get("qid"))).encode()).hexdigest())


def choose_balanced(rows, count, prefix, minimum_by_risk):
    chosen = []
    chosen_ids = set()
    risk_counts = Counter()
    for risk, minimum in minimum_by_risk.items():
        while risk_counts[risk] < minimum and len(chosen) < count:
            candidate = next((row for row in ordered(rows, prefix)
                              if row.get("id", row.get("qid")) not in chosen_ids and risk in row["risk_tags"]), None)
            if candidate is None:
                break
            chosen.append(candidate)
            chosen_ids.add(candidate.get("id", candidate.get("qid")))
            risk_counts.update(candidate["risk_tags"])
    for row in ordered(rows, prefix):
        if len(chosen) >= count:
            break
        row_id = row.get("id", row.get("qid"))
        if row_id not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(row_id)
            risk_counts.update(row["risk_tags"])
    if len(chosen) != count:
        raise RuntimeError(f"only {len(chosen)} of {count} distinct eligible questions")
    return chosen


def load_full(selected_ids):
    records = {}
    for path in TRAIN_FILES:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if row["id"] in selected_ids:
                    records[row["id"]] = row
    if set(records) != set(selected_ids):
        raise RuntimeError(f"missing selected source rows: {len(set(selected_ids) - set(records))}")
    return records


def documents_for(row):
    context = row["context"]
    result = []
    for title, sentences in zip(context["title"], context["sentences"]):
        text = " ".join(sentences)
        doc_id = hashlib.sha256((title + "\0" + text).encode()).hexdigest()[:20]
        result.append({"id": doc_id, "title": title, "text": text})
    return result


def support_facts(row):
    context = row["context"]
    title_to_sentences = dict(zip(context["title"], context["sentences"]))
    facts = []
    for title, sent_id in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        sentences = title_to_sentences.get(title, [])
        if type(sent_id) is int and 0 <= sent_id < len(sentences):
            quote = sentences[sent_id].strip()
            if quote:
                facts.append((title, quote))
    return facts


def facts_visible(required_facts, docs):
    return all(any(doc["title"] == title and quote in doc["text"] for doc in docs)
               for title, quote in required_facts)


def grounded_queries(question, docs, used):
    """Create queries only from the question and text already visible to the agent."""
    seen = set(used) | {norm(question)}
    variants = []
    for doc in docs:
        seeds = [doc.get("title", "")]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.get("text", "")) if len(s.strip()) >= 24]
        seeds.extend(sentences[:4])
        for seed in seeds:
            for query in (seed, (question + " " + seed).strip()):
                key = norm(query)
                if key and key not in seen:
                    seen.add(key)
                    variants.append(query[:500])
    return variants[:56]


def question_queries(question):
    """Narrow, non-oracle queries composed only of spans in the question."""
    queries = [question]
    for match in re.finditer(r"[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*){0,4}", question):
        phrase = match.group(0).strip()
        if phrase.casefold() not in {"which", "what", "who", "when", "where", "how", "the", "did", "is", "was"} and len(phrase) >= 5:
            queries.append(phrase)
            queries.append(question + " " + phrase)
    for clause in re.split(r"\b(?:or|and|but|while|whereas)\b", question, flags=re.I):
        clause = clause.strip(" ?.,")
        if len(clause) >= 12:
            queries.append(clause)
    tokens = re.findall(r"[A-Za-z0-9'’.-]+", question)
    content = [token for token in tokens if len(token) >= 3 and token.casefold() not in {
        "which", "what", "who", "when", "where", "how", "the", "did", "does", "was", "were",
        "are", "is", "with", "from", "that", "this", "same", "have", "has", "their", "they",
    }]
    for width in (2, 3, 4):
        for start in range(max(0, len(content) - width + 1)):
            queries.append(" ".join(content[start:start + width]))
    for token in content:
        queries.append(token)
    unique = []
    seen = set()
    for query in queries:
        key = norm(query)
        if key and key not in seen:
            seen.add(key)
            unique.append(query)
    return unique


def build_index(rows):
    from mini_agent_rl.tools.search import _BM25Index
    docs = {}
    for row in rows:
        for doc in documents_for(row):
            docs.setdefault(doc["id"], doc)
    return list(docs.values()), _BM25Index(list(docs.values()))


def build_examples(rows, index):
    from mini_agent_rl.prompts import DIRECT_SYSTEM

    examples = []
    for row in rows:
        qid = str(row["id"])
        question = row["question"]
        required_facts = support_facts(row)
        needed_titles = {title for title, _ in required_facts}
        if not required_facts:
            continue
        history = [
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": question},
        ]
        runs = []
        for query in question_queries(question):
            results = index.search(query, top_k=3)
            if not results:
                continue
            matched = [(title, quote) for title, quote in required_facts
                       if any(doc["title"] == title and quote in doc["text"] for doc in results)]
            if matched:
                runs.append((len(matched), query, results))
        if not runs:
            continue
        # Prefer a question-grounded first query that returns useful but
        # incomplete evidence. This produces a real missing-evidence decision.
        partial_runs = [item for item in runs if item[0] < len(required_facts)]
        partial_runs.sort(key=lambda item: (-item[0], len(item[1]), norm(item[1])))
        full_runs = [item for item in runs if item[0] == len(required_facts)]
        full_runs.sort(key=lambda item: (len(item[1]), norm(item[1])))
        # One empty-evidence action from a question-derived query. Its actual
        # top3 must contain at least one exact annotated support sentence.
        empty_choice = (partial_runs or full_runs)[0]
        _, empty_query, empty_results = empty_choice
        examples.append({
            "qid": qid, "question": question, "messages": list(history),
            "action": {"action": "search", "query": empty_query}, "evidence": [],
            "retrieval_top3": empty_results, "risk_tags": risk_tags(question),
            "state_kind": "empty_evidence_search",
            "why_search": "no tool evidence is currently visible",
            "source_support_titles_private": sorted(needed_titles),
        })

        answer_paths = []
        for _, query1, first_results in partial_runs[:4] + full_runs[:1]:
            answer_history = list(history)
            answer_history.extend([
                {"role": "assistant", "content": json.dumps({"action": "search", "query": query1}, ensure_ascii=False)},
                {"role": "tool", "content": json.dumps({"ok": True, "result": first_results}, ensure_ascii=False)},
            ])
            retrieved = list(first_results)
            seen = {doc["id"] for doc in first_results}
            queries_used = {norm(query1)}
            for _ in range(4):
                if facts_visible(required_facts, retrieved):
                    break
                missing_facts = [(title, quote) for title, quote in required_facts
                                 if not any(d["title"] == title and quote in d["text"] for d in retrieved)]
                query_options = []
                for query_next in grounded_queries(question, retrieved, queries_used):
                    next_results = index.search(query_next, top_k=3)
                    new_results = [doc for doc in next_results if doc["id"] not in seen]
                    closed = [(title, quote) for title, quote in missing_facts
                              if any(doc["title"] == title and quote in doc["text"] for doc in new_results)]
                    if closed:
                        query_options.append((len(closed), -len(query_next), query_next, next_results, new_results))
                if not query_options:
                    break
                _, _, query_next, next_results, new_results = max(query_options, key=lambda x: (x[0], x[1], x[2]))
                queries_used.add(norm(query_next))
                target_search = {"action": "search", "query": query_next}
                examples.append({
                    "qid": qid, "question": question, "messages": list(answer_history),
                    "action": target_search, "evidence": list(retrieved),
                    "retrieval_top3": next_results, "risk_tags": risk_tags(question),
                    "state_kind": "nonempty_evidence_search",
                    "why_search": "at least one annotated support sentence remains absent from visible evidence",
                    "source_support_titles_private": sorted(needed_titles),
                })
                answer_history.extend([
                    {"role": "assistant", "content": json.dumps(target_search, ensure_ascii=False)},
                    {"role": "tool", "content": json.dumps({"ok": True, "result": next_results}, ensure_ascii=False)},
                ])
                retrieved.extend(new_results)
                seen.update(doc["id"] for doc in new_results)
            if facts_visible(required_facts, retrieved):
                answer_paths.append((len(answer_history), answer_history, retrieved))
        if answer_paths:
            _, answer_history, retrieved = min(answer_paths, key=lambda item: item[0])
            examples.append({
                "qid": qid, "question": question, "messages": answer_history,
                "action": {"action": "answer", "answer": str(row["answer"])}, "evidence": retrieved,
                "retrieval_top3": [], "risk_tags": risk_tags(question),
                "state_kind": "supported_answer",
                "why_answer": "all annotated support sentences are present in visible tool observations",
                "source_support_titles_private": sorted(needed_titles),
            })
    return examples


def state_hash(example):
    messages = example["messages"] + [{"role": "assistant", "content": json.dumps(example["action"], ensure_ascii=False, sort_keys=True)}]
    return sha(messages)


def public_example(example, ordinal):
    case_id = hashlib.sha256(("v114-case:" + state_hash(example)).encode()).hexdigest()[:20]
    return {
        "case_id": case_id,
        "question": example["question"],
        "visible_messages": example["messages"],
        "proposed_action": example["action"],
        "visible_evidence_ids": [doc["id"] for doc in example["evidence"]],
        "retrieval_top3": [{k: doc[k] for k in ("id", "title", "text")} for doc in example["retrieval_top3"]],
        "evidence_refs": [doc["id"] for doc in example["evidence"] + example["retrieval_top3"]],
        "review": {
            "evidence_sufficient_for_action": None,
            "action_supported_or_justified": None,
            "search_query_grounded": None,
            "search_returns_relevant_new_evidence": None,
            "issue_tags": [],
            "verbatim_citations": [],
            "reviewer_id": None,
        },
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    if not TRAIN_FILES or len(TRAIN_FILES) != 2:
        raise RuntimeError("expected the two provenance-verified train parquet shards")
    OUT.mkdir(parents=True, exist_ok=True)
    history, history_files = collect_history()
    # Exact normalized question and stable HotpotQA ID match against all local
    # project records outside the sealed independent100/final-test directories.
    history_q = {value for value in history if not value.startswith("__id__")}
    history_ids = {value.removeprefix("__id__") for value in history if value.startswith("__id__")}
    unique = {}
    exclusion_counts = Counter()
    for meta in row_meta_batches():
        key = norm(meta["question"])
        if meta["id"].casefold() in history_ids or key in history_q:
            exclusion_counts["historical_question_or_id_overlap"] += 1
            continue
        if key in unique:
            exclusion_counts["duplicate_normalized_question"] += 1
            continue
        unique[key] = {**meta, "risk_tags": risk_tags(meta["question"])}
    eligible = list(unique.values())
    if len(eligible) < 250:
        raise RuntimeError(f"historical exclusions leave only {len(eligible)} eligible train questions")

    freeze_path = OUT / "internal-freeze.json"
    if freeze_path.exists():
        frozen = read_json(freeze_path)
        frozen_ids = set(frozen.get("ids", []))
        internal = [row for row in eligible if row["id"] in frozen_ids]
        if len(internal) != 50 or "SHA256(v114-internal-v1:" not in frozen.get("selection_rule", ""):
            raise RuntimeError("existing v114 internal freeze does not match the eligible source")
    else:
        internal = choose_balanced(eligible, 50, SEED_INTERNAL,
                                   {"relation": 5, "time": 5, "measurement": 5, "scope": 5})
    internal_ids = {row["id"] for row in internal}
    internal_norms = {norm(row["question"]) for row in internal}
    if not freeze_path.exists():
        write_json(freeze_path, {
            "version": "v114", "source_split": "official HotpotQA distractor train",
            "selection_rule": "NFKC/casefold normalized-question dedupe; SHA256(v114-internal-v1:<source id>) with risk coverage minima",
            "question_count": 50, "ids": sorted(internal_ids),
            "question_hashes": sorted(sha(norm(row["question"])) for row in internal),
            "risk_question_counts": dict(Counter(tag for row in internal for tag in row["risk_tags"])),
            "frozen_before_candidate_generation": True,
            "independent100_content_read": False, "clean_build_final_test_accessed": False,
            "prior_attempt_quarantined_for_final_test_filter_bug": True,
        })

    train_pool = [row for row in eligible if row["id"] not in internal_ids and norm(row["question"]) not in internal_norms]
    train_freeze_path = OUT / "training-question-freeze.json"
    if train_freeze_path.exists():
        frozen_train = read_json(train_freeze_path)
        train_ids = set(frozen_train.get("ids", []))
        training_meta = [row for row in train_pool if row["id"] in train_ids]
        if len(training_meta) != 220:
            raise RuntimeError("existing v114 training-question freeze does not match eligible source")
    else:
        training_meta = choose_balanced(train_pool, 220, SEED_TRAIN,
                                        {"relation": 10, "time": 10, "measurement": 10, "scope": 10})
        write_json(train_freeze_path, {
            "version": "v114", "source_split": "official HotpotQA distractor train",
            "selection_rule": "NFKC/casefold normalized-question dedupe; SHA256(v114-training-v1:<source id>) with risk coverage minima",
            "question_count": len(training_meta), "ids": sorted(row["id"] for row in training_meta),
            "question_hashes": sorted(sha(norm(row["question"])) for row in training_meta),
            "disjoint_from_internal_ids": True,
            "independent100_content_read": False, "clean_build_final_test_accessed": False,
            "prior_attempt_quarantined_for_final_test_filter_bug": True,
        })
    selected_ids = internal_ids | {row["id"] for row in training_meta}
    full_rows = load_full(selected_ids)
    train_rows = [full_rows[row["id"]] for row in training_meta]
    internal_rows = [full_rows[row["id"]] for row in internal]
    train_corpus, train_index = build_index(train_rows)
    internal_corpus, internal_index = build_index(internal_rows)
    write_jsonl(OUT / "training-corpus.jsonl", train_corpus)
    write_jsonl(OUT / "internal-corpus.jsonl", internal_corpus)

    train_cache = OUT / "candidate-cache-training-private.jsonl"
    internal_cache = OUT / "candidate-cache-internal-private.jsonl"
    if train_cache.exists():
        train_examples_all = [json.loads(line) for line in train_cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        train_examples_all = build_examples(train_rows, train_index)
        write_jsonl(train_cache, train_examples_all)
    if internal_cache.exists():
        internal_examples_all = [json.loads(line) for line in internal_cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        internal_examples_all = build_examples(internal_rows, internal_index)
        write_jsonl(internal_cache, internal_examples_all)
    # Freeze the internal questions first, then report the yield without replacing
    # questions based on observed semantic or model performance.
    def choose_states(examples, split, empty_limit, expected_questions):
        answer = [e for e in examples if e["state_kind"] == "supported_answer"]
        nonempty = [e for e in examples if e["state_kind"] == "nonempty_evidence_search"]
        empty = ordered([e for e in examples if e["state_kind"] == "empty_evidence_search"], f"{split}:empty:")
        target_min = 100 if split == "internal" else 300
        target_max = 100 if split == "internal" else 400
        answer = ordered(answer, f"{split}:answer:")
        nonempty = ordered(nonempty, f"{split}:nonempty:")
        if split == "internal":
            answer = [e for e in answer if e["qid"] in internal_ids]
            nonempty = [e for e in nonempty if e["qid"] in internal_ids]
            selected = answer[:50] + nonempty[:40] + empty[:10]
        else:
            # Choose 160 answer and 128 evidence-bearing search states, with 32
            # empty-evidence searches (10% of the 320-state corpus).
            answer = choose_balanced(answer, min(160, len(answer)), f"{split}:answer-balanced:",
                                     {"relation": 20, "time": 20, "measurement": 20, "scope": 20})
            nonempty = choose_balanced(nonempty, min(128, len(nonempty)), f"{split}:search-balanced:",
                                       {"relation": 20, "time": 20, "measurement": 20, "scope": 20})
            selected = answer + nonempty[:128] + empty[:32]
        seen = set()
        unique_states = []
        for item in selected:
            digest = state_hash(item)
            if digest not in seen:
                seen.add(digest)
                unique_states.append(item)
        qids = {e["qid"] for e in unique_states}
        risks = Counter(tag for e in unique_states for tag in e["risk_tags"])
        counts = Counter(e["state_kind"] for e in unique_states)
        gate = {
            "states": len(unique_states), "questions": len(qids), "state_kinds": dict(counts),
            "risk_state_counts": dict(risks), "empty_search_fraction": counts["empty_evidence_search"] / max(len(unique_states), 1),
            "answer_fraction": counts["supported_answer"] / max(len(unique_states), 1),
            "nonempty_search_fraction": counts["nonempty_evidence_search"] / max(len(unique_states), 1),
            "question_count_expected": expected_questions,
            "structural_gate_passed": (
                len(unique_states) >= target_min and len(unique_states) <= target_max
                and (split == "internal" and len(qids) == 50 or split == "train" and len(qids) >= 150)
                and counts["supported_answer"] / max(len(unique_states), 1) >= 0.30
                and counts["nonempty_evidence_search"] / max(len(unique_states), 1) >= 0.30
                and counts["empty_evidence_search"] / max(len(unique_states), 1) <= empty_limit
                and all(risks[k] >= (5 if split == "internal" else 20) for k in RISK_PATTERNS)
            ),
        }
        return unique_states, gate

    train_selected, train_gate = choose_states(train_examples_all, "train", 0.20, 160)
    internal_selected, internal_gate = choose_states(internal_examples_all, "internal", 0.20, 50)
    public_train = [public_example(e, i) for i, e in enumerate(train_selected)]
    public_internal = [public_example(e, i) for i, e in enumerate(internal_selected)]
    all_public = public_train + public_internal
    if len({row["case_id"] for row in all_public}) != len(all_public):
        raise RuntimeError("duplicate candidate case IDs")
    if {e["qid"] for e in train_selected} & {e["qid"] for e in internal_selected}:
        raise RuntimeError("question-level train/internal leakage")

    private = []
    for split, candidates in (("train", train_selected), ("internal", internal_selected)):
        for example in candidates:
            private.append({
                "case_id": hashlib.sha256(("v114-case:" + state_hash(example)).encode()).hexdigest()[:20],
                "source_id": example["qid"], "split": split,
                "gold_answer": str(full_rows[example["qid"]]["answer"]),
                "source_support_titles": example["source_support_titles_private"],
                "source": "hotpotqa/hotpot_qa distractor train",
            })
    write_jsonl(OUT / "blind-review-candidates.jsonl", all_public)
    write_jsonl(OUT / "private-key.jsonl", private)
    write_jsonl(OUT / "training-candidates.jsonl", public_train)
    write_jsonl(OUT / "internal-candidates.jsonl", public_internal)
    write_json(OUT / "source-audit.json", {
        "dataset": "hotpotqa/hotpot_qa", "config": "distractor", "split": "train",
        "source_rows": 90447, "historical_excluded_records": exclusion_counts["historical_question_or_id_overlap"],
        "duplicate_question_excluded_records": exclusion_counts["duplicate_normalized_question"],
        "eligible_unique_questions": len(eligible), "historical_exclusion_question_or_id_keys": len(history),
        "historical_artifacts_consulted": history_files,
        "historical_rule": "exclude exact HotpotQA id or normalized question found in readable local data/reports/backups/runs records",
        "independent100_content_read": False, "clean_build_final_test_accessed": False,
        "prior_attempt_quarantined_for_final_test_filter_bug": True,
        "caveat": "official train split and historical local records were checked; no claim is made about unindexed external exposure",
    })
    write_json(OUT / "candidate-gates.json", {
        "version": "v114", "training": train_gate, "internal": internal_gate,
        "review": {"semantic_review_status": "pending_independent_dual_review", "approved_states": 0,
                   "disputed_states_quarantined": "pending reviewer decisions", "no_teacher_output_imported": True},
        "data_gate_passed": train_gate["structural_gate_passed"] and internal_gate["structural_gate_passed"],
        "training_permitted": False,
        "reason": "all candidate actions require independent evidence-bound review before labels can be accepted",
    })
    manifest_files = [OUT / name for name in (
        "internal-freeze.json", "training-question-freeze.json", "source-audit.json",
        "training-corpus.jsonl", "internal-corpus.jsonl", "blind-review-candidates.jsonl",
        "training-candidates.jsonl", "internal-candidates.jsonl", "private-key.jsonl",
        "candidate-cache-training-private.jsonl", "candidate-cache-internal-private.jsonl", "candidate-gates.json",
    )]
    write_json(OUT / "manifest.json", {
        "version": "v114", "source_rows": 90447, "internal_questions_frozen": len(internal_ids),
        "train_questions_selected": len(training_meta), "training_candidates": len(public_train),
        "internal_candidates": len(public_internal), "train_corpus_documents": len(train_corpus),
        "internal_corpus_documents": len(internal_corpus),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in manifest_files},
        "candidate_gates": {"train": train_gate, "internal": internal_gate},
        "semantic_review_status": "pending_independent_dual_review",
        "training_permitted": False, "independent100_content_read": False, "clean_build_final_test_accessed": False,
        "prior_attempt_quarantined_for_final_test_filter_bug": True,
    })
    print(json.dumps({
        "historically_excluded": exclusion_counts["historical_question_or_id_overlap"],
        "eligible_questions": len(eligible), "train": train_gate, "internal": internal_gate,
        "candidate_gates_passed": train_gate["structural_gate_passed"] and internal_gate["structural_gate_passed"],
        "semantic_review": "pending_independent_dual_review", "output": str(OUT),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
