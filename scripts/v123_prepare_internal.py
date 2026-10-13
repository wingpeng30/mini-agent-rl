"""One bounded, performance-blind v123 internal candidate and corpus freeze."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from scripts.v123_cpu_core import ROOT, OUT, norm, read, readl, sha, source_manifest, write, writel

OLD = ROOT / "data/evidence-sft-v122-internal-candidates"
NEW = OUT / "internal"


def source_ids() -> tuple[set[str], set[str]]:
    inventories = [
        ROOT / "data/evidence-sft-v122-internal-candidates/history-private.json",
        ROOT / "data/evidence-sft-v122-train-candidates/history-private.json",
    ]
    ids, texts = set(), set()
    for path in inventories:
        value = read(path)
        ids.update(value["ids"])
        texts.update(value["normalized_questions"])
    for kind in ("internal", "train"):
        prefix = ROOT / f"data/evidence-sft-v122-{kind}-candidates"
        ids.update(read(prefix / "candidate-freeze.json")["ids"])
        for row in readl(prefix / "candidate-private.jsonl"):
            texts.add(norm(row["packet"]["question"]))
    return ids, texts


def old_approved() -> list[str]:
    details = readl(OLD / "review/merge-details.jsonl")
    grouped = defaultdict(list)
    for row in details:
        if row["accepted"]:
            grouped[row["source_id"]].append(row)
    frozen = read(OLD / "candidate-freeze.json")["ids"]
    approved = [qid for qid in frozen if len(grouped[qid]) == 2]
    if len(approved) != 45:
        raise ValueError("expected_v122_45_approved_pairs")
    return approved


def provisional_scope(question: str) -> bool:
    """Scheduling hint, never a semantic risk vote."""
    import re
    return bool(re.search(r"\b(both|among|between|only|all|same|largest|smallest|which state|which city|"
                          r"which county|which country|in what country|in which country)\b", question, re.I))


def metadata_candidates() -> tuple[list[dict], int]:
    import pyarrow.parquet as pq
    from scripts.prepare_v117 import wording_issue
    from scripts.build_v114_sft_candidates import support_facts
    from scripts.prepare_v116_data import full_rows, paired_examples
    from scripts.build_v114_sft_candidates import build_index

    exposed_ids, exposed_texts = source_ids()
    metadata = []
    seen_texts = set()
    for path in source_manifest()["files"]:
        for batch in pq.ParquetFile(path).iter_batches(columns=["id", "question"], batch_size=4096):
            for row in batch.to_pylist():
                key = norm(row["question"])
                if row["id"] in exposed_ids or key in exposed_texts or key in seen_texts:
                    continue
                seen_texts.add(key)
                metadata.append(row)
    metadata.sort(key=lambda row: hashlib.sha256(("v123-internal:" + row["id"]).encode()).hexdigest())
    metadata = metadata[:5000]
    rows = full_rows([item["id"] for item in metadata])
    scope, other, rejections = [], [], []
    checked = []
    for item in metadata:
        row = rows[item["id"]]
        checked.append({"id": item["id"], "question_sha256": hashlib.sha256(norm(row["question"]).encode()).hexdigest()})
        issue = wording_issue(row["question"])
        if not issue and len(support_facts(row)) < 2:
            issue = "insufficient_annotated_facts"
        if not issue:
            _, solo = build_index([row])
            if paired_examples(row, solo) is None:
                issue = "no_structural_pair_in_source"
        if issue:
            rejections.append({"id": row["id"], "reason": issue})
            continue
        (scope if provisional_scope(row["question"]) else other).append(row)
        if len(scope) >= 60 and len(other) >= 40:
            break
    chosen = scope[:60] + other[:40]
    if len(chosen) < 100:
        already = {row["id"] for row in chosen}
        chosen.extend(row for row in scope[60:] + other[40:] if row["id"] not in already)
        chosen = chosen[:100]
    writel(NEW / "screened-source-private.jsonl", checked)
    write(NEW / "preselection.json", {"screened": len(checked), "scope_hint_eligible": len(scope),
                                      "other_eligible": len(other), "selected": len(chosen),
                                      "reject_reasons": dict(Counter(r["reason"] for r in rejections)),
                                      "selected_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in chosen).encode()).hexdigest(),
                                      "selection_used_model_scores": False, "training_allowed": False})
    writel(NEW / "preselection-rejections-private.jsonl", rejections)
    return chosen, len(checked)


def packet_same(old: dict, new: dict) -> bool:
    keys = ("question", "visible_messages", "before_action", "proposed_action",
            "retrieved_after_search", "prompt_tokens", "target_tokens", "total_tokens",
            "prompt_ids_sha256", "sequence_ids_sha256", "target_mask_sha256")
    return all(old.get(key) == new.get(key) for key in keys)


def prepare() -> dict:
    from mini_agent_rl.tools.search import _BM25Index
    from scripts.build_v114_sft_candidates import documents_for, public_example
    from scripts.prepare_v116_data import full_rows, paired_examples
    from scripts.prepare_v116_review import PROCESSOR, packet
    from scripts.prepare_v117 import provisional_risks
    from transformers import AutoProcessor

    freeze = NEW / "candidate-freeze.json"
    if freeze.exists() or (NEW / "candidate-corpus-private.json").exists() or (NEW / "review/internal-packets.jsonl").exists():
        raise ValueError("one_time_internal_freeze_already_exists")
    if sha(OLD / "candidate-corpus-private.json") != read(OLD / "candidate-freeze.json")["corpus_sha256"]:
        raise ValueError("old_corpus_changed")
    old_ids = old_approved()
    new_rows, screened = metadata_candidates()
    if len(new_rows) != 100:
        report = {"version": "v123-internal-preparation-v1", "status": "blocked",
                  "reason": "fewer_than_100_structural_candidates_within_5000",
                  "new_candidates": len(new_rows), "screened": screened, "training_allowed": False}
        write(NEW / "preparation-status.json", report)
        return report
    old_corpus = read(OLD / "candidate-corpus-private.json")
    documents = {doc["id"]: doc for doc in old_corpus}
    for row in new_rows:
        for doc in documents_for(row):
            if doc["id"] in documents and documents[doc["id"]] != doc:
                raise ValueError("document_id_content_collision")
            documents.setdefault(doc["id"], doc)
    corpus = list(documents.values())
    index = _BM25Index(corpus)
    old_rows = full_rows(old_ids)
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    existing = defaultdict(list)
    # Keep the original case-to-source mapping separate from the anonymous packets.
    mapping = {case: row["source_id"] for row in readl(OLD / "review/eligible-private.jsonl")
               for case in row["case_ids"]}
    for old in readl(OLD / "review/internal-packets.jsonl"):
        if mapping[old["case_id"]] in set(old_ids):
            existing[(mapping[old["case_id"]], old["proposed_action"]["action"])].append(old)
    packets = []
    source_map = []
    reused = []
    structural_rejections = []
    for row in [old_rows[qid] for qid in old_ids] + new_rows:
        pair = paired_examples(row, index)
        if pair is None:
            structural_rejections.append({"id": row["id"], "old_approved": row["id"] in old_ids,
                                          "reason": "no_pair_in_new_shared_corpus"})
            continue
        try:
            candidates = [packet({"source_id": row["id"], "split": "internal",
                                  "source": "programmatically_constructed_official_train",
                                  "risk_tags_provisional": provisional_risks(row["question"]),
                                  "packet": public_example(example, 0)}, processor) for example in pair]
        except ValueError as error:
            structural_rejections.append({"id": row["id"], "old_approved": row["id"] in old_ids,
                                          "reason": "packet_invalid:" + str(error)[:100]})
            continue
        for candidate in candidates:
            matches = existing.get((row["id"], candidate["proposed_action"]["action"]), [])
            origin = next((old for old in matches if packet_same(old, candidate)), None)
            packets.append(candidate)
            source_map.append({"case_id": candidate["case_id"], "source_id": row["id"],
                               "old_approved": row["id"] in old_ids})
            if origin is not None:
                reused.append({"new_case_id": candidate["case_id"], "old_case_id": origin["case_id"],
                               "old_packet_sha256": origin["packet_sha256"],
                               "new_packet_sha256": candidate["packet_sha256"],
                               "same_reviewed_input": True})
    if len({p["case_id"] for p in packets}) != len(packets):
        raise ValueError("duplicate_packet_case_id")
    # Write the one-time freeze only after all CPU checks have completed.
    write(NEW / "candidate-corpus-private.json", corpus)
    writel(NEW / "review/internal-packets.jsonl", packets)
    writel(NEW / "review/source-map-private.jsonl", source_map)
    writel(NEW / "review/old-vote-reuse-private.jsonl", reused)
    writel(NEW / "structural-rejections-private.jsonl", structural_rejections)
    ordered_ids = old_ids + [row["id"] for row in new_rows]
    freeze_report = {"version": "v123-internal-candidates-v1", "old_candidate_ids": old_ids,
                     "new_candidate_ids": [row["id"] for row in new_rows],
                     "ordered_ids_sha256": hashlib.sha256("\n".join(ordered_ids).encode()).hexdigest(),
                     "old_corpus_sha256": sha(OLD / "candidate-corpus-private.json"),
                     "corpus_sha256": sha(NEW / "candidate-corpus-private.json"),
                     "packet_sha256": sha(NEW / "review/internal-packets.jsonl"),
                     "source_map_sha256": sha(NEW / "review/source-map-private.jsonl"),
                     "old_vote_reuse_sha256": sha(NEW / "review/old-vote-reuse-private.jsonl"),
                     "new_questions": len(new_rows), "screened": screened,
                     "all_questions": len(ordered_ids), "pairable_questions": len(packets) // 2,
                     "old_states_with_exact_reuse": len(reused),
                     "corpus_documents": len(corpus), "retriever": "bm25-v2 top3",
                     "max_input_tokens": 2048, "training_allowed": False}
    write(freeze, freeze_report)
    return freeze_report
