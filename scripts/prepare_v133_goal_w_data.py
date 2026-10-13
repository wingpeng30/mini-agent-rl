"""Prepare the bounded, unaudited Goal W HotpotQA candidate and corpus pools.

CPU only. This script does not load a model, call an API, create labels, or read
the sealed Goal V JSONL. Goal V overlap keys are reconstructed from the public
source parquet and checked against the sealed manifest's published ID digest.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-w-transfer/data"
SOURCE = ROOT / "data/hotpot-full-train-v114"
SOURCE_SHARDS = ("train-00000-of-00002.parquet", "train-00001-of-00002.parquet")
V041 = ROOT / "data/hotpot-agent-v041"
V072_MANIFEST = ROOT / "data/hotpot-agent-v072/manifest.json"
GOAL_G = ROOT / "scripts/run_v133_goal_g_cpu.py"
PROTOCOL = "goal-w-data-candidates-v1"
SPLITS = ("train", "dev", "confirm")
LIMITS = {"train": 512, "dev": 128, "confirm": 128}
CATEGORY_ORDER = ("occupation", "entity_type", "affiliation", "origin")
CATEGORY_QUOTAS = {"train": 16, "dev": 8, "confirm": 8}
COMMON = re.compile(r"\b(?:both|share|shared|common|same|mutual)\b", re.I)
PAIR = re.compile(r"\band\b|\bbetween\b", re.I)
ENTITY_STOP = {"what", "which", "who", "where", "when", "both", "the", "and", "or", "have", "has", "was", "were", "are", "is", "share", "shared", "common", "same", "mutual"}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def qhash(value: str) -> str:
    return sha_bytes(norm(value).encode("utf-8"))


def stable(seed: str, value: str) -> str:
    return sha_bytes((seed + "\0" + value).encode("utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def verify_source() -> tuple[list[Path], dict]:
    provenance = json.loads((SOURCE / "download-provenance.json").read_text(encoding="utf-8"))
    if (provenance.get("dataset"), provenance.get("config"), provenance.get("split"), provenance.get("rows")) != (
        "hotpotqa/hotpot_qa", "distractor", "train", 90447
    ):
        raise RuntimeError("source_provenance_mismatch")
    expected = {Path(row["path"]).name: row for row in provenance["files"]}
    if set(expected) != set(SOURCE_SHARDS):
        raise RuntimeError("source_shard_list_mismatch")
    files = []
    for name in SOURCE_SHARDS:
        path = SOURCE / name
        if not path.is_file() or sha_file(path) != expected[name]["sha256"]:
            raise RuntimeError(f"source_hash_mismatch:{name}")
        if pq.ParquetFile(path).metadata.num_rows != expected[name]["rows"]:
            raise RuntimeError(f"source_rows_mismatch:{name}")
        files.append(path)
    return files, provenance


def history_exposure() -> tuple[set[str], set[str], set[str], dict[str, str]]:
    # Reuse Goal G's frozen exposure manifest logic, which deliberately omits
    # Goal V's sealed content and records the checksums of every input used.
    import importlib.util

    spec = importlib.util.spec_from_file_location("goal_g_exposure", GOAL_G)
    if spec is None or spec.loader is None:
        raise RuntimeError("goal_g_exposure_module_unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids, questions, titles, inputs = module.exposure()
    return set(ids), set(questions), set(titles), inputs


def read_v041_exclusions() -> tuple[set[str], set[str], set[str], dict[str, str]]:
    manifest = json.loads((V041 / "manifest.json").read_text(encoding="utf-8"))
    sealed_manifest = json.loads(V072_MANIFEST.read_text(encoding="utf-8"))
    ids: set[str] = set()
    questions: set[str] = set()
    titles: set[str] = set()
    hashes = {}
    for split in ("train", "validation", "test"):
        path = V041 / f"{split}.jsonl"
        hashes[str(path.relative_to(ROOT))] = sha_file(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row["metadata"]
            task_id = str(meta["task_id"])
            ids.add(task_id.removeprefix("hotpot-"))
            for message in row["messages"]:
                if message.get("role") == "user":
                    questions.add(qhash(message["content"]))
            titles.update(norm(title) for title in meta.get("support_titles", []) if norm(title))
    expected_hashes = sealed_manifest.get("source_split_sha256", {})
    for split in ("train", "validation", "test"):
        path = V041 / f"{split}.jsonl"
        if sha_file(path) != expected_hashes.get(split):
            raise RuntimeError(f"v041_split_hash_mismatch:{split}")
    return ids, questions, titles, hashes


def official_support_docs(row: dict) -> list[dict]:
    facts = row.get("supporting_facts") or {}
    context = row.get("context") or {}
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    by_title = {}
    for title, body in zip(titles, sentences):
        if isinstance(title, str) and isinstance(body, list):
            by_title.setdefault(title, body)
    result = []
    seen = set()
    for title in facts.get("title") or []:
        title = str(title)
        if title in seen:
            continue
        seen.add(title)
        body = by_title.get(title)
        if not body or not all(isinstance(sentence, str) for sentence in body):
            return []
        text = " ".join(sentence.strip() for sentence in body if sentence.strip())
        if not text:
            return []
        result.append({"title": title, "text": text})
    return result


def category(question: str) -> str:
    q = question.casefold()
    if re.search(r"\b(occupation|profession|job|career|vocation)\b", q):
        return "occupation"
    if re.search(r"\b(type|kind|category|classification|genre|species|class)\b", q):
        return "entity_type"
    if re.search(r"\b(member|membership|belong|party|affiliation|group)\b", q):
        return "affiliation"
    if re.search(r"\b(nationality|country|origin|native|born|from)\b", q):
        return "origin"
    return "unclassified_common_attribute"


def question_entities(question: str, support_titles: list[str]) -> set[str]:
    qn = norm(question)
    entities = {norm(title) for title in support_titles if norm(title)}
    # Exact title mentions are high precision and cover titles whose case or
    # punctuation differs from the source question.
    for title in support_titles:
        tn = norm(title)
        if tn and tn in qn:
            entities.add(tn)
    # Retain capitalized multiword/name-like spans, excluding leading query
    # words and common comparator terms. This is a conservative collision
    # screen, not entity resolution.
    for span in re.findall(r"\b[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*){0,4}", question):
        key = norm(span)
        if key and key not in ENTITY_STOP and len(key.split()) <= 5:
            entities.add(key)
    return entities


def reconstruct_sealed(files: list[Path], v041_ids: set[str], v041_titles: set[str]) -> dict:
    manifest = json.loads(V072_MANIFEST.read_text(encoding="utf-8"))
    target = manifest["final_test"]
    selected_ids: list[str] = []
    selected_titles: set[str] = set()
    scanned = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["id", "supporting_facts", "context"], batch_size=128):
            for row in batch.to_pylist():
                scanned += 1
                facts = row.get("supporting_facts") or {}
                titles = list(dict.fromkeys(str(x) for x in (facts.get("title") or [])))
                source_id = str(row["id"])
                if (source_id in v041_ids
                        or len(titles) < 2 or any(norm(x) in v041_titles for x in titles)
                        or set(titles) & selected_titles):
                    if scanned >= target["scanned"]:
                        break
                    continue
                selected_ids.append(f"hotpot-{source_id}")
                selected_titles.update(titles)
                if len(selected_ids) >= target["count"] or scanned >= target["scanned"]:
                    break
            if len(selected_ids) >= target["count"] or scanned >= target["scanned"]:
                break
        if len(selected_ids) >= target["count"] or scanned >= target["scanned"]:
            break
    digest = sha_bytes("\n".join(selected_ids).encode("utf-8"))
    if scanned != target["scanned"] or len(selected_ids) != target["count"]:
        raise RuntimeError("sealed_reconstruction_cardinality_mismatch")
    if digest != target["task_id_sha256"]:
        raise RuntimeError("sealed_reconstruction_id_digest_mismatch")
    if len(selected_titles) != target["supporting_documents"]:
        raise RuntimeError("sealed_reconstruction_title_count_mismatch")
    return {"rows": len(selected_ids), "scanned": scanned, "task_id_sha256": digest,
            "supporting_documents": len(selected_titles),
            "id_set": set(x.removeprefix("hotpot-") for x in selected_ids),
            "title_set": {norm(x) for x in selected_titles},
            "question_hashes": set()}


def discover_metadata_candidates(files: list[Path], banned_ids: set[str], banned_questions: set[str]) -> tuple[list[dict], int]:
    candidates = []
    row_index = 0
    for path in files:
        shard_start = row_index
        for batch in pq.ParquetFile(path).iter_batches(columns=["id", "question"], batch_size=8192):
            for row in batch.to_pylist():
                question = str(row.get("question") or "")
                source_id = str(row["id"])
                # Cheap lexical tests first: the vast majority of Hotpot
                # questions are single-hop/look-up requests and need no hash.
                if (COMMON.search(question) and PAIR.search(question)
                        and category(question) != "unclassified_common_attribute"):
                    question_hash = qhash(question)
                    if source_id not in banned_ids and question_hash not in banned_questions:
                        candidates.append({"source_id": source_id, "source_row_index": row_index,
                                           "question": question, "question_sha256": question_hash,
                                           "attribute_category": category(question),
                                           "sort_key": stable(PROTOCOL + ":candidate", source_id)})
                row_index += 1
            print(f"[Goal W] metadata rows {row_index}/90447; shortlist={len(candidates)}", flush=True)
        print(f"[Goal W] metadata shard complete: {path.name} ({row_index - shard_start} rows)", flush=True)
    if row_index != 90447:
        raise RuntimeError("metadata_row_count_mismatch")
    # Balance the one-time question-only prefilter across the four named
    # classes so later review can detect actual category shortages.
    by_category = defaultdict(list)
    for row in candidates:
        by_category[row["attribute_category"]].append(row)
    for values in by_category.values():
        values.sort(key=lambda row: (row["sort_key"], row["source_id"]))
    ordered = []
    while any(by_category[name] for name in CATEGORY_ORDER):
        for name in CATEGORY_ORDER:
            if by_category[name]:
                ordered.append(by_category[name].pop(0))
    print(f"[Goal W] metadata prefilter complete: {len(ordered)} unreviewed candidates", flush=True)
    return ordered, row_index


def fetch_candidate_rows(files: list[Path], metadata: list[dict]) -> tuple[dict[str, dict], dict[str, str]]:
    wanted = {row["source_row_index"]: row for row in metadata}
    found = {}
    rejected = {}
    offset = 0
    columns = ["id", "question", "answer", "supporting_facts", "context"]
    for path in files:
        parquet = pq.ParquetFile(path)
        shard_end = offset + parquet.metadata.num_rows
        for group_index in range(parquet.metadata.num_row_groups):
            group_size = parquet.metadata.row_group(group_index).num_rows
            group_start, group_end = offset, offset + group_size
            selected_offsets = [index - group_start for index in wanted if group_start <= index < group_end]
            if selected_offsets:
                table = parquet.read_row_group(group_index, columns=columns)
                for local in selected_offsets:
                    index = group_start + local
                    meta = wanted[index]
                    row = {key: table[key][local].as_py() for key in columns}
                    if str(row["id"]) != meta["source_id"] or qhash(str(row["question"])) != meta["question_sha256"]:
                        raise RuntimeError("candidate_row_identity_mismatch")
                    support = official_support_docs(row)
                    if len(support) != 2:
                        rejected[meta["source_id"]] = "not_exactly_two_resolved_support_titles"
                        continue
                    answer = str(row.get("answer") or "").strip()
                    if not answer or norm(answer) in {"yes", "no"}:
                        rejected[meta["source_id"]] = "empty_or_boolean_official_answer"
                        continue
                    found[meta["source_id"]] = {**meta, "official_answer": answer,
                        "support_docs": support, "entity_keys": question_entities(meta["question"], [x["title"] for x in support])}
            offset = group_end
            print(f"[Goal W] full-row group {group_index + 1}/{parquet.metadata.num_row_groups} in {path.name}; candidates materialized={len(found)}", flush=True)
        if offset != shard_end:
            raise RuntimeError("candidate_row_group_offset_mismatch")
    if offset != 90447:
        raise RuntimeError("full_row_scan_count_mismatch")
    return found, rejected


def select_pools(ordered: list[dict], full_rows: dict[str, dict], rejected: dict[str, str], banned_titles: set[str], banned_entities: set[str]) -> tuple[dict[str, list[dict]], dict, list[dict]]:
    pools = {split: [] for split in SPLITS}
    split_title_sets = {split: set() for split in SPLITS}
    split_entity_sets = {split: set() for split in SPLITS}
    used_ids, used_questions = set(), set()
    counts = Counter()
    rejection_ledger = []
    assigned = set()
    def attempt(meta: dict, split: str) -> bool:
        source_id = meta["source_id"]
        row = full_rows.get(source_id)
        if row is None or source_id in used_ids or meta["question_sha256"] in used_questions:
            if source_id not in assigned:
                rejection_ledger.append({"source_id_sha256": stable(PROTOCOL + ":id-audit", source_id),
                    "question_sha256": meta["question_sha256"], "attribute_category": meta["attribute_category"],
                    "reason": rejected.get(source_id, "duplicate_or_unmaterialized_candidate")})
                assigned.add(source_id)
            return False
        titles = [doc["title"] for doc in row["support_docs"]]
        normalized_titles = {norm(title) for title in titles}
        entities = set(row["entity_keys"])
        if (len(normalized_titles) < 2 or normalized_titles & banned_titles
                or normalized_titles & split_title_sets[split]
                or entities & banned_entities or entities & split_entity_sets[split]):
            counts[f"{split}:overlap_or_support_reject"] += 1
            if source_id not in assigned:
                reason = "historical_or_within_split_title_overlap" if normalized_titles & (banned_titles | split_title_sets[split]) else "historical_or_within_split_entity_overlap"
                rejection_ledger.append({"source_id_sha256": stable(PROTOCOL + ":id-audit", source_id),
                    "question_sha256": meta["question_sha256"], "attribute_category": meta["attribute_category"], "reason": reason})
                assigned.add(source_id)
            return False
        other_splits = [s for s in SPLITS if s != split]
        other_titles = set().union(*(split_title_sets[s] for s in other_splits))
        other_entities = set().union(*(split_entity_sets[s] for s in other_splits))
        if normalized_titles & other_titles or entities & other_entities:
            counts[f"{split}:cross_split_title_or_entity_reject"] += 1
            if source_id not in assigned:
                rejection_ledger.append({"source_id_sha256": stable(PROTOCOL + ":id-audit", source_id),
                    "question_sha256": meta["question_sha256"], "attribute_category": meta["attribute_category"],
                    "reason": "cross_split_title_or_entity_overlap"})
                assigned.add(source_id)
            return False
        docs = [{"id": sha_bytes(norm(doc["title"]).encode("utf-8"))[:20],
                 "title": doc["title"], "text": doc["text"]} for doc in row["support_docs"]]
        row["corpus_support_docs"] = docs
        row["output_row"] = {"case_id": source_id, "question": row["question"],
            "official_answer": row["official_answer"], "support_ids": [doc["id"] for doc in docs],
            "attribute_category": row["attribute_category"], "source_id": source_id}
        pools[split].append(row)
        assigned.add(source_id)
        used_ids.add(source_id)
        used_questions.add(meta["question_sha256"])
        split_title_sets[split].update(normalized_titles)
        split_entity_sets[split].update(entities)
        counts[f"{split}:{row['attribute_category']}"] += 1
        return True

    # Reserve the required category minimums before filling remaining slots.
    # Dev and confirmation receive their disjoint representatives first, so a
    # large train pool cannot consume the only eligible examples of a class.
    for category_name in CATEGORY_ORDER:
        for split in ("dev", "confirm", "train"):
            needed = CATEGORY_QUOTAS[split]
            for meta in ordered:
                if len(pools[split]) >= LIMITS[split] or counts[f"{split}:{category_name}"] >= needed:
                    break
                if meta["attribute_category"] == category_name:
                    attempt(meta, split)

    # Fill each pool to its fixed size using the same frozen category/hash
    # ordering. No candidate is added after this pass.
    for split in SPLITS:
        for meta in ordered:
            if len(pools[split]) >= LIMITS[split]:
                break
            attempt(meta, split)
        if len(pools[split]) < LIMITS[split]:
            counts[f"{split}:pool_deficit"] = LIMITS[split] - len(pools[split])
    return pools, {"selection_counts": dict(counts),
                   "split_support_titles": {s: len(split_title_sets[s]) for s in SPLITS},
                   "split_entity_keys": {s: len(split_entity_sets[s]) for s in SPLITS}}, rejection_ledger


def gather_corpora(files: list[Path], pools: dict[str, list[dict]], excluded_titles: set[str], history_title_set: set[str]) -> tuple[dict[str, list[dict]], dict]:
    supports = {split: {} for split in SPLITS}
    all_support_titles = set()
    for split in SPLITS:
        for candidate in pools[split]:
            for doc in candidate["corpus_support_docs"]:
                key = norm(doc["title"])
                supports[split][key] = doc
                all_support_titles.add(key)
    need = {split: 2048 - len(supports[split]) for split in SPLITS}
    if any(value < 0 for value in need.values()):
        raise RuntimeError("support_documents_exceed_corpus_limit")
    heaps = {split: [] for split in SPLITS}
    seen_title = set()
    row_count = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(columns=["context"], batch_size=128):
            contexts = batch.column("context")
            for local in range(batch.num_rows):
                context = contexts[local].as_py() or {}
                for title, sentences in zip(context.get("title") or [], context.get("sentences") or []):
                    key = norm(title)
                    if not key or key in seen_title:
                        continue
                    seen_title.add(key)
                    if key in all_support_titles or key in excluded_titles or key in history_title_set:
                        continue
                    if not isinstance(sentences, list):
                        continue
                    text = " ".join(str(s).strip() for s in sentences if isinstance(s, str) and s.strip())
                    if not text:
                        continue
                    bucket = int(stable(PROTOCOL + ":corpus-split", key)[:16], 16) % len(SPLITS)
                    split = SPLITS[bucket]
                    cap = need[split]
                    if cap <= 0:
                        continue
                    rank = int(stable(PROTOCOL + ":corpus-doc", key), 16)
                    heap = heaps[split]
                    item = (-rank, key, {"id": sha_bytes(key.encode("utf-8"))[:20], "title": title, "text": text})
                    if len(heap) < cap:
                        heapq.heappush(heap, item)
                    elif rank < -heap[0][0]:
                        heapq.heapreplace(heap, item)
                row_count += 1
            if all(len(heaps[split]) >= need[split] for split in SPLITS):
                break
        if all(len(heaps[split]) >= need[split] for split in SPLITS):
            break
    if row_count == 0:
        raise RuntimeError("corpus_source_scan_empty")
    corpora = {}
    details = {}
    for split in SPLITS:
        docs = list(supports[split].values())
        docs.extend(item[2] for item in sorted(heaps[split], key=lambda item: (-item[0], item[1])))
        corpora[split] = docs
        details[split] = {"support_documents": len(supports[split]),
                          "natural_distractors": len(docs) - len(supports[split]),
                          "documents": len(docs), "title_set_sha256": sha_bytes("\n".join(sorted(norm(d["title"]) for d in docs)).encode())}
    title_sets = {split: {norm(doc["title"]) for doc in corpora[split]} for split in SPLITS}
    if any(title_sets[a] & title_sets[b] for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]):
        raise RuntimeError("corpus_title_leakage")
    return corpora, {"documents": details, "source_rows_scanned_for_distractors": row_count}


def prepare() -> dict:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite Goal W data directory: {OUT}")
    print("[Goal W] verify source and historical exposure", flush=True)
    files, provenance = verify_source()
    hist_ids, hist_questions, hist_titles, hist_inputs = history_exposure()
    v041_ids, v041_questions, v041_titles, v041_hashes = read_v041_exclusions()
    print("[Goal W] reconstruct sealed IDs from public source and verify published digest", flush=True)
    sealed = reconstruct_sealed(files, v041_ids, v041_titles)
    excluded_ids = hist_ids | v041_ids | sealed["id_set"]
    excluded_questions = hist_questions | v041_questions | sealed["question_hashes"]
    excluded_titles = hist_titles | v041_titles | sealed["title_set"]
    print("[Goal W] question-only common-attribute prefilter", flush=True)
    metadata, source_rows = discover_metadata_candidates(files, excluded_ids, excluded_questions)
    print(f"[Goal W] materialize shortlisted rows: {len(metadata)}", flush=True)
    full_rows, source_rejections = fetch_candidate_rows(files, metadata)
    # Question and title screens extend Goal G's historical exposure inventory.
    historical_entities = set(hist_titles) | set(v041_titles)
    pools, selection_report, rejection_ledger = select_pools(metadata, full_rows, source_rejections, excluded_titles, historical_entities)
    print("[Goal W] candidate pool counts " + json.dumps({s: len(pools[s]) for s in SPLITS})
          + "; categories " + json.dumps({s: dict(Counter(r["attribute_category"] for r in pools[s])) for s in SPLITS}), flush=True)
    print("[Goal W] build independent 2048-document corpora", flush=True)
    corpora, corpus_report = gather_corpora(files, pools, excluded_titles, hist_titles | v041_titles)

    # Prepare all content in memory before creating the run directory. This
    # ensures a failed prerequisite leaves no partial output that could look
    # frozen.
    candidate_rows = {split: [row["output_row"] for row in pools[split]] for split in SPLITS}
    actual_title_sets = {split: {norm(d["title"]) for d in corpora[split]} for split in SPLITS}
    if any(len(corpora[split]) != 2048 for split in SPLITS):
        raise RuntimeError("natural_corpus_pool_shortfall")
    # Split question / source / support title isolation.
    q_sets = {s: {qhash(r["question"]) for r in candidate_rows[s]} for s in SPLITS}
    id_sets = {s: {r["source_id"] for r in candidate_rows[s]} for s in SPLITS}
    if any(q_sets[a] & q_sets[b] or id_sets[a] & id_sets[b] or actual_title_sets[a] & actual_title_sets[b]
           for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]):
        raise RuntimeError("split_leakage_check_failed")

    OUT.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        write_jsonl(OUT / f"candidate-{split}.jsonl", candidate_rows[split])
        write_json(OUT / f"corpus-{split}.json", {"documents": corpora[split]})
    write_jsonl(OUT / "candidate-rejection-ledger.jsonl", rejection_ledger)

    quota_requirements = CATEGORY_QUOTAS
    category_counts = {split: dict(Counter(row["attribute_category"] for row in candidate_rows[split])) for split in SPLITS}
    deficits = {split: {category_name: max(0, quota_requirements[split] - category_counts[split].get(category_name, 0))
                        for category_name in CATEGORY_ORDER}
                for split in SPLITS}
    manifest = {
        "version": PROTOCOL,
        "status": "bounded_unreviewed_candidate_pools_prepared",
        "gpu_used": False, "model_or_api_used": False, "sealed_jsonl_read": False,
        "source": {"dataset": provenance["dataset"], "config": provenance["config"], "split": provenance["split"],
                   "rows": provenance["rows"], "files": {p.name: sha_file(p) for p in files},
                   "provenance_sha256": sha_file(SOURCE / "download-provenance.json")},
        "historical_exclusions": {
            "goal_g_exposure_input_sha256": hist_inputs,
            "v041_split_sha256": v041_hashes,
            "goal_g_ids": len(hist_ids), "goal_g_questions": len(hist_questions),
            "goal_g_titles": len(hist_titles), "v041_ids": len(v041_ids),
            "v041_questions": len(v041_questions), "v041_titles": len(v041_titles),
            "reconstructed_goal_v": {k: v for k, v in sealed.items() if k not in ("id_set", "title_set", "question_hashes")},
            "exclusion_key_counts": {"ids": len(excluded_ids), "questions": len(excluded_questions), "titles": len(excluded_titles)},
            "reconstruction_method": "official distractor/train parquet in source order; replay v072 selection rules excluding v041 train/validation/test IDs and supporting titles; verify count, scan count, title count, and task-id digest against v072 manifest; never open final-test.jsonl",
        },
        "candidate_prefilter": {
            "selection": "single source pass; question-only common/comparison lexical locator; deterministic per-category SHA256 ordering and round-robin across four lexical classes; then source ID/question/title/entity exclusion and >=2 official supporting titles",
            "unreviewed_candidates_seen": len(metadata), "candidate_rows_materialized": len(full_rows),
            "preselection_rejection_reason_counts": dict(Counter(source_rejections.values())),
            "postselection_rejection_reason_counts": dict(Counter(row["reason"] for row in rejection_ledger)),
            "selected_counts": {split: len(candidate_rows[split]) for split in SPLITS},
            "category_counts": category_counts,
            "category_quota_diagnostics_not_audit": {"requirements": quota_requirements, "deficits": deficits},
            "selection_diagnostics": selection_report,
            "not_final_or_audited": True,
        },
        "corpora": {"documents_per_split": 2048, "selection": "candidate official support paragraphs plus natural Hotpot context paragraphs from the source prefix needed to fill each split; document ranking is deterministic SHA256 within split; historical titles excluded; title sets disjoint",
                    "details": corpus_report},
        "checks": {"candidate_question_id_title_split_overlap": 0, "corpus_title_overlap": 0,
                   "confirm_questions_or_labels_used_in_training_or_dev": False},
        "outputs": {},
    }
    for path in sorted(OUT.iterdir()):
        manifest["outputs"][path.name] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    write_json(OUT / "manifest.json", manifest)
    return manifest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",))
    args = parser.parse_args()
    result = prepare()
    print(json.dumps({"status": result["status"], "output": str(OUT),
                      "candidate_counts": result["candidate_prefilter"]["selected_counts"],
                      "category_counts": result["candidate_prefilter"]["category_counts"],
                      "quota_diagnostics": result["candidate_prefilter"]["category_quota_diagnostics_not_audit"]["deficits"],
                      "sealed_reconstruction": result["historical_exclusions"]["reconstructed_goal_v"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
