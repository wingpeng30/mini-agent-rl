"""Bounded CPU-only construction of v133 Goal C evidence-to-answer candidates.

Reads exactly the official HotpotQA distractor train shards named below and
specified historical exposure indexes. It never loads a model or calls an API.
Output examples remain pending blind review and are not training-authorized.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "data/hotpot-full-train-v114"
OUT = ROOT / "data/evidence-sft-v133-goal-c-private"
TOKENIZER = Path(r"D:\qwen_08b")
V133_SETTINGS = ROOT / "runs/v133-search-budget-20260927-cuda-run01/settings.json"
V118_ADAPTER = ROOT / "backups/v118/extracted/runs/v118/formal/adapter"
SFT_CONFIG = ROOT / "configs/v133-goal-c-small-sft.json"
SCREEN_LIMIT = 300
CANDIDATE_LIMIT = 64
SEED = "v133-goal-c-screen-v1"
SPLIT_SEED = "v133-goal-c-split-v1"
SYSTEM = "You are an evidence-driven retrieval agent. Output one JSON search or answer action, using visible evidence only."
SHARDS = ("train-00000-of-00002.parquet", "train-00001-of-00002.parquet")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold()).strip()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def exposure_index() -> tuple[set[str], set[str]]:
    # Reuse the existing bounded v123/v122 source-history index, then add the
    # exact v117 selected IDs and the v123/v130 fixed source-screen ledgers.
    from scripts.v123_prepare_internal import source_ids

    ids, old_questions = source_ids()
    question_hashes = {sha_bytes(q.encode("utf-8")) for q in old_questions}
    v117 = ROOT / "data/evidence-sft-v117"
    ids.update(read_json(v117 / "candidate-freeze.json")["ids"])
    for row in read_jsonl(v117 / "candidate-private.jsonl"):
        q = row.get("packet", {}).get("question")
        if q:
            question_hashes.add(sha_bytes(norm(q).encode("utf-8")))
    ledgers = (
        ROOT / "data/evidence-sft-v123/internal/screened-source-private.jsonl",
        ROOT / "data/evidence-sft-v124-forensic/goal-v128/g4-new-pool/screened-source-private.jsonl",
    )
    for ledger in ledgers:
        for row in read_jsonl(ledger):
            if row.get("id"):
                ids.add(row["id"])
            if row.get("question_sha256"):
                question_hashes.add(row["question_sha256"])
    return ids, question_hashes


def sample_source_rows(files: list[Path], row_counts: list[int]) -> list[tuple[int, dict]]:
    total = sum(row_counts)
    if total != 90447:
        raise ValueError(f"unexpected_official_train_row_count:{total}")
    # The index sample is fixed and independent of answers, model outputs, or
    # support quality. Only these 300 row records are subsequently opened.
    indices = sorted(random.Random(SEED).sample(range(total), SCREEN_LIMIT))
    result: dict[int, dict] = {}
    global_start = 0
    columns = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]
    for path, shard_rows in zip(files, row_counts):
        pf = pq.ParquetFile(path)
        local_targets = [i - global_start for i in indices if global_start <= i < global_start + shard_rows]
        wanted: dict[int, list[tuple[int, int]]] = defaultdict(list)
        group_start = 0
        for gi in range(pf.metadata.num_row_groups):
            group_n = pf.metadata.row_group(gi).num_rows
            lo, hi = group_start, group_start + group_n
            for local_i in local_targets:
                if lo <= local_i < hi:
                    wanted[gi].append((local_i, local_i - lo))
            group_start = hi
        for gi, positions in wanted.items():
            table = pf.read_row_group(gi, columns=columns)
            for local_i, offset in positions:
                result[global_start + local_i] = {key: table[key][offset].as_py() for key in columns}
        global_start += shard_rows
    if set(result) != set(indices):
        raise ValueError("sampled_train_rows_missing")
    return [(i, result[i]) for i in indices]


def support_sentences(row: dict) -> list[dict] | None:
    facts = row.get("supporting_facts") or {}
    context = row.get("context") or {}
    titles = context.get("title") or []
    sentences = context.get("sentences") or []
    sf_titles, sf_sent_ids = facts.get("title") or [], facts.get("sent_id") or []
    if not sf_titles or len(sf_titles) != len(sf_sent_ids):
        return None
    selected = []
    for title, sentence_id in zip(sf_titles, sf_sent_ids):
        try:
            title_index = titles.index(title)
            sentence = sentences[title_index][sentence_id]
        except (ValueError, IndexError, TypeError):
            return None
        if not isinstance(sentence, str) or not sentence.strip():
            return None
        selected.append({"title": title, "sentence_index": int(sentence_id), "text": sentence.strip()})
    return selected


def make_documents(facts: list[dict]) -> list[dict]:
    grouped: dict[str, list[str]] = {}
    for fact in facts:
        grouped.setdefault(fact["title"], []).append(fact["text"])
    docs = []
    for title, texts in grouped.items():
        text = " ".join(texts)
        digest = sha_bytes((title + "\n" + text).encode("utf-8"))
        docs.append({
            "id": digest[:20], "title": title, "text": text,
            "source": "official-hotpotqa-train-supporting-fact:" + digest[:12],
        })
    return docs


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing_to_overwrite:{OUT}")
    provenance = read_json(SOURCE / "download-provenance.json")
    if (provenance.get("dataset"), provenance.get("config"), provenance.get("split"), provenance.get("rows")) != (
        "hotpotqa/hotpot_qa", "distractor", "train", 90447
    ):
        raise ValueError("official_train_provenance_mismatch")
    file_rows = {Path(item["path"]).name: item for item in provenance["files"]}
    if set(file_rows) != set(SHARDS):
        raise ValueError("official_train_shard_set_mismatch")
    files, row_counts, source_hashes = [], [], {}
    for name in SHARDS:
        path = SOURCE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        item = file_rows[name]
        actual_hash = sha_file(path)
        if actual_hash != item["sha256"]:
            raise ValueError("official_train_source_hash_mismatch:" + name)
        files.append(path)
        row_counts.append(int(item["rows"]))
        source_hashes[name] = actual_hash

    old_ids, old_question_hashes = exposure_index()
    baseline_settings = read_json(V133_SETTINGS)
    tokenizer_hashes = {
        name: sha_file(TOKENIZER / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")
    }
    expected_tokenizer_hashes = {
        "tokenizer.json": baseline_settings["model_files"]["tokenizer.json"],
        "tokenizer_config.json": baseline_settings["model_files"]["tokenizer_config.json"],
        "chat_template.jinja": baseline_settings["model_files"]["chat_template.jinja"],
        "vocab.json": baseline_settings["model_files"]["vocab.json"],
        "merges.txt": baseline_settings["model_files"]["merges.txt"],
    }
    if tokenizer_hashes != expected_tokenizer_hashes:
        raise ValueError("v118_tokenizer_asset_hash_mismatch")
    adapter_hash = sha_file(V118_ADAPTER / "adapter_model.safetensors")
    if adapter_hash != baseline_settings["adapter_sha256"]:
        raise ValueError("v118_adapter_asset_hash_mismatch")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, trust_remote_code=False)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("frozen_tokenizer_chat_template_missing")

    ledgers = []
    eligible = []
    reason_counts = Counter()
    seen_questions = set(old_question_hashes)
    for index, row in sample_source_rows(files, row_counts):
        source_id = str(row["id"])
        ledger = {"sample_index": index, "source_id": source_id}
        if source_id in old_ids:
            ledger["disposition"] = "exclude_previous_exposure_id"
            reason_counts[ledger["disposition"]] += 1
            ledgers.append(ledger)
            continue
        question = str(row.get("question") or "").strip()
        qnorm = norm(question)
        qhash = sha_bytes(qnorm.encode("utf-8"))
        ledger["question_sha256"] = qhash
        if not question or qhash in seen_questions:
            ledger["disposition"] = "exclude_empty_or_duplicate_question"
            reason_counts[ledger["disposition"]] += 1
            ledgers.append(ledger)
            continue
        seen_questions.add(qhash)
        facts = support_sentences(row)
        if not facts:
            ledger["disposition"] = "exclude_missing_or_invalid_supporting_fact"
            reason_counts[ledger["disposition"]] += 1
            ledgers.append(ledger)
            continue
        answer = str(row.get("answer") or "").strip()
        evidence_norm = norm(" ".join(x["text"] for x in facts))
        answer_norm = norm(answer)
        if not answer_norm or answer_norm not in evidence_norm:
            ledger["disposition"] = "exclude_answer_not_lexically_supported"
            reason_counts[ledger["disposition"]] += 1
            ledgers.append(ledger)
            continue
        docs = make_documents(facts)
        tool_result = json.dumps({"ok": True, "result": docs}, ensure_ascii=False, separators=(",", ":"))
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "tool", "content": tool_result},
            {"role": "assistant", "content": json.dumps({"action": "answer", "answer": answer}, ensure_ascii=False, separators=(",", ":"))},
        ]
        prompt_ids = tokenizer.apply_chat_template(messages[:3], tokenize=True, add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        prompt_tokens = len(prompt_ids["input_ids"])
        full_tokens = len(full_ids["input_ids"])
        target_tokens = len(tokenizer.encode(messages[-1]["content"], add_special_tokens=False))
        if prompt_tokens > 2048 or target_tokens > 128 or full_tokens > 2176:
            ledger["disposition"] = "exclude_v118_token_budget"
            ledger["tokens"] = {"prompt": prompt_tokens, "target": target_tokens, "full": full_tokens}
            reason_counts[ledger["disposition"]] += 1
            ledgers.append(ledger)
            continue
        item_id = sha_bytes(("v133-goal-c-anon-v1:" + source_id).encode("utf-8"))[:16]
        candidate = {
            "item_id": item_id,
            "source_id": source_id,
            "question": question,
            "question_sha256": qhash,
            "official_answer": answer,
            "supporting_facts": facts,
            "evidence_documents": docs,
            "messages": messages,
            "target_assistant_index": 3,
            "thinking_enabled": False,
            "tokens": {"prompt": prompt_tokens, "target": target_tokens, "full": full_tokens},
            "source_split": "official HotpotQA distractor train",
            "source_shard": next(name for name, start in zip(SHARDS, [0, row_counts[0]]) if start <= index < start + file_rows[name]["rows"]),
            "source_row_index": index,
        }
        eligible.append(candidate)
        ledger["disposition"] = "eligible"
        ledger["item_id"] = item_id
        ledger["support_fact_count"] = len(facts)
        ledger["token_counts"] = candidate["tokens"]
        reason_counts["eligible"] += 1
        ledgers.append(ledger)

    eligible.sort(key=lambda x: sha_bytes(("v133-goal-c-candidate-v1:" + x["source_id"]).encode("utf-8")))
    candidates = eligible[:CANDIDATE_LIMIT]
    if len(candidates) < CANDIDATE_LIMIT:
        # Shortfalls are reported; the script never reads beyond the frozen 300.
        pass
    for n, row in enumerate(candidates):
        row["precommitted_role"] = "train_candidate" if n < 32 else "dev_candidate" if n < 48 else "reserve_unreviewed"

    OUT.mkdir(parents=True)
    (OUT / "screen-ledger-private.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in ledgers), encoding="utf-8")
    (OUT / "candidate-supervision-private.pending-review.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in x.items() if k != "precommitted_role"}, ensure_ascii=False, sort_keys=True) + "\n" for x in candidates), encoding="utf-8")
    review_rows = []
    # Only the precommitted train/dev candidates are sent for adjudication.
    # The 16 reserve rows remain completely unreviewed and cannot be promoted
    # automatically after votes are observed.
    for x in candidates[:48]:
        review_rows.append({
            "item_id": x["item_id"],
            "question": x["question"],
            "proposed_answer": x["official_answer"],
            "visible_evidence": [{"title": d["title"], "text": d["text"]} for d in x["evidence_documents"]],
            "review_prompt": "Independently assess whether the proposed answer directly answers the question and is supported by the displayed evidence. Return supported / partial / unsupported and quote the exact supporting text. Do not infer from outside knowledge.",
        })
    (OUT / "anonymous-review-packet.pending-authorization.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in review_rows), encoding="utf-8")
    (OUT / "source-map-private.jsonl").write_text(
        "".join(json.dumps({"item_id": x["item_id"], "source_id": x["source_id"], "source_shard": x["source_shard"], "source_row_index": x["source_row_index"], "question_sha256": x["question_sha256"], "precommitted_role": x["precommitted_role"]}, sort_keys=True) + "\n" for x in candidates), encoding="utf-8")

    outputs = {}
    for path in sorted(OUT.iterdir()):
        outputs[path.name] = {"sha256": sha_file(path), "bytes": path.stat().st_size}
    manifest = {
        "version": "v133-goal-c-source-candidates-v1",
        "status": "pending_independent_review_and_user_training_authorization",
        "hypothesis": "When the question's requested attribute/value is explicitly present in visible evidence, one answer action should return that requested value concisely rather than a related entity or partial value.",
        "source": {"dataset": provenance["dataset"], "config": provenance["config"], "split": provenance["split"], "rows": provenance["rows"], "shards_sha256": source_hashes, "provenance_file_sha256": sha_file(SOURCE / "download-provenance.json")},
        "baseline_assets": {"model_path": str(TOKENIZER), "v118_adapter_path": str(V118_ADAPTER), "v118_adapter_sha256": adapter_hash, "tokenizer_sha256": tokenizer_hashes, "v133_run_settings_sha256": sha_file(V133_SETTINGS), "sft_encoder_path": "src/mini_agent_rl/training/sft.py", "sft_encoder_sha256": sha_file(ROOT / "src/mini_agent_rl/training/sft.py"), "sft_config_path": "configs/v133-goal-c-small-sft.json", "sft_config_sha256": sha_file(SFT_CONFIG), "answer_action_system_prompt_sha256": sha_bytes(SYSTEM.encode("utf-8"))},
        "sampling": {"seed": SEED, "method": "Python random.Random string seed; sample 300 distinct global row indices uniformly without replacement from the 90,447 fixed official train rows", "screen_limit": SCREEN_LIMIT, "screened_rows": len(ledgers), "opened_question_count": sum(1 for x in ledgers if "question_sha256" in x), "previous_exposure_skipped_by_id": reason_counts["exclude_previous_exposure_id"], "never_expanded": True},
        "construction": {"candidate_limit": CANDIDATE_LIMIT, "constructed": len(candidates), "eligible_in_screen": len(eligible), "criteria": ["official supporting_fact title/sentence references resolve exactly against that source row's context", "official answer occurs in the normalized concatenated supporting sentences", "question ID and normalized prompt are absent from the bounded historical exposure index", "v118 local tokenizer prompt <=2048, target <=128, full sequence <=2176"], "selection": "eligible rows sorted by SHA256('v133-goal-c-candidate-v1:'+source_id), first 64", "precommitted_roles": {"train_candidate": min(32, len(candidates)), "dev_candidate": max(0, min(16, len(candidates)-32)), "reserve_unreviewed": max(0, len(candidates)-48)}, "reviewers_see": ["item_id", "question", "proposed_answer", "official supporting evidence"], "reviewers_do_not_see": ["source_id", "source split/role", "A2/A3 group", "model score", "other review" ]},
        "exclusions": dict(reason_counts),
        "verification": {
            "candidate_rows": len(candidates),
            "source_map_rows": len(candidates),
            "anonymous_packet_rows": min(48, len(candidates)),
            "screen_ledger_rows": len(ledgers),
            "unique_source_ids": len({x["source_id"] for x in candidates}),
            "unique_normalized_questions": len({x["question_sha256"] for x in candidates}),
            "split_counts": {"train_candidate": min(32, len(candidates)), "dev_candidate": max(0, min(16, len(candidates) - 32)), "reserve_unreviewed": max(0, len(candidates) - 48)},
            "support_references_resolved_by_builder": True,
            "answers_lexically_present_in_displayed_support": True,
            "v118_tokenizer_asset_hashes_match_run_settings": tokenizer_hashes == expected_tokenizer_hashes,
            "v118_adapter_hash_matches_run_settings": adapter_hash == baseline_settings["adapter_sha256"],
            "blind_packet_excludes_source_and_split_metadata": True,
            "actual_token_ids_used_for_limits": True,
            "no_model_loaded_or_api_called": True,
        },
        "training_allowed": False,
        "new_model_inference_started": False,
        "sft_started": False,
        "sealed_data_accessed": False,
        "outputs_before_manifest": outputs,
        "builder_script_sha256": sha_file(Path(__file__).resolve()),
    }
    manifest_path = OUT / "manifest-private.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "screened": len(ledgers), "opened": manifest["sampling"]["opened_question_count"], "eligible": len(eligible), "candidates": len(candidates), "exclusions": dict(reason_counts), "training_allowed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
