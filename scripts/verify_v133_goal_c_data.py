"""Independent CPU-only provenance, split, token, and blind-packet audit."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_v133_goal_c_data import (
    OUT, ROOT, SHARDS, SOURCE, TOKENIZER, V118_ADAPTER, V133_SETTINGS, SFT_CONFIG, exposure_index, make_documents,
    norm, read_json, read_jsonl, sha_bytes, sha_file, support_sentences,
)


def source_row(global_index: int, files: list[Path], shard_rows: list[int]) -> dict:
    shard = 0 if global_index < shard_rows[0] else 1
    local_index = global_index - (shard_rows[0] if shard else 0)
    pf = pq.ParquetFile(files[shard])
    for gi in range(pf.metadata.num_row_groups):
        count = pf.metadata.row_group(gi).num_rows
        if local_index < count:
            table = pf.read_row_group(gi, columns=["id", "question", "answer", "supporting_facts", "context"])
            return {key: table[key][local_index].as_py() for key in table.schema.names}
        local_index -= count
    raise IndexError(global_index)


def main() -> None:
    manifest_path = OUT / "manifest-private.json"
    manifest = read_json(manifest_path)
    provenance = read_json(SOURCE / "download-provenance.json")
    by_name = {Path(x["path"]).name: x for x in provenance["files"]}
    files = [SOURCE / name for name in SHARDS]
    shard_rows = [int(by_name[name]["rows"]) for name in SHARDS]
    actual_shards = {name: sha_file(path) for name, path in zip(SHARDS, files)}
    assert actual_shards == manifest["source"]["shards_sha256"]
    settings = read_json(V133_SETTINGS)
    tokenizer_names = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")
    tokenizer_hashes = {name: sha_file(TOKENIZER / name) for name in tokenizer_names}
    assert all(tokenizer_hashes[name] == settings["model_files"][name] for name in tokenizer_names)
    adapter_hash = sha_file(V118_ADAPTER / "adapter_model.safetensors")
    assert adapter_hash == settings["adapter_sha256"] == manifest["baseline_assets"]["v118_adapter_sha256"]
    assert sha_file(SFT_CONFIG) == manifest["baseline_assets"]["sft_config_sha256"]

    candidates = list(read_jsonl(OUT / "candidate-supervision-private.pending-review.jsonl"))
    source_map = list(read_jsonl(OUT / "source-map-private.jsonl"))
    review = list(read_jsonl(OUT / "anonymous-review-packet.pending-authorization.jsonl"))
    screen = list(read_jsonl(OUT / "screen-ledger-private.jsonl"))
    assert len(candidates) == len(source_map) == 64
    assert len(review) == 48 and len(screen) == 300
    assert [x["item_id"] for x in candidates[:48]] == [x["item_id"] for x in review]
    assert len({x["source_id"] for x in source_map}) == 64
    assert len({x["question_sha256"] for x in source_map}) == 64
    roles = Counter(x["precommitted_role"] for x in source_map)
    assert roles == {"train_candidate": 32, "dev_candidate": 16, "reserve_unreviewed": 16}

    exposed_ids, exposed_q = exposure_index()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, trust_remote_code=False)
    map_by_item = {x["item_id"]: x for x in source_map}
    candidate_by_item = {x["item_id"]: x for x in candidates}
    for row in candidates:
        link = map_by_item[row["item_id"]]
        source = source_row(int(link["source_row_index"]), files, shard_rows)
        assert source["id"] == row["source_id"] == link["source_id"]
        assert source["question"].strip() == row["question"]
        assert str(source["answer"]).strip() == row["official_answer"]
        assert sha_bytes(norm(row["question"]).encode("utf-8")) == row["question_sha256"] == link["question_sha256"]
        assert source["id"] not in exposed_ids
        assert row["question_sha256"] not in exposed_q
        facts = support_sentences(source)
        assert facts == row["supporting_facts"]
        docs = make_documents(facts)
        assert docs == row["evidence_documents"]
        assert norm(row["official_answer"]) in norm(" ".join(d["text"] for d in docs))
        tool = json.loads(row["messages"][2]["content"])
        assert tool == {"ok": True, "result": docs}
        assert row["messages"][0]["role"] == "system"
        assert row["messages"][1]["role"] == "user"
        assert row["messages"][2]["role"] == "tool"
        assert row["messages"][3]["role"] == "assistant"
        assert json.loads(row["messages"][3]["content"]) == {"action": "answer", "answer": row["official_answer"]}
        prompt = tokenizer.apply_chat_template(row["messages"][:3], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
        prompt_n, full_n = len(prompt["input_ids"]), len(full["input_ids"])
        target_n = len(tokenizer.encode(row["messages"][3]["content"], add_special_tokens=False))
        assert (prompt_n, target_n, full_n) == (row["tokens"]["prompt"], row["tokens"]["target"], row["tokens"]["full"])
        assert prompt_n <= 2048 and target_n <= 128 and full_n <= 2176

    for packet in review:
        assert not ({"source_id", "precommitted_role", "normalized_exact_match", "token_f1", "gold", "arm"} & set(packet))
        assert set(packet) == {"item_id", "question", "proposed_answer", "visible_evidence", "review_prompt"}
    assert not (set(x["item_id"] for x in review) & set(x["item_id"] for x in candidates[48:]))

    role_hashes = {}
    for role in ("train_candidate", "dev_candidate", "reserve_unreviewed"):
        ids = [x["item_id"] for x in source_map if x["precommitted_role"] == role]
        rows = [candidate_by_item[x] for x in ids]
        role_hashes[role] = {
            "item_ids_sha256": sha_bytes("\n".join(ids).encode("utf-8")),
            "candidate_rows_sha256": sha_bytes("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in rows).encode("utf-8")),
            "count": len(rows),
        }
    report = {
        "version": "v133-goal-c-data-independent-audit-v1",
        "result": "passed",
        "screen_ledger_rows": len(screen),
        "candidate_rows_reopened_from_exact_source_row": len(candidates),
        "review_packet_rows": len(review),
        "split_counts": dict(roles),
        "split_hashes": role_hashes,
        "official_shard_hashes": actual_shards,
        "tokenizer_hashes": tokenizer_hashes,
        "v118_adapter_sha256": adapter_hash,
        "sft_config_sha256": sha_file(SFT_CONFIG),
        "audit_script_sha256": sha_file(Path(__file__).resolve()),
        "candidate_file_sha256": sha_file(OUT / "candidate-supervision-private.pending-review.jsonl"),
        "review_packet_sha256": sha_file(OUT / "anonymous-review-packet.pending-authorization.jsonl"),
        "token_ranges": {
            "prompt_min": min(x["tokens"]["prompt"] for x in candidates),
            "prompt_max": max(x["tokens"]["prompt"] for x in candidates),
            "target_max": max(x["tokens"]["target"] for x in candidates),
            "full_max": max(x["tokens"]["full"] for x in candidates),
        },
        "review_privacy": "reviewer packet omits source IDs, split roles, scores, model arms, and other votes; reserves omitted",
        "model_or_api_calls": 0,
        "training_allowed": False,
        "sealed_data_accessed": False,
    }
    audit_path = OUT / "audit-private.json"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["independent_audit_path"] = str(audit_path.relative_to(ROOT)).replace("\\", "/")
    manifest["independent_audit_sha256"] = sha_file(audit_path)
    manifest["frozen_candidate_split_hashes"] = role_hashes
    manifest["verification"]["independent_exact_row_reopen_audit"] = "passed"
    manifest["outputs_before_manifest"][audit_path.name] = {"sha256": sha_file(audit_path), "bytes": audit_path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": report["result"], "candidate_rows": len(candidates), "review_rows": len(review), "split_hashes": role_hashes, "audit_sha256": sha_file(audit_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
