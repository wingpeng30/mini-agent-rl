"""Fail-closed reconciliation and CPU replay of v117 repair proposals.

This script only reads the explicitly scoped v117 train states, v124 forensic
review ledger, and the frozen v114 training corpus. It writes a private audit
artifact; it never edits source rows or authorizes training.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoProcessor

from mini_agent_rl.tools.search import _BM25Index
from mini_agent_rl.training.sft import encode_assistant_only
from scripts.prepare_v116_review import PROCESSOR, render
from scripts.v123_cpu_core import ROOT, digest, readl, sha, write
from scripts.v124_forensic import OUT

TRAIN = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
CORPUS = ROOT / "data/evidence-sft-v114-clean/training-corpus.jsonl"
CORPUS_MANIFEST = ROOT / "data/evidence-sft-v114-clean/manifest.json"
OUTPUT = OUT / "repair-draft-reconciliation-cpu-v2-private.json"
PACKET_OUTPUT = OUT / "repair-draft-replay-inputs-v2-private.jsonl"


def target_action(row: dict) -> dict:
    return json.loads(row["messages"][row["target_assistant_index"]]["content"])


def question(row: dict) -> str:
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def load_primary():
    rows = [r for path in sorted(OUT.glob("primary-batch-*.jsonl")) for r in readl(path)]
    by_id = {r["record_id"]: r for r in rows}
    if len(by_id) != len(rows) or len(rows) != 1229:
        raise RuntimeError("primary_review_coverage_or_uniqueness_failed")
    for path in sorted(OUT.glob("primary-amendments*-private.jsonl")):
        for change in readl(path):
            current = by_id[change["record_id"]]
            if digest(current) != change["superseded_primary_review_sha256"]:
                raise RuntimeError("primary_amendment_chain_failed:" + change["record_id"])
            current = {**current, "disposition": change["effective_disposition"]}
            by_id[change["record_id"]] = current
    for change in readl(OUT / "primary-adjudication-overrides-private.jsonl"):
        current = by_id[change["record_id"]]
        if current["disposition"] != change["supersedes_effective_disposition"]:
            raise RuntimeError("primary_override_chain_failed:" + change["record_id"])
        by_id[change["record_id"]] = {**current, "disposition": change["effective_disposition"]}
    return by_id


def main():
    originals = readl(TRAIN)
    if len(originals) != 345:
        raise RuntimeError("unexpected_v117_train_length")
    by_id = {f"v117_train:{i:04d}": row for i, row in enumerate(originals, 1)}
    primary = load_primary()
    inventory_rows = readl(OUT / "inventory-private.jsonl")
    inventory = {r["record_id"]: r for r in inventory_rows}
    if len(inventory) != len(inventory_rows):
        raise RuntimeError("duplicate_inventory_record")
    batch_reviews = {r["record_id"]: r for path in sorted(OUT.glob("primary-batch-*.jsonl"))
                     for r in readl(path)}
    proposals = defaultdict(list)
    excluded = []

    def add(record_id: str, revised_q: str, revised_action: dict | None,
            source_hash: str, review_hash: str, origin: str,
            declared_group: list[str] | None = None):
        if record_id not in by_id or record_id not in primary:
            raise RuntimeError("unknown_repair_record:" + record_id)
        original = by_id[record_id]
        source_review = primary[record_id]
        inventory_record = inventory[record_id]
        if digest(original) != source_hash or digest(original) != inventory_record["source_row_sha256"]:
            raise RuntimeError("repair_source_hash_mismatch:" + record_id)
        # Hash-bind each proposal to the exact primary vote it was based on.
        batch_review = batch_reviews[record_id]
        if digest(batch_review) != review_hash:
            raise RuntimeError("repair_primary_review_hash_mismatch:" + record_id)
        if source_review["disposition"] not in {"repair_candidate", "quarantine_pending_repair"} and not (
                source_review["disposition"] == "retain_candidate"
                and origin in {"repair-drafts-v5-private.jsonl", "repair-drafts-v6-private.jsonl",
                               "repair-drafts-v7-private.jsonl", "repair-drafts-v8-private.jsonl"}):
            excluded.append({"record_id": record_id,
                             "reason": "superseded_or_not_currently_repair_candidate",
                             "effective_disposition": source_review["disposition"],
                             "origin": origin})
            return
        old_q = question(original)
        if declared_group is not None:
            actual = sorted(rid for rid, row in by_id.items() if question(row) == old_q)
            if sorted(declared_group) != actual:
                raise RuntimeError("declared_question_group_mismatch:" + record_id)
        actual_group = tuple(sorted(rid for rid, row in by_id.items() if question(row) == old_q))
        if revised_q and not revised_q.strip():
            raise RuntimeError("empty_revised_question:" + record_id)
        proposals[actual_group].append({"record_id": record_id,
            "revised_question": (revised_q or old_q) if (revised_q or old_q) != old_q else None,
            "revised_action": revised_action or target_action(original),
            "target_action_changed": revised_action is not None and revised_action != target_action(original),
            "source_disposition": source_review["disposition"],
            "origin": origin})

    # Legacy draft files are immutable candidate inputs; revisions are merged only
    # after their parent row and primary-review hashes validate.
    amendments = json.loads((OUT / "repair-draft-amendments-v5-private.json").read_text(encoding="utf-8"))
    retired = {}
    for amendment in amendments["amendments"]:
        if amendment["status"] not in {"retire_repair_query_candidate_keep_source_quarantined",
                                       "superseded_by_v5_group_revision"}:
            raise RuntimeError("unknown_repair_draft_amendment_status")
        retired[amendment["record_id"]] = amendment
    draft_filenames = ("repair-drafts-private.jsonl", "repair-drafts-v2-private.jsonl",
                       "repair-drafts-v3-private.jsonl", "repair-drafts-v4-private.jsonl",
                       "repair-drafts-v5-private.jsonl", "repair-drafts-v6-private.jsonl",
                       "repair-drafts-v7-private.jsonl", "repair-drafts-v8-private.jsonl")
    for filename in draft_filenames[:4]:
        for draft in readl(OUT / filename):
            if draft["record_id"] in retired:
                expected = retired[draft["record_id"]]["superseded_draft_sha256"]
                if digest(draft) != expected:
                    raise RuntimeError("retired_repair_draft_hash_mismatch:" + draft["record_id"])
                excluded.append({"record_id": draft["record_id"], "reason": retired[draft["record_id"]]["reason"],
                                 "origin": filename, "status": "superseded_repair_proposal"})
                continue
            revised = draft["revised_row"]
            q = question(revised)
            proposed = target_action(revised)
            declared = draft.get("same_question_record_ids")
            add(draft["record_id"], q, proposed, draft["original_row_sha256"],
                draft["primary_review_sha256"], filename, declared)
    for filename in draft_filenames[4:]:
        for draft in readl(OUT / filename):
            revised = draft["revised_row"]
            add(draft["record_id"], question(revised), target_action(revised),
                draft["original_row_sha256"], draft["primary_review_sha256"], filename,
                draft.get("same_question_record_ids"))

    # Include the later source-scoped candidates. A differing proposal for the
    # same frozen question group becomes a hard conflict, never last-write-wins.
    cpu = json.loads((OUT / "v117-repair-candidates-cpu-private.json").read_text(encoding="utf-8"))
    for group in cpu["candidates"]:
        for state in group["states"]:
            if state["source_record_sha256"] != inventory[state["record_id"]]["record_sha256"]:
                raise RuntimeError("cpu_candidate_inventory_hash_mismatch:" + state["record_id"])
            add(state["record_id"], state["revised_question"], None,
                digest(by_id[state["record_id"]]), digest(batch_reviews[state["record_id"]]),
                "v117-repair-candidates-cpu-private.json",
                group["record_ids"])

    corpus_hash = sha(CORPUS)
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    if corpus_hash != manifest["files"]["training-corpus.jsonl"]:
        raise RuntimeError("training_corpus_manifest_hash_mismatch")
    corpus = readl(CORPUS)
    index = _BM25Index(corpus)
    processor = AutoProcessor.from_pretrained(str(PROCESSOR), local_files_only=True)
    lineage = {r["record_id"]: r for r in readl(OUT / "lineage-private.jsonl")}
    replay_rows, group_reports = [], []
    legacy_searches = [r for r in inventory_rows if r["cohort"] == "v117_train"
                       and r["proposed_action"]["action"] == "search"]
    legacy_exact = 0
    for row in legacy_searches:
        prior = lineage[row["record_id"]]["matches"]
        query = row["proposed_action"]["query"]
        got = [{k: d.get(k) for k in ("id", "title", "text")} for d in index.search(query, top_k=3)]
        if prior and got == prior[0].get("retrieved_after_search"):
            legacy_exact += 1
    if legacy_exact != len(legacy_searches):
        raise RuntimeError("frozen_training_corpus_does_not_reproduce_all_legacy_searches")
    for ids, entries in sorted(proposals.items(), key=lambda item: item[0]):
        questions = {e["revised_question"] for e in entries if e["revised_question"] is not None}
        if not questions:
            questions = {question(by_id[ids[0]])}
        actions_by_id = defaultdict(list)
        for entry in entries:
            if entry["target_action_changed"]:
                actions_by_id[entry["record_id"]].append(entry["revised_action"])
        action_conflicts = {rid: actions for rid, actions in actions_by_id.items()
                            if len({json.dumps(a, sort_keys=True, ensure_ascii=False) for a in actions}) > 1}
        if len(questions) != 1 or action_conflicts:
            group_reports.append({"record_ids": list(ids), "status": "conflict_blocked",
                "revised_question_variants": len(questions),
                "action_conflict_record_ids": sorted(action_conflicts),
                "draft_origins": sorted({e["origin"] for e in entries}),
                "training_allowed": False})
            continue
        revised_q = next(iter(questions))
        edit_actions = {rid: actions[0] for rid, actions in actions_by_id.items()}
        tokens = []
        retrievals = []
        group_ok = True
        for rid in ids:
            original = by_id[rid]
            row = json.loads(json.dumps(original, ensure_ascii=False))
            row["messages"][1]["content"] = revised_q
            if rid in edit_actions:
                row["messages"][row["target_assistant_index"]]["content"] = json.dumps(
                    edit_actions[rid], ensure_ascii=False, separators=(",", ":"))
            target_idx = row["target_assistant_index"]
            prompt = row["messages"][:target_idx]
            prompt_ids = render(processor, [SimpleNamespace(**m) for m in prompt])
            target_msg = {"role": "assistant", "content": row["messages"][target_idx]["content"]}
            encoded = encode_assistant_only(processor, prompt + [target_msg], 2176,
                                            len(prompt), False)
            target_tokens = sum(token != -100 for token in encoded["labels"])
            within = len(prompt_ids) <= 2048 and target_tokens <= 128 and len(encoded["input_ids"]) <= 2176
            if not within:
                group_ok = False
            tokens.append({"record_id": rid, "prompt_tokens": len(prompt_ids),
                "target_tokens": target_tokens, "total_tokens": len(encoded["input_ids"]),
                "within_limits": within,
                "prompt_ids_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
                "target_action": target_action(row),
                "target_mask_sha256": hashlib.sha256(json.dumps(encoded["labels"]).encode()).hexdigest()})
            action = target_action(row)
            if action["action"] == "search":
                docs = index.search(action["query"], top_k=3)
                projection = [{k: d.get(k) for k in ("id", "title", "text")} for d in docs]
                original_query = target_action(original)["query"]
                prior = lineage[rid]["matches"]
                baseline_exact = bool(prior) and projection == prior[0].get("retrieved_after_search")
                retrievals.append({"record_id": rid, "query_changed": action["query"] != original_query,
                    "query": action["query"], "results": projection,
                    "unchanged_query_exact_to_legacy_lineage": baseline_exact})
                if action["query"] == original_query and not baseline_exact:
                    group_ok = False
        replay_rows.append({"question": revised_q, "record_ids": list(ids), "states": tokens,
                            "search_replays": retrievals,
                            "revised_search_requires_downstream_context_rebuild": any(r["query_changed"] for r in retrievals),
                            "training_allowed": False})
        group_reports.append({"record_ids": list(ids), "status": "cpu_replay_complete_pending_semantic_double_review",
            "revised_question": revised_q, "state_count": len(ids),
            "source_dispositions": {rid: primary[rid]["disposition"] for rid in ids},
            "token_limits_passed": group_ok and all(s["within_limits"] for s in tokens),
            "search_states": len(retrievals),
            "query_changes": sum(r["query_changed"] for r in retrievals),
            "baseline_unchanged_query_replays_exact": all(
                r["unchanged_query_exact_to_legacy_lineage"] for r in retrievals if not r["query_changed"]),
            "downstream_context_rebuild_required": any(r["query_changed"] for r in retrievals),
            "draft_origins": sorted({e["origin"] for e in entries}),
            "training_allowed": False})
    report = {"version": "v124-repair-draft-reconciliation-v1",
        "source_sha256": sha(TRAIN), "primary_ledger_sha256": sha(OUT / "primary-verification.json"),
        "corpus_sha256": corpus_hash, "corpus_documents": len(corpus),
        "legacy_search_replay_validation": {"search_states": len(legacy_searches), "exact_top3": legacy_exact,
            "basis": "independently recomputed against manifest-hash-verified v114 training corpus"},
        "eligible_revision_groups": len(replay_rows),
        "state_records_replayed": sum(len(r["states"]) for r in replay_rows),
        "groups": group_reports, "excluded": excluded,
        "semantic_review": "not_run", "independent_double_review": "not_run",
        "training_allowed": False}
    write(OUTPUT, report)
    OUTPUT_PACKET_TMP = PACKET_OUTPUT.with_suffix(".jsonl.tmp")
    OUTPUT_PACKET_TMP.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                                        for r in replay_rows), encoding="utf-8")
    OUTPUT_PACKET_TMP.replace(PACKET_OUTPUT)
    print(json.dumps({"eligible_revision_groups": len(replay_rows),
        "states_replayed": report["state_records_replayed"],
        "groups_conflicted": sum(r["status"] == "conflict_blocked" for r in group_reports),
        "excluded_superseded_or_not_repair": len(excluded),
        "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
