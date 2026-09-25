"""Fail-closed merge of independent v117 audits and frozen local data export."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.build_v114_sft_candidates import norm
from scripts.finalize_v116 import export, valid_vote
from scripts.prepare_v117 import OUT, ROOT
from scripts.prepare_v117_review import loadl, sha, write, writel
from scripts.prepare_v116_data import full_rows
from mini_agent_rl.tools.search import _BM25Index

REVIEW = OUT / "review"
RISKS = ("relation", "time", "scope", "measurement")


def valid_v117_vote(packet, vote):
    valid, reason = valid_vote(packet, vote)
    if not valid or not vote.get("approved"):
        return valid, reason
    if packet["proposed_action"]["action"] == "search":
        stages = {quote.get("stage") for quote in vote["citations"]}
        if stages != {"before_action", "retrieved_after_search"}:
            return False, "search_needs_before_and_after_quotes"
        if not vote.get("missing_fact", "").strip():
            return False, "search_gap_missing"
    return True, "valid"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def reviewer_records(prefix, batches, allowed_names=None):
    votes = {}
    for name, spec in batches.items():
        if allowed_names is not None and name not in allowed_names:
            continue
        path = REVIEW / f"{prefix}-{name}"
        if not path.exists():
            continue
        rows = loadl(path)
        for row in rows:
            key = row.get("case_id")
            if not key or key in votes:
                raise ValueError("duplicate_or_missing_reviewer_case")
            votes[key] = row
    for name in batches:
        if allowed_names is not None and name not in allowed_names:
            continue
        sidecar = REVIEW / f"{prefix}-retry-{name}"
        if prefix != "api" or not sidecar.exists():
            continue
        seen_retry = set()
        for row in loadl(sidecar):
            key = row.get("case_id")
            if key in seen_retry or key not in votes or votes[key].get("decision") is not None:
                raise ValueError("invalid_api_retry")
            seen_retry.add(key)
            if row.get("decision") is not None:
                votes[key] = row
    if prefix == "luna":
        for filename in ("luna-corrections.jsonl", "luna-risk-corrections.jsonl",
                         "luna-taxonomy-corrections.jsonl"):
            corrections = REVIEW / filename
            seen = set()
            if corrections.exists():
                for row in loadl(corrections):
                    key = row.get("case_id")
                    if key not in votes or key in seen or not row.get("correction_reason"):
                        raise ValueError("invalid_luna_correction")
                    seen.add(key)
                    votes[key] = row
    return votes


def select_pairs(pairs, count=50):
    """Choose the lexicographically first feasible hash-ordered 50-question set."""
    ordered = sorted(pairs, key=lambda x: hashlib.sha256(("v117-final:" + x[0]).encode()).hexdigest())
    n = len(ordered)
    if n < count:
        return None
    risk = [Counter(tag for _, labels in records for tag in labels) for _, records in ordered]
    suffix = [Counter() for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + risk[i]
    selected = []
    failed = set()

    def visit(i, needs):
        if len(selected) == count:
            return all(x == 0 for x in needs)
        if n - i < count - len(selected) or any(suffix[i][r] < needs[j] for j, r in enumerate(RISKS)):
            return False
        key = (i, count - len(selected), needs)
        if key in failed:
            return False
        if i < n:
            selected.append(ordered[i])
            new_needs = tuple(max(0, needs[j] - risk[i][r]) for j, r in enumerate(RISKS))
            if visit(i + 1, new_needs):
                return True
            selected.pop()
            if visit(i + 1, needs):
                return True
        failed.add(key)
        return False

    return list(selected) if visit(0, (5, 5, 5, 5)) else None


def main():
    manifest = json.loads((REVIEW / "manifest.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "candidate-freeze.json").read_text(encoding="utf-8"))
    inventory = json.loads((OUT / "history-inventory.json").read_text(encoding="utf-8"))
    if not inventory["ready"] or inventory["failures"]:
        raise RuntimeError("history_gate_failed")
    scratch = json.loads((OUT / "history-scratch-verification.json").read_text(encoding="utf-8"))
    expected_scratch = {"reports/v113-stage-oj4awatt", "backups/v113-stage-nqptz9px"}
    if (set(scratch.get("empty_paths", [])) != expected_scratch or scratch.get("child_counts") != [0, 0]
            or not expected_scratch <= set(inventory.get("excluded_scratch_paths", []))):
        raise RuntimeError("unverified_historical_scratch_exclusion")
    if sha(OUT / "candidate-corpus-private.json") != manifest["corpus_sha256"] != freeze["corpus_sha256"]:
        raise RuntimeError("corpus_hash_mismatch")
    if sha(REVIEW / "internal-packets.jsonl") != manifest["packet_sha256"]:
        raise RuntimeError("packet_hash_mismatch")
    token_report = json.loads((OUT / "token-replay.json").read_text(encoding="utf-8"))
    if (not token_report.get("all_exact") or token_report.get("packet_sha256") != manifest["packet_sha256"]
            or token_report.get("internal_states") != manifest["states"]):
        raise RuntimeError("local_token_replay_missing_or_stale")
    for name, spec in manifest["batches"].items():
        file = REVIEW / name
        if sha(file) != spec["sha256"] or len(loadl(file)) != spec["states"] or spec["states"] > 20:
            raise RuntimeError("batch_hash_or_size_mismatch")
    private = loadl(REVIEW / "eligible-private.jsonl")
    ids = {case: row["source_id"] for row in private for case in row["case_ids"]}
    packets = loadl(REVIEW / "internal-packets.jsonl")
    api_config = json.loads((REVIEW / "api-config.json").read_text(encoding="utf-8"))
    if api_config.get("host") != "api.deepseek.com" or api_config.get("credential_exported") is not False:
        raise RuntimeError("api_review_config_invalid")
    if (len(packets) != manifest["states"] or len({p["case_id"] for p in packets}) != len(packets)
            or len(ids) != len(packets) or set(ids) != {p["case_id"] for p in packets}):
        raise RuntimeError("private_packet_mapping_mismatch")
    stop_path = REVIEW / "audit-stop.json"
    stop = json.loads(stop_path.read_text(encoding="utf-8")) if stop_path.exists() else None
    allowed_luna = None
    if stop is not None:
        if stop.get("last_batch") != 12 or stop.get("audited_states") != 240 or stop.get("later_batches_used") is not False:
            raise RuntimeError("invalid_audit_stop")
        allowed_luna = set(list(manifest["batches"])[:12])
        for name in allowed_luna:
            if sha(REVIEW / ("luna-" + name)) != stop["luna_batch_sha256"][name]:
                raise RuntimeError("stopped_luna_batch_changed")
    luna = reviewer_records("luna", manifest["batches"], allowed_luna)
    api = reviewer_records("api", manifest["batches"])
    approved = defaultdict(list)
    details = []
    for p in packets:
        key = p["case_id"]
        a = luna.get(key)
        b = api.get(key)
        decision = b.get("decision") if b else None
        av, ar = valid_v117_vote(p, a)
        bv, br = valid_v117_vote(p, decision)
        if a and a.get("packet_sha256") != p["packet_sha256"]:
            av, ar = False, "luna_packet_mismatch"
        batch = next((name for name in manifest["batches"]
                      if key in {x["case_id"] for x in loadl(REVIEW / name)}), None)
        if b and (b.get("packet_sha256") != digest(p) or
                  b.get("input_file_sha256") != manifest["batches"][batch]["sha256"] or
                  b.get("system_prompt_sha256") != api_config["system_prompt_sha256"] or
                  b.get("reviewer") != api_config["model"] + "-independent"):
            bv, br = False, "api_packet_or_batch_mismatch"
        accepted = bool(av and bv and a["approved"] and decision["approved"])
        risks = sorted(set(a["risk_labels"]) & set(decision["risk_labels"])) if accepted else []
        if accepted:
            approved[ids[key]].append((p, risks))
        details.append({"case_id": key, "source_id": ids[key], "accepted": accepted,
                        "luna": ar, "api": br, "luna_approved": a.get("approved") if av else None,
                        "api_approved": decision.get("approved") if bv else None,
                        "risk_labels": risks, "packet_sha256": p["packet_sha256"]})
    pairs = [(qid, records) for qid, records in approved.items()
             if len(records) == 2 and {p["proposed_action"]["action"] for p, _ in records} == {"search", "answer"}]
    chosen = select_pairs(pairs)
    if stop is not None:
        if chosen is None or len(chosen) != 50:
            raise RuntimeError("stopped_selection_no_longer_valid")
        encoded_ids = json.dumps([qid for qid, _ in chosen], ensure_ascii=False, separators=(",", ":")).encode()
        encoded_cases = json.dumps([p["case_id"] for _, records in chosen for p, _ in records],
                                   ensure_ascii=False, separators=(",", ":")).encode()
        if (hashlib.sha256(encoded_ids).hexdigest() != stop["selected_question_ids_sha256"] or
                hashlib.sha256(encoded_cases).hexdigest() != stop["selected_case_ids_sha256"]):
            raise RuntimeError("stopped_selection_changed")
    pool_risks = Counter(tag for _, records in pairs for _, labels in records for tag in labels)
    train = ROOT / "data/evidence-sft-v116/staged-train/train-input-target.jsonl"
    prior_archive = json.loads((ROOT / "backups/v116-pre-sft-manifest.json").read_text(encoding="utf-8"))
    if sha(ROOT / "backups/v116-pre-sft.zip") != prior_archive["archive_sha256"]:
        raise RuntimeError("v116_lineage_backup_changed")
    for old_path in (train, train.parent / "manifest.json", ROOT / "reports/v116-sft-preflight.json",
                     ROOT / "data/evidence-sft-v116/token-replay.json"):
        name = old_path.relative_to(ROOT).as_posix()
        if sha(old_path) != prior_archive["files"][name]["sha256"]:
            raise RuntimeError("v116_training_lineage_changed:" + name)
    train_manifest = json.loads((train.parent / "manifest.json").read_text(encoding="utf-8"))
    old_gate = json.loads((ROOT / "reports/v116-sft-preflight.json").read_text(encoding="utf-8"))
    train_ok = sha(train) == train_manifest["sha256"] and old_gate["train_data_only_passed"] and len(loadl(train)) == 345
    history = json.loads((OUT / "history-private.json").read_text(encoding="utf-8"))
    train_qids = set()
    train_questions = set()
    for row in loadl(ROOT / "data/evidence-sft-v114-review/accepted.jsonl"):
        if row["split"] == "train":
            train_qids.add(row["source_id"])
            train_questions.add(norm(row["packet"]["question"]))
    for row in loadl(ROOT / "data/evidence-sft-v116/train-supplement-private.jsonl"):
        if row["case_id"] in {r["case_id"] for r in old_gate["review"]["train"] if r["approved"]}:
            train_qids.add(row["source_id"])
            train_questions.add(norm(row["packet"]["question"]))
    chosen_qids = {qid for qid, _ in chosen or []}
    selected_questions = {norm(p["question"]) for _, records in chosen or [] for p, _ in records}
    isolated = not (chosen_qids & train_qids or selected_questions & train_questions or
                    chosen_qids & set(history["ids"]) or selected_questions & set(history["normalized_questions"]))
    risk_counts = Counter(tag for _, records in chosen or [] for _, labels in records for tag in labels)
    retrieval_exact = False
    if chosen is not None:
        index = _BM25Index(json.loads((OUT / "candidate-corpus-private.json").read_text(encoding="utf-8")))
        selected_searches = [p for _, records in chosen for p, _ in records
                             if p["proposed_action"]["action"] == "search"]
        def project(docs):
            return [{k: doc[k] for k in ("id", "title", "text")} for doc in docs]
        retrieval_exact = len(selected_searches) == 50 and all(
            project(index.search(p["proposed_action"]["query"], top_k=3)) == p["retrieved_after_search"]
            for p in selected_searches)
    reject_counts = Counter()
    for row in details:
        if row["accepted"]:
            continue
        if row["luna"] != "valid" or row["api"] != "valid":
            reject_counts["missing_or_invalid_review"] += 1
        elif row["luna_approved"] != row["api_approved"]:
            reject_counts["reviewer_disagreement"] += 1
        else:
            reject_counts["both_rejected"] += 1
    checks = {"history_complete": inventory["ready"], "fixed_corpus_hash": True,
              "reused_train_valid": train_ok, "train_345_states_186_questions_75_pairs":
              old_gate["train_states"] == 345 and old_gate["train_questions"] == 186 and old_gate["train_pairs"] == 75,
              "train_actions_and_risks": all(old_gate["checks"].get(k, False) for k in
                  ("answer_45_to_55_percent", "nonempty_search_at_least_40_percent", "empty_search_at_most_10_percent",
                   "train_four_risks_at_least_20")),
              "internal_50_complete_pairs": chosen is not None and len(chosen) == 50,
              "internal_risk_5_each": all(risk_counts[r] >= 5 for r in RISKS),
              "selected_search_top3_exact": retrieval_exact,
              "question_isolation": isolated,
              "unique_states": len({digest(export(p)) for _, records in chosen or [] for p, _ in records}) ==
                               2 * len(chosen or [])}
    report = {"version": "v117-local-preflight-v1", "checks": checks, "local_data_ready": all(checks.values()),
              "training_allowed": False, "remote_preflight_pending": True,
              "training_started": False, "independent100_accessed": False, "final_test_accessed": False,
              "candidate_questions": freeze["source_count"], "audited_states": len(luna),
              "api_reviewed_states": len(api), "complete_approved_pairs": len(pairs),
              "complete_pair_pool_risks": dict(pool_risks),
              "rejected_state_categories": dict(reject_counts),
              "selected_pairs": len(chosen or []), "selected_risks": dict(risk_counts),
              "gaps": {"pairs_needed": max(0, 50-len(pairs)),
                       "scope_states_needed_from_pool": max(0, 5-pool_risks["scope"]),
                       "risk_states_needed_from_pool": {r: max(0, 5-pool_risks[r]) for r in RISKS},
                       "unreviewed_states": len(packets)-min(len(luna), len(api))}}
    write(ROOT / "reports/v117-local-preflight.json", report)
    writel(REVIEW / "merge-details.jsonl", details)
    if report["local_data_ready"]:
        frozen = OUT / "frozen"
        frozen.mkdir(exist_ok=True)
        exports = [export(p) for _, records in chosen for p, _ in records]
        writel(frozen / "internal.jsonl", exports)
        writel(frozen / "train.jsonl", loadl(train))
        source_rows = full_rows([qid for qid, _ in chosen])
        tasks = []
        states = []
        for qid, records in chosen:
            source = source_rows[qid]
            tasks.append({"id": qid, "question": source["question"], "answer": source["answer"],
                          "metadata": {"source": "hotpotqa-official-train", "source_id": qid,
                                       "support_titles": list(dict.fromkeys(source["supporting_facts"]["title"]))}})
            for p, _ in records:
                states.append({"state_id": p["case_id"], "task_id": qid, "question": p["question"],
                               "messages": p["visible_messages"], "target": p["proposed_action"],
                               "evidence": p["before_action"],
                               "retrieval_top3": p.get("retrieved_after_search") or []})
        writel(frozen / "internal-tasks-private.jsonl", tasks)
        writel(frozen / "internal-states-private.jsonl", states)
        corpus_path = frozen / "internal-corpus.json"
        corpus_path.write_bytes((OUT / "candidate-corpus-private.json").read_bytes())
        write(frozen / "internal-environment-manifest.json", {
              "version": "v117-internal-environment-v1", "candidate_corpus_sha256": freeze["corpus_sha256"],
              "tasks_sha256": sha(frozen / "internal-tasks-private.jsonl"),
              "states_sha256": sha(frozen / "internal-states-private.jsonl"),
              "corpus_sha256": sha(corpus_path), "question_count": 50,
              "state_count": 100, "document_count": freeze["documents"],
              "retriever": "bm25-v2 top3", "source": "official train only",
              "model_scores_used_for_selection": False})
        write(frozen / "internal-private.json", {"question_ids": [qid for qid, _ in chosen],
              "risk_counts": dict(risk_counts), "case_ids": [p["case_id"] for _, records in chosen for p, _ in records]})
        write(frozen / "manifest.json", {"version": "v117-frozen-local-data-v1",
              "train_sha256": sha(frozen / "train.jsonl"), "internal_sha256": sha(frozen / "internal.jsonl"),
              "corpus_sha256": freeze["corpus_sha256"], "gate_sha256": sha(ROOT / "reports/v117-local-preflight.json"),
              "environment_manifest_sha256": sha(frozen / "internal-environment-manifest.json"),
              "train_states": 345, "internal_states": 100,
              "remote_preflight_pending": True, "training_started": False})
    print(json.dumps({"complete_approved_pairs": len(pairs), "selected": len(chosen or []),
                      "risks": dict(risk_counts), "local_data_ready": report["local_data_ready"],
                      "gaps": report["gaps"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
