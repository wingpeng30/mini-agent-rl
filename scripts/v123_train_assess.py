"""Read-only assessment of the existing v122 training supplement.

The supplement is never promoted while the v123 internal set is unfinished.
No model, API, or training entry point is imported here.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from scripts.v123_cpu_core import ROOT, OUT, digest, norm, read, readl, sha, write

SOURCE = ROOT / "data/evidence-sft-v122-train-candidates"
REVIEW = SOURCE / "review"
REPORT = OUT / "train/assessment.json"


def assess() -> dict:
    from scripts.v123_api_protocol import valid_vote

    manifest = read(REVIEW / "manifest.json")
    frozen = read(SOURCE / "candidate-freeze.json")
    replay = read(SOURCE / "token-replay.json")
    packets = readl(REVIEW / "internal-packets.jsonl")
    if len(packets) != 212 or len({p["case_id"] for p in packets}) != len(packets):
        raise ValueError("v122_train_packet_count_or_duplicates")
    if sha(REVIEW / "internal-packets.jsonl") != manifest["packet_sha256"] or not replay["all_exact"] or replay["packet_sha256"] != manifest["packet_sha256"]:
        raise ValueError("v122_train_packet_or_token_replay_changed")
    if sha(SOURCE / "candidate-corpus-private.json") != frozen["corpus_sha256"]:
        raise ValueError("v122_train_corpus_changed")
    source_map = {case: row["source_id"] for row in readl(REVIEW / "eligible-private.jsonl") for case in row["case_ids"]}
    if set(source_map) != {p["case_id"] for p in packets}:
        raise ValueError("train_source_map_invalid")
    decisions = {}
    counts = Counter()
    approved_by_question = defaultdict(list)
    for name, spec in manifest["batches"].items():
        batch = readl(REVIEW / name)
        api = readl(REVIEW / ("api-" + name))
        if len(batch) != spec["states"] or len(batch) > 20 or sha(REVIEW / name) != spec["sha256"] or len(api) != len(batch):
            raise ValueError("train_batch_or_api_coverage_invalid")
        if {r["case_id"] for r in batch} != {r["case_id"] for r in api}:
            raise ValueError("train_api_batch_case_mismatch")
        for row in api:
            case = row["case_id"]
            if case in decisions or row.get("input_file_sha256") != spec["sha256"]:
                raise ValueError("train_duplicate_or_unbound_api_vote")
            decisions[case] = row
    for p in packets:
        row = decisions[p["case_id"]]
        decision = row.get("decision")
        valid, reason = valid_vote(p, decision)
        valid = valid and row.get("packet_sha256") == digest(p)
        if not valid:
            counts["invalid_api"] += 1
        elif decision["approved"]:
            counts["approved_api"] += 1
            approved_by_question[source_map[p["case_id"]]].append((p, decision))
        else:
            counts["rejected_api"] += 1
    pairs = [(qid, approved_by_question[qid]) for qid in frozen["ids"] if
             len(approved_by_question[qid]) == 2 and
             {p["proposed_action"]["action"] for p, _ in approved_by_question[qid]} == {"search", "answer"}]
    risk = Counter(label for _, records in pairs for _, d in records for label in d["risk_labels"])
    # Generic relation votes are only an upper bound on the target behavior.
    mechanism_candidates = sum(bool(search.get("missing_fact", "").strip()) and
                               search.get("query_grounded") == "yes" and
                               search.get("retrieval_relevant") == "yes"
                               for _, records in pairs for p, search in records
                               if p["proposed_action"]["action"] == "search")
    internal_frozen = (OUT / "internal/frozen/manifest.json").exists()
    if internal_frozen:
        internal = read(OUT / "internal/frozen/manifest.json")
        internal_frozen = (internal.get("questions") == 50 and internal.get("states") == 100 and
                           internal.get("training_allowed") is False)
    all_internal_ids = set(read(OUT / "internal/candidate-freeze.json")["new_candidate_ids"])
    # The reused 45 are already excluded by the v122 training candidate provenance;
    # check the entire new candidate pool, not merely a selected final 50.
    all_internal_text = {norm(p["question"]) for p in readl(OUT / "internal/review/internal-packets.jsonl")}
    train_ids = set(frozen["ids"])
    train_text = {norm(p["question"]) for p in packets}
    isolated = not (all_internal_ids & train_ids or all_internal_text & train_text)
    alignment_path = OUT / "train/target-alignment.json"
    alignment = read(alignment_path) if alignment_path.exists() else {}
    approved_pair_ids = {qid for qid, _ in pairs}
    aligned_ids = alignment.get("pair_ids", [])
    alignment_valid = (alignment.get("status") == "passed" and
                       len(aligned_ids) >= 25 and len(aligned_ids) == len(set(aligned_ids)) and
                       set(aligned_ids) <= approved_pair_ids and
                       alignment.get("uses_development_gold") is False and
                       alignment.get("training_allowed") is False)
    reason = ("new_internal_not_frozen" if not internal_frozen else
              "training_target_alignment_not_verified" if not alignment_valid else None)
    result = {"version": "v123-train-assessment-v1", "status": "blocked" if reason else "ready_for_luna_only",
              "reason": reason,
              "candidate_states": len(packets), "candidate_questions": len(frozen["ids"]),
              "api_valid": counts["approved_api"] + counts["rejected_api"],
              "api_approved": counts["approved_api"], "api_rejected": counts["rejected_api"],
              "api_invalid": counts["invalid_api"], "api_approved_complete_pairs": len(pairs),
              "api_relation_risk_states_upper": risk["relation"],
              "mechanism_candidate_pairs_upper": mechanism_candidates,
              "target_alignment_valid": alignment_valid,
              "entire_new_internal_pool_isolated": isolated,
              "luna_review": "not_run", "training_allowed": False}
    if not isolated:
        result.update(status="failed", reason="training_internal_candidate_overlap")
    write(REPORT, result)
    return result
