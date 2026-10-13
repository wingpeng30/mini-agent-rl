"""Bounded, append-only review transport and v123 data gates."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.v123_cpu_core import ROOT, OUT, digest, norm, read, readl, sha, validate_selected_pairs, write, writel

INTERNAL = OUT / "internal"
REVIEW = INTERNAL / "review"
OLD = ROOT / "data/evidence-sft-v122-internal-candidates/review"


def packets_and_source():
    freeze = read(INTERNAL / "candidate-freeze.json")
    path = REVIEW / "internal-packets.jsonl"
    mapping_path = REVIEW / "source-map-private.jsonl"
    if sha(path) != freeze["packet_sha256"] or sha(mapping_path) != freeze["source_map_sha256"]:
        raise ValueError("frozen_packet_or_source_changed")
    tokens = read(INTERNAL / "token-replay.json")
    if not tokens.get("all_exact") or tokens.get("packet_sha256") != freeze["packet_sha256"]:
        raise ValueError("candidate_token_replay_missing_or_stale")
    packets = readl(path)
    if tokens.get("states") != len(packets):
        raise ValueError("candidate_token_replay_count_mismatch")
    mapping = {row["case_id"]: row["source_id"] for row in readl(mapping_path)}
    if len(mapping) != len(packets) or len({p["case_id"] for p in packets}) != len(packets) or set(mapping) != {p["case_id"] for p in packets}:
        raise ValueError("packet_source_coverage_or_duplicate")
    return freeze, packets, mapping


def _read_old_votes(prefix: str) -> dict:
    result = {}
    for path in sorted(OLD.glob(f"{prefix}-internal-batch-*.jsonl")):
        for row in readl(path):
            if row["case_id"] in result:
                raise ValueError("duplicate_old_review")
            result[row["case_id"]] = row
    return result


def reused_votes() -> dict[str, dict[str, dict]]:
    from scripts.v123_api_protocol import valid_vote
    freeze, packets, _ = packets_and_source()
    by_case = {p["case_id"]: p for p in packets}
    old_packets = {p["case_id"]: p for p in readl(OLD / "internal-packets.jsonl")}
    old_api = _read_old_votes("api")
    old_luna = _read_old_votes("luna")
    maps = readl(REVIEW / "old-vote-reuse-private.jsonl")
    if sha(REVIEW / "old-vote-reuse-private.jsonl") != freeze["old_vote_reuse_sha256"]:
        raise ValueError("old_reuse_map_changed")
    result = {"api": {}, "luna": {}}
    for item in maps:
        new_case, old_case = item["new_case_id"], item["old_case_id"]
        if new_case not in by_case or old_case not in old_packets or item["old_packet_sha256"] != old_packets[old_case]["packet_sha256"]:
            raise ValueError("old_vote_reuse_mapping_invalid")
        if item["new_packet_sha256"] != by_case[new_case]["packet_sha256"]:
            raise ValueError("new_reuse_packet_changed")
        for reviewer, source in (("api", old_api), ("luna", old_luna)):
            record = source.get(old_case)
            vote = record.get("decision") if reviewer == "api" and record else record
            if not record or not isinstance(vote, dict):
                continue
            expected_hash = digest(old_packets[old_case]) if reviewer == "api" else old_packets[old_case]["packet_sha256"]
            if record.get("packet_sha256") != expected_hash:
                continue
            old_valid, _ = valid_vote(old_packets[old_case], vote)
            rebound_vote = {**vote, "case_id": new_case}
            new_valid, _ = valid_vote(by_case[new_case], rebound_vote)
            if old_valid and new_valid and vote.get("approved") is True:
                result[reviewer][new_case] = {"case_id": new_case, "packet_sha256": digest(by_case[new_case]),
                                              "decision": rebound_vote, "reviewer": reviewer,
                                              "reuse_source_case": old_case,
                                              "reuse_source_packet_sha256": old_packets[old_case]["packet_sha256"],
                                              "reuse_source_vote_sha256": digest(record)}
    return result


def new_votes(reviewer: str) -> dict:
    if reviewer not in {"api", "luna"}:
        raise ValueError("reviewer_must_be_api_or_luna")
    path = REVIEW / f"{reviewer}-v123.jsonl"
    rows = readl(path) if path.exists() else []
    by_case = {r["case_id"]: r for r in rows}
    if len(by_case) != len(rows):
        raise ValueError("duplicate_v123_review")
    return by_case


def all_votes(reviewer: str, packets: list[dict]) -> dict:
    old = reused_votes()[reviewer]
    fresh = new_votes(reviewer)
    by_case = {p["case_id"]: p for p in packets}
    if set(fresh) & set(old) or not set(fresh) <= set(by_case):
        raise ValueError("repeated_or_unknown_review_case")
    for case, row in fresh.items():
        if row.get("packet_sha256") != digest(by_case[case]):
            raise ValueError("new_review_packet_binding_mismatch")
    return old | fresh


def valid_review(packet: dict, record: dict | None) -> tuple[bool, bool, list[str]]:
    from scripts.v123_api_protocol import valid_vote
    if not record:
        return False, False, []
    if record.get("packet_sha256") != digest(packet):
        return False, False, []
    vote = record.get("decision")
    valid, _ = valid_vote(packet, vote)
    if not valid:
        return False, False, []
    return True, vote["approved"] is True, vote["risk_labels"]


def status() -> dict:
    freeze, packets, source = packets_and_source()
    api, luna = all_votes("api", packets), all_votes("luna", packets)
    grouped = defaultdict(list)
    counts = Counter()
    for packet in packets:
        case = packet["case_id"]
        av, aa, ar = valid_review(packet, api.get(case))
        lv, la, lr = valid_review(packet, luna.get(case))
        if case in api:
            counts["api_records"] += 1
            counts["api_valid" if av else "api_invalid"] += 1
        if case in luna:
            counts["luna_records"] += 1
            counts["luna_valid" if lv else "luna_invalid"] += 1
        if av and lv and aa and la:
            grouped[source[case]].append((packet, sorted(set(ar) & set(lr))))
            counts["approved_states"] += 1
    ordered_ids = freeze["old_candidate_ids"] + freeze["new_candidate_ids"]
    pairs = [(qid, grouped[qid]) for qid in ordered_ids if len(grouped[qid]) == 2 and
             {p["proposed_action"]["action"] for p, _ in grouped[qid]} == {"search", "answer"}]
    risk = Counter(label for _, records in pairs for _, labels in records for label in labels)
    all_groups = defaultdict(list)
    for packet in packets:
        all_groups[source[packet["case_id"]]].append(packet)

    def may_pass(packet):
        for side in (api, luna):
            record = side.get(packet["case_id"])
            if record is None:
                continue
            valid, approved, _ = valid_review(packet, record)
            if not valid or not approved:
                return False
        return True

    possible = [records for records in all_groups.values() if len(records) == 2 and
                {p["proposed_action"]["action"] for p in records} == {"search", "answer"} and
                all(may_pass(p) for p in records)]
    risk_upper = Counter()
    for records in possible:
        for packet in records:
            for label in ("relation", "time", "scope", "measurement"):
                if all(not valid or (approved and label in labels) for valid, approved, labels in
                       (valid_review(packet, side.get(packet["case_id"])) for side in (api, luna))):
                    risk_upper[label] += 1
    impossible_pairs = len(possible) < 50
    impossible_risk = any(risk_upper[r] < 5 for r in ("relation", "time", "scope", "measurement"))
    report = {"version": "v123-internal-review-status-v1", "candidate_states": len(packets),
              "candidate_questions": len(ordered_ids), "counts": dict(counts),
              "complete_double_approved_pairs": len(pairs), "complete_pair_risks": dict(risk),
              "possible_complete_pairs_upper": len(possible),
              "possible_risk_state_upper": dict(risk_upper),
              "stop_for_risk_bound": impossible_risk,
              "stop_for_pair_bound": impossible_pairs,
              "review_complete": len(pairs) >= 50 and all(risk[r] >= 5 for r in ("relation", "time", "scope", "measurement")),
              "training_allowed": False}
    return report


def next_batch(reviewer: str) -> dict:
    from scripts.v123_prepare_internal import provisional_scope
    freeze, packets, source = packets_and_source()
    current = status()
    if current["stop_for_risk_bound"] or current["stop_for_pair_bound"] or current["review_complete"]:
        return {"status": "stopped", "reason": "goal_met_or_upper_bound_failed", "cases": 0,
                "training_allowed": False}
    votes = all_votes(reviewer, packets)
    other = all_votes("luna" if reviewer == "api" else "api", packets)
    groups = defaultdict(list)
    for packet in packets:
        groups[source[packet["case_id"]]].append(packet)
    ids = freeze["old_candidate_ids"] + freeze["new_candidate_ids"]
    # Schedule by provisional question features, never by model scores or reviewer preferences.
    position = {qid: n for n, qid in enumerate(ids)}
    ids = sorted((qid for qid in ids if len(groups.get(qid, [])) == 2),
                 key=lambda q: (not provisional_scope(groups[q][0]["question"]), position[q]))
    rows = []
    for qid in ids:
        pair = groups.get(qid, [])
        if len(pair) != 2:
            continue
        if reviewer == "luna" and any(not valid_review(p, other.get(p["case_id"]))[:2] == (True, True) for p in pair):
            continue
        pending = [p for p in pair if p["case_id"] not in votes]
        if len(rows) + len(pending) > 20:
            break
        rows.extend(pending)
        if len(rows) == 20:
            break
    if not rows:
        return {"status": "blocked", "reason": "no_eligible_unreviewed_complete_pairs", "cases": 0,
                "training_allowed": False}
    batch_id = hashlib.sha256((reviewer + freeze["packet_sha256"] + ":" +
                               ",".join(p["case_id"] for p in rows)).encode()).hexdigest()[:12]
    # The transport packet omits model scores, gold fields, provisional risk tags and provenance.
    public = [{key: p[key] for key in ("case_id", "question", "proposed_action",
                                       "before_action", "retrieved_after_search")} for p in rows]
    path = REVIEW / f"work-{reviewer}-{batch_id}.jsonl"
    if path.exists() and readl(path) != public:
        raise ValueError("existing_review_batch_changed")
    if not path.exists():
        writel(path, public)
    result = {"version": "v123-review-batch-v1", "reviewer": reviewer, "batch": path.name,
              "cases": len(rows), "packet_file_sha256": sha(path),
              "private_case_ids_sha256": digest([p["case_id"] for p in rows]),
              "training_allowed": False}
    write(path.with_suffix(".manifest.json"), result)
    return result


def append_vote(reviewer: str, case: str, decision: dict | None, packet_hash: str,
                model: str, attempts: list[dict]) -> None:
    _, packets, _ = packets_and_source()
    by_case = {p["case_id"]: p for p in packets}
    if case not in by_case or packet_hash != digest(by_case[case]):
        raise ValueError("review_case_binding_invalid")
    if case in all_votes(reviewer, packets):
        raise ValueError("review_already_recorded")
    if reviewer == "luna":
        from scripts.v123_api_protocol import valid_vote
        if "luna" not in model.casefold() or not valid_vote(by_case[case], decision)[0]:
            raise ValueError("luna_model_or_decision_invalid")
    record = {"case_id": case, "packet_sha256": packet_hash, "reviewer": reviewer,
              "model": model, "decision": decision, "attempts": attempts}
    path = REVIEW / f"{reviewer}-v123.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def finalize_internal() -> dict:
    from scripts.v123_cpu_core import export_packet, select_risk_pairs
    from mini_agent_rl.tools.search import _BM25Index
    freeze, packets, source = packets_and_source()
    current = status()
    if not current["review_complete"]:
        return {"status": "blocked", "reason": "double_review_and_risk_gate_not_met",
                "review": current, "training_allowed": False}
    api, luna = all_votes("api", packets), all_votes("luna", packets)
    grouped = defaultdict(list)
    for p in packets:
        av, aa, ar = valid_review(p, api.get(p["case_id"]))
        lv, la, lr = valid_review(p, luna.get(p["case_id"]))
        if av and aa and lv and la:
            grouped[source[p["case_id"]]].append((p, sorted(set(ar) & set(lr))))
    ordered = [(qid, grouped[qid]) for qid in freeze["old_candidate_ids"] + freeze["new_candidate_ids"]
               if len(grouped[qid]) == 2]
    selected = select_risk_pairs(ordered)
    if not selected:
        return {"status": "blocked", "reason": "no_feasible_risk_balanced_selection", "training_allowed": False}
    history = read(ROOT / "data/evidence-sft-v122-internal-candidates/history-private.json")
    train = readl(ROOT / "data/evidence-sft-v117/frozen/train.jsonl")
    training_text = {norm(next(m["content"] for m in row["messages"] if m["role"] == "user")) for row in train}
    packets_selected = [p for _, records in selected for p, _ in records]
    risks = {p["case_id"]: labels for _, records in selected for p, labels in records}
    gate = validate_selected_pairs(packets_selected, source, [qid for qid, _ in selected], risks,
                                   training_text, set(history["ids"]), set(history["normalized_questions"]))
    index = _BM25Index(read(INTERNAL / "candidate-corpus-private.json"))
    gate["checks"]["selected_top3_replay_exact"] = all(
        [{k: d[k] for k in ("id", "title", "text")} for d in index.search(p["proposed_action"]["query"], top_k=3)] ==
        [{k: d[k] for k in ("id", "title", "text")} for d in p["retrieved_after_search"]]
        for p in packets_selected if p["proposed_action"]["action"] == "search")
    gate["passed"] = all(gate["checks"].values())
    if not gate["passed"]:
        write(INTERNAL / "gate.json", gate)
        return {"status": "failed", "gate": gate, "training_allowed": False}
    rows = [export_packet(p) for p in packets_selected]
    path = INTERNAL / "frozen/internal.jsonl"
    if path.exists():
        raise ValueError("internal_already_frozen")
    writel(path, rows)
    write(INTERNAL / "frozen/internal-corpus.json", read(INTERNAL / "candidate-corpus-private.json"))
    if sha(INTERNAL / "frozen/internal-corpus.json") != freeze["corpus_sha256"]:
        raise ValueError("frozen_corpus_copy_mismatch")
    write(INTERNAL / "frozen/manifest.json", {"version": "v123-internal-freeze-v1",
          "questions": 50, "states": 100, "selected_ids_sha256": digest([q for q, _ in selected]),
          "internal_sha256": sha(path), "corpus_sha256": freeze["corpus_sha256"],
          "gate_sha256": digest(gate), "training_allowed": False,
          "remote_token_replay": "not_run"})
    write(INTERNAL / "gate.json", gate)
    return {"status": "passed", "gate": gate, "training_allowed": False}
