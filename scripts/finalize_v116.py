"""Fail-closed v116 SFT preparation gate; never starts a trainer."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.build_v114_sft_candidates import norm
from scripts.review_v116_api import validate_citations

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/evidence-sft-v114-review"
WORK = ROOT / "data/evidence-sft-v116"
REVIEW = WORK / "review"
RISKS = {"relation", "time", "scope", "measurement"}


def loadl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_risks(values):
    return {"relation" if x == "entity" else x for x in values}


def agent_reviews(split):
    result = {}
    for path in sorted(REVIEW.glob(f"luna-{split}-batch-*.jsonl")):
        for row in loadl(path):
            key = row.get("case_id")
            if not key or key in result:
                raise ValueError("duplicate_or_missing_luna_id")
            result[key] = row
    correction_path = REVIEW / f"luna-corrections-{split}.jsonl"
    corrected = set()
    for row in loadl(correction_path):
        key = row.get("case_id")
        if key not in result or key in corrected or not row.get("supersedes_batch") or not row.get("correction_reason"):
            raise ValueError("invalid_or_duplicate_luna_correction")
        corrected.add(key)
        result[key] = row
    return result


def api_reviews(split):
    result = {}
    for row in loadl(REVIEW / f"api-{split}.jsonl"):
        key = row.get("case_id")
        if not key or key in result:
            raise ValueError("duplicate_or_missing_api_id")
        result[key] = row
    retried = set()
    for row in loadl(REVIEW / f"api-retry-{split}.jsonl"):
        key = row.get("case_id")
        if key not in result or result[key].get("decision") is not None or key in retried:
            raise ValueError("invalid_or_duplicate_api_retry")
        retried.add(key)
        if row.get("decision") is not None:
            result[key] = row
    return result


def valid_vote(packet, vote):
    if not isinstance(vote, dict) or type(vote.get("approved")) is not bool or not vote.get("reason"):
        return False, "incomplete_vote"
    if not isinstance(vote.get("risk_labels"), list) or set(vote["risk_labels"]) - RISKS:
        return False, "invalid_risk_labels"
    try:
        validate_citations(packet, vote)
    except (KeyError, ValueError, TypeError):
        return False, "invalid_citation"
    if vote["approved"]:
        if not vote.get("citations"):
            return False, "missing_citation"
        if packet["proposed_action"]["action"] == "answer":
            if not any(c.get("stage") == "before_action" for c in vote["citations"]):
                return False, "answer_citation_not_visible"
        else:
            if vote.get("search_needed") != "yes" or vote.get("query_grounded") != "yes" or vote.get("retrieval_relevant") != "yes":
                return False, "search_semantics_not_approved"
            if not vote.get("missing_fact") or not any(c.get("stage") == "retrieved_after_search" for c in vote["citations"]):
                return False, "search_gap_or_result_citation_missing"
    return True, "valid"


def review_pool(split):
    packet_file = REVIEW / f"{split}-packets.jsonl"
    packets = loadl(packet_file)
    source_file_sha = hashlib.sha256(packet_file.read_bytes()).hexdigest()
    a, b = agent_reviews(split), api_reviews(split)
    accepted, audit = [], []
    for packet in packets:
        key = packet["case_id"]
        luna = a.get(key)
        api_record = b.get(key)
        api = api_record.get("decision") if api_record else None
        av, ae = valid_vote(packet, luna)
        bv, be = valid_vote(packet, api)
        packet_full_sha = digest(packet)
        if luna is not None and luna.get("packet_sha256") != packet["packet_sha256"]:
            av, ae = False, "luna_packet_hash_mismatch"
        if api_record is not None and api_record.get("packet_sha256") != packet_full_sha:
            bv, be = False, "api_packet_hash_mismatch"
        if api_record is not None and api_record.get("input_file_sha256") != source_file_sha:
            bv, be = False, "api_input_file_hash_mismatch"
        approved = av and bv and luna["approved"] and api["approved"]
        risk = sorted(set(luna["risk_labels"]) & set(api["risk_labels"])) if approved else []
        if approved:
            accepted.append((packet, risk))
        audit.append({"case_id": key, "approved": approved, "luna_status": ae, "api_status": be,
                      "luna_approved": luna.get("approved") if av else None,
                      "api_approved": api.get("approved") if bv else None,
                      "risk_labels": risk, "packet_sha256": packet["packet_sha256"]})
    return accepted, audit


def source_ids():
    return {r.get("case_id", r["packet"]["case_id"]): r["source_id"]
            for name in ("train-supplement-private.jsonl", "internal-candidate-private.jsonl")
            for r in loadl(WORK / name)}


def old_pool():
    result = []
    for row in loadl(OLD / "accepted.jsonl"):
        if row["split"] != "train":
            continue
        packet = row["packet"]
        risks = normalize_risks(row["review_a"].get("risk_labels", [])) & normalize_risks(row["review_b"].get("risk_labels", []))
        result.append((packet, row["source_id"], sorted(risks)))
    if len(result) != 332:
        raise ValueError("old_approved_pool_changed")
    return result


def export(packet):
    messages = packet.get("visible_messages", packet.get("messages"))
    action = packet.get("proposed_action", packet.get("action"))
    target = {"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))}
    return {"messages": messages + [target], "target_assistant_index": len(messages), "thinking_enabled": False}


def choose_train_supplements(base, approved, ids):
    """Select the smallest valid paired supplement without using model outcomes."""
    present = defaultdict(set)
    for packet, source_id, _ in base:
        present[source_id].add(packet["proposed_action"]["action"])
    eligible = []
    for packet, risks in approved:
        source_id = ids[packet["case_id"]]
        action = packet["proposed_action"]["action"]
        if action not in {"search", "answer"} or ({"answer", "search"} - {action}) - present[source_id]:
            continue
        if action == "search" and not packet["before_action"]:
            continue
        eligible.append((packet, source_id, risks))
    kinds = {kind: sorted((r for r in eligible if r[0]["proposed_action"]["action"] == kind),
                          key=lambda r: hashlib.sha256(("v116-final:" + r[1] + r[0]["case_id"]).encode()).hexdigest())
             for kind in ("answer", "search")}
    base_answers = sum(p["proposed_action"]["action"] == "answer" for p, _, _ in base)
    base_search = sum(p["proposed_action"]["action"] == "search" and bool(p.get("before_action", p.get("visible_evidence")))
                      for p, _, _ in base)
    for size in range(len(eligible) + 1):
        for answers in range(size + 1):
            searches = size - answers
            if answers > len(kinds["answer"]) or searches > len(kinds["search"]):
                continue
            picked = kinds["answer"][:answers] + kinds["search"][:searches]
            if len({sid for _, sid, _ in picked}) != size:
                continue
            total = len(base) + size
            if not 300 <= total <= 400 or not .45 <= (base_answers + answers) / total <= .55:
                continue
            if (base_search + searches) / total < .40 or (len(base) - base_answers - base_search) / total > .10:
                continue
            if sum("answer" in present[sid] and p["proposed_action"]["action"] == "search"
                   or "search" in present[sid] and p["proposed_action"]["action"] == "answer"
                   for p, sid, _ in picked) < 13:
                continue
            return picked
    return eligible  # Gate below fails closed and reports the exact remaining gap.


def run():
    history = json.loads((WORK / "history-inventory.json").read_text(encoding="utf-8"))
    if not history["ready"] or history["failures"]:
        raise RuntimeError("history_gate_failed")
    manifest = json.loads((REVIEW / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["source_files"].items():
        if hashlib.sha256((WORK / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("review_source_hash_mismatch")
    for split in ("train", "internal"):
        path = REVIEW / f"{split}-packets.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["packets"][split]["sha256"]:
            raise RuntimeError("review_packet_hash_mismatch")
    token_report = json.loads((WORK / "token-replay.json").read_text(encoding="utf-8"))
    if not token_report.get("all_exact") or token_report.get("counts") != {
            "train": 60, "internal": 100, "reused_v114_train": 332}:
        raise RuntimeError("local_token_replay_failed_or_incomplete")
    for filename, expected in token_report["input_sha256"].items():
        path = OLD / "accepted.jsonl" if filename == "accepted.jsonl" else REVIEW / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("local_token_replay_stale")
    revision_path = REVIEW / "internal-revision-manifest.json"
    if revision_path.exists():
        revision = json.loads(revision_path.read_text(encoding="utf-8"))
        pending_path = REVIEW / "internal-revision-pending.jsonl"
        if (revision["canonical_packets_sha256"] != manifest["packets"]["internal"]["sha256"]
                or hashlib.sha256(pending_path.read_bytes()).hexdigest() != revision["pending_packets_sha256"]
                or loadl(pending_path) != loadl(REVIEW / "internal-packets.jsonl")):
            raise RuntimeError("revision_packet_binding_mismatch")
        for name, spec in revision["batches"].items():
            path = REVIEW / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"] or len(loadl(path)) != spec["rows"]:
                raise RuntimeError("revision_batch_hash_mismatch")
    ids = source_ids()
    train_new, train_audit = review_pool("train")
    internal_new, internal_audit = review_pool("internal")
    train = old_pool()
    selected_new = choose_train_supplements(train, train_new, ids)
    train.extend(selected_new)
    by_question = defaultdict(set)
    action_count, risk_counts = Counter(), Counter()
    for packet, source_id, risks in train:
        action = packet["proposed_action"]["action"]
        evidence = packet.get("before_action", packet.get("visible_evidence", []))
        kind = "answer" if action == "answer" else "nonempty_search" if evidence else "empty_search"
        action_count[kind] += 1
        risk_counts.update(risks)
        by_question[source_id].add(kind)
    pairs = sum("answer" in kinds and "nonempty_search" in kinds for kinds in by_question.values())
    total = len(train)
    internal_by_question = defaultdict(set)
    for packet, risks in internal_new:
        internal_by_question[ids[packet["case_id"]]].add(packet["proposed_action"]["action"])
    internal_complete_ids = {source_id for source_id, actions in internal_by_question.items()
                             if actions == {"search", "answer"}}
    internal_complete = [(packet, risks) for packet, risks in internal_new
                         if ids[packet["case_id"]] in internal_complete_ids]
    internal_risk = Counter(tag for _, risks in internal_complete for tag in risks)
    train_questions = {norm(p["question"]) for p, _, _ in train}
    internal_questions = {norm(p["question"]) for p, _ in internal_new}
    train_ids = {sid for _, sid, _ in train}
    internal_ids = {ids[p["case_id"]] for p, _ in internal_new}
    private_history = json.loads((WORK / "history-private.json").read_text(encoding="utf-8"))
    history_ids = set(private_history["ids"])
    history_questions = set(private_history["normalized_questions"])
    checks = {
        "selected_train_independent_approval": all(
            packet["case_id"] in {item["case_id"] for item in train_audit if item["approved"]}
            for packet, _, _ in selected_new),
        "complete_independent_reviews_internal": all(r["luna_status"] == "valid" and r["api_status"] == "valid" for r in internal_audit),
        "train_300_to_400": 300 <= total <= 400,
        "train_at_least_150_questions": len(by_question) >= 150,
        "train_at_least_75_pairs": pairs >= 75,
        "answer_45_to_55_percent": total > 0 and .45 <= action_count["answer"] / total <= .55,
        "nonempty_search_at_least_40_percent": total > 0 and action_count["nonempty_search"] / total >= .40,
        "empty_search_at_most_10_percent": total > 0 and action_count["empty_search"] / total <= .10,
        "train_four_risks_at_least_20": all(risk_counts[r] >= 20 for r in RISKS),
        "internal_50_paired_questions": len(internal_complete_ids) == 50,
        "internal_100_states": len(internal_complete) == 100,
        "internal_four_risks_at_least_5": all(internal_risk[r] >= 5 for r in RISKS),
        "question_isolation": not (train_ids & internal_ids or train_questions & internal_questions),
        "internal_history_isolation": not (internal_ids & history_ids or internal_questions & history_questions),
        "unique_states": len({digest(export(p)) for p, _, _ in train}) == len(train)
                         and len({digest(export(p)) for p, _ in internal_complete}) == len(internal_complete),
    }
    gaps = {"train_pairs_needed": max(0, 75 - pairs),
            "internal_paired_states_needed": max(0, 100 - len(internal_complete)),
            "internal_paired_questions_needed": max(0, 50 - len(internal_complete_ids)),
            "train_luna_missing_or_invalid": sum(r["luna_status"] != "valid" for r in train_audit),
            "train_api_missing_or_invalid": sum(r["api_status"] != "valid" for r in train_audit),
            "internal_luna_missing_or_invalid": sum(r["luna_status"] != "valid" for r in internal_audit),
            "internal_api_missing_or_invalid": sum(r["api_status"] != "valid" for r in internal_audit)}
    report = {"version": "v116-sft-preflight-v1", "train_states": total, "train_questions": len(by_question),
              "train_pairs": pairs, "train_actions": dict(action_count), "train_risks": dict(risk_counts),
              "new_train_dual_approved": len(train_new), "new_train_selected": len(selected_new),
              "internal_dual_approved_states": len(internal_new),
              "internal_complete_pair_states": len(internal_complete),
              "internal_complete_pairs": len(internal_complete_ids),
              "internal_approved_questions_including_singletons": len(internal_by_question),
              "internal_risks": dict(internal_risk), "gaps": gaps,
              "train_candidates_with_invalid_or_missing_second_review": gaps["train_api_missing_or_invalid"],
              "review": {"train": train_audit, "internal": internal_audit},
              "checks": checks, "local_data_ready": all(checks.values()),
              "remote_gpu_and_token_audit_pending": True, "training_started": False,
              "independent100_accessed": False, "final_test_accessed": False}
    train_gate_names = ("selected_train_independent_approval", "train_300_to_400",
                        "train_at_least_150_questions", "train_at_least_75_pairs",
                        "answer_45_to_55_percent", "nonempty_search_at_least_40_percent",
                        "empty_search_at_most_10_percent", "train_four_risks_at_least_20",
                        "question_isolation", "unique_states")
    report["train_data_only_passed"] = all(checks[name] for name in train_gate_names)
    write(ROOT / "reports/v116-sft-preflight.json", report)
    if report["train_data_only_passed"]:
        stage = WORK / "staged-train";stage.mkdir(exist_ok=True)
        path = stage / "train-input-target.jsonl"
        path.write_text("".join(json.dumps(export(p), ensure_ascii=False, separators=(",", ":")) + "\n"
                                for p, _, _ in train), encoding="utf-8")
        write(stage / "manifest.json", {"version": "v116-train-side-freeze-v1",
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "states": len(train),
              "questions": len(by_question), "complete_pairs": pairs,
              "global_preflight_passed": report["local_data_ready"],
              "training_allowed": False, "reason": "internal_pair_and_risk_gate_failed"})
    if report["local_data_ready"]:
        out = WORK / "frozen";out.mkdir(exist_ok=True)
        for name, values in (("train", [export(p) for p, _, _ in train]),
                             ("internal", [export(p) for p, _ in internal_complete])):
            (out / f"{name}.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False, separators=(",", ":")) + "\n"
                                                   for v in values), encoding="utf-8")
        write(out / "manifest.json", {"version": "v116-sft-freeze-v1", "train_sha256": hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest(),
                                     "internal_sha256": hashlib.sha256((out / "internal.jsonl").read_bytes()).hexdigest(),
                                     "gate_sha256": hashlib.sha256((ROOT / "reports/v116-sft-preflight.json").read_bytes()).hexdigest(),
                                     "train_states": total, "internal_states": len(internal_complete),
                                     "training_started": False, "independent100_accessed": False})
    print(json.dumps({k: v for k, v in report.items() if k not in {"review"}}, ensure_ascii=False))


if __name__ == "__main__":
    run()
