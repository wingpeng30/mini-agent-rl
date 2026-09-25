"""Merge independent train-candidate audits and export one 395-state SFT set."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.build_v114_sft_candidates import norm
from scripts.finalize_v116 import export
from scripts.finalize_v117 import valid_v117_vote
from scripts.finalize_v122_internal import digest, loadl
from scripts.prepare_v116_data import ROOT, sha, write, writel

OUT = ROOT / "data/evidence-sft-v122-train-candidates"
REVIEW = OUT / "review"
FROZEN = OUT / "frozen"
BASE = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
EXPECTED_BASE = "26ced5bafb5210fefe9b007915598c8d58d2ebec156e938a48d86e1f1577ad8b"


def choose_relation_pairs(pairs: list[tuple[str, list]], count: int = 25, relation_min: int = 20):
    if len(pairs) < count:
        return None
    relation = [sum("relation" in labels for _, labels in records) for _, records in pairs]
    suffix = [0] * (len(pairs) + 1)
    for i in range(len(pairs) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + relation[i]
    selected: list[tuple[str, list]] = []
    failed = set()

    def visit(i: int, need: int) -> bool:
        if len(selected) == count:
            return need == 0
        if len(pairs) - i < count - len(selected) or suffix[i] < need:
            return False
        key = (i, count - len(selected), need)
        if key in failed:
            return False
        selected.append(pairs[i])
        if visit(i + 1, max(0, need - relation[i])):
            return True
        selected.pop()
        if visit(i + 1, need):
            return True
        failed.add(key)
        return False

    return list(selected) if visit(0, relation_min) else None


def main() -> None:
    history = json.loads((OUT / "history-inventory.json").read_text(encoding="utf-8"))
    freeze = json.loads((OUT / "candidate-freeze.json").read_text(encoding="utf-8"))
    manifest = json.loads((REVIEW / "manifest.json").read_text(encoding="utf-8"))
    tokens = json.loads((OUT / "token-replay.json").read_text(encoding="utf-8"))
    if not history.get("ready") or history.get("failures") or history.get("sealed_content_read"):
        raise RuntimeError("historical_exposure_gate_failed")
    if sha(BASE) != EXPECTED_BASE or sha(OUT / "candidate-corpus-private.json") != freeze["corpus_sha256"]:
        raise RuntimeError("source_train_or_candidate_corpus_changed")
    if sha(REVIEW / "internal-packets.jsonl") != manifest["packet_sha256"] or not tokens.get("all_exact") or tokens.get("packet_sha256") != manifest["packet_sha256"]:
        raise RuntimeError("final_input_or_token_replay_changed")
    packets = loadl(REVIEW / "internal-packets.jsonl")
    if len(packets) != manifest["states"] or len({p["case_id"] for p in packets}) != len(packets):
        raise RuntimeError("packet_coverage_or_duplicate")
    source_map = {case: row["source_id"] for row in loadl(REVIEW / "eligible-private.jsonl") for case in row["case_ids"]}
    if set(source_map) != {p["case_id"] for p in packets}:
        raise RuntimeError("private_source_mapping_invalid")
    api: dict[str, dict] = {}
    luna: dict[str, dict] = {}
    luna_batches = []
    luna_gap_seen = False
    for batch, spec in manifest["batches"].items():
        batch_packets = loadl(REVIEW / batch)
        expected_cases = {p["case_id"] for p in batch_packets}
        if sha(REVIEW / batch) != spec["sha256"] or len(batch_packets) != spec["states"] or spec["states"] > 20:
            raise RuntimeError("review_batch_changed")
        for prefix, result in (("api", api), ("luna", luna)):
            path = REVIEW / f"{prefix}-{batch}"
            if not path.exists():
                if prefix == "luna":
                    luna_gap_seen = True
                continue
            if prefix == "luna":
                if luna_gap_seen or len(loadl(path)) != spec["states"]:
                    raise RuntimeError("partial_or_noncontiguous_luna_review")
                luna_batches.append(batch)
            reviewed_rows = loadl(path)
            if {row.get("case_id") for row in reviewed_rows} != expected_cases:
                raise RuntimeError(f"{prefix}_batch_case_coverage_mismatch")
            for row in reviewed_rows:
                key = row.get("case_id")
                if not key or key in result:
                    raise RuntimeError(f"duplicate_{prefix}_review")
                if prefix == "api" and row.get("input_file_sha256") != spec["sha256"]:
                    raise RuntimeError("api_batch_hash_mismatch")
                result[key] = row
    approved: dict[str, list] = defaultdict(list)
    details: list[dict] = []
    for packet in packets:
        case = packet["case_id"]
        lrow = luna.get(case)
        arow = api.get(case)
        lv, lr = valid_v117_vote(packet, lrow)
        av, ar = valid_v117_vote(packet, arow.get("decision") if arow else None)
        if lrow and lrow.get("packet_sha256") != packet["packet_sha256"]:
            lv, lr = False, "luna_packet_hash_mismatch"
        if arow and arow.get("packet_sha256") != digest(packet):
            av, ar = False, "api_packet_hash_mismatch"
        accepted = bool(lv and av and lrow["approved"] and arow["decision"]["approved"])
        labels = sorted(set(lrow["risk_labels"]) & set(arow["decision"]["risk_labels"])) if accepted else []
        if accepted:
            approved[source_map[case]].append((packet, labels))
        details.append({"case_id": case, "source_id": source_map[case], "accepted": accepted,
                        "luna_validation": lr, "api_validation": ar, "risk_labels": labels})
    pairs = [(qid, approved[qid]) for qid in freeze["ids"] if len(approved[qid]) == 2 and
             {p["proposed_action"]["action"] for p, _ in approved[qid]} == {"search", "answer"}]
    selected = choose_relation_pairs(pairs) or []
    target_risks = Counter(tag for _, records in selected for _, labels in records for tag in labels)
    old = loadl(BASE)
    supplement = [export(p) for _, records in selected for p, _ in records]
    combined = old + supplement
    action_counts = Counter(json.loads(row["messages"][row["target_assistant_index"]]["content"])["action"] for row in combined)
    candidate_questions = {norm(p["question"]) for _, records in selected for p, _ in records}
    prior = json.loads((OUT / "history-private.json").read_text(encoding="utf-8"))
    internal = json.loads((ROOT / "data/evidence-sft-v122-internal-candidates/candidate-freeze.json").read_text(encoding="utf-8"))
    isolated = not ({qid for qid, _ in selected} & (set(prior["ids"]) | set(internal["ids"])) or
                    candidate_questions & set(prior["normalized_questions"]))
    reviewed_prefix = {p["case_id"] for name in luna_batches for p in loadl(REVIEW / name)}
    checks = {"contiguous_complete_double_reviewed_prefix": bool(luna_batches) and
              set(luna) == reviewed_prefix and reviewed_prefix <= set(api),
              "25_new_complete_pairs": len(selected) == 25,
              "at_least_20_new_relation_risk_states": target_risks["relation"] >= 20,
              "395_states_211_questions_100_pairs": len(combined) == 395 and len(selected) == 25,
              "answer_45_to_55_percent": 0.45 <= action_counts["answer"] / max(1, len(combined)) <= 0.55,
              "question_isolation": isolated,
              "unique_states": len({digest(row) for row in combined}) == len(combined),
              "v117_training_gate_preserved": json.loads((ROOT / "reports/v117-local-preflight.json").read_text(encoding="utf-8"))["checks"]["train_actions_and_risks"]}
    report = {"version": "v122-targeted-train-gate-v1", "checks": checks,
              "local_train_ready": all(checks.values()), "training_allowed": False,
              "candidate_questions": freeze["source_count"], "candidate_states": len(packets),
              "api_reviewed": len(api), "luna_reviewed": len(luna), "luna_batches": luna_batches,
              "approved_pairs": len(pairs), "selected_pairs": len(selected),
              "combined_states": len(combined), "combined_actions": dict(action_counts),
              "new_risk_states": dict(target_risks), "gaps": {"pairs": max(0,25-len(pairs)),
                                                               "relation_states": max(0,20-target_risks["relation"])},
              "independent100_accessed": False, "final_test_accessed": False}
    write(ROOT / "reports/v122-train-preflight.json", report)
    writel(REVIEW / "merge-details.jsonl", details)
    if not report["local_train_ready"]:
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return
    FROZEN.mkdir(parents=True, exist_ok=True)
    writel(FROZEN / "train.jsonl", combined)
    write(FROZEN / "manifest.json", {"version": "v122-targeted-train-freeze-v1",
          "states": 395, "questions": 211, "complete_pairs": 100,
          "train_sha256": sha(FROZEN / "train.jsonl"), "source_v117_train_sha256": EXPECTED_BASE,
          "selected_ids_sha256": digest([qid for qid, _ in selected]),
          "training_allowed": False, "remote_preflight_pending": True})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
