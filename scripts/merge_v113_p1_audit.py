"""Validate and aggregate blinded fixed-input P1 semantic reviews."""
import hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    packet = read(R / "v113-p1-blind-review.jsonl")
    private = json.loads((R / "v113-p1-private-key.json").read_text(encoding="utf-8"))
    inputs = {row["input_hash"]: row for row in read(R / "v113-cross-inputs.jsonl")}
    results = read(R / "v113-cross-results-stable.jsonl")
    by_id = defaultdict(dict)
    for row in results:
        by_id[row["input_id"]][row["arm"]] = row
    reviews = [read(R / f"v113-p1-agent-review-{i}.jsonl") for i in range(3)]
    review_maps = [{row["case_id"]: row for row in group} for group in reviews]
    packet_by_case = {row["blind_input_id"]: row for row in packet}
    if len(packet) != 170 or len(private) != 170 or set(packet_by_case) != set(private):
        raise ValueError("complete_blinded_170_required")
    expected_coverage = [57, 57, 56]
    if [len(group) for group in reviews] != expected_coverage:
        raise ValueError("unexpected_review_coverage")
    if any(not set(m) <= set(packet_by_case) for m in review_maps):
        raise ValueError("review_contains_unrecognized_case")
    if set.union(*(set(m) for m in review_maps)) != set(packet_by_case):
        raise ValueError("review_union_does_not_cover_packet")
    if len(by_id) != 170 or any(set(pair) != {"baseline", "candidate"} for pair in by_id.values()):
        raise ValueError("complete_paired_cross_outputs_required")

    counts = {arm: Counter() for arm in ("baseline", "candidate")}
    relative = Counter()
    disagreements = Counter()
    verified_quotes = 0
    per_case = {}
    for case_id, item in packet_by_case.items():
        input_hash = private[case_id]["input_id"]
        source_input = inputs[input_hash]
        if hashlib.sha256(json.dumps(source_input["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest() != input_hash:
            raise ValueError("input_token_hash_mismatch:" + case_id)
        if len(source_input["prompt_token_ids"]) != len(by_id[input_hash]["baseline"]["prompt_token_ids"]):
            raise ValueError("cross_output_prompt_token_count_mismatch:" + case_id)
        for arm in ("baseline", "candidate"):
            out = by_id[input_hash][arm]
            if out["prompt_token_ids"] != source_input["prompt_token_ids"]:
                raise ValueError("cross_output_prompt_token_mismatch:" + case_id + ":" + arm)
            expected_action = out["action_parsed"]
            pubarm = "A" if private[case_id]["A_arm"] == arm else "B"
            if item[pubarm]["action"] != expected_action:
                raise ValueError("blind_packet_action_mismatch:" + case_id + ":" + arm)
        docs = {d["document_id"]: d["text"] for d in item["current_evidence"]}
        for pubarm in ("A", "B"):
            action = item[pubarm]["action"].get("action")
            present_reviews = [(reviewer, review_map) for reviewer, review_map in enumerate(review_maps) if case_id in review_map]
            for reviewer, review_map in present_reviews:
                verdict = review_map[case_id][pubarm]
                counts[private[case_id][pubarm + "_arm"]]["reviewed_actions"] += 1
                for field in ("answer_support", "search_needed", "query_grounded", "retrieval_gap_fit"):
                    val = verdict.get(field, "missing")
                    counts[private[case_id][pubarm + "_arm"]][field + ":" + str(val)] += 1
                counts[private[case_id][pubarm + "_arm"]]["reviewer_unresolved"] += bool(review_map[case_id].get("unresolved"))
                for cite in verdict.get("citations", []):
                    verified_quotes += 1
                    source = cite.get("source")
                    if source in ("before", "current_evidence"):
                        valid = cite.get("quote") in docs.get(cite.get("document_id"), "")
                    elif source in ("after", "top3_retrieval_after_action"):
                        after = {d["document_id"]: d["text"] for d in item[pubarm]["top3_retrieval_after_action"]}
                        valid = action == "search" and cite.get("quote") in after.get(cite.get("document_id"), "")
                    else:
                        valid = False
                    if not valid:
                        raise ValueError(f"invalid_or_temporally_illegal_quote:{reviewer}:{case_id}:{pubarm}:{source}")
                    if action == "answer" and source not in ("before", "current_evidence"):
                        raise ValueError("answer_used_future_retrieval:" + case_id)
        rel = [review_map[case_id].get("relative_evidence", "missing") for review_map in review_maps if case_id in review_map]
        relative.update(rel)
        disagreements["relative_unanimous" if len(set(rel)) == 1 else "relative_disagreement"] += 1
        per_case[case_id] = {"reviewers": [i for i, m in enumerate(review_maps) if case_id in m], "relative_reviews": rel, "unresolved_reviewers": sum(bool(m[case_id].get("unresolved")) for m in review_maps if case_id in m)}

    # Paired behavior/action matrix. New docs are descriptive; gap-fit labels decide usefulness.
    paired = Counter()
    for input_hash, pair in by_id.items():
        actions = (pair["baseline"]["action_parsed"]["action"], pair["candidate"]["action_parsed"]["action"])
        paired[f"{actions[0]}->{actions[1]}"] += 1
    summary = {
        "version": "v113-p1",
        "review_mode": "blinded task-agent semantic audits over three disjoint chunks; chunks are not three-way overlapping; no external API review",
        "cases": 170,
        "reviewer_coverage": [len(x) for x in reviews],
        "reviewer_overlap": dict(Counter(str(sum(case_id in m for m in review_maps)) for case_id in packet_by_case)),
        "three_way_reviews": sum(all(case_id in m for m in review_maps) for case_id in packet_by_case),
        "one_way_reviews": sum(sum(case_id in m for m in review_maps) == 1 for case_id in packet_by_case),
        "verified_quotes": verified_quotes,
        "token_identity_verified": True,
        "action_matrix_baseline_to_candidate": dict(paired),
        "relative_evidence_reviews": dict(relative),
        "relative_review_agreement": dict(disagreements),
        "per_arm_reviewer_counts": {arm: dict(c) for arm, c in counts.items()},
        "case_reviews": per_case,
        "fixed_state_only_not_closed_loop_causal_effect": True,
        "independent100_accessed": False,
        "final_test_accessed": False,
        "source_packet_sha256": hashlib.sha256((R / "v113-p1-blind-review.jsonl").read_bytes()).hexdigest(),
        "stable_result_sha256": hashlib.sha256((R / "v113-cross-results-stable.jsonl").read_bytes()).hexdigest(),
    }
    out = R / "v113-p1-agent-audit-summary.json"
    if out.exists():
        raise FileExistsError("refuse_to_overwrite_p1_audit_summary")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("case_reviews", "per_arm_reviewer_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
