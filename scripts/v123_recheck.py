"""Independent structural recount from frozen rows and original review tickets."""
from __future__ import annotations

from collections import Counter, defaultdict

from scripts.v123_api_protocol import valid_vote
from scripts.v123_cpu_core import ROOT, OUT, action, digest, nonempty_search, norm, question, read, readl, sha, write


def recheck() -> dict:
    base_path = ROOT / "data/evidence-sft-v117/frozen/train.jsonl"
    base = readl(base_path)
    if sha(base_path) != "26ced5bafb5210fefe9b007915598c8d58d2ebec156e938a48d86e1f1577ad8b":
        raise ValueError("v117_train_changed")
    grouped_train = defaultdict(set)
    train_actions = Counter()
    for row in base:
        grouped_train[norm(question(row))].add("nonempty_search" if nonempty_search(row) else action(row))
        train_actions[action(row)] += 1
    old = ROOT / "data/evidence-sft-v122-internal-candidates"
    packets = readl(old / "review/internal-packets.jsonl")
    source_map = {case: row["source_id"] for row in readl(old / "review/eligible-private.jsonl") for case in row["case_ids"]}
    if len(packets) != 326 or set(source_map) != {p["case_id"] for p in packets}:
        raise ValueError("old_internal_packet_coverage_invalid")
    api, luna = {}, {}
    from scripts.v123_review import _read_old_votes
    api.update(_read_old_votes("api"))
    luna.update(_read_old_votes("luna"))
    counts = Counter()
    accepted = defaultdict(list)
    for p in packets:
        case = p["case_id"]
        avote = api.get(case, {}).get("decision")
        lvote = luna.get(case)
        av, _ = valid_vote(p, avote)
        lv, _ = valid_vote(p, lvote)
        av = av and api[case].get("packet_sha256") == digest(p) if case in api else False
        lv = lv and luna[case].get("packet_sha256") == p["packet_sha256"] if case in luna else False
        for side, recorded, valid, vote in (("api", case in api, av, avote), ("luna", case in luna, lv, lvote)):
            if recorded:
                counts[side + "_records"] += 1
                counts[side + "_valid" if valid else side + "_invalid"] += 1
                if valid and vote["approved"]:
                    counts[side + "_approved"] += 1
        if av and lv and avote["approved"] and lvote["approved"]:
            accepted[source_map[case]].append((p, sorted(set(avote["risk_labels"]) & set(lvote["risk_labels"]))))
    approved_pairs = [(qid, records) for qid, records in accepted.items() if
                      len(records) == 2 and {p["proposed_action"]["action"] for p, _ in records} == {"search", "answer"}]
    risks = Counter(label for _, records in approved_pairs for _, labels in records for label in labels)
    result = {"version": "v123-historical-recheck-v1", "status": "passed",
              "v117": {"states": len(base), "questions": len(grouped_train),
                       "complete_pairs": sum({"answer", "nonempty_search"} <= x
                                             for x in grouped_train.values()),
                       "actions": dict(train_actions)},
              "v122_internal": {"candidate_states": len(packets), "candidate_questions": len(set(source_map.values())),
                                "review_counts": dict(counts), "complete_double_approved_pairs": len(approved_pairs),
                                "risks_in_complete_pairs": dict(risks), "scope_upper_in_frozen_votes": risks["scope"],
                                "full_data_gate_passed": len(approved_pairs) >= 50 and all(risks[r] >= 5 for r in
                                                        ("relation", "time", "scope", "measurement"))},
              "training_allowed": False}
    write(OUT / "diagnosis/historical-recheck.json", result)
    return result
