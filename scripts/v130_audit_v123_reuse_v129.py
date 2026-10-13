"""Recount only exact v122 vote reuses embedded in the exposed v123 internal freeze."""
from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.v123_prepare_internal import packet_same
from scripts.v123_cpu_core import digest, sha
from scripts.v128_review_protocol import valid_vote as valid_v128
from scripts.v129_review_protocol import valid_vote as valid_v129

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/evidence-sft-v122-internal-candidates/review"
NEW = ROOT / "data/evidence-sft-v123/internal/review"
OUT = ROOT / "data/evidence-sft-v124-forensic/goal-v128/g4-v123-reuse-v129-audit-v2-private.json"


def _readl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def _for_validation(packet: dict) -> dict:
    """Treat null post-action retrieval on an answer as the same empty list."""
    result = copy.deepcopy(packet)
    if result.get("proposed_action", {}).get("action") == "answer" and result.get("retrieved_after_search") is None:
        result["retrieved_after_search"] = []
    return result


def audit() -> dict:
    old_packets = {p["case_id"]: p for p in _readl(OLD / "internal-packets.jsonl")}
    new_packets = {p["case_id"]: p for p in _readl(NEW / "internal-packets.jsonl")}
    source_map = {r["case_id"]: r["source_id"] for r in _readl(NEW / "source-map-private.jsonl")}
    old_api, old_luna = {}, {}
    for path in sorted(OLD.glob("api-internal-batch-*.jsonl")):
        for row in _readl(path):
            old_api[row["case_id"]] = row
    for path in sorted(OLD.glob("luna-internal-batch-*.jsonl")):
        for row in _readl(path):
            old_luna[row["case_id"]] = row
    reuse = _readl(NEW / "old-vote-reuse-private.jsonl")
    errors, valid_approvals = Counter(), Counter()
    per_case: dict[str, dict] = {}
    grouped = defaultdict(list)
    for link in reuse:
        old_id, new_id = link["old_case_id"], link["new_case_id"]
        old_packet, new_packet = old_packets[old_id], new_packets[new_id]
        if (link.get("same_reviewed_input") is not True or not packet_same(old_packet, new_packet)
                or link.get("old_packet_sha256") != old_packet["packet_sha256"]
                or link.get("new_packet_sha256") != new_packet["packet_sha256"]):
            raise ValueError("old_vote_packet_equivalence_not_proven")
        rebound = {}
        for reviewer, ledger in (("api", old_api), ("luna", old_luna)):
            record = ledger.get(old_id)
            if not record:
                errors[reviewer + "_vote_missing"] += 1
                continue
            vote = copy.deepcopy(record.get("decision") if reviewer == "api" else record)
            expected = digest(old_packet) if reviewer == "api" else old_packet["packet_sha256"]
            if not isinstance(vote, dict) or record.get("packet_sha256") != expected:
                errors[reviewer + "_old_ticket_binding_invalid"] += 1
                continue
            vote["case_id"] = new_id  # Explicit container-ID mapping; decision is unchanged.
            checked_packet = _for_validation(new_packet)
            ok128, why128 = valid_v128(checked_packet, vote)
            ok129, why129 = valid_v129(checked_packet, vote)
            if not ok129:
                errors[f"{reviewer}:{new_packet['proposed_action']['action']}:{why129}"] += 1
            if ok129 and vote.get("approved") is True:
                valid_approvals[reviewer] += 1
            rebound[reviewer] = {"vote": vote, "valid_v128": ok128, "valid_v129": ok129,
                                 "approved": ok129 and vote.get("approved") is True}
        sid = source_map[new_id]
        grouped[sid].append((new_id, new_packet, rebound))
        per_case[new_id] = {"source_id": sid,
                            "action": new_packet["proposed_action"]["action"],
                            "old_case_id": old_id,
                            "old_new_visible_input_equivalent": True,
                            "answer_null_normalized_only_for_validation":
                                new_packet["proposed_action"]["action"] == "answer" and
                                new_packet.get("retrieved_after_search") is None,
                            "api_valid_v129_approved": rebound.get("api", {}).get("approved", False),
                            "luna_valid_v129_approved": rebound.get("luna", {}).get("approved", False)}
    complete_pairs, risk = 0, Counter()
    for source_id, states in grouped.items():
        if len(states) != 2 or {p["proposed_action"]["action"] for _, p, _ in states} != {"search", "answer"}:
            continue
        if all(all(s.get(r, {}).get("approved", False) for r in ("api", "luna"))
               for _, _, s in states):
            complete_pairs += 1
            for _, _, reviews in states:
                a, l = reviews["api"]["vote"], reviews["luna"]["vote"]
                risk.update(set(a.get("risk_labels", [])) & set(l.get("risk_labels", [])))
    inputs = [OLD / "internal-packets.jsonl", NEW / "internal-packets.jsonl",
              NEW / "source-map-private.jsonl", NEW / "old-vote-reuse-private.jsonl"]
    inputs.extend(sorted(OLD.glob("api-internal-batch-*.jsonl")))
    inputs.extend(sorted(OLD.glob("luna-internal-batch-*.jsonl")))
    inputs.extend([Path(__file__), Path(__file__).with_name("v128_review_protocol.py"),
                   Path(__file__).with_name("v129_review_protocol.py"),
                   Path(__file__).with_name("v123_prepare_internal.py")])
    report = {"version": "v130-v123-old-vote-reuse-under-v129-v2",
        "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        "scope": "Only the 42 exact packet-equivalent vote mappings already recorded in v123; no votes rewritten.",
        "mapped_states": len(reuse), "valid_approved_states": dict(valid_approvals),
        "complete_dual_approved_pairs_under_v129": complete_pairs,
        "risk_intersection_states_in_complete_pairs": dict(risk),
        "mechanical_exclusions": dict(errors),
        "answer_null_compatibility": "For validation only, answer packets with retrieved_after_search=null are viewed as having []; visible messages, action, top-3, token hashes and saved packet bytes remain unchanged.",
        "semantic_status": "Mechanical v129 validity and old independent votes only; this is not a new semantic review, nor evidence that any pool meets the 50-question gate.",
        "training_eligible_states": 0, "training_allowed": False,
        "data_ready_for_experiment": False, "case_audit": per_case}
    return report


def main() -> None:
    report = audit()
    if OUT.exists():
        if json.loads(OUT.read_text(encoding="utf-8")) != report:
            raise RuntimeError("existing_v123_reuse_audit_differs_refuse_overwrite")
    else:
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mapped_states": report["mapped_states"],
                      "complete_dual_approved_pairs_under_v129": report["complete_dual_approved_pairs_under_v129"],
                      "risk_intersections": report["risk_intersection_states_in_complete_pairs"],
                      "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
