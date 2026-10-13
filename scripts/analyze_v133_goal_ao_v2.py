"""CPU-only final evidence packet preparation for Goal AO v2.

Builds on Goal AM's fixed citation/review packet logic while keeping all
outputs inside the isolated AO v2 run directory. This script never calls a
model or search tool.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_v133_goal_am as am_review

OUT = ROOT / "runs/v133-goal-ao-relation-contrast-sft-v2"
AM = ROOT / "runs/v133-goal-am-action-sft-v2"


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_new(path: Path, records):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "correct-citations":
        tok = am_review.am.tokenizer_from_base()
        corrections = json.loads((AM / "citation-corrections.json").read_text(encoding="utf-8"))
        overrides = {(r["case_id"], r["edge_id"]): r for r in corrections}
        packets = rows(OUT / "review-packets-seed42.jsonl")
        revisions = []
        for packet in packets:
            for edge in packet["edges"]:
                correction = overrides.get((packet["case_id"], edge["edge_id"]))
                if correction is None:
                    continue
                citation = am_review.prior.cite(packet["prompt_token_ids"], tok, correction["document_id"], correction["quote"])
                if citation is None:
                    raise RuntimeError(f"corrected source quote absent from AO input: {packet['case_id']}:{edge['edge_id']}")
                revisions.append(dict(case_id=packet["case_id"], model=packet["model"], edge_id=edge["edge_id"],
                                      previous_citation=edge.get("citation"), corrected_citation=citation,
                                      reason=correction["reason"]))
                edge["citation"] = citation
                edge["actual_visible"] = True
                edge["first_return_search"] = next((t["step"] for t in packet["trace"] if any(
                    d.get("id") == correction["document_id"] and correction["quote"] in d.get("text", "")
                    for d in t.get("observation", {}).get("result", []))), None)
            packet["complete_chain_visible"] = all(e["actual_visible"] for e in packet["edges"])
        (OUT / "review-packets-seed42-corrected.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in packets), encoding="utf-8"
        )
        (OUT / "citation-revisions-seed42.json").write_text(
            json.dumps(revisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        templates = {r["case_id"]: r for r in packets if r["model"] == "42-A"}
        old_probe_evidence = {r["probe_id"]: r for r in rows(AM / "probe-evidence-seed42.jsonl")}
        frozen_probe_inputs = {r["probe_id"]: r for r in rows(AM / "probe-review-packets-seed42.jsonl")
                               if r["category"] == "complete"}
        probe_packets = rows(OUT / "probe-review-packets-seed42.jsonl")
        probe_evidence = []
        for probe in probe_packets:
            if probe["category"] != "complete":
                continue
            case = probe["probe_id"].removeprefix("complete:")
            if probe["prompt_token_ids"] != frozen_probe_inputs[probe["probe_id"]]["prompt_token_ids"]:
                raise RuntimeError(f"AO complete probe input differs from reviewed AM input: {case}")
            edges = []
            for edge in old_probe_evidence[probe["probe_id"]]["edges"]:
                old_citation = edge["citation"]
                citation = am_review.prior.cite(probe["prompt_token_ids"], tok, old_citation["document_id"], old_citation["quote"])
                if citation is None:
                    raise RuntimeError(f"necessary quotation absent from AO complete probe: {case}:{edge['edge_id']}")
                edges.append({k: edge[k] for k in ("edge_id", "subject", "relation", "object")} | {"citation": citation})
            probe_evidence.append(dict(model=probe["model"], probe_id=probe["probe_id"], edges=edges, training_use_forbidden=True))
        (OUT / "probe-evidence-seed42.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in probe_evidence), encoding="utf-8"
        )
        print(json.dumps({"citation_revisions": len(revisions), "complete_probe_evidence": len(probe_evidence)}))
        return

    # Adapt the already-reviewed AO output names to AM's CPU-only packet builder.
    for label in ("42-A", "42-B"):
        combined = []
        for kind in ("complete", "partial"):
            combined.extend(rows(OUT / f"probe-{label}-{kind}.jsonl"))
        combined = [{k: v for k, v in row.items() if k != "model"} for row in combined]
        (OUT / f"probe-{label}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined), encoding="utf-8"
        )

    old_out = am_review.OUT
    am_review.OUT = OUT
    am_review.packets(42)
    am_review.OUT = old_out

    # Reuse only exact problem+actual-input+answer verdicts from the frozen
    # AM audit. New combinations remain explicitly pending human review.
    historic = rows(AM / "semantic-review-seed42.jsonl")
    historic_by_key = {r["combination_sha256"]: r for r in historic}
    packets = rows(OUT / "review-packets-seed42.jsonl")
    pending = []
    exact = []
    for packet in packets:
        previous = historic_by_key.get(packet["combination_sha256"])
        packet["exact_am_verdict"] = previous
        packet["adjudication_status"] = "reused_exact" if previous else "needs_primary_review"
        (exact if previous else pending).append(packet)
    (OUT / "review-packets-seed42.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in packets), encoding="utf-8"
    )
    write_new(OUT / "pending-answer-adjudications-seed42.jsonl", pending)

    historic_probes = rows(AM / "probe-review-seed42.jsonl")
    probe_by_key = {}
    for record in historic_probes:
        # Probe decisions are reusable only for the same fixed state and
        # exact generated action payload.
        probe_by_key[(record["model"], record["probe_id"], record["action"], record.get("answer"), record.get("query"))] = record
    probe_packets = rows(OUT / "probe-review-packets-seed42.jsonl")
    pending_probes = []
    for packet in probe_packets:
        previous = probe_by_key.get(("42-A", packet["probe_id"], packet["action"], packet.get("answer"), packet.get("query")))
        packet["exact_am_verdict"] = previous
        packet["adjudication_status"] = "reused_exact" if previous else "needs_primary_review"
        if packet["category"] == "partial":
            # This is a mechanical missing-edge next-action test, not a
            # semantic answer adjudication.
            packet["adjudication_status"] = "mechanical_review"
        elif previous is None:
            pending_probes.append(packet)
    (OUT / "probe-review-packets-seed42.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in probe_packets), encoding="utf-8"
    )
    write_new(OUT / "pending-probe-adjudications-seed42.jsonl", pending_probes)
    print(json.dumps({
        "answer_combinations": len(packets),
        "exact_am_reuse": len(exact),
        "pending_answers": len(pending),
        "probe_outputs": len(probe_packets),
        "pending_complete_probes": sum(p["category"] == "complete" for p in pending_probes),
        "partial_mechanical": sum(p["category"] == "partial" for p in probe_packets),
    }))


if __name__ == "__main__":
    main()
