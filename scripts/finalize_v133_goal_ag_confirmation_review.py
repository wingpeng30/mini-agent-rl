"""Freeze the main-agent adjudication of Goal AG's confirmation review."""
from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "v133-goal-ag-failure-repair"
CATEGORIES = {
    "correct_sufficient",
    "factual_error",
    "missing_required_information",
    "unsupported_by_visible_evidence",
    "indeterminate",
}

# Main-agent adjudication after reviewing every answer marked correct and the
# specifically disputed relation cases. Matching is by the full question and
# answer because packet rows can contain repeated trace variants.
OVERRIDES = {
    ("What is the record label of the Thrill of a Lifetime performer?", "Capitol Records"):
        ("unsupported_by_visible_evidence",
         "Visible text says the Thrill of a Lifetime album was released by Capitol Records, but does not establish that Capitol was the performer’s label or signing relationship. The same visible context lists a different King Kobra album released by New Renaissance Records, so the album label cannot be generalized to the performer."),
    ("What record label is the performer of Almost Made Ya signed to?", "Universal Records"):
        ("unsupported_by_visible_evidence",
         "Visible text says the album was released through both Derrty Entertainment and Universal Records; it does not establish which label the performer was signed to. The response selects one co-release label without support for the asked signing relation."),
    ("Where is the location of the headquarters of the publisher of Bamboo Among the Oaks?", "Saint Paul, Minnesota"):
        ("missing_required_information",
         "The visible evidence identifies the Minnesota History Center as the publisher's headquarters and places it near downtown Saint Paul. The answer gives only the city, omitting the actual headquarters location asked for."),
    ("What province shares a border with the province where Lago District is located?", "Cabo Delgado"):
        ("unsupported_by_visible_evidence",
         "The visible pages establish Lago District is in Niassa Province and Niassa Reserve has territory in Niassa and Cabo Delgado. A reserve occupying both provinces does not explicitly establish their shared border; apply the same missing-relation rule to all identical evidence combinations."),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    packet = read_jsonl(OUT / "confirmation-review-packet.jsonl")
    if len({row["review_id"] for row in packet}) != len(packet):
        raise RuntimeError("duplicate confirmation review_id in packet")
    packet_by_id = {row["review_id"]: row for row in packet}
    raw_votes = read_jsonl(OUT / "confirmation-review-a.jsonl") + read_jsonl(OUT / "confirmation-review-b.jsonl")
    votes_by_id = {row["review_id"]: row for row in raw_votes}
    if len(votes_by_id) != len(raw_votes) or set(votes_by_id) != set(packet_by_id):
        raise RuntimeError("blind confirmation reviews are missing, duplicated, or out of scope")

    final_rows: list[dict] = []
    ledger: list[dict] = []
    seen_override: set[tuple[str, str]] = set()
    for review_id, original in packet_by_id.items():
        packet_row = original
        vote = votes_by_id[review_id]
        category = vote["category"]
        if category not in CATEGORIES:
            raise RuntimeError(f"invalid reviewer category {category!r}")
        doc_ids = {doc["document_id"] for doc in packet_row["visible_documents"]}
        if not set(vote.get("evidence_ids", [])) <= doc_ids:
            raise RuntimeError(f"review cites a document absent from actual final input: {review_id}")
        if category == "correct_sufficient" and not vote.get("evidence_ids"):
            raise RuntimeError(f"success label has no visible-evidence citation: {review_id}")

        key = (packet_row["question"], packet_row["answer"])
        if category == "correct_sufficient" and key in OVERRIDES:
            category, reason = OVERRIDES[key]
            seen_override.add(key)
            ledger.append({
                "review_id": review_id,
                "question": packet_row["question"],
                "answer": packet_row["answer"],
                "old_category": vote["category"],
                "new_category": category,
                "reason": reason,
                "visible_document_ids": sorted(doc_ids),
                "reviewer_evidence_ids": vote.get("evidence_ids", []),
            })
        final_rows.append({
            "review_id": review_id,
            "category": category,
            "reason": (ledger[-1]["reason"] if ledger and ledger[-1]["review_id"] == review_id
                       else vote["reason"]),
            "evidence_ids": vote.get("evidence_ids", []),
            "reviewer_category": vote["category"],
        })
    if seen_override != set(OVERRIDES):
        raise RuntimeError(f"expected manual adjudications absent: {set(OVERRIDES) - seen_override}")

    write_jsonl(OUT / "confirmation-review-final.jsonl", final_rows)
    write_jsonl(OUT / "confirmation-review-adjudication-ledger.jsonl", ledger)
    summary = {"rows": len(final_rows), "categories": dict(Counter(r["category"] for r in final_rows)),
               "main_agent_overrides": len(ledger), "reviewers": {"a": 55, "b": 54}}
    (OUT / "confirmation-review-final-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    decision = calculate_decision(packet_by_id, final_rows)
    (OUT / "confirmation-decision-local.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def calculate_decision(packet_by_id: dict[str, dict], final_rows: list[dict]) -> dict:
    mapping = read_jsonl(OUT / "confirmation-review-map.jsonl")
    by_vote = {row["review_id"]: row for row in final_rows}
    by_case = {(row["model"], row["case_id"]): by_vote[row["review_id"]]["category"] for row in mapping}
    scores: dict[str, dict] = {}
    raw_by_model: dict[str, dict[str, dict]] = {}
    for label in ("baseline", "42-B", "43-B"):
        raw = read_jsonl(OUT / f"eval-confirmation-{label}.jsonl")
        if len(raw) != 64:
            raise RuntimeError(f"expected 64 complete outputs for {label}, got {len(raw)}")
        if len({row["case_id"] for row in raw}) != 64:
            raise RuntimeError(f"duplicate evaluation case for {label}")
        raw_by_model[label] = {row["case_id"]: row for row in raw}
        counts = Counter(by_case[(label, row["case_id"])] for row in raw)
        counts["bad_total"] = counts["factual_error"] + counts["unsupported_by_visible_evidence"]
        counts["invalid"] = sum(row.get("reason") == "invalid_action" for row in raw)
        counts["em"] = sum(row.get("em", 0) for row in raw)
        counts["f1_mean"] = sum(row.get("f1", 0) for row in raw) / len(raw)
        counts["searches_mean"] = sum(row.get("searches", 0) for row in raw) / len(raw)
        counts["calls_mean"] = sum(row.get("model_calls", 0) for row in raw) / len(raw)
        counts["latency_mean_seconds"] = sum(row.get("elapsed_seconds", 0) for row in raw) / len(raw)
        scores[label] = dict(counts)
    baseline = scores["baseline"]
    expected_cases = set(raw_by_model["baseline"])
    if any(set(raw) != expected_cases for raw in raw_by_model.values()):
        raise RuntimeError("confirmation models evaluated different cases")
    case_path = OUT / "confirmation-cases-snapshot.jsonl"
    if not case_path.is_file():
        case_path = OUT.parent / "v133-goal-ab-evidence-decision" / "fresh-confirm-cases.jsonl"
    access = json.loads((OUT / "confirmation-access-record.json").read_text(encoding="utf-8"))
    if hashlib.sha256(case_path.read_bytes()).hexdigest() != access["case_sha256"]:
        raise RuntimeError("confirmation case snapshot differs from the frozen evaluated file")
    cases = {str(row["id"]): row for row in read_jsonl(case_path)}
    if set(cases) != expected_cases:
        raise RuntimeError("confirmation case snapshot does not match evaluation IDs")
    review_map = {(row["model"], row["case_id"]): row["review_id"] for row in mapping}
    protected = set()
    for case_id, row in raw_by_model["baseline"].items():
        if by_case[("baseline", case_id)] != "correct_sufficient":
            continue
        titles = cases[case_id].get("support_titles", [])
        if len(titles) != 2:
            continue
        first_steps = {}
        for index, step in enumerate(row["trace"]):
            for doc in step.get("observation", {}).get("result", []):
                first_steps.setdefault(doc["title"], index)
        final_docs = packet_by_id[review_map[("baseline", case_id)]]["visible_documents"]
        visible_titles = {doc["title"] for doc in final_docs}
        if all(title in first_steps and title in visible_titles for title in titles) and \
                first_steps[titles[0]] != first_steps[titles[1]]:
            protected.add(case_id)
    write_jsonl(OUT / "confirmation-paired-results.jsonl", [
        {"case_id": case_id, "question": raw_by_model["baseline"][case_id]["question"],
         "protected_continuous_search": case_id in protected,
         "models": {label: {"answer": raw[case_id]["answer"],
                            "category": by_case[(label, case_id)],
                            "em": raw[case_id]["em"], "searches": raw[case_id]["searches"]}
                    for label, raw in raw_by_model.items()}}
        for case_id in sorted(expected_cases)])
    seeds = {}
    for seed in (42, 43):
        candidate = scores[f"{seed}-B"]
        gained = sorted(case_id for case_id in expected_cases
                        if by_case[("baseline", case_id)] != "correct_sufficient"
                        and by_case[(f"{seed}-B", case_id)] == "correct_sufficient")
        lost = sorted(case_id for case_id in expected_cases
                      if by_case[("baseline", case_id)] == "correct_sufficient"
                      and by_case[(f"{seed}-B", case_id)] != "correct_sufficient")
        protected_lost = sorted(protected & set(lost))
        seeds[str(seed)] = {
            "success_gain": candidate["correct_sufficient"] - baseline["correct_sufficient"],
            "bad_total_nonincrease": candidate["bad_total"] <= baseline["bad_total"],
            "invalid_nonincrease": candidate["invalid"] <= baseline["invalid"],
            "improvement_gate_passed": candidate["correct_sufficient"] >= baseline["correct_sufficient"] + 4,
            "gained_cases": gained,
            "lost_cases": lost,
            "protected_lost_cases": protected_lost,
            "protected_preservation_passed": len(protected_lost) <= 1,
        }
    passed = all(row["improvement_gate_passed"] and row["bad_total_nonincrease"]
                 and row["invalid_nonincrease"] and row["protected_preservation_passed"]
                 for row in seeds.values())
    return {
        "status": "complete_published" if passed else "complete_confirmation_failed",
        "selected_route": "B",
        "published": passed,
        "published_adapter": str(OUT / "seed42-B" / "adapter-final") if passed else None,
        "retained_product_adapter": "Goal Z full-trajectory seed42",
        "scores": scores,
        "seed_gates": seeds,
        "protected_cases": sorted(protected),
        "protection_rule": "same-rubric baseline successes, two frozen support pages first returned at different search steps, both visible in reviewed final input",
    }


if __name__ == "__main__":
    main()
