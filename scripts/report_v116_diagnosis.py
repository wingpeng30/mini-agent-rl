"""Join blinded evidence judgments to v115 score deltas only after review."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def support_change(before, after):
    if before not in {"supported", "unsupported"} or after not in {"supported", "unsupported"}:
        return "unresolved"
    if before == after:
        return "no_change"
    return "improved" if after == "supported" else "regressed"


def main():
    score = read(ROOT / "reports/v116-v115-diagnosis.json")
    merge = read(ROOT / "reports/v116-v115-semantic-merge.json")
    mapping = read(ROOT / "reports/v115-private-run/packets/private-mapping.json")
    old = read(ROOT / "reports/v115-private-run/semantic-merge.json")
    semantic = {}
    for row in merge["cases_detail"]:
        private = mapping["states" if row["kind"] == "state" else "tasks"][row["audit_id"]]
        item = {"source": "v116_corrected_two_review", "models": {}}
        for alias, value in row["candidates"].items():
            model = private["anonymous_to_model"][alias]
            item["models"][model] = value["evidence_support"] if value["status"] == "agreed" else "unresolved"
        semantic[private["state_id"] if row["kind"] == "state" else private["task_id"]] = item
    for row in old["joined_reviews"]:
        private = mapping["states" if row["section"] == "states" else "tasks"][row["audit_id"]]
        key = private["state_id"] if row["section"] == "states" else private["task_id"]
        if key in semantic:
            continue
        semantic[key] = {"source": "v115_prior_agreement_revalidated_visible_quote", "models": {
            vote["model"]: vote["unanimous"] if vote["unanimous"] in {"supported", "unsupported"} else "unresolved"
            for vote in row["candidate_reviews"]}}
    states = []
    for row in score["state_cases"]:
        item = semantic.get(row["state_id"], {"source": "missing", "models": {}})
        models = item["models"]
        states.append({**row, "semantic_support": models,
                       "evidence_support_change": support_change(models.get("v101"), models.get("v115")),
                       "review_source": item["source"]})
    tasks = []
    for row in score["task_official_changes"]:
        item = semantic.get(row["task_id"], {"source": "missing", "models": {}})
        models = item["models"]
        tasks.append({**row, "semantic_support": models,
                      "evidence_support_change": support_change(models.get("v101"), models.get("v115")),
                      "review_source": item["source"]})
    report = {"version": "v116-v115-score-vs-evidence-v1", "state_cases": states,
              "changed_task_cases": tasks,
              "state_evidence_change_counts": dict(Counter(r["evidence_support_change"] for r in states)),
              "changed_task_official_counts": dict(Counter(r["official_change"] for r in tasks)),
              "changed_task_evidence_counts": dict(Counter(r["evidence_support_change"] for r in tasks)),
              "diagnosis_cases_total": len(semantic), "corrected_cases": len(merge["cases_detail"]),
              "prior_retained_cases": len(semantic) - len(merge["cases_detail"]),
              "score_is_not_evidence_support": True}
    path = ROOT / "reports/v116-v115-score-vs-evidence.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"state_cases", "changed_task_cases"}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
