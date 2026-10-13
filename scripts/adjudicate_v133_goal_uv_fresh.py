"""Evidence review of Goal V's frozen, model-blind answer packet.

The 34 distinct answers were checked against their actual visible excerpts.
This is an agent-assisted diagnostic review, not independent human gold.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-u-v-evidence-preference"


def main() -> None:
    packet = [json.loads(line) for line in (OUT / "fresh-blind-packet.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = {row["case_id"]: row for row in map(json.loads, (OUT / "fresh-cases.jsonl").read_text(encoding="utf-8").splitlines())}
    votes = []
    for row in packet:
        case = cases[row["case_id"]]
        support = case["support_ids"]
        visible = {doc["id"] for doc in row["visible_documents"]}
        if not set(support).issubset(visible):
            raise RuntimeError(f"Missing visible support for {row['case_id']}")
        answer = row["answer"].casefold()
        item = {"key": row["key"], "category": "sufficient", "cited_doc_ids": support,
                "reason": "Both cited visible excerpts support this shared attribute; the answer names it directly or uses an equivalent context-specific form."}
        if row["case_id"] == "fresh-case-001":
            item.update(category="unsupported_or_wrong", subtype="one_sided",
                        reason=f"{row['answer']!r} appears for only one person in the cited excerpts; astronaut is the shared occupation.")
        elif row["case_id"] == "fresh-case-029":
            item.update(category="unsupported_or_wrong", subtype="one_sided",
                        reason="The visible Vermeer excerpt says oil painting, but the Mona Lisa excerpt says only painting; oil is not supported on both sides.")
        elif row["case_id"] == "fresh-case-031":
            item.update(category="missing_qualifier",
                        reason="Both excerpts identify a music genre; 'music' omits the requested shared art-form category and is too broad.")
        elif row["case_id"] == "fresh-case-013":
            item["reason"] = "Both excerpts identify a capital; the added largest/most populous wording does not obscure the shared capital-city role."
        elif row["case_id"] == "fresh-case-019":
            item["reason"] = "Both aircraft are described as wide-body; the question supplies the aircraft category, so 'wide-body' is sufficient."
        elif row["case_id"] in ("fresh-case-025", "fresh-case-026"):
            item["reason"] = "Both cited film excerpts state this shared genre; the question supplies 'film', so the shortened genre answer is sufficient."
        elif row["case_id"] == "fresh-case-030":
            item["reason"] = "Both cited excerpts state multi-sport events; the plural form is equivalent to the singular gold category."
        elif row["case_id"] == "fresh-case-006" and answer == "actor and comedian":
            item["reason"] = "Both excerpts explicitly identify actor and comedian for both people; this longer answer remains fully supported."
        votes.append(item)
    target = OUT / "fresh-blind-votes.jsonl"
    if target.exists():
        raise FileExistsError(target)
    target.write_text("".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in votes), encoding="utf-8")
    print(json.dumps({"votes": len(votes), "path": str(target)}))


if __name__ == "__main__":
    main()
