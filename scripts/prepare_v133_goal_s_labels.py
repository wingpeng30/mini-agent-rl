"""Freeze the Goal S training-only label audit from Goal R's visible traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/v133-goal-r-evidence-order/train-inputs.jsonl"
OUT = ROOT / "runs/v133-goal-s-label-granularity"

# These two changes follow the same rule: both named entities' visible passages
# explicitly contain the narrower common type. No development answer is used.
REVISIONS = {
    "5a8b8eec55429949d91db5e0": (
        "rock band", "Both passages explicitly identify a rock band; punk is only supported for SWMRS."),
    "5ab3a3c4554299233954ff6e": (
        "film director", "Both passages explicitly identify a film director."),
}
AMBIGUITIES = {
    "5ab563365542992aa134a304": "One passage says documentary film; the other says documentary. Keep the shared visible wording.",
    "5abadb4355429939ce03dd61": "Only Josh Trank's visible passage specifies film director; Jonathan Frakes is called director.",
    "5adfeffb55429942ec259ba0": "Question asks industry; film is an industry label, not a profession to expand.",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path = SOURCE) -> dict:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 16 or len({x["case_id"] for x in rows}) != 16:
        raise RuntimeError("Expected exactly 16 unique training inputs")
    if set(REVISIONS) - {x["case_id"] for x in rows}:
        raise RuntimeError("Revision case missing from frozen inputs")
    audit = []
    for row in rows:
        case_id = row["case_id"]
        original = json.loads(row["target_raw"])["answer"]
        revised, reason = REVISIONS.get(case_id, (original, AMBIGUITIES.get(case_id, "Original label matches the common attribute visible for both entities.")))
        documents = []
        for step in row["rollout"]["trace"]:
            for document in step.get("observation", {}).get("result", []):
                documents.append({"id": document["id"], "title": document["title"], "text": document["text"]})
        if not row["eligible"] or not documents:
            raise RuntimeError(f"Missing visible evidence: {case_id}")
        audit.append({"case_id": case_id, "question": row["rollout"]["question"],
                      "original_answer": original, "revised_answer": revised,
                      "changed": revised != original, "reason": reason,
                      "visible_documents": documents})
    return {"version": "goal-s-labels-v1", "source_sha256": sha(source),
            "rule": "Keep the question's common attribute and any domain qualifier explicitly supported by both visible entities; retain ambiguous originals.",
            "changes": sum(x["changed"] for x in audit), "cases": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT / "labels.json")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    item = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(args.out), "changes": item["changes"]}))


if __name__ == "__main__":
    main()
