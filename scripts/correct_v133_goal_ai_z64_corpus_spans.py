"""Version source quote offsets against exact frozen-corpus document bodies."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ai-z64-audit-repair-v7"
CORPUS = ROOT / "runs/v133-goal-z-multihop-sft/corpus-dev.json"


def main():
    old = OUT / "edge-evidence-ledger.final.jsonl"
    snapshot = OUT / "edge-evidence-ledger.pre-span-correction.jsonl"
    correction_path = OUT / "source-span-corrections.jsonl"
    if snapshot.exists() or correction_path.exists():
        raise FileExistsError("span-correction stage already ran")
    rows = [json.loads(s) for s in old.read_text(encoding="utf-8").splitlines() if s.strip()]
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    docs = {str(x["id"]): x for x in corpus}
    corrections = []
    for row in rows:
        quote = row.get("quote")
        if not quote:
            continue
        body = str(docs[str(row["document_id"])].get("text", ""))
        positions = []
        pos = body.find(quote)
        while pos >= 0:
            positions.append(pos)
            pos = body.find(quote, pos + 1)
        if not positions:
            corrections.append({"case_id": row["case_id"], "edge_id": row["edge_id"],
                                "document_id": row["document_id"], "status": "unresolved_quote_not_found",
                                "old_span": row.get("source_char_span"), "reason": "Exact citation string not found in frozen document body."})
            continue
        old_span = row.get("source_char_span")
        old_start = old_span[0] if old_span else positions[0]
        new_start = min(positions, key=lambda x: (abs(x - old_start), x))
        new_span = [new_start, new_start + len(quote)]
        if old_span != new_span:
            corrections.append({"case_id": row["case_id"], "edge_id": row["edge_id"],
                                "document_id": row["document_id"], "old_span": old_span,
                                "new_span": new_span, "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                                "reason": "Recomputed exact character offsets in the frozen corpus body identified by document_id; quote text and semantic decision unchanged.",
                                "reviewer": "primary-agent provenance correction", "diagnostic_only": True,
                                "training_allowed": False})
            row["source_char_span"] = new_span
            row["review_history"] = str(row.get("review_history", "")) + "; corpus-body span corrected in versioned provenance pass"
    if any(x.get("status") == "unresolved_quote_not_found" for x in corrections):
        raise RuntimeError("Some quotes could not be found; preserving original ledger and stopping.")
    snapshot.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
    correction_path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in corrections), encoding="utf-8")
    old.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")
    print(json.dumps({"corrected_spans": len(corrections), "edge_rows": len(rows), "quote_texts_changed": 0,
                      "semantic_decisions_changed": 0, "snapshot": snapshot.name}, ensure_ascii=False))


if __name__ == "__main__":
    main()

