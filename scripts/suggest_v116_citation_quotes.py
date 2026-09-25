"""Offer non-authoritative exact-text suggestions for blinded citation repairs."""
from __future__ import annotations

import difflib
import json
from pathlib import Path

from scripts.review_v116_api import expected_docs

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evidence-sft-v116/diagnosis-blind/rereview"


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def main():
    packets = {r["audit_id"]: r for r in rows(BASE / "selected.jsonl")}
    audits = {r["audit_id"]: r for path in BASE.glob("luna-batch-*.jsonl") for r in rows(path)}
    suggestions = []
    for error in rows(BASE / "citation-errors.jsonl"):
        key, alias, index = error["audit_id"], error["alias"], error["citation_index"]
        quote = audits[key]["candidates"][alias]["citations"][index]
        source = expected_docs(packets[key], alias).get((quote.get("step"), quote.get("stage"), quote.get("document_id")))
        suggestion = None
        if source and quote.get("quote"):
            wanted = quote["quote"]
            match = difflib.SequenceMatcher(None, wanted, source, autojunk=False).find_longest_match()
            left = max(0, match.b - match.a)
            right = min(len(source), left + len(wanted))
            sample = source[left:right]
            suggestion = {"quote": sample,
                          "similarity": round(difflib.SequenceMatcher(None, wanted, sample).ratio(), 4)}
        suggestions.append({**error, "original_quote": quote.get("quote"), "suggestion": suggestion,
                            "requires_independent_reviewer_confirmation": True})
    target = BASE / "citation-suggestions.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in suggestions), encoding="utf-8")
    print(json.dumps({"suggestions": len(suggestions), "with_candidate": sum(bool(x["suggestion"]) for x in suggestions),
                      "file": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
