"""Rebuild Goal AG's corrected ledger with citations from the actual final prompt."""
from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from pathlib import Path

import run_v133_goal_af_retrieval_validation as af

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ag-failure-repair"
PACKET = af.OUT / "anonymous-semantic-review-packet.jsonl"
RAW_REVIEWS = (OUT / "review-first-half.jsonl", OUT / "review-second-half.jsonl")
SOURCE_LEDGER = OUT / "correction-ledger-final.jsonl"
DEST = OUT / "correction-ledger-final-v2.jsonl"

# The prior lead pass inferred this county from world knowledge, but that
# relation is not present in the final model input. Preserve the answer as an
# unsupported response under the frozen visible-evidence rule.
OVERRIDES = {
    "musique:2hop__506035_325512": {
        "category": "unsupported_by_visible_evidence",
        "reason": "The final input says Larkin was born in Poplarville and separately mentions Pearl River County among his police positions, but it does not state that Poplarville is in Pearl River County. The answer is not established by the visible evidence.",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def docs_in_prompt(prompt: str) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for match in re.finditer(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", prompt, re.DOTALL):
        payload = json.loads(match.group(1))
        for doc in payload.get("result") or []:
            if isinstance(doc, dict) and doc.get("id") is not None:
                found[str(doc["id"])] = doc
    return found


def main() -> dict:
    packet_rows = read_jsonl(PACKET)
    packet = {row["review_id"]: row for row in packet_rows}
    raw = [row for path in RAW_REVIEWS for row in read_jsonl(path)]
    raw_by_id = {row["review_id"]: row for row in raw}
    if len(packet) != 624 or len(raw_by_id) != 624 or set(packet) != set(raw_by_id):
        raise RuntimeError("Expected complete, matching 624-row packet and independent reviews")

    source = read_jsonl(SOURCE_LEDGER)
    if len(source) != 624:
        raise RuntimeError("Expected the existing 624-row lead-adjudicated ledger")
    output = []
    invalid_successes = []
    for prior in source:
        row = dict(prior)
        review = raw_by_id[row["review_id"]]
        case = packet[row["review_id"]]
        final_prompt = (case.get("calls") or [])[-1].get("actual_prompt", "")
        visible_docs = docs_in_prompt(final_prompt)

        override = OVERRIDES.get(str(row["case_id"])) if row.get("arm") == "A" else None
        if override:
            row.update(override)

        if row.get("category") == "correct_sufficient":
            selected_ids = []
            for quote in row.get("evidence_quotes") or []:
                if isinstance(quote, dict) and quote.get("source_id"):
                    selected_ids.append(str(quote["source_id"]))
            if not selected_ids and review.get("category") == "correct_sufficient":
                selected_ids = [str(q["source_id"]) for q in review.get("evidence_quotes") or []
                                if isinstance(q, dict) and q.get("source_id")]

            # These eight baseline upgrades were manually adjudicated against
            # their final prompts. Supply the exact visible passage as an
            # auditable citation instead of inheriting an empty quote list.
            if not selected_ids and row.get("arm") == "A":
                manual_sources = {
                    "musique:2hop__864101_135373": ["11c75d69418f832f2c81e009"],
                    "musique:2hop__703493_162341": ["bac49d947f636c1889289acf"],
                    "2wiki:4b9f70180bda11eba7f7acde48001122": ["9c6f0119569c8211a8b689fb"],
                    "2wiki:ef9d40a40bdb11eba7f7acde48001122": ["74ef23844d4985c143d5cb79"],
                    "2wiki:d94d1d720bdd11eba7f7acde48001122": ["9a2ec60145d4b02646c49305"],
                    "2wiki:2e92333a0bde11eba7f7acde48001122": ["d504b488e188e6e003db5c99"],
                    "2wiki:8796fb620bdb11eba7f7acde48001122": ["956c438ea4fdbda813ea0ed7"],
                }
                selected_ids = manual_sources.get(str(row["case_id"]), [])

            citations = []
            for source_id in dict.fromkeys(selected_ids):
                doc = visible_docs.get(source_id)
                if not doc or not doc.get("text"):
                    invalid_successes.append({"review_id": row["review_id"], "case_id": row["case_id"],
                                              "arm": row.get("arm"), "source_id": source_id,
                                              "issue": "citation source absent from final actual prompt"})
                    continue
                first_call = next((i for i, call in enumerate(case.get("calls") or [])
                                   if any(str(d.get("id")) == source_id
                                          for d in call.get("actual_tool_results") or []
                                          if isinstance(d, dict))), None)
                citations.append({"call_index": first_call, "source_id": source_id,
                                  "text": str(doc["text"])})

            if not citations:
                row.update(category="unsupported_by_visible_evidence",
                           reason="A success classification requires a source passage actually visible in the final answer input; no cited support passage could be recovered.",
                           evidence_quotes=[], first_sufficient_call=None)
                invalid_successes.append({"review_id": row["review_id"], "case_id": row["case_id"],
                                          "arm": row.get("arm"), "issue": "success downgraded: no final-input citation"})
            else:
                row["evidence_quotes"] = citations
                row["first_sufficient_call"] = max(int(x["call_index"]) for x in citations
                                                    if x["call_index"] is not None)
                row["actual_prompt_sha256"] = hashlib.sha256(final_prompt.encode("utf-8")).hexdigest()
        output.append(row)

    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output),
                    encoding="utf-8")

    # Validate every resulting success against documents in the actual final
    # prompt, including citations whose text had to be recovered from JSON.
    final_categories = Counter(row["category"] for row in output)
    summary = {"rows": len(output), "categories": dict(final_categories),
               "successes_without_final_input_citations": len(invalid_successes),
               "citation_recovery_or_downgrade_log": invalid_successes,
               "source_ledger_sha256": af.sha(SOURCE_LEDGER), "packet_sha256": af.sha(PACKET)}
    summary_path = OUT / "corrected-baseline-summary-v2.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
