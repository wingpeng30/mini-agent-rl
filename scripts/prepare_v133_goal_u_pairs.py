"""Freeze training-only evidence preference pairs for Goal U.

This script deliberately reads only Goal S's training labels and Goal R's
persisted training inputs. It never imports evaluation data or a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-u-v-evidence-preference"
LABELS = ROOT / "runs/v133-goal-s-label-granularity/labels.json"
INPUTS = ROOT / "runs/v133-goal-r-evidence-order/train-inputs.jsonl"


# One manually audited rejected answer per included case. Each negative is a
# property explicitly stated for one entity in the visible passages, while
# the passages do not establish it for the other entity. “Unsupported” here
# is evidential, not a claim that the property is false in the world.
SELECTIONS = {
    "5adcea155542994d58a2f713": {
        "chosen_docs": ["d17cb93661348e9ede8c", "61d0afb6a32e2a7f785d"],
        "rejected": "actor",
        "negative_doc": "d17cb93661348e9ede8c",
        "counterpart_doc": "61d0afb6a32e2a7f785d",
        "negative_entity": "Mack V. Wright",
        "reason": "The question asks for a common profession. Mack V. Wright is explicitly an actor and film director; Daniel Myrick is explicitly a film director, with no actor claim in the visible passage. The rejected profession is directly relevant but one-sided.",
    },
    "5a89297255429951533612cc": {
        "chosen_docs": ["c90454ca48f61a3dbcec", "5f9a4bf57661a5746880"],
        "rejected": "vocalist",
        "negative_doc": "c90454ca48f61a3dbcec",
        "counterpart_doc": "5f9a4bf57661a5746880",
        "negative_entity": "Aidan Moffat",
        "reason": "The question asks what both people are. Aidan Moffat is explicitly a vocalist and musician; Vic Ruggiero is explicitly a musician, songwriter, and producer, with no vocalist claim in the visible passage. The rejected profession is related and one-sided in the evidence.",
    },
    "5a8b8eec55429949d91db5e0": {
        "chosen_docs": ["66de641f1e683fa04efc", "b4a1c23f8f387338b26d"],
        "rejected": "punk rock band",
        "negative_doc": "b4a1c23f8f387338b26d",
        "counterpart_doc": "66de641f1e683fa04efc",
        "negative_entity": "SWMRS",
        "reason": "The question asks for the group type in common. SWMRS is explicitly a punk rock band; We the Kings is explicitly a rock band, without the punk qualifier in the visible passage. The rejected subtype is stated for only one side. This records missing evidence for the qualifier, not proof that We the Kings is never punk rock.",
    },
    "5ab3a3c4554299233954ff6e": {
        "chosen_docs": ["5f7803a602b41ff15acf", "295a995223c93622615e"],
        "rejected": "actor",
        "negative_doc": "5f7803a602b41ff15acf",
        "counterpart_doc": "295a995223c93622615e",
        "negative_entity": "Djibril Diop Mambéty",
        "reason": "The question asks what the two people have in common. Mambéty is explicitly an actor and film director; Jacques Tourneur is explicitly a film director, with no actor claim in the visible passage. The rejected profession is directly relevant but one-sided.",
    },
    "5abe6da155429976d4830b19": {
        "chosen_docs": ["69dee9d554a19cc07868", "062513133b2c70b42e19"],
        "rejected": "crime writer",
        "negative_doc": "69dee9d554a19cc07868",
        "counterpart_doc": "062513133b2c70b42e19",
        "negative_entity": "William Kent Krueger",
        "reason": "The question asks for a shared attribute. Krueger is explicitly an author and crime writer; Borges is explicitly a short-story writer, essayist, poet, and translator, with no crime-writer claim in the visible passage. This is a directly related but one-sided writer subtype.",
    },
    "5a90923f5542995651fb5192": {
        "chosen_docs": ["b968ab4d58bbee52c394", "fa6b124fbd575ba29582"],
        "rejected": "county-level city",
        "negative_doc": "fa6b124fbd575ba29582",
        "counterpart_doc": "b968ab4d58bbee52c394",
        "negative_entity": "Linqing",
        "reason": "The question asks what the places have in common. Linqing is explicitly a county-level city; Luzhou is explicitly a prefecture-level city. Both are cities, but the county-level designation is supported only for Linqing and is not supported for Luzhou.",
    },
    "5abadb4355429939ce03dd61": {
        "chosen_docs": ["3e4c679cb366348b36a7", "0660e32d1b1871851b7a"],
        "rejected": "screenwriter",
        "negative_doc": "3e4c679cb366348b36a7",
        "counterpart_doc": "0660e32d1b1871851b7a",
        "negative_entity": "Josh Trank",
        "reason": "The question asks what both people are. Josh Trank is explicitly a film director and screenwriter; Jonathan Frakes is explicitly an actor, author, and director, with no screenwriter claim in the visible passage. The rejected profession is directly relevant but one-sided.",
    },
    "5ae30bfb55429928c42395c7": {
        "chosen_docs": ["ce29ac7a9f9895d6ea08", "4a30a21db78798b664ec"],
        "rejected": "animator",
        "negative_doc": "ce29ac7a9f9895d6ea08",
        "counterpart_doc": "4a30a21db78798b664ec",
        "negative_entity": "Trey Parker",
        "reason": "The question asks for an attribute the two people share. Trey Parker's passage explicitly lists animator among his occupations; Pietro Germi's passage lists actor, screenwriter, and director but does not support animator. Animator is an occupation directly relevant to the open-ended common-attribute question, and is evidenced on only one side. This marks evidence support, not real-world falsity.",
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {row["case_id"]: row for row in rows}
    if len(rows) != 16 or len(result) != 16:
        raise RuntimeError("Expected exactly 16 unique Goal R training inputs")
    return result


def build() -> tuple[dict, dict]:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    inputs = read_inputs(INPUTS)
    cases = {case["case_id"]: case for case in labels["cases"]}
    if len(cases) != 16 or set(cases) != set(inputs):
        raise RuntimeError("Goal S labels and Goal R inputs must match all 16 training cases")
    if labels.get("source_sha256") != sha(INPUTS):
        raise RuntimeError("Goal S labels do not reference the current Goal R training-input file")

    pairs = []
    audit_cases = []
    for case_id, case in cases.items():
        source_input = inputs[case_id]
        if not source_input.get("eligible"):
            raise RuntimeError(f"Ineligible frozen training context: {case_id}")
        if len(source_input["input_ids"]) != len(source_input["input_mask"]):
            raise RuntimeError(f"Input ids/mask length mismatch: {case_id}")
        if len(source_input["input_ids"]) > 2048:
            raise RuntimeError(f"Frozen training input exceeds product context limit: {case_id}")

        selection = SELECTIONS.get(case_id)
        docs = {doc["id"]: doc for doc in case["visible_documents"]}
        if selection:
            positive_docs = [docs[doc_id] for doc_id in selection["chosen_docs"]]
            if len(positive_docs) != 2 or any(
                case["revised_answer"].casefold() not in doc["text"].casefold()
                and case["revised_answer"].casefold() not in doc["title"].casefold()
                for doc in positive_docs
            ):
                raise RuntimeError(f"Chosen answer is not visible in both entity passages: {case_id}")
            negative_doc = docs[selection["negative_doc"]]
            counterpart_doc = docs[selection["counterpart_doc"]]
            if selection["rejected"].casefold() not in negative_doc["text"].casefold():
                raise RuntimeError(f"Rejected answer is not verbatim in its cited passage: {case_id}")
            pair = {
                "case_id": case_id,
                "question": case["question"],
                "chosen": case["revised_answer"],
                "rejected": selection["rejected"],
                "input_ids": source_input["input_ids"],
                "input_mask": source_input["input_mask"],
                "chosen_evidence": [
                    {"doc_id": doc["id"], "title": doc["title"], "text": doc["text"]}
                    for doc in positive_docs[:2]
                ],
                "rejected_evidence": {
                    "doc_id": negative_doc["id"],
                    "title": negative_doc["title"],
                    "entity": selection["negative_entity"],
                    "text": negative_doc["text"],
                },
                "other_entity_evidence": {
                    "doc_id": counterpart_doc["id"],
                    "title": counterpart_doc["title"],
                    "text": counterpart_doc["text"],
                    "note": "This visible passage is the basis for marking the rejected attribute unsupported for the other entity; omission is not evidence of real-world falsity.",
                },
                "reason": selection["reason"],
                "evidence_scope": "visible training context only; unsupported for the other entity is not asserted false in the world",
                "answer_protocol": "Encode each answer with the existing Goal R answer_target convention: JSON {action: answer, answer: <text>} followed by <|im_end|>.",
            }
            pairs.append(pair)
            audit_cases.append({
                "case_id": case_id,
                "question": case["question"],
                "chosen": pair["chosen"],
                "decision": "included",
                "candidates_reviewed": [{
                    "candidate": pair["rejected"],
                    "evidence": pair["rejected_evidence"],
                    "decision": "eligible_one_sided_evidence_negative",
                    "reason": selection["reason"],
                }],
                "pair_index": len(pairs) - 1,
            })
        else:
            # Exclude cases with no clean one-sided answer on the queried
            # attribute. The audit explains the evidence-based exclusion.
            exclusion_reasons = {
                "5ab563365542992aa134a304": "The only initially considered negative adds the year 2010, while the question asks what type the works are. Year is not the queried type attribute, so it is excluded.",
                "5a8b98c85542995d1e6f1415": "Both passages identify the work as an opera; no comparably clear one-sided answer to the asked performance type is available.",
                "5ab5c27d5542997d4ad1f1da": "The passages support the same country, England. The visible one-sided band descriptor is unrelated to the country asked for.",
                "5a75b52f5542992d0ec05fe7": "Both passages identify the people as mathematicians; no other one-sided career answer to the requested scientific career is clearly supported.",
                "5abda14f55429965af743d90": "Both passages identify the animals as dogs. No one-sided animal answer can be separated from the shared animal category.",
                "5ab698c75542995eadef002c": "Both passages identify the entities as genera of flowering plants. Family or geographic details are not the asked common classification level.",
                "5adfeffb55429942ec259ba0": "Both passages support the film industry. The visible one-sided nationality or job details do not answer the industry question.",
                "5aba9cff5542994dbf01997e": "Both passages identify the entities as genera. The different plant-family detail is not a one-sided answer to the shared classification level.",
            }
            if case_id not in exclusion_reasons:
                raise RuntimeError(f"Missing explicit audit exclusion: {case_id}")
            reviewed = []
            if case_id == "5ab563365542992aa134a304":
                one_sided = docs["d7645758605e7c1d227e"]
                counterpart = docs["7ac07fcc813434f285b4"]
                reviewed.append({
                    "candidate": "2010 documentary",
                    "evidence": [
                        {"doc_id": one_sided["id"], "title": one_sided["title"], "text": one_sided["text"]},
                        {"doc_id": counterpart["id"], "title": counterpart["title"], "text": counterpart["text"]},
                    ],
                    "decision": "excluded_not_same_queried_attribute",
                    "reason": exclusion_reasons[case_id],
                })
            audit_cases.append({
                "case_id": case_id,
                "question": case["question"],
                "chosen": case["revised_answer"],
                "decision": "excluded",
                "candidates_reviewed": reviewed,
                "reason": exclusion_reasons[case_id],
            })

    if len(pairs) < 8:
        raise RuntimeError(f"Only {len(pairs)} defensible pairs; Goal U minimum is 8")
    if len(pairs) != 8 or len(audit_cases) != 16:
        raise RuntimeError("Unexpected pair or audit count")
    metadata = {
        "version": "goal-u-pairs-v1",
        "scope": "Goal S's 16 training labels and Goal R's 16 persisted training inputs only",
        "selection_rule": "At most one pair per training case; chosen is the frozen Goal S revised answer supported by both entity passages; rejected must be a directly relevant attribute explicitly stated for one entity but not supported for the other by its visible passage.",
        "interpretation": "Evidence absence is treated as unsupported, not false in the world.",
        "minimum_pairs": 8,
        "pair_count": len(pairs),
        "case_count": len(audit_cases),
        "hashes": {"labels": sha(LABELS), "training_inputs": sha(INPUTS)},
        "pairs": pairs,
    }
    audit = {
        "version": "goal-u-pair-audit-v1",
        "reviewer": "manual evidence audit encoded by the preparation script; not an independent human gold standard",
        "candidate_limit_per_case": 3,
        "included_count": len(pairs),
        "excluded_count": len(audit_cases) - len(pairs),
        "cases": audit_cases,
    }
    return metadata, audit


def main() -> None:
    if OUT.exists():
        if any(OUT.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty Goal U directory: {OUT}")
    else:
        OUT.mkdir(parents=True)
    pairs, audit = build()
    (OUT / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "pair-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": pairs["pair_count"], "included": audit["included_count"],
                      "excluded": audit["excluded_count"], "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
