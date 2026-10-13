"""Review the eight new blinded answer/evidence combinations after Goal U training."""
from __future__ import annotations

import json

import run_v133_goal_u_v_evidence_preference as uv

NEW = {
    ("g010", "promoter and manager"):
        ("unsupported_or_wrong", "one_sided", "Only Chuck Wein is described as a promoter and manager; Edie Sedgwick is an actress and model."),
    ("g010", "actress"):
        ("unsupported_or_wrong", "one_sided", "Only Edie Sedgwick is described as an actress; Chuck Wein is a promoter and manager."),
    ("g009", "Democratic Party"):
        ("sufficient", None, "Both passages support Democratic Party affiliation at the September 2004 primary; one calls Clyne a Democrat."),
    ("g018", "American attorney and politician"):
        ("unsupported_or_wrong", "one_sided", "Tim Kaine is described as an American attorney and politician; Aneesh Chopra is described as an American executive and technology officer."),
    ("g001", "director"):
        ("missing_qualifier", None, "Both passages also identify screenwriter; the question asks for common occupations in the plural, so director alone is incomplete."),
    ("5a73583e55429901807dafd0", "musician"):
        ("sufficient", None, "The Rev is explicitly called a musician; Mike Sparks is identified as a guitarist and vocalist in Canterbury, which establishes the shared musician occupation."),
    ("g018", "politician"):
        ("unsupported_or_wrong", "one_sided", "Tim Kaine is described as a politician and senator; the Aneesh Chopra passage describes an executive and technology officer, without evidence of politician as the shared role."),
    ("g001", "film director and screenwriter"):
        ("sufficient", None, "Both James Young and Michael Jones are explicitly identified as film directors and screenwriters; the full plural answer is supplied."),
}


def main():
    packet = uv.rows(uv.OUT / "old-blind-packet.jsonl")
    base = {x["key"]: x for x in uv.rows(uv.OUT / "baseline-blind-votes.jsonl")}
    new_seen = {(x["case_id"], x["answer"]) for x in packet if x["key"] not in base}
    if new_seen != set(NEW): raise RuntimeError(f"Unexpected unseen answer/evidence combinations: {new_seen ^ set(NEW)}")
    output = uv.OUT / "old-blind-votes.jsonl"
    if output.exists(): raise FileExistsError(output)
    with output.open("w", encoding="utf-8") as stream:
        for row in packet:
            if row["key"] in base:
                value = base[row["key"]]
            else:
                category, subtype, reason = NEW[(row["case_id"], row["answer"])]
                # The two required entities were actually retrieved in every
                # completed product trace; cite those documents, not a gold map
                # embedded in the model input.
                case = next(x for x in uv.q.dev_cases() if x["case_id"] == row["case_id"])
                visible = {x["id"] for x in row["visible_documents"]}
                citations = [x for x in case["support_ids"] if x in visible]
                if len(citations) != 2: raise RuntimeError("Relevant passages not actually visible")
                value = {"key": row["key"], "case_id": row["case_id"],
                         "category": category, "subtype": subtype,
                         "reason": reason, "cited_doc_ids": citations}
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"votes": str(output), "unique": len(packet), "new": len(NEW)}))


if __name__ == "__main__": main()
