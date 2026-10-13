"""Evidence judgments for the 32 blinded Goal U/V starting-model answers."""
from __future__ import annotations

import json
from pathlib import Path

import run_v133_goal_u_v_evidence_preference as uv

SUFFICIENT = {
    "5a7789215542997042120a99", "5a842f375542990548d0b25c",
    "5a8673585542991e77181662", "5a8c717f5542995e66a475ef",
    "5a8d92a3554299068b959d40", "5ac00c72554299012d1db589",
    "5ac169c25542994ab5c67d1a", "5ac259bc5542992f1f2b386a",
    "5add23bd5542994734353870", "5add27645542992ae4cec4b5",
    "5adde5005542990dbb2f7eee", "5ae0e4a55542990adbacf6b6",
    "5ae30de055429928c42395cf", "5ae3be675542992e3233c46e",
    "g003", "g008", "g009", "g011", "g012", "g013", "g014", "g015", "g017",
}
MISSING = {
    "g001": "The question asks for common occupations in the plural. Both passages also identify screenwriter; film director alone omits that required common role.",
    "g004": "Both passages explicitly specify women's magazine; magazine omits the distinguishing type requested by the question.",
    "g006": "Both passages explicitly specify rock band; band omits the evidenced type of group.",
}
WRONG = {
    "5a73583e55429901807dafd0": "The Rev is a drummer, but Mike Sparks is listed as guitarist and vocalist; the Canterbury drummer is Chris Velissarides.",
    "5a78cc3f554299029c4b5e9b": "Pearl Lowe is a designer, but the Lukas Rossi passage identifies a singer/musician, not a designer.",
    "g000": "Takako Honda is called a voice actress, while Milla Jovovich is called an actress without voice-acting evidence.",
    "g005": "Isaac Hayes is a singer-songwriter, but Albert Raisner is described as a harmonica player, TV/radio host and producer.",
    "g010": "Promoter and manager of entertainment acts is Chuck Wein's role; Edie Sedgwick is an actress and model.",
    "g018": "Tim Kaine is a senator; Aneesh Chopra is described as an executive and technology officer.",
}


def main():
    packet = uv.rows(uv.OUT / "baseline-blind-packet.jsonl")
    expected = SUFFICIENT | set(MISSING) | set(WRONG)
    if len(packet) != 32 or {row["case_id"] for row in packet} != expected:
        raise RuntimeError("Baseline evidence judgment list does not cover exactly 32 cases")
    cases = {x["case_id"]: x for x in uv.q.dev_cases()}
    output = uv.OUT / "baseline-blind-votes.jsonl"
    if output.exists(): raise FileExistsError(output)
    with output.open("w", encoding="utf-8") as stream:
        for row in packet:
            case_id = row["case_id"]
            visible = {d["id"]: d for d in row["visible_documents"]}
            support = [x for x in cases[case_id]["support_ids"] if x in visible]
            if len(support) != 2: raise RuntimeError(f"Support not actually visible {case_id}")
            if case_id in SUFFICIENT:
                category, subtype = "sufficient", None
                reason = (f"The visible passages for {visible[support[0]]['title']} and "
                          f"{visible[support[1]]['title']} both establish the asked common answer "
                          f"'{row['answer']}' (including an equivalent shorter wording where applicable).")
            elif case_id in MISSING:
                category, subtype, reason = "missing_qualifier", None, MISSING[case_id]
            else:
                category, subtype, reason = "unsupported_or_wrong", "one_sided", WRONG[case_id]
            stream.write(json.dumps({"key": row["key"], "case_id": case_id,
                "category": category, "subtype": subtype, "reason": reason,
                "cited_doc_ids": support}, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"votes": str(output), "sufficient": len(SUFFICIENT),
                      "missing": len(MISSING), "one_sided": len(WRONG)}))


if __name__ == "__main__": main()
