"""Primary semantic triage for v123's 22 structurally rejected questions.

This binds each interpretation to the frozen source and annotated support facts.
It does not create new states, independent votes, or training labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.prepare_v116_data import full_rows
from scripts.build_v114_sft_candidates import support_facts

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data/evidence-sft-v124-forensic/remaining-unpaired-diagnostic-private.jsonl"
OUT = ROOT / "data/evidence-sft-v124-forensic/v123-unpaired-semantic-triage-private.jsonl"

# Codes: replay_candidate requires only structural/query work; rewrite_question must
# be re-authored and double reviewed; exclude_original rejects the frozen wording.
SPECS = [
 ("exclude_original", "Asking for an unspecified word common to two paragraphs has no task-specific answer: the gold is the stopword 'in'.", "Do not make a search/answer pair from this lexical-overlap prompt."),
 ("replay_candidate", "Both named works are explicitly called documentary films; structural non-pairability is separate from their shared genre.", "Try a bounded retrieval replay without changing the answer, then review any new pair."),
 ("replay_candidate", "The inventor Julius Kahn is linked to the company and his birthplace source explicitly says Germany.", "Replay a Julius Kahn birthplace query, retaining source identity."),
 ("rewrite_question", "Redinha is documented against an Anglo-Portuguese force, while the Waterloo fact is routed through Wellington's British victory. The question's 'what nation was allied against the French in both' needs a precise Britain/coalition relation.", "Ask for the British force/coalition component with battles named; verify both battles directly."),
 ("replay_candidate", "Nahuelito is placed in Nahuel Huapi Lake, and that lake is documented between Río Negro and Neuquén.", "Replay the lake-to-provinces chain in the frozen corpus."),
 ("replay_candidate", "Cara Delevingne is identified by the two films and her biography gives the requested birth date.", "Replay with the shared cast member as a grounded search target."),
 ("replay_candidate", "The source explicitly calls EMF British and Brand New American, so 'both American' is false as the gold no states.", "Replay country-of-origin evidence for both bands; preserve the negative comparison support."),
 ("rewrite_question", "The cited actor Shōhei Hino played Hideki Tōjō in Emperor and was born in Tokyo, but the question does not name that film; other portrayals are not excluded.", "Specify Emperor and the actor role, then review the birthplace chain."),
 ("exclude_original", "The federal-operation source concerns sites such as the Statue of Liberty and Ellis Island within New York Harbor, not federal operation of the harbor itself. The original question transfers an attribute from sites to their containing harbor.", "Remove the federal-operation clause and review a straightforward Port of New York and New Jersey question if needed."),
 ("replay_candidate", "Howard Burnham's brother is explicitly Frederick Russell Burnham, whose source says he helped inspire the international Scouting Movement.", "Replay the brother-to-movement chain."),
 ("replay_candidate", "North Williams Avenue's southern terminus is stated near Moda Center, and the arena is stated to have been called Rose Garden.", "Replay the road-to-arena-to-former-name chain."),
 ("rewrite_question", "The four-letter EGOT label is documented for a collection of four awards associated with composer Marvin Hamlisch, but 'achievement' is too vague and the date text in the composer passage is damaged.", "Ask explicitly for the four-award acronym and repair the source rendering before a fresh review."),
 ("replay_candidate", "The album photograph is attributed to Anton Corbijn and the biography calls him a Dutch photographer.", "Replay the photographer-to-nationality chain."),
 ("rewrite_question", "The source specifies Crestwood, Missouri in Greater St. Louis; the bare name Crestwood can denote other places.", "Add Missouri to the question, then replay."),
 ("replay_candidate", "The cited royal-family passage explicitly identifies Umberto II as Italy's last king and Queen Marie José as his wife.", "Replay or direct-answer using the same dated historical relation."),
 ("replay_candidate", "George Mason's acting passage names Home and Away, whose program passage identifies Alan Bateman as creator.", "Replay the actor-to-series-to-creator chain."),
 ("exclude_original", "The question says 23 World Rally Championships, but the cited biography says 23 individual world rallies and four drivers' championship titles. Two Lancia titles are 1987 and 1991. The gold person cannot validate the false 23-title premise.", "Replace with exact counts and relation only as a new question; do not salvage the frozen wording."),
 ("replay_candidate", "Skelewu names Emmanuel Adebayor as an endorser and his biography identifies him as Togolese.", "Replay the song-to-endorser-to-nationality chain."),
 ("replay_candidate", "Jodi Benson is tied to Hawaiian Vacation, and its passage identifies Gary Rydstrom as director of the 2011 Pixar short.", "Replay the actor/short/director chain."),
 ("rewrite_question", "The Professor source lists Betsy Brandt and Rick Peters as stars, but only Betsy Brandt is linked here to a CBS sitcom; the question does not uniquely select her and 'recently' has no date.", "Name Betsy Brandt or Life in Pieces and bound any time-sensitive wording."),
 ("replay_candidate", "John Sackville's biography links him to House of Anubis and the series source identifies it as a Nickelodeon mystery series.", "Replay the actor-to-program qualification chain."),
 ("rewrite_question", "The question misspells Tracie Thoms as Thomas and 'orrurred', and compares birth with when a film was 'produced' though the cited film passage supplies its 2006 film date. The 1975 birth-year answer is only safe after a precise event definition.", "Correct her name and ask which year is earlier: her 1975 birth or the film's 2006 release."),
]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [json.loads(line) for line in IN.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["cohort"] == "v123_internal_unpaired"]
    assert len(rows) == len(SPECS) == 22
    source = full_rows([r["source_id"] for r in rows])
    output = []
    for ordinal, (r, spec) in enumerate(zip(rows, SPECS), 1):
        disposition, rationale, repair = spec
        assert r["ordinal"] == ordinal and disposition in {"replay_candidate", "rewrite_question", "exclude_original"}
        original = source[r["source_id"]]
        assert original["question"] == r["question"] and original["answer"] == r["original_answer_private"]
        facts = [{"title": title, "quote": quote} for title, quote in support_facts(original)]
        output.append({
            "ordinal": ordinal, "source_id": r["source_id"], "question": r["question"],
            "original_answer_private": original["answer"], "source_facts": facts,
            "source_facts_sha256": hashlib.sha256(json.dumps(facts, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "disposition": disposition, "rationale": rationale, "next_action": repair,
            "structural_rejection": r["frozen_rejection_reason"],
            "status": "primary_semantic_triage_only_not_independent_approval",
            "training_allowed": False,
        })
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in output), encoding="utf-8")
    print(json.dumps({"rows": len(output),
                      "dispositions": {code: sum(r["disposition"] == code for r in output)
                                       for code in ("replay_candidate", "rewrite_question", "exclude_original")},
                      "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
                      "training_allowed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
