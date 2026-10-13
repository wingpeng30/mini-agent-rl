"""Primary semantic triage for v122 train supplement's 13 unpaired questions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.prepare_v116_data import full_rows
from scripts.build_v114_sft_candidates import support_facts

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data/evidence-sft-v124-forensic/remaining-unpaired-diagnostic-private.jsonl"
OUT = ROOT / "data/evidence-sft-v124-forensic/v122-train-unpaired-semantic-triage-private.jsonl"

SPECS = [
 ("exclude_original", "'A time travel drama' does not identify Predestination among many possible films; the intended directors are not uniquely recoverable from the question alone.", "Name Predestination (2014) and review a new film-director question if useful."),
 ("replay_candidate", "Saul Davies is linked to James, and the James article places the rock band in Manchester.", "Replay the musician-to-band-to-city chain in the frozen corpus."),
 ("rewrite_question", "The source does connect Sarah Vowell to the documentary and Violet in The Incredibles, and supplies the festival, but the question misspells 'premiered' and has broken syntax.", "Ask where Gigantic (A Tale of Two Johns), featuring Sarah Vowell, premiered; replay and review."),
 ("rewrite_question", "The Legend of Bonnie & Clyde is identified as a Merle Haggard album and his birth year is 1937; 'singer who made' is vague about the album relation.", "Name the album and ask for its singer Merle Haggard's birth year."),
 ("replay_candidate", "Trainwreck is linked to Vanessa Bayer and is described as directed by Judd Apatow and written by Amy Schumer.", "Replay a film-identity query without altering the answer."),
 ("exclude_original", "The cited 2002 Hairspray is the original Broadway cast recording of an adaptation of the 1988 film, not a 2002 revival. Linda Hart is a cast performer, but the question's revival premise is false.", "Ask about the 2002 original Broadway cast recording as a new question."),
 ("rewrite_question", "Ron Webster's comparison to Seth Johnson and Johnson's one England international in 2000 are documented, but 'recently' lacks a reference date and can age into a false claim.", "Remove 'recently' and ask for the year of the named compared player's England appearance."),
 ("replay_candidate", "Sad Girl credits Mike Daly of Whiskeytown, and Whiskeytown is documented as an alternative country band formed in 1994.", "Replay the album-contributor-to-band chain."),
 ("replay_candidate", "Georg Unger is directly tied to Siegfried in Der Ring des Nibelungen, which is described as a four-drama cycle.", "Replay or direct-answer depending on the action-time evidence."),
 ("rewrite_question", "The intended Harry Potter film and Rupert Grint are identifiable in context, but the question's 'a boy who had previously had no film experience' is not unique; the distributor name is split across two source sentences, breaking a contiguous citation.", "Name Rupert Grint and the first Harry Potter film, and repair source tokenization before replay."),
 ("rewrite_question", "Carl is explicitly the South New Jersey neighbor with repeated non-canon deaths, but the question has duplicated 'show', misspelling and broken grammar.", "Rewrite a single-target question about the Aqua Teen Hunger Force neighbor and replay."),
 ("rewrite_question", "Workaholics has 86 episodes and Jillian Bell played Jillian Belk, but the question asks 'How many episodes has run by' and miscapitalizes the character.", "Ask for the total episode count of Workaholics, identifying Jillian Belk cleanly."),
 ("replay_candidate", "The Birds and Lysistrata are both linked to Aristophanes in the source facts.", "Replay or direct-answer according to the available action-time evidence."),
]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [json.loads(line) for line in IN.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["cohort"] == "v122_train_unpaired"]
    assert len(rows) == len(SPECS) == 13
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
