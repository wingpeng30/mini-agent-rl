"""Create versioned, lead-adjudicated Goal AG development reviews."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ag-failure-repair"
PACKETS = {arm: OUT / f"development-review-{arm}-packet.jsonl" for arm in "ab"}
REVIEWS = {arm: OUT / f"development-review-{arm}.jsonl" for arm in "ab"}
FINAL = OUT / "development-review-final.jsonl"
LEDGER = OUT / "development-review-adjudication-ledger.jsonl"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in values),
                    encoding="utf-8")


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


# Independent reviewer A's relation/identity misses, mapped to the five
# categories frozen by the Goal AG evaluator. Prefixes retain the reviewer audit
# trail while matching duplicate rows with identical prompt and answer.
PREFIX_CORRECTIONS = {
    "0c40a776": ("factual_error", "The cited source identifies Ishmael as Hagar's son, not as the person Abraham married after Sarah died."),
    "0c80ccf0": ("factual_error", "The cited Chief Judge entry is for Rivers State, while Gloria Kemasuode's birthplace is in Delta State."),
    "1093af61": ("unsupported_by_visible_evidence", "The visible sources do not establish Frances Tupper's birthplace; the English birthplace belongs to a different Frances."),
    "1963eb51": ("factual_error", "Farid Khan is Sher Shah Suri's name, not his father's birthplace."),
    "1bb711a1": ("factual_error", "The visible author/source chain does not connect this title to the author of The Timothy Files."),
    "1fab7ab1": ("unsupported_by_visible_evidence", "The visible merger is dated 2003 and does not establish the queried 2002 merger."),
    "20258e58": ("factual_error", "The cited record label belongs to a different Moby Grape release, not the queried album."),
    "2de1c183": ("unsupported_by_visible_evidence", "P&O's merger is unrelated to the CEMM developer in the visible evidence."),
    "2fc2220e": ("unsupported_by_visible_evidence", "The evidence locates Laren in the Gooi region but does not identify a person in charge of it."),
    "f07ff7a3": ("unsupported_by_visible_evidence", "The evidence locates Laren in the Gooi region but does not identify a person in charge of it."),
    "37aeb151": ("unsupported_by_visible_evidence", "The visible documents do not establish which country contains the water body that inspired the Mara Region's name."),
    "468fc472": ("factual_error", "The cited song entry distinguishes the Shrek 2 performer from the performer of the unrelated cover-up theme."),
    "65cbeaa9": ("factual_error", "The source identifies the Orange River's origin in Lesotho, not the Northern Cape."),
    "7688a4c7": ("unsupported_by_visible_evidence", "The source says Annapurna Devi later moved to Bombay but does not establish her birthplace."),
    "7db2c0db": ("unsupported_by_visible_evidence", "The visible evidence does not identify the child or the child's movement."),
    "811c6566": ("unsupported_by_visible_evidence", "The visible records do not connect Tom Owen to the horse that fathered Tompion."),
    "7a6b4408": ("factual_error", "Honey appears in Bee's Knees but not in the visible Between the Sheets ingredients; it is not common to both."),
    "980abdf4": ("factual_error", "Honey appears in Bee's Knees but not in the visible Between the Sheets ingredients; it is not common to both."),
    "915d3852": ("unsupported_by_visible_evidence", "The visible May 2017 date is unrelated and does not establish when the show resumed in 2018."),
    "92937072": ("unsupported_by_visible_evidence", "The visible Dell record does not establish a merger with the CEMM developer."),
    "99467155": ("missing_required_information", "The evidence says Hoffman was nominated; 'Academy Award' omits the nomination status and category."),
    "a87a48f6": ("unsupported_by_visible_evidence", "The cited territory is not established as Wenceslaus's father's place of death."),
    "b40d5555": ("factual_error", "The Walker cast table identifies Cordell Walker as Chuck Norris's role; Simon Moon is another character."),
    "c7c35508": ("missing_required_information", "The answer names the agency but omits the food-safety system asked for; the prompt does not establish FDA itself as the system."),
    "c814a9f8": ("missing_required_information", "The answer names the agency but omits the food-safety system asked for; the prompt does not establish FDA itself as the system."),
    "c9dbc9e5": ("missing_required_information", "Nissan identifies the manufacturer, not the vehicle model requested."),
    "d4d0b60d": ("unsupported_by_visible_evidence", "The visible records do not state whom the creator followed."),
    "e07a0f34": ("factual_error", "The cited recording history identifies Sharp/MGM, not Asian Man Records, as the label for the original singer."),
    "e4d7ce85": ("factual_error", "1608 is the father's birth year; the cited source gives his death date as 25 July 1666."),
    "e4724c30": ("unsupported_by_visible_evidence", "The visible source identifies Jennifer Saunders as the performer but does not identify her spouse."),
    "f4592b33": ("factual_error", "19 February 1752 is the father's birth date; the cited source gives his death on 8 March 1829."),
    "26ef5842": ("unsupported_by_visible_evidence", "The sources identify the mother but do not give her birthday; the displayed date belongs to Philipp's father."),
}

QUESTION_CORRECTIONS = {
    ("Which two features were played up by the highest paid person in the world?", "comedic and dramatic performances"):
        ("unsupported_by_visible_evidence", "The answer describes Amy Adams, whose source says she was among the highest-paid actresses; the visible evidence does not establish her as the person asked about or connect her features to the question's intended person."),
    ("Which two features were played up by the highest paid person in the world?", "skateboarder and actress"):
        ("unsupported_by_visible_evidence", "The answer combines occupations of different people and does not answer the queried person's two features."),
    ("Who was the first black student admitted to the school that owns Swayze Field?", "Winona Cargile Alexander"):
        ("factual_error", "The evidence identifies Winona Cargile Alexander as the first Black student at the New York School of Philanthropy; Swayze Field belongs to the University of Mississippi."),
    ("What do Tim Kaine and Aneesh Chopra have in common?", "senator"):
        ("factual_error", "The evidence identifies Aneesh Chopra as a technology executive, not a senator; the shared supported attribute is American."),
    ("The Rev and Canterbury member Mike sparks were both what?", "drummer"):
        ("factual_error", "The evidence does not support drummer as their shared role; the frozen answer and cited entries identify vocalist."),
    ("What type of territory does Northern New Hampshire Correctional Facility and New Hampshire Department of Corrections have in common?", "state prison"):
        ("factual_error", "The facility is a state prison, while the Department of Corrections is a state agency; 'state prison' is not common to both."),
    ("What profession do both Milla Jovovich and Takako Honda share?", "voice actress"):
        ("unsupported_by_visible_evidence", "The visible cited evidence does not establish Milla Jovovich as a voice actress; it supports only the broader shared profession actress."),
    ("What type of collection does Etarcomol and Medb have in common?", "character"):
        ("missing_required_information", "Both are described as characters, but the question asks for their shared collection, which the evidence identifies as the Ulster Cycle."),
    ("Where does the performer of song Balance-Toi work at?", "ASVEL Basket"):
        ("indeterminate", "The source identifies Parker as a majority owner of ASVEL Basket and says he played for the San Antonio Spurs; it does not resolve which affiliation the ambiguous 'work at' wording asks for."),
}

CITATION_REPAIRS = {
    # The visible Reinhold Schünzel entry states that he was born in St. Pauli.
    "45fbde6614476d7deb04601ebc75b98f286beb90ee3c8d0fbe78dc3659fe7eb9":
        ("956c438ea4fdbda813ea0ed7", "The Reinhold Schünzel entry states that he was born in St. Pauli, Hamburg."),
}


def main() -> dict:
    if FINAL.exists() or LEDGER.exists():
        raise FileExistsError("Refusing to overwrite a prior Goal AG adjudication")
    packets = {arm: {x["review_id"]: x for x in load(PACKETS[arm])} for arm in "ab"}
    reviews = {arm: load(REVIEWS[arm]) for arm in "ab"}
    all_reviews: list[dict] = []
    ledger: list[dict] = []
    seen: set[str] = set()
    for arm in "ab":
        if set(packets[arm]) != {x["review_id"] for x in reviews[arm]}:
            raise RuntimeError(f"{arm} review packet coverage mismatch")
        for original in reviews[arm]:
            vote = dict(original)
            rid = str(vote["review_id"])
            if rid in seen:
                raise RuntimeError(f"Duplicate review ID across arms: {rid}")
            seen.add(rid)
            packet = packets[arm][rid]
            old = vote["category"]
            # Normalize the second reviewer’s equivalent category spelling to
            # the canonical labels used by the frozen scorer.
            vote["category"] = {"factually_wrong": "factual_error",
                                 "missing_necessary_information": "missing_required_information"}.get(old, old)
            correction = next((value for prefix, value in PREFIX_CORRECTIONS.items()
                               if arm == "a" and rid.startswith(prefix)), None)
            if correction is None:
                correction = QUESTION_CORRECTIONS.get((packet["question"], norm(packet.get("answer"))))
            if correction:
                vote["category"], reason = correction
                vote["reason"] = reason
                # For unsupported/ambiguous outputs, avoid leaving a citation
                # that the lead has just ruled insufficient for the claim.
                if vote["category"] in {"unsupported_by_visible_evidence", "indeterminate"}:
                    vote["evidence_ids"] = []
                    vote["evidence_snippets"] = []
            if rid in CITATION_REPAIRS and vote["category"] == "correct_sufficient":
                vote["evidence_ids"] = [CITATION_REPAIRS[rid][0]]
                vote["reason"] = CITATION_REPAIRS[rid][1]
            if vote["category"] == "correct_sufficient" and not vote.get("evidence_ids"):
                raise RuntimeError(f"Success lacks evidence citation: {rid}")
            if not vote.get("reason"):
                raise RuntimeError(f"Review lacks rationale: {rid}")
            final = {**vote, "review_arm": arm}
            all_reviews.append(final)
            if vote["category"] != old or vote != original:
                ledger.append({"review_id": rid, "review_arm": arm,
                    "question": packet["question"], "answer": packet.get("answer"),
                    "previous_category": old, "final_category": vote["category"],
                    "previous_reason": original.get("reason"), "final_reason": vote["reason"],
                    "previous_evidence_ids": original.get("evidence_ids", []),
                    "final_evidence_ids": vote.get("evidence_ids", [])})
    all_reviews.sort(key=lambda x: x["review_id"])
    ledger.sort(key=lambda x: x["review_id"])
    write_jsonl(FINAL, all_reviews)
    write_jsonl(LEDGER, ledger)
    from collections import Counter
    return {"rows": len(all_reviews), "adjudications": len(ledger),
            "categories": dict(Counter(x["category"] for x in all_reviews)),
            "final_sha256": __import__("hashlib").sha256(FINAL.read_bytes()).hexdigest(),
            "ledger_sha256": __import__("hashlib").sha256(LEDGER.read_bytes()).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, sort_keys=True))
