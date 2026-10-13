"""Apply pre-inference evidence review; keep natural and constructed strata separate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from uuid import uuid4

from mini_agent_rl.retrieval.local import ROOT, config, corpus, sha
from prepare_v133_goal_z_multihop_data import norm

# Positions in the audited source-order pool, reviewed before any inference.
NATURAL_ACCEPT = {
    "musique": [0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 14, 17, 18, 25, 26, 27, 28, 30, 32, 36, 38, 40, 44, 46, 47, 48],
    "2wiki": [1, 4, 5, 9, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30],
}
REASONS = {
    "musique": {
        5: "Birth in Wenzhou does not establish birth in Yongjia County; church is near Wenzhou.",
        8: "Question asks a state but evidence identifies a Portuguese district.",
        11: "Battle has multiple participants; question does not identify which participant or refugee group.",
        13: "Co-starring in a soundtrack description does not establish marriage.",
        15: "Question accepts study or work, but evidence gives both Clifton College and London grain trade.",
        16: "Historical SAP/SR administrative classification cannot answer an undated current country question.",
        19: "Jack Clay had multiple employers and enrollment lacks a question reference year.",
        20: "Undated provincial leader question uses historical commissioner snapshot.",
        21: "First passage lists multiple starring actors without establishing a uniquely leading actor.",
        22: "Evidence states FDA registration and an Act, not ownership and a food safety system as asked.",
        23: "Question asserts one million votes, while evidence states 913664.",
        24: "Question omits the 2016 population reference year.",
        29: "Co-authorship does not establish Carrie Hamilton as Carol Burnett's child.",
        31: "First Governor-General and first Viceroy are conflated.",
        33: "Question omits 2006 ranking context; education bridge also needs stronger evidence.",
        34: "Co-starring in On Broadway does not establish marriage.",
        35: "Undated agency leader question uses a historical present marker.",
        37: "Co-starring in Flesh and Bone does not establish marriage.",
        39: "Reference physic is malformed; physics-law evidence alone does not explicitly state the person's field.",
        41: "Question asks overall NBA single-season record, evidence limits Durant claim to active players.",
        42: "No Tears Left to Cry has multiple songwriters; bridge is not uniquely specified.",
        43: "A religious congregation's name does not establish Saint Joseph's birthplace.",
        45: "Beginning regular in-person addresses after 1913 does not establish first-ever in-person address.",
        49: "Two album passages give different labels at different dates; undated performer label is ambiguous.",
    },
    "2wiki": {
        0: "Past article writing does not establish current employment.",
        2: "Broadcast on Comedy Central does not establish employment there.",
        3: "Footballer Dave Bird is conflated with television creator Dave Bird; birth location also absent.",
        6: "Serving in late Eastern Han dynasty is not a nationality statement.",
        7: "Reference Buenos Aires loses the explicitly named Vicente Lopez birth locality.",
        8: "Montgomery is a bus-boycott location, not stated birthplace.",
        10: "Lordship of Writtle does not establish birth there.",
        11: "Residence and burial in Medina do not establish birthplace.",
        12: "Florentine polity affiliation does not establish a birthplace.",
        13: "German is a language label for a name, not nationality evidence.",
        14: "Passage explicitly gives alternative birth years 1146 or 1147.",
        22: "Moving to Hollywood followed by death does not explicitly establish place of death.",
        23: "Birmingham theatre work does not establish birthplace.",
        24: "Anatomy professor Matthew Kaufman is conflated with film director Matthew Kaufman.",
        25: "Chinese state media reference does not establish Fruit Chan's birthplace.",
        26: "Hong Kong origin and making films in China do not explicitly establish country-of-origin answer China.",
        27: "Song has multiple writers/composers; singular bridge not uniquely specified.",
        31: "Founding Turkic Khaganate does not establish place of death.",
    },
}

# New single-document diagnostics, with explicit factual answer spans. These are
# constructed from an already known corpus and never treated as a held-out set.
DIAGNOSTICS = [
    ("Days of Betrayal", "In what year was Days of Betrayal released?", "1973"),
    ("Otakar Vávra", "What is the birth date of Otakar Vávra?", "28 February 1911"),
    ("Richard Marshal, 3rd Earl of Pembroke", "On what date did Richard Marshal succeed his brother to the Earldom of Pembroke?", "6 April 1231"),
    ("Clarice Orsini", "Who was the mother of Clarice Orsini?", "Maddalena Orsini"),
    ("Lorenzo de' Medici", "In which chapel is Lorenzo de' Medici buried?", "Medici Chapel"),
    ("Man at Bath", "On what date was Man at Bath released in cinemas?", "22 September 2010"),
    ("Christophe Honoré", "Where was Christophe Honoré born?", "Carhaix, Finistère"),
    ("Trey Parker", "What production studio did Trey Parker and Matt Stone establish in 2013?", "Important Studios"),
    ("Xiahou Yuan", "At which battle was Xiahou Yuan killed?", "Battle of Mount Dingjun"),
    ("Xiahou Ba", "Which state did Xiahou Ba defect to around 249?", "Shu Han"),
    ("José A. Ferreyra", "What illness caused José A. Ferreyra's death?", "throat cancer"),
    ("Virginia Foster Durr", "In which city was Virginia Foster Durr born?", "Birmingham, Alabama"),
    ("Clifford Durr", "Which university did Clifford Durr attend as a Rhodes Scholar?", "Oxford"),
    ("Incubus HQ Live", "In which month and year was the Incubus HQ Live special box set released?", "July 2012"),
    ("Marc Scarpa", "On what date was Marc Scarpa born?", "September 25, 1969"),
    ("Nigel de Brus", "At which castle was Nigel de Brus captured in 1306?", "Kildrummy Castle"),
    ("Hasan ibn Ali", "In which cemetery was Hasan ibn Ali buried?", "Jannat al-Baqi"),
    ("Dietrich IX, Count of Mark", "In what year did Dietrich IX receive the title of Count of Mark?", "1393"),
    ("Ida May Park", "Who was the spouse of Ida May Park?", "Joseph De Grasse"),
    ("Fruit Chan", "Which film released in 1997 made Fruit Chan a household name?", "Made in Hong Kong"),
    ("Hissa Abdulla Ahmed Al-Otaiba", "Where was Hissa Abdulla Ahmed Al-Otaiba born?", "Abu Dhabi"),
    ("Billie Eilish", "What is the name of Billie Eilish's debut EP?", "Don't Smile at Me"),
    ("Vsevolod II of Kiev", "Which brother did Vsevolod II choose as his successor?", "Igor"),
    ("Irvin Yeaworth", "For how many years was Irvin Yeaworth married to Jean Yeaworth?", "59"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draft", type=Path)
    args = parser.parse_args()
    pool = [json.loads(x) for x in (args.draft / "pool.jsonl").read_text(encoding="utf-8").splitlines()]
    old_manifest = json.loads((args.draft / "manifest.json").read_text(encoding="utf-8"))
    documents = corpus(config())
    by_title = {d["title"]: d for d in documents}
    accepted, reviews = [], []
    for source in ("musique", "2wiki"):
        values = [r for r in pool if r["source"] == source]
        for i, row in enumerate(values):
            ok = i in NATURAL_ACCEPT[source]
            reviews.append(dict(id=row["id"], accepted=ok,
                reason="Two support paragraphs support the intended relation; retained original reference and aliases" if ok else REASONS[source][i]))
            if ok:
                accepted.append(row | dict(stratum="natural_supplement"))
    assert len(accepted) == 40
    constructed = []
    for i, (title, question, answer) in enumerate(DIAGNOSTICS):
        if title not in by_title:
            raise ValueError("Missing diagnostic title: " + title)
        document = by_title[title]
        assert norm(answer) in norm(document["text"])
        constructed.append(dict(id=f"constructed:hybrid-diagnostic-{i+1:02d}", source="constructed",
            stratum="constructed_singlehop", kind="singlehop", question=question, answer=answer,
            answer_aliases=[], support_titles=[title], support_document_ids=[document["id"]],
            support_documents=[document], chain=[]))
    selected = accepted + constructed
    previous = []
    root = ROOT / "runs/v133-goal-z-multihop-sft"
    for name in ("selected-train.jsonl", "selected-dev.jsonl", "selected-confirm.jsonl"):
        previous.extend(json.loads(x) for x in (root / name).read_text(encoding="utf-8").splitlines() if x.strip())
    old_questions = {norm(r["question"]) for r in previous}
    assert not any(norm(r["question"]) in old_questions for r in selected)
    assert len({norm(r["question"]) for r in selected}) == 64
    random.Random(20261006).shuffle(selected)
    folder = Path(config()["run_dir"]) / ("supplement-reviewed-64-" + uuid4().hex)
    folder.mkdir(parents=True)
    for name, values in (("selected.jsonl", selected), ("natural.jsonl", accepted), ("constructed.jsonl", constructed), ("semantic-review.jsonl", reviews)):
        (folder / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values), encoding="utf-8")
    manifest = dict(kind="40 natural same-corpus supplementary questions plus 24 constructed single-hop diagnostics; report separately",
        questions=64, strata=dict(Counter(r["stratum"] for r in selected)),
        source_counts=dict(Counter(r["source"] for r in selected)),
        cases_sha256=sha(folder / "selected.jsonl"), corpus_sha256=config()["corpus_sha256"],
        draft_manifest_sha256=sha(args.draft / "manifest.json"), draft=str(args.draft),
        semantic_review_sha256=sha(folder / "semantic-review.jsonl"),
        selection_seed=20261006, selection_uses_model_or_retrieval_results=False,
        historical_local_run_references={r['id']: old_manifest['historical_local_run_references'][r['id']] for r in accepted},
        exposure_note="Some natural questions occur in old candidate pools; same corpus, not independent held-out validation. Constructed questions may reuse known facts.",
        training=False, frozen_before_inference=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(folder=str(folder), strata=manifest["strata"], sources=manifest["source_counts"]), ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
