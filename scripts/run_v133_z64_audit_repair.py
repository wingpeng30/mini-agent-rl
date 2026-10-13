"""Prepare a keyed, source-bound Goal-Z 64-case repair audit (CPU only).

This supersedes the positional finalizer from the immutable v2 run. It reads
only frozen Goal-Z materials and never queries or loads a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "runs/v133-goal-ai-z64-source-admission-v2"
OUT = ROOT / "runs/v133-goal-ai-z64-audit-repair-v7"
CASES = ROOT / "runs/v133-goal-ah-entity-relation/evaluation-cases-goal_z_dev.jsonl"
CORPUS = ROOT / "runs/v133-goal-z-multihop-sft/corpus-dev.json"
ADAPTER_DIR = ROOT / "runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final"
ADAPTER = ADAPTER_DIR / "adapter_model.safetensors"
AA = ROOT / "runs/v133-goal-aa-audit-confirmation/product-decision.json"
EXPECTED_ADAPTER = "d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52"

# Per-case, per-edge decisions made from the complete official source bodies.
# Keys are stable case IDs; values are in that case's explicitly identified
# edge_id order. "unconfirmed" never means false or corpus-wide absence.
EDGE_STATUS = {
    "musique:2hop__46376_158105": ["unconfirmed", "pass"],
    "musique:2hop__864101_135373": ["pass", "pass"],
    "musique:2hop__235471_565925": ["pass", "unconfirmed"],
    "musique:2hop__478978_124498": ["pass", "pass"],
    "musique:2hop__862994_69048": ["pass", "pass"],
    "musique:2hop__95673_121319": ["pass", "unconfirmed"],
    "musique:2hop__829358_754719": ["pass", "pass"],
    "musique:2hop__789565_416401": ["pass", "pass"],
    "musique:2hop__506035_325512": ["pass", "pass"],
    "musique:2hop__337223_56873": ["pass", "pass"],
    "musique:2hop__121145_561444": ["pass", "unconfirmed"],
    "musique:2hop__44522_58556": ["pass", "pass"],
    "musique:2hop__475070_532404": ["pass", "unconfirmed"],
    "musique:2hop__796221_18803": ["pass", "pass"],
    "musique:2hop__199216_33564": ["pass", "unconfirmed"],
    "musique:2hop__151750_141308": ["pass", "pass"],
    "musique:2hop__418360_67370": ["pass", "pass"],
    "musique:2hop__374495_68633": ["pass", "pass"],
    "musique:2hop__96194_78303": ["pass", "pass"],
    "musique:2hop__463353_66723": ["unconfirmed", "pass"],
    "musique:2hop__703493_162341": ["pass", "pass"],
    "musique:2hop__158275_35739": ["pass", "pass"],
    "musique:2hop__435087_55984": ["pass", "pass"],
    "musique:2hop__109242_113442": ["pass", "pass"],
    "musique:2hop__69765_43945": ["pass", "pass"],
    "musique:2hop__71034_343058": ["pass", "unconfirmed"],
    "musique:2hop__817096_15463": ["pass", "pass"],
    "2wiki:91c13a780bd911eba7f7acde48001122": ["pass", "unconfirmed"],
    "2wiki:982f8e440bdb11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:b587d3d20bdd11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:d52aa2a20bdb11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:67ecf3100bdc11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:3a3d60020bdd11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:66616e2a0bde11eba7f7acde48001122": ["pass", "unconfirmed"],
    "2wiki:1b911ab00bdb11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:7b3f24c80bdd11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:4b9f70180bda11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:8a77eae20bdd11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:74f39a760bde11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:69af3b6c0bdd11eba7f7acde48001122": ["pass", "unconfirmed"],
    "2wiki:66772eb80bde11eba7f7acde48001122": ["pass", "unconfirmed"],
    "2wiki:f27e01300bdd11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:ef9d40a40bdb11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:5dac1e8e0bdd11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:1dd33f080bde11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:a703e1160bdd11eba7f7acde48001122": ["pass", "unconfirmed"],
    "2wiki:764039340bde11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:d94d1d720bdd11eba7f7acde48001122": ["pass", "unconfirmed"],
    "2wiki:7bda387a0bdb11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:df13ddf40bdd11eba7f7acde48001122": ["pass", "unconfirmed"],
    "2wiki:015a72a60bde11eba7f7acde48001122": ["pass", "unconfirmed"],
    "2wiki:2e92333a0bde11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:114313dc0bdc11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:d8c06f220bdc11eba7f7acde48001122": ["pass", "unconfirmed"],
    "2wiki:92d51c3a0bda11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:f299dd600bdd11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:49d965aa0bde11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:8796fb620bdb11eba7f7acde48001122": ["pass", "pass"],
    "2wiki:ee3233920bda11eba7f7acde48001122": ["pass", "pass"],
    "musique:2hop__560534_91104": ["pass", "unconfirmed"],
    "musique:2hop__498251_126089": ["pass", "pass"],
    "musique:2hop__530579_13529": ["pass", "pass"],
    "musique:2hop__468778_88165": ["unconfirmed", "pass"],
    "musique:2hop__442175_56873": ["pass", "pass"],
}

CASE_UNCONFIRMED = {
    "musique:2hop__46376_158105": ("unconfirmed", "unconfirmed", "The first source calls Gates the richest man, not the highest-paid person; this distinction leaves the bridge unsupported."),
    "musique:2hop__235471_565925": ("unconfirmed", "unconfirmed", "The label source establishes Columbia released Moby Grape albums; it does not establish the requested signing or label membership relation."),
    "musique:2hop__95673_121319": ("unconfirmed", "unconfirmed", "The film source lists three stars, so the singular featured person is not uniquely identified; winning an academy scholarship also does not state study or employment there."),
    "musique:2hop__475070_532404": ("unconfirmed", "unconfirmed", "The second source is about Arthur Rupert Dickey and never gives Charles Tupper's birthplace."),
    "musique:2hop__199216_33564": ("unconfirmed", "unconfirmed", "The source says DRDO enterprises were established in Hyderabad during the period; that does not establish the organization's founding date."),
    "musique:2hop__44522_58556": ("unconfirmed", "unconfirmed", "The cited law grants FDA food-regulation authorities, but the question's phrase 'food safety system' does not map unambiguously to the act's name."),
    "musique:2hop__463353_66723": ("unconfirmed", "unconfirmed", "The source calls the stadium the team's home; it does not state that the university owns it."),
    "musique:2hop__435087_55984": ("pass", "pass", "The song source itself names Carlile as singer and Hanseroth as writer; that body supports both relevant relations."),
    "musique:2hop__71034_343058": ("unconfirmed", "unconfirmed", "The second source says Jennifer Saunders wrote Mirrorball and Adrian Edmondson directed it; it does not say they were spouses."),
    "2wiki:91c13a780bd911eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The father is described as a Seleucid king; the source does not establish that 'which country is he from' means citizenship in the Seleucid Empire."),
    "2wiki:66616e2a0bde11eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The father source lists rule over Szprotawa, not death there."),
    "2wiki:69af3b6c0bdd11eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The source mentions Kyoto as a later political seat, but does not state Go-Daigo was born there."),
    "2wiki:66772eb80bde11eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The source says Gerstad retired to Palm Springs, not that he died there."),
    "2wiki:a703e1160bdd11eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The source says Sigismondo was lord of Rimini; it does not give his death place."),
    "2wiki:d94d1d720bdd11eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The biography says Parker is a former player and describes his former Spurs career; it does not establish current employment with the Spurs."),
    "2wiki:df13ddf40bdd11eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The source gives Sasaram as Sher Shah's capital, not birthplace."),
    "2wiki:015a72a60bde11eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The source identifies Bernard's family and titles, but gives no birthplace."),
    "2wiki:2e92333a0bde11eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The source gives historical appointments at Bergen Museum and the University of Bergen; the present-tense question about a person who died in 1957 has unclear time scope."),
    "2wiki:d8c06f220bdc11eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The source mentions a film festival in Buenos Aires, not Becher's place of death."),
    "musique:2hop__560534_91104": ("unconfirmed", "unconfirmed", "The source establishes ATS-6 participated in an experiment with ISRO; it does not establish the stated forerunner relation to an ISRO communications satellite."),
    "musique:2hop__468778_88165": ("unconfirmed", "unconfirmed", "The source calls Hagar Sarah's handmaid and Ishmael's mother; it does not establish Hagar as Abraham's spouse."),
}

EDGE_REASONS = {
    ("musique:2hop__121145_561444", 1): "The text says Chaya Mushka was Dovber's daughter and Menachem Mendel's wife. It does not establish that Menachem Mendel followed Dovber.",
    ("musique:2hop__95673_121319", 1): "Winning an American Academy of Dramatic Arts scholarship does not establish that Robinson studied or worked at the institution.",
    ("musique:2hop__475070_532404", 1): "The body names Arthur Rupert Dickey as born in Amherst; it does not identify him as Charles Tupper or support Charles's birthplace.",
    ("musique:2hop__199216_33564", 1): "The body groups DRDO among enterprises established in Hyderabad during a broad period. Establishment in a city is not necessarily an organization's founding date.",
    ("musique:2hop__463353_66723", 0): "The body says Swayze Field is the home of the university baseball team; home is not an ownership statement.",
    ("musique:2hop__71034_343058", 1): "The body identifies Jennifer Saunders as writer and Adrian Edmondson as director of Mirrorball; it supplies no spouse relation.",
    ("2wiki:66616e2a0bde11eba7f7acde48001122", 1): "The body states Henry VIII ruled over Szprotawa but gives no death location.",
    ("2wiki:69af3b6c0bdd11eba7f7acde48001122", 1): "Kyoto appears as a court/political location; the body does not state a birthplace.",
    ("2wiki:66772eb80bde11eba7f7acde48001122", 1): "Retiring to Palm Springs does not establish dying there.",
    ("2wiki:a703e1160bdd11eba7f7acde48001122", 1): "Lordship over Rimini does not establish dying there.",
    ("2wiki:d94d1d720bdd11eba7f7acde48001122", 1): "The body describes a former Spurs playing career, not a current employer relationship.",
    ("2wiki:df13ddf40bdd11eba7f7acde48001122", 1): "The body identifies Sasaram as the empire's capital, not Sher Shah's birthplace.",
    ("2wiki:015a72a60bde11eba7f7acde48001122", 1): "The body does not state Bernard's birthplace.",
    ("2wiki:2e92333a0bde11eba7f7acde48001122", 1): "This historical text supports appointments at two institutions but does not resolve the present-tense wording for a deceased person.",
    ("2wiki:d8c06f220bdc11eba7f7acde48001122", 1): "Screening a film in Buenos Aires does not establish a person's death place.",
    ("musique:2hop__560534_91104", 1): "Joint experiment participation does not establish a forerunner relation.",
    ("musique:2hop__468778_88165", 0): "The cited text calls Hagar a handmaid of Sarah; it does not support the asserted spouse edge.",
    ("musique:2hop__46376_158105", 0): "The document says richest man, not highest-paid person.",
    ("musique:2hop__235471_565925", 1): "Columbia released Moby Grape albums; that does not prove the band's signing or affiliation.",
    ("musique:2hop__44522_58556", 1): "The act's FDA authorities are present, but 'food safety system' is not a clearly equivalent relation in this passage.",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", str(s).casefold(), flags=re.UNICODE).strip()


def source_triple(packet: dict, i: int) -> dict:
    edge = packet["official_chain_or_evidence"][i]
    if "official_evidence" in edge:
        s, r, o = edge["official_evidence"]
        return {"subject": str(s), "relation": str(r), "object": str(o), "qualifiers": {}}
    step = edge.get("official_decomposition", edge)
    q = str(step.get("question", ""))
    obj = str(step.get("answer", edge.get("object", "")))
    if ">>" in q:
        subject, relation = [x.strip() for x in q.split(">>", 1)]
        if subject == "#1":
            prior = packet["official_chain_or_evidence"][i - 1].get("official_decomposition", {})
            subject = str(prior.get("answer", "#1 unresolved"))
        subject = re.sub(r"#1", str(packet["official_chain_or_evidence"][i - 1].get("official_decomposition", {}).get("answer", "#1")), subject)
    elif "#1" in q and i > 0:
        prior = str(packet["official_chain_or_evidence"][i - 1].get("official_decomposition", {}).get("answer", "#1"))
        subject = prior
        relation = q.replace("#1", prior)
    else:
        # Ground the first-hop edge in its explicit named entity where present.
        m = re.search(r"(?:Who was the star of|Who featured in the film|Who created|What is the body of water that|Who is the author of)\s+(.+?)(?:\?|$)", q, re.I)
        if m:
            subject = m.group(1).strip()
            relation = q[:m.start(1)].strip() or "identity requested"
        elif "highest paid person" in q.casefold():
            subject, relation = "highest paid person in the world", "identity at maximum pay rank"
        else:
            subject = q
            relation = "answer to official subquestion: " + q
    return {"subject": subject, "relation": relation, "object": obj,
            "qualifiers": {"source_question": q, "decomposition_step_id": step.get("id")}}


def edge_id(packet: dict, i: int) -> str:
    edge = packet["official_chain_or_evidence"][i]
    ev = edge.get("official_evidence")
    if ev:
        return f"edge{i}:{norm(ev[0])}|{norm(ev[1])}|{norm(ev[2])}"
    step = edge.get("official_decomposition", edge)
    return f"edge{i}:decomp-{step.get('id')}"


def doc_for_edge(packet: dict, i: int) -> dict:
    """Join evidence by explicit source paragraph ID or fact subject/title."""
    edge = packet["official_chain_or_evidence"][i]
    if edge.get("official_decomposition"):
        step = edge["official_decomposition"]
        idx = step.get("paragraph_support_idx")
        matches = [d for d in packet["official_support_documents"]
                   if d.get("source_paragraph_idx") == idx]
        if len(matches) != 1:
            raise ValueError(f"No unique paragraph-ID source join: {packet['case_id']} {idx}")
        return matches[0]
    subj = norm(edge["official_evidence"][0])
    matches = [d for d in packet["official_support_documents"]
               if norm(d.get("title", "")) == subj]
    if len(matches) != 1:
        matches = [d for d in packet["official_support_documents"]
                   if subj and subj in norm(d.get("text", ""))]
    if len(matches) != 1:
        raise ValueError(f"No unique subject-to-document join: {packet['case_id']} edge {i}")
    return matches[0]


def phrase_terms(s: str) -> list[str]:
    words = [x for x in norm(s).split() if len(x) > 2 and x not in
             {"the", "who", "what", "when", "where", "which", "that", "from", "into", "with", "was", "were", "his", "her", "their", "one", "name", "answer", "official", "question"}]
    return list(dict.fromkeys(words))


def quote_for(packet: dict, i: int, triple: dict, doc: dict) -> tuple[str | None, list[int] | None]:
    cid = packet["case_id"]
    if EDGE_STATUS[cid][i] != "pass":
        return None, None
    text = str(doc["text"])
    rel = triple["relation"].casefold()
    cues = {
        "father": ["son of", "daughter of", "father", "child"], "mother": ["daughter of", "son of", "mother"],
        "spouse": ["wife", "husband", "married"], "place of birth": ["born", "birth"],
        "date of birth": ["born"], "place of death": ["died", "death"], "date of death": ["died", "death"],
        "director": ["directed"], "performer": ["song by", "songs by", "recorded", "singer", "starring"],
        "member of sports team": ["played for", "joined", "team"], "employer": ["professor", "director", "employer", "worked"],
        "publisher": ["published"], "inception": ["founded"], "country of citizenship": ["citizenship", "american", "born", "country"],
        "record label": ["record label", "released", "label"], "owned by": ["tanker", "shell"],
        "located in the administrative territorial entity": ["located", "city in", "county", "province", "state"],
        "place of death": ["died", "death"], "chief judge": ["chief justice", "chief judge", "succeeded"],
    }
    words = cues.get(rel)
    if words is None:
        words = phrase_terms(rel)
        q = str(triple.get("qualifiers", {}).get("source_question", "")).casefold()
        synonym_rules = {
            "feature": ["starring", "starred", "featured", "highlighting"],
            "star": ["starring", "starred"], "created": ["authored", "written"],
            "creator": ["authored", "written", "created", "painted"],
            "father": ["son of", "daughter of", "father"], "mother": ["son of", "daughter of", "mother"],
            "study": ["studied", "attended", "graduated", "school", "scholarship"],
            "author": ["book by", "written by", "author"], "named": ["named for", "named after"],
            "located": ["located", "part of", "in the", "city in", "state of"],
            "follow": ["followed", "disciple", "follower"], "part": ["division", "subsidiary", "part of"],
            "president": ["president", "incumbent"], "child": ["son", "daughter", "child"],
            "married": ["married", "wife", "husband"], "performer": ["singer", "recorded", "performed", "song by"],
            "recognition": ["nominated", "award", "won"], "merge": ["merged", "merger"],
            "work": ["works", "professor", "director", "employer", "appointed"],
            "name": ["name", "called", "known as"], "child": ["son", "daughter", "child", "father", "parent"],
            "date": ["born", "died", "date"],
            "after": ["after", "succeeded"], "announce": ["announced", "announcement"],
        }
        for key, vals in synonym_rules.items():
            if key in q:
                words += vals
    sentences = [(m.start(), m.end(), m.group(0).strip())
                 for m in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text) if m.group(0).strip()]
    obj_terms = phrase_terms(triple["object"])
    subj_terms = phrase_terms(triple["subject"])
    scored = []
    for a, b, sent in sentences:
        ns = norm(sent)
        obj_hit = any(norm(x) in ns for x in [triple["object"]] if norm(x)) or any(w in ns.split() for w in obj_terms)
        subj_hit = norm(doc.get("title", "")) == norm(triple["subject"]) or any(w in ns.split() for w in subj_terms)
        rel_hit = any(w in ns for w in words if w)
        score = int(obj_hit) * 4 + int(subj_hit) * 2 + int(rel_hit) * 3
        if obj_hit and rel_hit:
            scored.append((score, -(b-a), a, b, sent))
    if not scored:
        # A reviewed PASS without an extractable source sentence is rejected.
        return None, None
    _, _, a, b, sent = max(scored)
    return sent, [a, b]


def candidates_for(triple: dict, docs: list[dict], official_ids: set[str]) -> list[dict]:
    subj = phrase_terms(triple["subject"])
    rel = phrase_terms(triple["relation"])
    rows = []
    for d in docs:
        did = str(d.get("id", ""))
        if did in official_ids:
            continue
        title = norm(d.get("title", "")); body = norm(d.get("text", ""))
        title_hits = sum(w in title.split() for w in subj)
        body_hits = sum(w in body.split() for w in subj)
        rel_hits = sum(w in body.split() for w in rel)
        if not title_hits and body_hits < max(1, min(2, len(subj))):
            continue
        rows.append((-(title_hits*5 + body_hits*2 + rel_hits), did,
                     {"document_id": did, "title": d.get("title"), "subject_title_hits": title_hits,
                      "subject_body_hits": body_hits, "relation_term_hits": rel_hits}))
    rows.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in rows[:12]]


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite repair run: {OUT}")
    cases = read_jsonl(CASES)
    packets = read_jsonl(OLD / "official-source-packets.jsonl")
    traj = read_jsonl(OLD / "trajectory-source-index.jsonl")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    aa = json.loads(AA.read_text(encoding="utf-8"))
    if len(cases) != 64 or len(packets) != 64 or len(traj) != 64 or len(corpus) != 2048:
        raise RuntimeError("Frozen Goal Z input cardinality mismatch")
    ids = [x["case_id"] for x in cases]
    if ids != [x["case_id"] for x in packets] or set(EDGE_STATUS) != set(ids):
        raise RuntimeError("Case identity/order or explicit edge-review map mismatch")
    if aa.get("published") is not True or aa.get("selected_seed") != 42 or aa.get("selected_route") != "trajectory" or aa.get("adapter") != str(ADAPTER_DIR):
        raise RuntimeError("AA release record does not identify the Goal-Z seed42 adapter")
    actual_hash = sha(ADAPTER)
    if actual_hash != EXPECTED_ADAPTER:
        raise RuntimeError("Goal Z adapter identity mismatch")
    corpus_by_id = {str(d.get("id")): d for d in corpus}
    if len(corpus_by_id) != 2048:
        raise RuntimeError("Frozen corpus document IDs are not unique")
    OUT.mkdir(parents=True)

    freeze = {
        "run_id": OUT.name, "scope": "fixed Goal Z dev 64; source and existing-trajectory audit only",
        "case_ids_in_order": ids,
        "limits": {"cases": 64, "edges": 128, "official_bodies": 128,
                   "candidate_docs_per_edge": 12, "candidate_edge_doc_pairs_total": 1536,
                   "new_searches": 0, "new_model_calls": 0, "gpu_calls": 0, "updates": 0},
        "identity": {"aa_product_decision": str(AA.relative_to(ROOT)), "published": True,
                     "selected_seed": 42, "selected_route": "trajectory", "adapter_sha256": actual_hash,
                     "adapter_loaded": False},
        "sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                   [CASES, CORPUS, ADAPTER, AA, OLD / "freeze.json", OLD / "official-source-packets.jsonl",
                    OLD / "trajectory-source-index.jsonl", Path(__file__)]},
        "history": "All source decisions were exposed through Goal Z questions, official decomposition/evidence and support bodies; not a blind review.",
    }
    (OUT / "freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    doc_by_id = corpus_by_id
    edge_rows, candidate_rows, admission_rows = [], [], []
    calls_rows = []
    source_case = {c["case_id"]: c for c in cases}
    traj_by = {x["case_id"]: x for x in traj}
    q_override = {
        "musique:2hop__95673_121319": ("unconfirmed", "pass", "The source names three film stars. The requested person is not uniquely specified."),
        "musique:2hop__44522_58556": ("unconfirmed", "unconfirmed", "The source ties FSMA to FDA food-regulation authority, but the question's 'system' wording leaves the requested relation/category uncertain."),
        "2wiki:91c13a780bd911eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The father is described as a Seleucid king, but the question does not specify whether 'from' means citizenship, realm, or origin."),
        "2wiki:d94d1d720bdd11eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The source describes a former Spurs playing role, while the question asks in present tense where the performer works."),
        "2wiki:2e92333a0bde11eba7f7acde48001122": ("unconfirmed", "unconfirmed", "The source gives historical appointments, while the question is present tense about a person who died in 1957."),
    }
    for p in packets:
        cid = p["case_id"]
        statuses = EDGE_STATUS[cid]
        if len(statuses) != len(p["official_chain_or_evidence"]):
            raise RuntimeError(f"Explicit edge count mismatch: {cid}")
        proofs = []
        for i, status in enumerate(statuses):
            eid = edge_id(p, i)
            triple = source_triple(p, i)
            d = doc_for_edge(p, i)
            ids_here = [str(x) for x in d.get("frozen_corpus_matches", [])]
            matches = [doc_by_id[x] for x in ids_here if x in doc_by_id and
                       doc_by_id[x].get("title") == d.get("title") and doc_by_id[x].get("text") == d.get("text")]
            if len(matches) != 1:
                raise RuntimeError(f"Literal support body-to-corpus identity mismatch: {cid} {eid}")
            corpus_doc = matches[0]
            quote, span = quote_for(p, i, triple, d)
            if status == "pass" and not quote:
                # Conservative fail-closed behavior: inability to locate a
                # passage never promotes a status. Record as unconfirmed.
                statuses[i] = "unconfirmed"
                status = "unconfirmed"
            if status == "unconfirmed":
                general = CASE_UNCONFIRMED.get(cid, ("unconfirmed", "unconfirmed", "Required relation not established by this cited body."))[2]
                reason = EDGE_REASONS.get((cid, i), general)
                if not quote and statuses[i] == "unconfirmed" and EDGE_STATUS[cid][i] == "pass":
                    reason += " A passage-level source quote could not be localized by the frozen reviewer; status was downgraded to unconfirmed."
            else:
                reason = "The cited body sentence states the relation or an equivalent expression with the same entity binding and scope."
            record = {"case_id": cid, "edge_id": eid, "edge_index_for_source_join": i,
                      "subject": triple["subject"], "subject_role": "subject in the source relation frame",
                      "relation": triple["relation"], "object": triple["object"],
                      "qualifiers": triple["qualifiers"], "status": status,
                      "document_id": str(corpus_doc["id"]), "document_title": d["title"],
                      "document_body_sha256": hashlib.sha256(d["text"].encode()).hexdigest(),
                      "quote": quote, "source_char_span": span,
                      "reason": reason, "reviewer": "primary-agent source review",
                      "review_history": "exposed official source packet; new keyed adjudication",
                      "diagnostic_only": True, "training_allowed": False}
            edge_rows.append(record)
            if quote:
                proofs.append(record)
            if status == "unconfirmed":
                case_official_ids = {str(cid_) for source_doc in p["official_support_documents"]
                                     for cid_ in source_doc.get("frozen_corpus_matches", [])}
                cands = candidates_for(triple, corpus, case_official_ids)
                for c in cands:
                    candidate_rows.append({"case_id": cid, "edge_id": eid, **c,
                                           "body_read": False, "status": "frozen_candidate_pending_review"})

        complete = all(s == "pass" for s in statuses)
        qstat, lstat, override_reason = q_override.get(cid, ("pass", "pass", None))
        if not complete:
            qstat, lstat = "unconfirmed", "unconfirmed"
        gates = {"chain_supported": "pass" if complete else "unconfirmed",
                 "question_resolvable": qstat, "label_compatible": lstat}
        admitted = all(v == "pass" for v in gates.values())
        admission_rows.append({"case_id": cid, "question": p["question"], "official_answer": p["official_answer"],
                               **gates, "admission": "admitted" if admitted else "not_admitted",
                               "reason": override_reason or ("All required source-bound edges and the answer label are supported." if admitted else "At least one required edge is not confirmed."),
                               "edge_ids": [edge_id(p, i) for i in range(len(statuses))],
                               "diagnostic_only": True, "training_allowed": False})
        t = traj_by[cid]["actual_inputs_record"]
        for call in t.get("calls", []):
            calls_rows.append({"case_id": cid, "call_index": call["call_index"],
                               "actual_input_tokens_recorded": call["actual_input_tokens"],
                               "prompt_token_ids": call["prompt_token_ids"],
                               "prompt_sha256": call["prompt_sha256"],
                               "actual_prompt": call["actual_prompt"],
                               "parsed_action": call["parsed_action"], "raw_action": call["raw_action"],
                               "tool_result": call.get("search_observation"),
                               "source_admitted": admitted, "diagnostic_only": True,
                               "training_allowed": False})

    # Deduplicate the frozen candidate edge/document pairs and enforce both caps.
    seen = set(); dedup = []
    for row in candidate_rows:
        k = (row["case_id"], row["edge_id"], row["document_id"])
        if k not in seen:
            seen.add(k); dedup.append(row)
    if len(dedup) > 1536:
        raise RuntimeError("Candidate review cap exceeded")
    counts = {}
    for row in dedup:
        counts[(row["case_id"], row["edge_id"])] = counts.get((row["case_id"], row["edge_id"]), 0) + 1
    if max(counts.values(), default=0) > 12:
        raise RuntimeError("Per-edge candidate review cap exceeded")
    write_jsonl(OUT / "edge-evidence-ledger.jsonl", edge_rows)
    write_jsonl(OUT / "source-admission-ledger.jsonl", admission_rows)
    write_jsonl(OUT / "candidate-freeze.jsonl", dedup)
    write_jsonl(OUT / "trajectory-call-ledger.jsonl", calls_rows)
    summary = {"cases": 64, "edges": len(edge_rows), "admitted": sum(x["admission"] == "admitted" for x in admission_rows),
               "unconfirmed_edges": sum(x["status"] == "unconfirmed" for x in edge_rows),
               "candidate_edge_document_pairs": len(dedup), "historical_calls": len(calls_rows),
               "new_searches": 0, "new_model_calls": 0, "gpu_calls": 0, "updates": 0,
               "status": "source review frozen; candidate review and tokenizer/trace verification remain"}
    (OUT / "prepare-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def materialize_candidates() -> None:
    """Materialize bodies only after the candidate-ID freeze exists."""
    freeze_path = OUT / "candidate-freeze.jsonl"
    target = OUT / "candidate-documents.jsonl"
    if not freeze_path.is_file() or target.exists():
        raise RuntimeError("Candidate freeze missing or candidate bodies already materialized")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    by_id = {str(d.get("id")): d for d in corpus}
    rows = []
    for row in read_jsonl(freeze_path):
        doc = by_id.get(row["document_id"])
        if doc is None:
            raise RuntimeError("Frozen candidate document disappeared")
        rows.append({**row, "text": doc.get("text", ""),
                     "body_sha256": hashlib.sha256(str(doc.get("text", "")).encode()).hexdigest(),
                     "body_read": False})
    write_jsonl(target, rows)
    print(json.dumps({"materialized_candidates": len(rows), "new_searches": 0,
                      "new_model_calls": 0, "gpu_calls": 0, "updates": 0}))


def candidate_sample(start: int, limit: int, skip_official: bool = False, rank: int = 0) -> None:
    rows = read_jsonl(OUT / "candidate-documents.jsonl")
    official_ids = set()
    if skip_official:
        for packet in read_jsonl(OLD / "official-source-packets.jsonl"):
            for doc in packet["official_support_documents"]:
                official_ids.update(str(x) for x in doc.get("frozen_corpus_matches", []))
    groups = {}
    for row in rows:
        groups.setdefault((row["case_id"], row["edge_id"]), []).append(row)
    selected = list(groups.items())[start:start + limit]
    for (cid, eid), docs in selected:
        eligible = [d for d in docs if str(d["document_id"]) not in official_ids]
        if rank >= len(eligible):
            print(json.dumps({"case_id": cid, "edge_id": eid, "status": "candidate_list_exhausted"}, ensure_ascii=False))
            continue
        d = eligible[rank]
        print(json.dumps({"case_id": cid, "edge_id": eid, "document_id": d["document_id"],
                          "title": d["title"], "candidate_rank_after_official": rank + 1, "text": d["text"]}, ensure_ascii=False))


def verify_calls() -> None:
    """Round-trip all saved prompts with the frozen base tokenizer (no model load)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/qwen_08b", local_files_only=True)
    rows = read_jsonl(OUT / "trajectory-call-ledger.jsonl")
    by_case = {}
    failures = []
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
        ids = row["prompt_token_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        roundtrip = tokenizer(decoded, add_special_tokens=False)["input_ids"]
        if roundtrip != ids:
            failures.append({"case_id": row["case_id"], "call_index": row["call_index"], "failure": "decode_encode_token_ids_mismatch"})
        if len(ids) != row["actual_input_tokens_recorded"] or len(ids) > 2048:
            failures.append({"case_id": row["case_id"], "call_index": row["call_index"], "failure": "input_token_count_mismatch_or_over_budget"})
        if hashlib.sha256(decoded.encode()).hexdigest() != row["prompt_sha256"]:
            failures.append({"case_id": row["case_id"], "call_index": row["call_index"], "failure": "decoded_prompt_hash_mismatch"})
    budget = {}
    for cid, calls in by_case.items():
        calls.sort(key=lambda x: x["call_index"])
        attempted = sum(x["parsed_action"].get("action") == "search" for x in calls)
        executed = sum(x["parsed_action"].get("action") == "search" and x.get("tool_result") is not None for x in calls)
        overbudget = max(0, attempted - 3)
        if len(calls) > 4 or executed > 3:
            failures.append({"case_id": cid, "failure": "executed_call_or_search_budget_exceeded"})
        budget[cid] = {"calls": len(calls), "searches_attempted": attempted,
                       "searches_executed": executed, "overbudget_search_attempts": overbudget,
                       "max_input_tokens": max((x["actual_input_tokens_recorded"] for x in calls), default=0)}
    overbudget_total = sum(x["overbudget_search_attempts"] for x in budget.values())
    report = {"status": "pass_with_invalid_actions_recorded" if not failures and len(rows) == 194 and len(by_case) == 64 else "fail",
              "cases": len(by_case), "historical_calls": len(rows), "tokenizer_roundtrip_failures": failures,
              "new_searches": 0, "new_model_calls": 0, "gpu_calls": 0, "updates": 0,
              "budget_by_case": budget, "tokenizer_path": "/root/autodl-tmp/qwen_08b",
              "model_weights_loaded": False}
    target = OUT / "tokenizer-replay-verification-v2.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(by_case), "calls": len(rows),
                      "failures": len(failures), "overbudget_search_attempts": overbudget_total,
                      "max_input_tokens": max(x["max_input_tokens"] for x in budget.values())}, ensure_ascii=False))


def replay_production_inputs() -> None:
    """Rebuild each historical prompt through the production Goal-Q _inputs path."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import close_v133_goal_ai as production

    client, _tokenizer = production.renderer()
    prompts = {r["case_id"]: r["question"] for r in read_jsonl(CASES)}
    rows = read_jsonl(OUT / "trajectory-call-ledger.jsonl")
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)

    checks = []
    failures = []
    for cid, calls in by_case.items():
        if cid not in prompts:
            failures.append({"case_id": cid, "failure": "case_question_missing"})
            continue
        calls.sort(key=lambda x: x["call_index"])
        messages = production.initial(prompts[cid])
        for call_pos, row in enumerate(calls):
            try:
                enc = client._inputs(messages)
                ids = enc["input_ids"][0].tolist()
                decoded = client._processor.tokenizer.decode(ids, skip_special_tokens=False)
                check = {"case_id": cid, "call_index": row["call_index"],
                         "rebuilt_tokens": len(ids),
                         "recorded_tokens": row["actual_input_tokens_recorded"],
                         "token_ids_match": ids == row["prompt_token_ids"],
                         "prompt_hash_match": hashlib.sha256(decoded.encode()).hexdigest() == row["prompt_sha256"],
                         "budget_usage": enc.get("_context_usage")}
                checks.append(check)
                if not check["token_ids_match"] or len(ids) != row["actual_input_tokens_recorded"] or len(ids) > 2048:
                    failures.append({"case_id": cid, "call_index": row["call_index"],
                                     "failure": "production_input_reconstruction_mismatch"})
                action = row.get("parsed_action") or {}
                stored_obs = row.get("tool_result")
                if action.get("action") == "search" and stored_obs is not None:
                    # The call ledger's decoded JSON reordered keys. Restore the
                    # production search serializer's stable key order before
                    # _inputs applies the historical context budget and truncation.
                    obs = {k: stored_obs[k] for k in ("ok", "result", "searched_queries")
                           if k in stored_obs}
                    if "result" in obs:
                        ordered_docs = []
                        for doc in obs["result"]:
                            ordered_docs.append({k: doc[k] for k in ("id", "title", "text", "score") if k in doc}
                                                | {k: v for k, v in doc.items() if k not in {"id", "title", "text", "score"}})
                        obs["result"] = ordered_docs
                    messages = production.extend(messages, action["query"], obs)
            except Exception as exc:
                failures.append({"case_id": cid, "call_index": row["call_index"],
                                 "failure": type(exc).__name__, "detail": str(exc)[:500]})
                break

    report = {"status": "pass" if not failures and len(checks) == 194 and len(by_case) == 64 else "fail",
              "cases": len(by_case), "historical_calls": len(checks), "checks": checks,
              "failures": failures, "production_renderer": str(ROOT / "scripts/close_v133_goal_ai.py"),
              "production_renderer_sha256": sha(ROOT / "scripts/close_v133_goal_ai.py"),
              "local_model_client_sha256": sha(ROOT / "src/mini_agent_rl/model/local.py"),
              "prompt_source_sha256": sha(ROOT / "src/mini_agent_rl/prompts.py"),
              "question_file_sha256": sha(CASES), "new_searches": 0, "new_model_calls": 0,
              "gpu_calls": 0, "updates": 0, "model_weights_loaded": False}
    target = OUT / "production-input-replay-v4.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(by_case), "calls": len(checks),
                      "failures": len(failures), "renderer_sha256": report["production_renderer_sha256"]}, ensure_ascii=False))


def audit_mechanisms() -> None:
    admission = {x["case_id"]: x for x in read_jsonl(OUT / "source-admission-ledger.jsonl")}
    edges = read_jsonl(OUT / "edge-evidence-ledger.jsonl")
    by_edges = {}
    for e in edges:
        by_edges.setdefault(e["case_id"], []).append(e)
    calls = read_jsonl(OUT / "trajectory-call-ledger.jsonl")
    by_calls = {}
    for c in calls:
        by_calls.setdefault(c["case_id"], []).append(c)
    for items in by_calls.values():
        items.sort(key=lambda x: x["call_index"])
    result_rows = []
    answer_error_notes = {
        "musique:2hop__829358_754719": "The visible chain states Darley was born in Pedro Leopoldo and that Pedro Leopoldo is in Minas Gerais; output Minsk Region is a different place.",
        "musique:2hop__506035_325512": "The visible chain identifies the requested county as Pearl River County; the answer omits 'County' and names a less specific entity.",
        "musique:2hop__703493_162341": "The visible chain states a nomination for Best Supporting Actor; the answer 'Academy Award' omits nomination status and category.",
        "2wiki:b587d3d20bdd11eba7f7acde48001122": "The visible chain identifies Peter as Maria's spouse and gives his death place as Pinos Puente; output Sijena is Maria's death place.",
        "2wiki:f299dd600bdd11eba7f7acde48001122": "The visible chain identifies the fourth duke as the fifth duke's father and gives the father's death year as 1575; output 1544 is the son's birth year.",
    }
    for cid, gate in admission.items():
        cs = by_calls[cid]
        es = sorted(by_edges[cid], key=lambda x: x["edge_index_for_source_join"])
        reports = []
        for e in es:
            q = e.get("quote") or ""
            escaped = json.dumps(q, ensure_ascii=False)[1:-1] if q else ""
            returned = []
            retained = []
            for i, c in enumerate(cs):
                tr = c.get("tool_result") or {}
                docs = tr.get("result") or []
                for d in (docs if isinstance(docs, list) else []):
                    if str(d.get("id")) == str(e["document_id"]) and q and q in str(d.get("text", "")):
                        returned.append(c["call_index"])
                        next_call = cs[i + 1] if i + 1 < len(cs) else None
                        if next_call and escaped in next_call["actual_prompt"]:
                            retained.append(c["call_index"])
            visible_calls = [c["call_index"] for c in cs if escaped and escaped in c["actual_prompt"]]
            reports.append({"edge_id": e["edge_id"], "status": e["status"],
                            "support_document_id": e["document_id"], "support_quote": q,
                            "returned_call_indices": returned, "retained_in_immediate_next_input_after": retained,
                            "visible_input_call_indices": visible_calls,
                            "direct_context_loss": bool(returned and any(x not in retained for x in returned)),
                            "evidence_review_status": "source_quote_directly_checked" if q else "source_edge_unconfirmed"})
        last = cs[-1]
        action = last.get("parsed_action") or {}
        complete_visible = gate["admission"] == "admitted" and all(bool(e["quote"]) and
                            json.dumps(e["quote"], ensure_ascii=False)[1:-1] in last["actual_prompt"]
                            for e in es if e["status"] == "pass")
        errors = answer_error_notes.get(cid)
        direct_answer = bool(complete_visible and action.get("action") == "answer" and errors)
        result_rows.append({"case_id": cid, "source_admitted": gate["admission"] == "admitted",
                            "actual_calls": len(cs),
                            "searches_attempted": sum(c["parsed_action"].get("action") == "search" for c in cs),
                            "searches_executed": sum(c["parsed_action"].get("action") == "search" and c.get("tool_result") is not None for c in cs),
                            "overbudget_search_attempts": max(0, sum(c["parsed_action"].get("action") == "search" for c in cs)-3),
                            "per_edge": reports,
                            "complete_chain_visible_at_final_action": complete_visible,
                            "final_action": action.get("action"), "final_answer": action.get("answer"),
                            "direct_answer_decision_error": direct_answer,
                            "answer_error_reason": errors if direct_answer else None,
                            "query_generation": "not_checked_without_a_valid_saved_counterfactual",
                            "retrieval": "not_checked_without_direct_ranking_confirmation",
                            "diagnostic_only": True, "training_allowed": False})
    direct = {axis: [] for axis in ("context", "retrieval", "query", "answer")}
    for r in result_rows:
        if not r["source_admitted"]:
            continue
        if any(e["direct_context_loss"] for e in r["per_edge"]):
            direct["context"].append(r["case_id"])
        if r["direct_answer_decision_error"]:
            direct["answer"].append(r["case_id"])
    # Reassess already-saved query branches and rankings using the corrected
    # source quotes. No branch is regenerated and no new retrieval is run.
    branch_path = ROOT / "runs/v133-goal-ai-fixed-z-coverage-v1/branches.jsonl"
    ranking_path = ROOT / "runs/v133-goal-ai-fixed-z-coverage-v1/historical-ranking.jsonl"
    branches = read_jsonl(branch_path)
    rankings = read_jsonl(ranking_path)
    original_rank = {}
    for row in rankings:
        if row.get("kind") == "original":
            original_rank.setdefault(row["case_id"], []).append(row)
    branch_rows = []
    direct_by_axis = {k: set(v) for k, v in direct.items()}
    admitted_ids = {cid for cid, g in admission.items() if g["admission"] == "admitted"}
    for b in branches:
        cid = b["case_id"]
        if cid not in admitted_ids:
            continue
        es = [e for e in by_edges[cid] if e["status"] == "pass"]
        prefix_docs = b.get("original_prefix_documents", [])
        result_docs = b.get("result", [])
        visible_text = b.get("answer_context", {}).get("actual_prompt", "")
        def has_quote(doclist, e):
            return any(str(d.get("document_id", d.get("id", ""))) == str(e["document_id"])
                       and e["quote"] and e["quote"] in str(d.get("text", "")) for d in doclist)
        returned = {e["edge_id"]: has_quote(prefix_docs + result_docs, e) for e in es}
        kept = {e["edge_id"]: bool(e["quote"] and json.dumps(e["quote"], ensure_ascii=False)[1:-1] in visible_text) for e in es}
        full_returned = all(returned.values()) if es else False
        full_retained = all(kept.values()) if es else False
        eligible = bool(b.get("grounded") and b.get("legal_budget") and not b.get("repeated")
                        and b.get("calls_if_answer", 99) <= 4 and b.get("searches_if_answer", 99) <= 3
                        and b.get("answer_context", {}).get("actual_input_tokens", 99999) <= 2048)
        raw = by_calls[cid]
        original_call = next((c for c in raw if c["call_index"] == b.get("anchor_call")), None)
        original_query = ((original_call or {}).get("parsed_action") or {}).get("query")
        alt = b.get("query", "")
        def relation_terms(e):
            r = e["relation"].casefold()
            rules = [("birth", ["birth", "born"]), ("death", ["death", "died"]),
                     ("father", ["father", "son", "daughter", "child"]),
                     ("mother", ["mother", "son", "daughter", "child"]),
                     ("spouse", ["spouse", "wife", "husband", "married"]),
                     ("performer", ["performer", "singer", "song", "recorded"]),
                     ("director", ["director", "directed"]), ("employer", ["work", "employer", "job"]),
                     ("publisher", ["publisher", "published"]), ("inception", ["founded", "inception"]),
                     ("team", ["team", "club", "played"]), ("label", ["label", "record company"]),
                     ("located", ["located", "location", "county", "province", "state"]),
                     ("death", ["date of death", "died", "death"]),
                     ("recognized", ["award", "recognition", "nominated"]),
                     ("citizenship", ["citizenship", "nationality", "country"])]
            for key, vals in rules:
                if key in r:
                    return vals
            return phrase_terms(r)
        missing_edges = [e for e in es if not has_quote(prefix_docs, e)]
        alt_has_relation = all(any(t in alt.casefold() for t in relation_terms(e)) for e in missing_edges) if missing_edges else False
        original_has_relation = all(any(t in str(original_query or "").casefold() for t in relation_terms(e)) for e in missing_edges) if missing_edges else True
        recovered = bool(eligible and full_returned and full_retained and missing_edges)
        query_direct = bool(recovered and alt_has_relation and not original_has_relation)
        if query_direct:
            direct_by_axis["query"].add(cid)
        ranks = b.get("ranking", {}).get("ranked", [])
        rank_by_id = {str(x.get("id")): x.get("rank") for x in ranks}
        retrieval_edges = []
        if eligible:
            for e in missing_edges:
                rr = rank_by_id.get(str(e["document_id"]))
                if rr is not None and rr > 3 and any(t in alt.casefold() for t in relation_terms(e)):
                    retrieval_edges.append({"edge_id": e["edge_id"], "support_document_id": e["document_id"], "rank": rr})
        if retrieval_edges:
            direct_by_axis["retrieval"].add(cid)
        branch_rows.append({"case_id": cid, "branch_id": b["branch_id"], "query": alt,
                            "original_query_at_anchor": original_query,
                            "eligible_saved_branch": eligible, "source_edges_returned": returned,
                            "source_edges_retained": kept, "complete_chain_returned": full_returned,
                            "complete_chain_retained": full_retained,
                            "original_prefix_missing_edge_ids": [e["edge_id"] for e in missing_edges],
                            "original_query_mentions_required_relation": original_has_relation,
                            "alternative_query_mentions_required_relation": alt_has_relation,
                            "direct_query_repair": query_direct,
                            "direct_retrieval_rank_misses": retrieval_edges,
                            "diagnostic_only": True, "training_allowed": False})
    direct = {k: sorted(v) for k, v in direct_by_axis.items()}
    counts = {k: len(v) for k, v in direct.items()}
    precedence = ["context", "retrieval", "query", "answer"]
    maximum = max(counts.values(), default=0)
    tied = [x for x in precedence if counts[x] == maximum and maximum >= 4]
    decision = tied[0] if tied else "mechanism_evidence_insufficient"
    write_jsonl(OUT / "trajectory-mechanism-ledger-v2.jsonl", result_rows)
    write_jsonl(OUT / "saved-branch-replay-ledger.jsonl", branch_rows)
    summary = {"fixed_case_denominator": 64, "source_admitted_denominator": sum(x["admission"] == "admitted" for x in admission.values()),
               "historical_call_rows": len(calls), "direct_mechanism_case_ids": direct,
               "direct_mechanism_counts": counts, "minimum_route_cases": 4,
               "next_route": decision,
               "note": "Query and retrieval remain not checked where no direct stored counterfactual/rank suffices; not_checked is not a zero-case finding.",
               "new_searches": 0, "new_model_calls": 0, "gpu_calls": 0, "updates": 0}
    target = OUT / "mechanism-decision-v2.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"admitted": summary["source_admitted_denominator"], "direct_counts": counts,
                      "next_route": decision, "cases": 64, "calls": len(calls)}, ensure_ascii=False))


def scan_candidates() -> None:
    """Locate explicit subject/object/predicate sentences in frozen candidates.

    This is a locator only. A human must read any emitted sentence before a
    candidate can change an edge decision.
    """
    edge_rows = read_jsonl(OUT / "edge-evidence-ledger.jsonl")
    by_eid = {x["edge_id"]: x for x in edge_rows}
    docs = read_jsonl(OUT / "candidate-documents.jsonl")
    cue_map = {
        "father": ["son of", "daughter of", "father", "child of"],
        "mother": ["son of", "daughter of", "mother", "child of"],
        "spouse": ["wife", "husband", "married"],
        "place of birth": ["born", "birth"], "date of birth": ["born"],
        "place of death": ["died", "death"], "date of death": ["died", "death"],
        "director": ["directed"], "publisher": ["published"], "inception": ["founded", "established"],
        "country of citizenship": ["citizenship", "american", "nationality", "from"],
        "employer": ["professor", "director", "employed", "appointed", "worked"],
        "educated at": ["graduated", "studied", "attended", "school"],
        "record label": ["record label", "released on", "label"],
        "owned by": ["owned by", "owned", "owner"],
        "performer": ["performed", "recorded", "song by", "singer", "starring"],
        "place of death": ["died", "death"],
    }
    result = []
    for d in docs:
        e = by_eid[d["edge_id"]]
        subject, obj, rel = e["subject"], e["object"], e["relation"]
        ns, no, nr = norm(subject), norm(obj), norm(rel)
        cues = cue_map.get(nr, [])
        if not cues:
            q = str(e.get("qualifiers", {}).get("source_question", "")).casefold()
            for k, vals in {"father": cue_map["father"], "mother": cue_map["mother"],
                            "birth": ["born", "birth"], "death": ["died", "death"],
                            "study": ["studied", "graduated", "attended", "school"],
                            "feature": ["starred", "starring", "featured", "highlighted"],
                            "award": ["nominated", "award", "won"], "performer": cue_map["performer"],
                            "label": cue_map["record label"], "employer": cue_map["employer"],
                            "found": cue_map["inception"], "paid": ["highest paid", "richest", "most paid"],
                            "located": ["located", "in the province", "city in", "county"]}.items():
                if k in q or k in nr:
                    cues += vals
        text = str(d.get("text", ""))
        sentences = [m.group(0).strip() for m in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text) if m.group(0).strip()]
        for sent in sentences:
            low = sent.casefold()
            no_hit = bool(no and no in norm(sent))
            # Accept an exact target entity in the document title as identity
            # context, but require predicate and object in the cited sentence.
            subj_hit = bool(ns and (ns in norm(sent) or ns in norm(d.get("title", ""))))
            cue_hit = [c for c in cues if c in low]
            if no_hit and subj_hit and cue_hit:
                result.append({"case_id": d["case_id"], "edge_id": d["edge_id"],
                               "document_id": d["document_id"], "title": d["title"],
                               "candidate_sentence": sent, "object_literal_found": True,
                               "subject_identity_found": True, "relation_cues_found": cue_hit,
                               "human_review_status": "pending", "diagnostic_only": True,
                               "training_allowed": False})
    path = OUT / "candidate-locator-hits.jsonl"
    if path.exists():
        raise FileExistsError(path)
    write_jsonl(path, result)
    print(json.dumps({"candidate_documents_screened": len(docs), "potential_sentence_hits": len(result),
                      "unique_edges_with_hits": len({x['edge_id'] for x in result}),
                      "human_review_required": True, "new_searches": 0, "new_model_calls": 0,
                      "gpu_calls": 0, "updates": 0}))


def revise_scholarship_edge() -> None:
    """Correct the exposed scholarship/attendance boundary as a versioned row."""
    cid = "musique:2hop__95673_121319"
    eid = "edge1:decomp-121319"
    revision_path = OUT / "source-review-revisions.jsonl"
    if revision_path.exists():
        raise FileExistsError(revision_path)
    edges = read_jsonl(OUT / "edge-evidence-ledger.jsonl")
    admissions = read_jsonl(OUT / "source-admission-ledger.jsonl")
    old_edge = next(x for x in edges if x["case_id"] == cid and x["edge_id"] == eid)
    if old_edge["status"] != "pass":
        raise RuntimeError("Expected original scholarship edge to be pass before revision")
    revision = {"case_id": cid, "edge_id": eid, "old_status": "pass", "new_status": "unconfirmed",
                "reason": "Winning a scholarship at the American Academy of Dramatic Arts does not establish attendance, study, or employment there.",
                "old_quote": old_edge["quote"], "review_history": "exposed case; explicit source-bound correction",
                "diagnostic_only": True, "training_allowed": False}
    revised_edges = []
    for e in edges:
        if e["case_id"] == cid and e["edge_id"] == eid:
            e = dict(e, status="unconfirmed", quote=None, source_char_span=None,
                     reason=revision["reason"], revision_id="source-review-revision-001")
        revised_edges.append(e)
    revised_admission = []
    for a in admissions:
        if a["case_id"] == cid:
            a = dict(a, chain_supported="unconfirmed", question_resolvable="unconfirmed",
                     label_compatible="unconfirmed", admission="not_admitted",
                     reason=revision["reason"], revision_id="source-review-revision-001")
        revised_admission.append(a)
    packet = next(x for x in read_jsonl(OLD / "official-source-packets.jsonl") if x["case_id"] == cid)
    official_ids = {str(i) for d in packet["official_support_documents"] for i in d.get("frozen_corpus_matches", [])}
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    entity = norm("Edward G. Robinson")
    terms = ["academy", "study", "studied", "attended", "school", "scholarship", "work", "worked"]
    ranked = []
    for d in corpus:
        did = str(d.get("id", ""))
        if did in official_ids:
            continue
        title, body = norm(d.get("title", "")), norm(d.get("text", ""))
        title_hit, body_hit = int(entity in title), int(entity in body)
        rel_hits = sum(t in body.split() for t in terms)
        if title_hit or body_hit:
            ranked.append((-(title_hit*20 + body_hit*5 + rel_hits), did,
                           {"case_id": cid, "edge_id": eid, "document_id": did,
                            "title": d.get("title"), "entity_title_match": bool(title_hit),
                            "entity_body_match": bool(body_hit), "relation_term_hits": rel_hits,
                            "status": "frozen_candidate_pending_review", "body_read": False}))
    ranked.sort(key=lambda x: (x[0], x[1]))
    addendum = [x[2] for x in ranked[:12]]
    write_jsonl(revision_path, [revision])
    write_jsonl(OUT / "edge-evidence-ledger.final.jsonl", revised_edges)
    write_jsonl(OUT / "source-admission-ledger.final.jsonl", revised_admission)
    write_jsonl(OUT / "candidate-freeze-addendum.jsonl", addendum)
    summary = {"revision_id": "source-review-revision-001", "case_id": cid,
               "edge_status": "unconfirmed", "candidate_count": len(addendum),
               "admitted_final": sum(x["admission"] == "admitted" for x in revised_admission),
               "new_searches": 0, "new_model_calls": 0, "gpu_calls": 0, "updates": 0}
    (OUT / "source-revision-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def materialize_revision_candidates() -> None:
    freeze_path = OUT / "candidate-freeze-addendum.jsonl"
    target = OUT / "candidate-documents-addendum.jsonl"
    if not freeze_path.is_file() or target.exists():
        raise RuntimeError("Revision candidate freeze missing or already materialized")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    by_id = {str(d.get("id")): d for d in corpus}
    rows = []
    for row in read_jsonl(freeze_path):
        d = by_id[row["document_id"]]
        rows.append({**row, "text": d.get("text", ""),
                     "body_sha256": hashlib.sha256(str(d.get("text", "")).encode()).hexdigest()})
    write_jsonl(target, rows)
    print(json.dumps({"materialized_revision_candidates": len(rows), "new_searches": 0,
                      "new_model_calls": 0, "gpu_calls": 0, "updates": 0}))


def review_revision_candidates() -> None:
    rows = read_jsonl(OUT / "candidate-documents-addendum.jsonl")
    if len(rows) != 1:
        raise RuntimeError("Unexpected scholarship-edge candidate count")
    d = rows[0]
    if "Edward G. Robinson" not in d["text"]:
        raise RuntimeError("Candidate identity does not match the frozen subject")
    quote = "The film stars Charlton Heston in the lead role, Yul Brynner as Rameses, Anne Baxter as Nefretiri, Edward G. Robinson as Dathan"
    start = d["text"].find(quote)
    if start < 0:
        raise RuntimeError("Candidate citation span cannot be located")
    row = {"case_id": d["case_id"], "edge_id": d["edge_id"], "document_id": d["document_id"],
           "title": d["title"], "status": "reviewed_no_support",
           "quote": quote, "source_char_span": [start, start + len(quote)],
           "reason": "The candidate binds Edward G. Robinson to a film role; it does not state study, attendance, or work at the American Academy of Dramatic Arts.",
           "body_sha256": d["body_sha256"], "reviewer": "primary-agent",
           "diagnostic_only": True, "training_allowed": False}
    target = OUT / "candidate-review-addendum.jsonl"
    if target.exists():
        raise FileExistsError(target)
    write_jsonl(target, [row])
    print(json.dumps({"candidate_reviewed": 1, "support_found": 0, "edge_status": "unconfirmed"}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--materialize-candidates", action="store_true")
    ap.add_argument("--candidate-sample", action="store_true")
    ap.add_argument("--verify-calls", action="store_true")
    ap.add_argument("--audit-mechanisms", action="store_true")
    ap.add_argument("--scan-candidates", action="store_true")
    ap.add_argument("--revise-scholarship-edge", action="store_true")
    ap.add_argument("--materialize-revision-candidates", action="store_true")
    ap.add_argument("--review-revision-candidates", action="store_true")
    ap.add_argument("--replay-production-inputs", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--skip-official", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()
    stages = [args.prepare, args.materialize_candidates, args.candidate_sample, args.verify_calls, args.audit_mechanisms, args.scan_candidates, args.revise_scholarship_edge, args.materialize_revision_candidates, args.review_revision_candidates, args.replay_production_inputs]
    if sum(stages) != 1:
        ap.error("Select exactly one CPU stage")
    if args.prepare:
        prepare()
    elif args.materialize_candidates:
        materialize_candidates()
    elif args.candidate_sample:
        candidate_sample(args.start, args.limit, args.skip_official, args.rank)
    elif args.verify_calls:
        verify_calls()
    elif args.audit_mechanisms:
        audit_mechanisms()
    elif args.scan_candidates:
        scan_candidates()
    elif args.revise_scholarship_edge:
        revise_scholarship_edge()
    elif args.materialize_revision_candidates:
        materialize_revision_candidates()
    elif args.review_revision_candidates:
        review_revision_candidates()
    else:
        replay_production_inputs()


if __name__ == "__main__":
    main()

