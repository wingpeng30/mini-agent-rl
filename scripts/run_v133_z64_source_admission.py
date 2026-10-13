"""Freeze and prepare the bounded Goal-Z-dev source-admission audit.

CPU-only: reads only the fixed 64 Goal Z development cases, their two official
dataset dev archives, the frozen dev corpus, and prior audit/trajectory files.
It does not search, load model weights, or call a model.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ai-z64-source-admission-v2"
CASES = ROOT / "runs/v133-goal-ah-entity-relation/evaluation-cases-goal_z_dev.jsonl"
CORPUS = ROOT / "runs/v133-goal-z-multihop-sft/corpus-dev.json"
SOURCES = ROOT.parent / "goal-z-sources"
MUSIQUE = SOURCES / "musique_v1.0.zip"
WIKI = SOURCES / "2wiki_data_ids_april7.zip"
PRIOR = ROOT / "runs/v133-goal-ai-coverage-source-v1"
AH = ROOT / "runs/v133-goal-ah-entity-relation"
FIXED = ROOT / "runs/v133-goal-ai-fixed-z-coverage-v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    required = [CASES, CORPUS, MUSIQUE, WIKI,
                PRIOR / "case-source-review.jsonl", PRIOR / "edge-review-v2.jsonl",
                AH / "actual-inputs-goal_z_dev-baseline.jsonl",
                AH / "final-source-review-baseline.jsonl",
                FIXED / "diagnostic-ledger.jsonl", FIXED / "historical-ranking.jsonl"]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(p)
    cases = read_jsonl(CASES)
    if len(cases) != 64 or len({c["case_id"] for c in cases}) != 64:
        raise RuntimeError("Goal Z dev freeze is not exactly 64 unique cases")
    if Counter(c["case_id"].split(":", 1)[0] for c in cases) != {"musique": 32, "2wiki": 32}:
        raise RuntimeError("Unexpected source composition")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if len(corpus) != 2048:
        raise RuntimeError("Frozen Goal Z dev corpus size changed")
    corpus_by_norm = {}
    for d in corpus:
        corpus_by_norm.setdefault((norm(d.get("title", "")), norm(d.get("text", ""))), []).append(d)

    prior_cases = {x["case_id"]: x for x in read_jsonl(PRIOR / "case-source-review.jsonl")}
    prior_edges = {}
    for x in read_jsonl(PRIOR / "edge-review-v2.jsonl"):
        prior_edges.setdefault(x["case_id"], []).append(x)
    ah_reviews = {x["case_id"]: x for x in read_jsonl(AH / "final-source-review-baseline.jsonl")
                  if x.get("split") == "goal_z_dev" and x.get("model") == "baseline"}
    actual_inputs = {x["case_id"]: x for x in read_jsonl(AH / "actual-inputs-goal_z_dev-baseline.jsonl")}
    diagnostics = {x["case_id"]: x for x in read_jsonl(FIXED / "diagnostic-ledger.jsonl")}
    rankings = read_jsonl(FIXED / "historical-ranking.jsonl")

    wanted_m = {c["case_id"].split(":", 1)[1] for c in cases if c["case_id"].startswith("musique:")}
    wanted_w = {c["case_id"].split(":", 1)[1] for c in cases if c["case_id"].startswith("2wiki:")}
    source_rows = {}
    with zipfile.ZipFile(MUSIQUE) as z:
        with z.open("data/musique_ans_v1.0_dev.jsonl") as f:
            for i, line in enumerate(f, 1):
                row = json.loads(line)
                if row.get("id") in wanted_m:
                    source_rows["musique:" + row["id"]] = ("musique_v1.0.zip", "data/musique_ans_v1.0_dev.jsonl", i, row)
    with zipfile.ZipFile(WIKI) as z:
        rows = json.load(z.open("dev.json"))
    for i, row in enumerate(rows, 1):
        if row.get("_id") in wanted_w:
            source_rows["2wiki:" + row["_id"]] = ("2wiki_data_ids_april7.zip", "dev.json", i, row)
    if set(source_rows) != {c["case_id"] for c in cases}:
        raise RuntimeError(f"Official source ID mismatch: missing={sorted({c['case_id'] for c in cases}-set(source_rows))[:5]}")

    OUT.mkdir(parents=True)
    # Freeze only permitted inputs and hashes. Source records are exported only for the exact fixed IDs.
    code_path = Path(__file__)
    freeze = {
        "run_id": OUT.name,
        "scope": "goal_z_dev only; 64 fixed IDs/order/questions; diagnostic-only; no training",
        "limits": {"cases": 64, "required_edges": 128, "official_support_bodies": 128,
                   "candidate_documents_per_edge": 12, "edge_document_pairs": 1536,
                   "new_searches": 0, "model_calls": 0, "gpu_calls": 0, "updates": 0},
        "case_ids_in_order": [c["case_id"] for c in cases],
        "case_ids_order_sha256": hashlib.sha256("\n".join(c["case_id"] for c in cases).encode()).hexdigest(),
        "files_sha256": {str(p.relative_to(ROOT.parent)): sha(p) for p in required + [code_path]},
        "source_archives_sha256": {MUSIQUE.name: sha(MUSIQUE), WIKI.name: sha(WIKI)},
        "corpus_count": len(corpus),
        "official_records_located": len(source_rows),
        "prior_source_review_reuse_count": len(set(prior_cases) & set(source_rows)),
        "model_identity_reference": "Goal Z seed42; AH/AA frozen identity verification references recorded, no model loaded",
        "review_history": "12 Goal-Z-dev rows inherited from finite prior source audit; not blind. Other rows use official source records plus fixed AH final review references.",
    }
    (OUT / "freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    packets, source_case_rows, edge_rows = [], [], []
    for c in cases:
        cid = c["case_id"]
        source_name, member, record_num, row = source_rows[cid]
        if norm(row.get("question", "")) != norm(c["question"]):
            raise RuntimeError(f"Question mismatch for {cid}")
        answer = str(row.get("answer", ""))
        if norm(answer) != norm(c["answer"]):
            raise RuntimeError(f"Official answer mismatch for {cid}: {answer!r} vs {c['answer']!r}")
        docs = []
        if cid.startswith("musique:"):
            chain = row.get("question_decomposition", [])
            paragraphs = row.get("paragraphs", [])
            if len(chain) != 2:
                raise RuntimeError(f"MuSiQue chain not 2 edges: {cid}")
            for ei, step in enumerate(chain):
                idx = step.get("paragraph_support_idx")
                p = next((p for p in paragraphs if p.get("idx") == idx), None)
                if p is None:
                    raise RuntimeError(f"Official support paragraph missing {cid} edge {ei}")
                docs.append({"edge_index": ei, "title": p["title"], "text": p["paragraph_text"],
                             "official_support_flag": bool(p.get("is_supporting")), "source_paragraph_idx": idx,
                             "source_id": str(step.get("id")), "subquestion": step.get("question"),
                             "subanswer": step.get("answer")})
            edges = [{"subject": str(s.get("question", "")), "relation": "official decomposition subquestion",
                      "object": str(s.get("answer", "")), "official_decomposition": s}
                     for s in chain]
        else:
            facts = row.get("supporting_facts", [])
            context = row.get("context", [])
            titles = list(dict.fromkeys(f[0] for f in facts))
            if len(titles) < 2:
                raise RuntimeError(f"2Wiki support titles <2: {cid}")
            for ei, title in enumerate(titles[:2]):
                sentences = next((ss for t, ss in context if t == title), None)
                if sentences is None:
                    raise RuntimeError(f"2Wiki supporting context missing {cid} {title}")
                docs.append({"edge_index": ei, "title": title, "text": " ".join(sentences),
                             "official_support_flag": True,
                             "supporting_facts": [f for f in facts if f[0] == title]})
            edges = [{"edge_index": i, "official_evidence": ev} for i, ev in enumerate(row.get("evidences", []))]
        source_docs = []
        for d in docs:
            matches = corpus_by_norm.get((norm(d["title"]), norm(d["text"])), [])
            source_docs.append({**d, "frozen_corpus_matches": [x.get("id") for x in matches],
                                "exact_body_preserved": bool(matches),
                                "frozen_corpus_match_count": len(matches)})
            edge_rows.append({"case_id": cid, "edge_index": d["edge_index"],
                              "official_subquestion_or_fact": d.get("subquestion", d.get("supporting_facts")),
                              "official_answer_hint": d.get("subanswer"),
                              "official_support_flag": d["official_support_flag"],
                              "support_document_title": d["title"],
                              "support_document_body": d["text"],
                              "source_char_span": [0, len(d["text"])],
                              "frozen_corpus_matches": [x.get("id") for x in matches],
                              "source_authority_only": True,
                              "semantic_review_status": "pending_one_pass_adjudication"})
        ah = ah_reviews.get(cid)
        if ah is None or cid not in actual_inputs:
            raise RuntimeError(f"Missing baseline source/trajectory records for {cid}")
        packets.append({"case_id": cid, "question": c["question"], "official_answer": answer,
                        "goal_z_order": len(packets), "source": source_name, "archive_member": member,
                        "record_number": record_num, "official_chain_or_evidence": edges,
                        "official_support_documents": source_docs,
                        "prior_source_audit": prior_cases.get(cid),
                        "ah_final_source_review_reference": {k: ah.get(k) for k in
                            ("category", "evidence_complete", "entity_mismatch", "relation_mismatch", "ambiguity", "missing_relations", "reason", "review_id", "scoring_version")},
                        "diagnostic_mechanism_reference": ({k: diagnostics[cid].get(k) for k in
                            ("primary_block", "coexisting_verified_mechanisms", "required_edges")}
                            if cid in diagnostics else None),
                        "trajectory_calls": len(actual_inputs[cid].get("calls", [])),
                        "diagnostic_only": True, "training_allowed": False})
        source_case_rows.append({"case_id": cid, "question": c["question"], "official_answer": answer,
                                 "source_record_exact_question": True, "source_record_exact_answer": True,
                                 "source_archive": source_name, "member": member, "record_number": record_num,
                                 "official_support_documents": len(docs),
                                 "all_support_bodies_exact_in_corpus": all(x["exact_body_preserved"] for x in source_docs),
                                 "prior_audit_reused": cid in prior_cases,
                                 "chain_supported": "unconfirmed", "question_resolvable": "unconfirmed",
                                 "label_compatible": "unconfirmed",
                                 "admission": "pending_one_pass_adjudication", "diagnostic_only": True,
                                 "training_allowed": False})
    write_jsonl(OUT / "official-source-packets.jsonl", packets)
    write_jsonl(OUT / "edge-evidence-ledger.jsonl", edge_rows)
    write_jsonl(OUT / "source-admission-ledger.jsonl", source_case_rows)
    # Preserve trajectory-source evidence for later offline adjudication; no new outputs are generated.
    write_jsonl(OUT / "trajectory-source-index.jsonl", [
        {"case_id": c["case_id"], "actual_inputs_record": actual_inputs[c["case_id"]],
         "diagnostic_record": diagnostics.get(c["case_id"]),
         "existing_rankings": [r for r in rankings if r.get("case_id") == c["case_id"]],
         "existing_prior_edges": prior_edges.get(c["case_id"], [])}
        for c in cases])
    summary = {"cases": len(cases), "official_records": len(packets),
               "official_support_bodies": len(edge_rows),
               "prior_source_audits_reused": sum(x["prior_audit_reused"] for x in source_case_rows),
               "all_support_bodies_exact_in_corpus": sum(x["all_support_bodies_exact_in_corpus"] for x in source_case_rows),
               "pending_case_gates": len(source_case_rows), "new_searches": 0,
               "model_calls": 0, "gpu_calls": 0, "updates": 0}
    (OUT / "prepare-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
