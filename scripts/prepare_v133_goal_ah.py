"""Goal AH CPU preparation. Never loads a model or authorizes GPU execution.

All proposals are drafts until explicit source-bound review. Existing AG files
are immutable inputs. An unmet quota is a terminal material gate, not a reason
to sample more data or change the experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ah-entity-relation"
Z = ROOT / "runs/v133-goal-z-multihop-sft"
AB = ROOT / "runs/v133-goal-ab-evidence-decision"
AG = ROOT / "runs/v133-goal-ag-failure-repair"
EXPECTED_ADAPTER = "d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52"
VERDICTS = ("supported", "entity_mismatch", "relation_mismatch", "missing_link")


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(value):
    return hashlib.sha256(("goal-ah-v1\0" + value).encode()).hexdigest()


def write_once(path, value, *, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        if jsonl:
            for x in value:
                f.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            f.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def documents(prompt):
    docs = []
    for block, m in enumerate(re.finditer(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", prompt, re.S)):
        payload = json.loads(m.group(1))
        for index, d in enumerate(payload.get("result", [])):
            docs.append({"reference": f"E{len(docs)+1}", "document_id": d["id"],
                         "title": d["title"], "text": d["text"],
                         "tool_block": block, "document_index": index,
                         "input_char_span": [m.start(), m.end()]})
    return docs


def normalize(value):
    return re.sub(r"\W+", "", str(value).casefold())


def proposal(case, call, verdict, claim, facts, missing, origin, real_error):
    target = {"verdict": verdict, "facts": facts, "missing": missing}
    key = stable(case["id"] + json.dumps([call["prompt_sha256"], verdict, claim], sort_keys=True))
    return {"judgment_id": key, "case_id": case["id"], "question": case["question"],
            "verdict": verdict, "claim": claim, "target": target,
            "actual_prompt": call["actual_prompt"], "actual_input_token_ids": call["prompt_token_ids"],
            "actual_prompt_sha256": call["prompt_sha256"], "call_index": call["call_index"],
            "visible_documents": documents(call["actual_prompt"]),
            "origin": origin, "from_real_error_case": real_error,
            "review_status": "pending", "label_is_factually_false": False}


def prepare_proposals():
    if (OUT / "material-proposals.jsonl").exists():
        raise FileExistsError("AH proposal pool already frozen; do not regenerate")
    decoded = rows(OUT / "cpu-source-contexts.jsonl")
    contexts = {(x["kind"], x["case_id"]): x for x in decoded}
    selected = {x["id"]: x for x in rows(Z / "selected-train.jsonl")}
    approved = {x["case_id"]: x for x in rows(AB / "decision-pairs.jsonl")}
    failures = {x["case_id"]: x for x in rows(AB / "train-decision-review.jsonl")
                if x["on_policy_stop_error"]}
    proposals = []
    wiki = sorted((x for x in selected.values() if x["source"] == "2wiki"), key=lambda x: stable(x["id"]))
    for c in wiki:
        call = contexts[("teacher", c["id"])]["calls"][-1]
        ds = documents(call["actual_prompt"])
        chain = c["chain"]
        facts = []
        for triple in chain:
            doc = next((d for d in ds if normalize(d["title"]) == normalize(triple[0])), None)
            if not doc:
                break
            facts.append(list(triple) + [doc["reference"]])
        if len(facts) != 2:
            continue
        proposals.append(proposal(c, call, "supported", chain, facts, [],
                                  "original_teacher_actual_context", c["id"] in failures))
    if len(proposals) != 32:
        raise RuntimeError(f"Expected 32 bounded positive proposals, got {len(proposals)}")
    # The negative claim is a candidate for auditing, never a fabricated source.
    core = [x for x in wiki if x["id"] in approved]
    if len(core) != 16:
        raise RuntimeError("Expected the existing sixteen reviewed 2Wiki state pairs")
    for c in core:
        call = contexts[("teacher", c["id"])]["calls"][-1]
        ds = documents(call["actual_prompt"])
        first, second = c["chain"]
        source = next(d for d in ds if normalize(d["title"]) == normalize(first[0]))
        alt = next(d for d in reversed(ds) if normalize(d["title"]) not in
                   {normalize(first[0]), normalize(second[0])})
        proposals.append(proposal(c, call, "entity_mismatch",
            [[first[0], first[1], alt["title"]]], [list(first)+[source["reference"]]], [],
            "counterfactual_claim_in_unchanged_teacher_context", False))
        wrong_relation = {"date of birth": "date of death", "date of death": "date of birth",
            "place of birth": "place of death", "place of death": "place of birth",
            "country of citizenship": "place of birth", "spouse": "child", "child": "spouse",
            "father": "child", "cause of death": "place of death"}[second[1]]
        second_doc = next(d for d in ds if normalize(d["title"]) == normalize(second[0]))
        proposals.append(proposal(c, call, "relation_mismatch",
            [[second[0], wrong_relation, second[2]]], [list(second)+[second_doc["reference"]]], [],
            "counterfactual_claim_in_unchanged_teacher_context", False))
    # A bounded proposal heuristic only: absent answer text is NOT an acceptance
    # rule. The reviewer must check every visible paragraph and missing edge.
    missing = []
    for cid in sorted(failures, key=stable):
        c = selected[cid]
        if "chain" not in c:
            continue  # Legacy rows lack an auditable source relation chain.
        context = contexts[("greedy", cid)]
        for call in reversed(context["calls"]):
            ds = documents(call["actual_prompt"])
            if not ds or any(normalize(c["answer"]) in normalize(d["text"]) for d in ds):
                continue
            if c["source"] == "2wiki":
                needed = [c["chain"][-1][0], c["chain"][-1][1], "?"]
            else:
                ch = c["chain"]
                needed = [ch[0]["answer"], ch[1]["question"].replace("#1", ch[0]["answer"]), "?"]
            missing.append(proposal(c, call, "missing_link", [], [], [needed],
                "actual_greedy_error_context", True))
            break
        if len(missing) == 32:
            break
    proposals += missing
    if len(proposals) > 96 or len({x["judgment_id"] for x in proposals}) != len(proposals):
        raise RuntimeError("Proposal pool exceeds the frozen cap or contains duplicates")
    write_once(OUT / "material-proposals.jsonl", proposals, jsonl=True)
    write_once(OUT / "proposal-freeze.json", {"count": len(proposals), "cap": 96, "new_official_training_candidates": 0,
        "categories": dict(Counter(x["verdict"] for x in proposals)), "reviewed": 0,
        "sha256": sha(OUT / "material-proposals.jsonl"),
        "selection_rule": "review once; stable judgment-ID order within each class; sixteen per class",
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in
                    [Z/'selected-train.jsonl', Z/'training-inputs.jsonl', AB/'decision-pairs.jsonl',
                     AB/'train-decision-review.jsonl', OUT/'cpu-source-contexts.jsonl']}})


def material_gate(proposals, reviews):
    by_id = {r["judgment_id"]: r for r in reviews}
    if len(by_id) != len(reviews) or set(by_id) != {p["judgment_id"] for p in proposals}:
        raise ValueError("Exactly one source-bound review per frozen proposal is required")
    accepted, shortages = [], {}
    for verdict in VERDICTS:
        eligible = []
        for p in sorted(proposals, key=lambda p: p["judgment_id"]):
            r = by_id[p["judgment_id"]]
            if p["verdict"] != verdict or not r.get("accepted"):
                continue
            if not r.get("reason") or not r.get("source_checked"):
                raise ValueError("Acceptance requires source checking and rationale")
            if r.get("corrected_target") is None:
                raise ValueError("Review must supply the exact reviewed target")
            target = r["corrected_target"]
            if target.get('verdict') != verdict or set(target) != {'verdict','facts','missing'}:
                raise ValueError('Reviewed target has a different category or schema')
            refs = {d['reference']:d for d in p['visible_documents']}
            quotes = r.get('source_quotes', [])
            if not quotes:
                raise ValueError('Every accepted row needs visible source quotes')
            for quote in quotes:
                if quote['reference'] not in refs or quote['quote'] not in refs[quote['reference']]['text']:
                    raise ValueError('Review quote not present verbatim in actual tool input')
            for fact in target['facts']:
                if len(fact) != 4 or not any(q['reference']==fact[3] for q in quotes):
                    raise ValueError('Every fact must cite a source-bound reviewed passage')
            if verdict == 'supported' and (len(target['facts']) < 2 or target['missing']):
                raise ValueError('Supported training examples require a complete two-edge chain')
            if verdict == 'missing_link' and not target['missing']:
                raise ValueError('Missing-link target must name the missing relation')
            q = dict(p, target=r["corrected_target"], review=r)
            eligible.append(q)
        shortages[verdict] = max(0, 16-len(eligible))
        accepted += eligible[:16]
    real = [p for p in accepted if p["verdict"] != "supported" and p["from_real_error_case"]]
    failure_cases = {p["case_id"] for p in real}
    passed = not any(shortages.values()) and len(real) >= 16 and len(failure_cases) >= 8
    return accepted, {"passed": passed, "selected": len(accepted), "required": 64,
        "shortages": shortages, "real_error_negative_rows": len(real),
        "real_error_question_count": len(failure_cases), "required_real_negative_rows": 16,
        "required_real_error_questions": 8, "gpu_authorized": False}


def finalize_material():
    proposals = rows(OUT/'material-proposals.jsonl')
    reviews = rows(OUT/'material-review-main.jsonl')
    selected, gate = material_gate(proposals, reviews)
    write_once(OUT/'material-gate.json', gate)
    write_once(OUT/'state.json', {"status": "awaiting_cpu_token_freeze" if gate['passed'] else
                                "stopped_material_shortfall", "gpu_started": False,
                                "gpu_authorized": False})
    if gate['passed']:
        write_once(OUT/'judgments-reviewed.jsonl', selected, jsonl=True)
    print(json.dumps(gate, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['propose', 'finalize-material'])
    a = p.parse_args()
    prepare_proposals() if a.command == 'propose' else finalize_material()
