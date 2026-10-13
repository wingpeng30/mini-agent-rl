"""Freeze the reviewed Goal AG repair set from existing Goal Z traces.

This helper does not sample or generate model output. It reuses the one-time
Goal AG supplement rollout and the already reviewed AC repair paths, checks
that answer-correction evidence is present in the actual saved final inputs,
and tokenizes only the frozen target answer actions.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import run_v133_goal_ag_failure_repair as ag
import run_v133_goal_q_search as q
import run_v133_goal_r_evidence_order_v2 as r
import run_v133_goal_z_multihop_sft as z


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ag-failure-repair"
SUPP = OUT / "supplemental"
TRACES = SUPP / "attempt-02/start-model-traces.jsonl"
POOL = SUPP / "candidate-pool.jsonl"
REPAIRS = OUT / "repair-material-final.jsonl"
FROZEN = OUT / "repair-material-final-ag.jsonl"
GATE = OUT / "repair-material-gate-ag.json"
REVIEW = OUT / "supplemental-repair-review-ag.jsonl"

# These four training questions had a wrong/incomplete final answer even though
# the answer-bearing evidence was present in the saved final model input. The
# query/bridge repair type remains the dominant selected material: the 12
# already lead-approved AC repairs are all retrieval/bridge corrections.
ANSWER_REPAIRS = {
    "musique:2hop__104341_411087": {
        "reason": "The actual input identifies Luis Villafañe as Puerto Rican and the pipeline as connecting Puerto Rico to the United States Virgin Islands. The frozen target names the destination island group, while the model answered only Puerto Rico.",
        "failure_class": "complete_evidence_but_answer_wrong",
    },
    "musique:2hop__407314_80178": {
        "reason": "The actual input identifies Justin Bieber as the Believe Tour performer and says Never Say Never features rap interludes from Jaden Smith. The model returned the lead performer instead of the other singer requested.",
        "failure_class": "complete_evidence_but_answer_wrong",
    },
    "musique:2hop__86360_64214": {
        "reason": "The actual input links I Need You to George Harrison and states that Something was written for Pattie Boyd. The model answered the songwriter rather than the requested dedicatee.",
        "failure_class": "complete_evidence_but_answer_wrong",
    },
    "2wiki:b2addc160bdc11eba7f7acde48001122": {
        "reason": "The actual input says Otto Donner was professor at the University of Helsinki. The model returned the job title but omitted the requested workplace.",
        "failure_class": "complete_evidence_but_answer_wrong",
    },
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()
            if x.strip()]


def write_once(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def main() -> None:
    for path in (TRACES, POOL, REPAIRS):
        if not path.is_file():
            raise FileNotFoundError(path)
    if any(path.exists() for path in (FROZEN, GATE, REVIEW)):
        raise FileExistsError("Goal AG supplemental repair freeze already exists; refusing overwrite")

    traces = read(TRACES)
    pool = read(POOL)
    original = read(REPAIRS)
    if len(traces) != 128 or len(pool) != 128 or len(original) != 12:
        raise RuntimeError(f"Expected 128 sampled rows, 128 frozen candidates and 12 AC repairs; got {len(traces)}/{len(pool)}/{len(original)}")
    trace_by_id = {str(x["case_id"]): x for x in traces}
    case_by_id = {str(x["id"]): x for x in pool}
    if len(trace_by_id) != 128 or len(case_by_id) != 128:
        raise RuntimeError("Supplement identifiers are not unique")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(q.p.BASE, local_files_only=True)
    tok = processor.tokenizer
    review_rows: list[dict] = []
    supplemental: list[dict] = []
    for case_id, audit in ANSWER_REPAIRS.items():
        trace_row = trace_by_id[case_id]
        case = case_by_id[case_id]
        steps = trace_row["trace"]
        if not steps or steps[-1].get("action") != "answer":
            raise RuntimeError(f"Expected saved final answer state for {case_id}")
        gold = str(case["answer"])
        model_answer = str(trace_row.get("answer") or "")
        if q.normalize_answer(model_answer) == q.normalize_answer(gold):
            raise RuntimeError(f"Selected supplement is not an actual answer failure: {case_id}")
        actual_prompt = tok.decode(steps[-1]["prompt_token_ids"], skip_special_tokens=False)
        visible = ag.visible_tool_documents(actual_prompt)
        visible_titles = {str(x.get("title")) for x in visible}
        support_docs = [d for d in case.get("docs", []) if d.get("support")]
        support_titles = sorted(str(d["title"]) for d in support_docs)
        missing = sorted(set(support_titles) - visible_titles)
        if missing:
            raise RuntimeError(f"Selected repair evidence absent from actual final input for {case_id}: {missing}")
        # The evidence text is drawn from the final token replay, not the full trace.
        visible_by_title = {str(d.get("title")): d for d in visible}
        evidence = [{"title": title, "document_id": visible_by_title[title].get("document_id"),
                     "input_start": visible_by_title[title].get("prompt_start"),
                     "input_end": visible_by_title[title].get("prompt_end"),
                     "text": visible_by_title[title]["text"]}
                    for title in support_titles]
        answer_ids, raw_answer = r.answer_target(processor, gold)
        if len(answer_ids) > 128:
            raise RuntimeError(f"Target action exceeds product output-token limit for {case_id}")
        final_step = steps[-1]
        action = {"action": "answer", "answer": gold, "raw": raw_answer,
                  "prompt_token_ids": final_step["prompt_token_ids"],
                  "prompt_attention_mask": final_step["prompt_attention_mask"],
                  "action_token_ids": answer_ids}
        item = {"case_id": case_id, "question": case["question"], "answer_target": gold,
                "original_answer": model_answer, "primary_error": "answer_target_binding",
                "primary_cause_category": audit["failure_class"], "eligible": True,
                "lead_review": audit["reason"], "actual_final_prompt": actual_prompt,
                "visible_documents": visible, "support_evidence": evidence,
                "insufficient_state": {"step": final_step.get("step"),
                    "prompt_token_ids": final_step["prompt_token_ids"],
                    "prompt_attention_mask": final_step["prompt_attention_mask"],
                    "incorrect_action_token_ids": final_step["action_token_ids"]},
                "repair_actions": [action],
                "repair_path_sha256": hashlib.sha256(json.dumps([action], sort_keys=True,
                    ensure_ascii=False).encode("utf-8")).hexdigest(),
                "model_identity": "goal-z-trajectory-seed42",
                "adapter_sha256": trace_row.get("adapter_sha256"),
                "corpus_sha256": trace_row.get("corpus_sha256"),
                "selection_sha256": hashlib.sha256(case_id.encode("utf-8")).hexdigest()}
        if item["adapter_sha256"] != "d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52":
            raise RuntimeError(f"Supplement trace adapter mismatch: {case_id}")
        supplemental.append(item)
        review_rows.append({"case_id": case_id, "question": case["question"],
            "model_answer": model_answer, "target": gold,
            "decision": "approved_answer_correction", "failure_class": audit["failure_class"],
            "actual_input_support_titles": support_titles,
            "evidence": evidence, "reason": audit["reason"],
            "selection_sha256": item["selection_sha256"]})

    # Add the dominant primary cause to the AC-approved examples without
    # changing their original review records.
    focus = "corpus_evidence_present_but_query_or_bridge_missed"
    merged = []
    for row in original:
        updated = dict(row)
        updated["primary_cause_category"] = focus
        updated["source_material"] = "AC lead-approved repair"
        merged.append(updated)
    # The answer corrections are selected in stable hash order, after verifying
    # each one against the same frozen evidence rubric.
    supplemental.sort(key=lambda x: x["selection_sha256"])
    merged.extend(supplemental)
    if len({x["case_id"] for x in merged}) != len(merged):
        raise RuntimeError("Duplicate case in combined repair material")
    focus_n = sum(x.get("primary_cause_category") == focus for x in merged)
    gate = {"reviewed_ac_repairs": len(original), "supplemental_reviewed": len(supplemental),
            "selected_repairs": len(merged), "primary_cause": focus,
            "primary_cause_repairs": focus_n,
            "primary_cause_fraction": focus_n / len(merged),
            "minimum_repairs": 16, "maximum_repairs": 32,
            "minimum_primary_cause_fraction": 0.5,
            "passed": 16 <= len(merged) <= 32 and focus_n * 2 >= len(merged),
            "supplement_candidate_pool_size": len(pool),
            "supplement_trace_count": len(traces),
            "confirmation_accessed": False,
            "source_hashes": {"ac_material": sha(REPAIRS), "supplement_traces": sha(TRACES),
                              "supplement_pool": sha(POOL)},
            "selection": "12 lead-approved query/bridge repairs, plus the four eligible answer-correction failures in stable SHA-256 case-ID order",
            "excluded_near_misses": {
                "musique:2hop__102777_757032": "The visible Serdar Denktaş passage only states a parent relationship to a former TRNC president; it does not explicitly establish his own country of origin.",
                "musique:2hop__128979_90736": "The retrieved Golden Nugget passage gives an opening date, not an explicit construction date.",
                "musique:2hop__161751_34844": "The wording and source distinguish abandoning efforts in December from evacuation by February 1943; the frozen label does not resolve that timing ambiguity.",
                "musique:2hop__776701_47102": "The frozen corpus lacks the Crazy Horse birthplace evidence; the result links the memorial, not his birthplace, to South Dakota.",
            }}
    if not gate["passed"]:
        raise RuntimeError(f"Repair material gate did not pass: {gate}")
    write_once(FROZEN, merged)
    write_once(REVIEW, review_rows)
    GATE.write_text(json.dumps(gate, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps({"gate": gate, "material": str(FROZEN), "review": str(REVIEW)},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
